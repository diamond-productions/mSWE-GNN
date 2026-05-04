# Libraries
import torch
import wandb
import PIL
import pickle
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from copy import copy
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import WandbLogger
import matplotlib as mpl

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['figure.figsize'] = [7, 5]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.bbox'] = 'tight'

mpl.rcParams['font.size'] = 19
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['font.family'] = 'serif'

from utils.dataset import create_model_dataset, get_temporal_test_dataset_parameters
from utils.dataset import NUM_WATER_VARS, create_prob_test_dataset, add_BC_to_data
from utils.miscellaneous import read_config
from utils.visualization import ProbabilisticSpatialAnalysis, plot_percentage_plausible_volumes_vs_ARME, plot_valid_runs_breach_distribution, plot_breach_distribution_and_quantiles
from utils.visualization import plot_percentiles, plot_percentile_diff, plot_ARME_thresholds_analysis, plot_WD_max_probability, analyze_training_vs_good_tests
from utils.miscellaneous import get_model, stack_rollout_different_BC, WD_to_FAT, set_cpu_affinity_LUMI
from database.dhydro_utils import generate_realistic_hydrograph
from training.train import LightningTrainer

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

import warnings
import psutil
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(config_file='config_test.yaml'):
    # Read configuration file with parameters
    cfg = read_config(config_file)
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    partition = os.environ.get("SLURM_JOB_PARTITION", None)
    if partition == 'standard-g':
        set_cpu_affinity_LUMI(rank, local_rank)
    
    wandb.finish()
    wandb.init(config=cfg, 
               project="mSWEGNN", 
                mode='online', # online, offline, disabled
               tags=["prob_testing"]
               )

    wandb_logger = WandbLogger(log_model=True)

    config = wandb.config

    L.seed_everything(config.models['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parameters = config.dataset_parameters
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    save_folder = config.get('save_folder', 'results')
    os.makedirs(save_folder, exist_ok=True)

    # Create scalers
    train_dataset, _, test_dataset, scalers = create_model_dataset(
        scalers=scalers, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )
    
    with_polygon_mesh = dataset_parameters.get('dataset_folder').split('/')[-1] == 'polygon_meshes'

    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(config, config.temporal_dataset_parameters)

    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']
    previous_t = temporal_test_dataset_parameters['previous_t']
    future_t = temporal_test_dataset_parameters['future_t']
    temporal_res = dataset_parameters['temporal_res']
    model_parameters = config.models
    model_type = model_parameters.pop('model_type')

    num_node_features = NUM_WATER_VARS*previous_t + sum(selected_node_features.values())
    num_edge_features = sum(selected_edge_features.values())

    print('Temporal resolution:\t', temporal_res, 'min')

    if model_type == 'MSGNN':
        num_scales = test_dataset[0].mesh.num_meshes
        model_parameters['num_scales'] = num_scales

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=previous_t,
        future_t=future_t,
        **model_parameters)

    trainer_options = config.trainer_options
    lr_info = config['lr_info']

    # info for testing dataset
    plmodel = LightningTrainer(model, lr_info, trainer_options, temporal_test_dataset_parameters)

    # Load trained model
    plmodule_kwargs = {'model': model, 
                       'lr_info': lr_info, 
                       'trainer_options': trainer_options, 
                       'temporal_test_dataset_parameters': temporal_test_dataset_parameters}

    num_GPUs = torch.cuda.device_count()
    print('Number of GPUs:\t', num_GPUs)

    accelerator="gpu" if torch.cuda.is_available() else 'auto'

    # Define trainer
    trainer = L.Trainer(accelerator=accelerator, 
                        devices=num_GPUs if num_GPUs > 0 else 'auto',
                        strategy='ddp' if num_GPUs > 1 else 'auto',
                        enable_progress_bar=False,
                        precision='bf16-mixed',
                        logger=wandb_logger)
    
    # Load the best model checkpoint
    plmodel = LightningTrainer.load_from_checkpoint(config['saved_model'], **plmodule_kwargs)
    model = plmodel.model
    
    ####################################################################################
    save_folder = config.get('save_folder', 'results')
    # save_folder = os.path.join(save_folder, 'dk43')
    os.makedirs(save_folder, exist_ok=True)

    breach_id = config.get('breach_location', 'all')
    return_period = config.get('return_period', None)

    save_folder = os.path.join(config.get('save_folder', 'results'), breach_id)
    os.makedirs(save_folder, exist_ok=True)

    case_study = 'dk41'
    dataset_folder = f"database/raw_datasets_{case_study}"

    dijkpalen_file = os.path.join(dataset_folder, "dijkpalen.gpkg")
    dijkpalen = gpd.read_file(dijkpalen_file)
    dijkpalen.sort_values(by='CODE', inplace=True)
    dijkpalen = dijkpalen[dijkpalen.WS_DIJKRIN == 'DR41']
    dijkpalen_coords = np.array([[geom.x, geom.y] for geom in dijkpalen.geometry])

    if breach_id != 'all':
        if isinstance(breach_id, list):
            breach_index = np.where(dijkpalen.CODE.isin([b + '.' for b in breach_id]))[0]
        else:
            breach_index = np.where(dijkpalen.CODE == (breach_id + '.'))[0]
        BC_loc = np.array([(geom.x, geom.y) for geom in dijkpalen.iloc[breach_index].geometry])
    else:
        # selected_dijkpalen = dijkpalen[dijkpalen.CODE.str.startswith(('HD', 'ND'))][10::21]
        selected_dijkpalen_HD = dijkpalen[dijkpalen.CODE.str.startswith(('HD'))][::9]
        selected_dijkpalen_ND = dijkpalen[dijkpalen.CODE.str.startswith(('ND'))][::8][::-1]

        selected_dijkpalen = pd.concat([selected_dijkpalen_HD, selected_dijkpalen_ND])
        # dijkpalen = dijkpalen[dijkpalen.CODE.str.startswith(('HD'))][::8]
        BC_loc = np.array([(geom.x, geom.y) for geom in selected_dijkpalen.geometry])

    # point_id = 2
    # BC_loc = points[[point_id]]
    # BC_loc = portion[::5]

    time_stop = test_dataset[0].WD.shape[1]

    # Print available CPU RAM after deletion
    print("Used CPU RAM: ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), " GB")

    print("Building meshes...")
    new_mesh_file = os.path.join(save_folder, "base_meshes.pkl")
    mesh_file = os.path.join(dataset_folder, "base_meshes.pkl")

    if not os.path.exists(new_mesh_file):
        base_datas = []
        for bc_loc in tqdm(BC_loc, total=len(BC_loc)):
            j = 1
            while True:
                with open(mesh_file, 'rb') as f:
                    meshes = pickle.load(f)
                    if with_polygon_mesh:
                        meshes.pop(2)
                        meshes.pop(2)
                        meshes.pop(2)
                    else:
                        meshes.pop(0)
                        meshes.pop(0)
                base_data = add_BC_to_data(copy(meshes), np.array([bc_loc]), np.zeros(time_stop), type_BC=2)
                if base_data:
                    _, closest_idx = cKDTree(dijkpalen_coords).query(bc_loc)
                    base_data.CODE = dijkpalen.iloc[closest_idx].CODE
                    base_datas.append(base_data)
                    break
                else:
                    _, closest_idx = cKDTree(dijkpalen_coords).query(bc_loc, k=j+1)
                    closest_dijkpalen = dijkpalen.iloc[closest_idx[j:]]
                    bc_loc = np.array([(geom.x, geom.y) for geom in closest_dijkpalen.geometry]).squeeze()
                    j += 1

        mesh_counts = np.unique_counts([data.DEM.shape for data in base_datas])
        best_cell_num = mesh_counts[0][mesh_counts[1].argmax()]

        base_datas = [data for data in base_datas if data.DEM.shape[0] == best_cell_num]
            
        with open(new_mesh_file, 'wb') as f:
            pickle.dump(base_datas, f)
    else:
        with open(new_mesh_file, 'rb') as f:
            base_datas = pickle.load(f)
        
    with open(mesh_file, 'rb') as f:
        meshes = pickle.load(f)
        if with_polygon_mesh:
            meshes.pop(2)
            meshes.pop(2)
            meshes.pop(2)
        else:
            meshes.pop(0)
            meshes.pop(0)

    num_breaches = len(base_datas)
    print(f'Number of breach locations: {num_breaches}')

    ID_locations = ['ND234', 'HD073', 'ND111']
    return_periods = [10000, 1000, 100]

    id_return_period_to_index = {(ID_locations[i // 3], return_periods[i % 3]): i for i in range(len(test_dataset))}
    test_dataset_id = id_return_period_to_index.get((breach_id, return_period), None)

    # generate BCs
    if return_period is not None and breach_id != 'all':
        test_BC = pd.read_csv(os.path.join('database', "raw_datasets_dk41", "Probabilistic breach outflow", "processed", f"{breach_id}_T{return_period}.csv"), header=None)
        all_hydrographs = test_BC.values[:,1:]
        num_scenarios_per_location = len(all_hydrographs)
        num_scenarios = num_breaches * num_scenarios_per_location
        # plt.plot(all_hydrographs.T)
    else:
        num_scenarios_per_location = config.get('num_scenarios_per_location', 5)
        num_scenarios = num_breaches * num_scenarios_per_location

        # generate BCs
        peak_value=(6.2, 0.4) # loc, scale of the lognormal distribution
        peak_value=(100, 1500) # uniform distribution (change below)
        min_discharge=0
        param1=(1, 0.4)
        param2=(0., 0.2)
        param3=(5,15)
        param4=0.6
        x_window=(0.2, 1.6)

        all_hydrographs = []
        for i in range(num_scenarios):
            np.random.seed(i)

            hydrograph_time_series = generate_realistic_hydrograph(time_stop-1, 1, 
                                                                    np.random.uniform(*peak_value),
                                                                    min_discharge, 
                                                                    np.random.lognormal(*param1), 
                                                                    np.random.uniform(*param2), 
                                                                    np.random.uniform(*param3), 
                                                                    param4,
                                                                    x_window)
            all_hydrographs.append(hydrograph_time_series[1])
            # plt.plot(hydrograph_time_series[1])

        all_hydrographs = np.array(all_hydrographs)
        all_hydrographs[all_hydrographs < 1e-10] = 0

    if num_GPUs >= 1:
        print("Testing and saving results...")
        # create prob_dataset
        for k in tqdm(range(len(base_datas))):
            prob_test_dataset = create_prob_test_dataset([base_datas[k]], all_hydrographs[num_scenarios_per_location*k:num_scenarios_per_location*(k+1)], 
                                                        for_execution=True, temporal_res=temporal_res, scalers=scalers,
                                                        **temporal_test_dataset_parameters, **selected_node_features, **selected_edge_features)
            
            batch_size = trainer_options['batch_size']

            prob_test_dataloader = DataLoader(prob_test_dataset, batch_size=batch_size, shuffle=False)

            start_time = time.time()
            predicted_rollout = trainer.predict(plmodel, dataloaders=prob_test_dataloader)
            prediction_time = time.time() - start_time
            predicted_rollout = [item for roll in predicted_rollout for item in roll]
            batch_pred = stack_rollout_different_BC(predicted_rollout, prob_test_dataset, scale=0)

            wandb.log({'prediction_time_location': prediction_time})
                        
            FAT_005 = torch.stack([WD_to_FAT(data[:, 0], temporal_res, 0.05, time_start) for data in batch_pred])
            FAT_03 = torch.stack([WD_to_FAT(data[:, 0], temporal_res, 0.3, time_start) for data in batch_pred])
            WD_max = torch.stack([data.max(2).values[:, 0] for data in batch_pred])
            V_max = torch.stack([data.max(2).values[:, 1] for data in batch_pred])
            WD_end = torch.stack([data[:, 0, -1] for data in batch_pred])

            variables_array = torch.stack([FAT_005, FAT_03, WD_max, V_max, WD_end], dim=2).cpu().numpy()
            volumes_array = torch.einsum('snt,n->st', batch_pred[:, :, 0].to(torch.float32),
                                        torch.as_tensor(meshes[-1].face_area, device=batch_pred.device, dtype=torch.float32)).cpu().numpy()

            if local_rank == 0:
                np.save(os.path.join(save_folder, f"prediction_{k}.npy"), variables_array)
                np.save(os.path.join(save_folder, f"volumes_{k}.npy"), volumes_array)

                df = pd.DataFrame([{'breach_id': base_datas[k].CODE, 'total_computation_time[s]': prediction_time}])
                df.to_csv(os.path.join(save_folder, "prediction_times.csv"), mode='a', sep=',', index=False,
                        header=not os.path.exists(os.path.join(save_folder, "prediction_times.csv")))

    # Print available CPU RAM
    print("Used CPU RAM: ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), " GB")

    print("Loading predictions...")
    # load all predictions
    ensemble_selected_prediction = []
    for i in range(num_breaches):
        file_path = os.path.join(save_folder, f"prediction_{i}.npy")
        if os.path.exists(file_path):
            ensemble_selected_prediction.append(np.load(file_path, mmap_mode='r'))
    ensemble_selected_prediction = np.concatenate(ensemble_selected_prediction, 0)
    print("Predictions shape: ", ensemble_selected_prediction.shape)

    # Print available CPU RAM
    print("Used CPU RAM: ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), " GB")

    # load all volumes
    ensemble_predicted_volumes = []
    for i in range(num_breaches):
        file_path = os.path.join(save_folder, f"volumes_{i}.npy")
        if os.path.exists(file_path):
            ensemble_predicted_volumes.append(np.load(file_path, mmap_mode='r'))
    ensemble_predicted_volumes = np.concatenate(ensemble_predicted_volumes, 0)
    print("Predicted volumes shape: ", ensemble_predicted_volumes.shape)

    prob_test_dataset = create_prob_test_dataset(base_datas, all_hydrographs, for_execution=False, temporal_res=temporal_res, scalers=scalers,
                         **temporal_test_dataset_parameters, **selected_node_features, **selected_edge_features)
    print("Probabilistic test dataset length: ", len(prob_test_dataset))

    # Print available CPU RAM
    print("Used CPU RAM: ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), " GB")

    fig = plot_breach_distribution_and_quantiles(ensemble_predicted_volumes, prob_test_dataset, num_breach_groups=10, 
                                                q_ranges=[0, 5, 25, 50, 75, 95, 100], max_ARME=2, show_percentage_runs=True, time_start=time_start)

    breach_distribution_file = os.path.join(save_folder, "breach_distribution_quantiles.png")
    plt.savefig(breach_distribution_file)
    img = PIL.Image.open(breach_distribution_file).convert("RGB")
    image = wandb.Image(img, caption="breach distribution quantiles")
    wandb.log({"breach distribution quantiles": image})
    plt.close(fig)

    base_DEM = base_datas[0].DEM if hasattr(base_datas[0], 'DEM') else None
    base_mesh = meshes[-1]
    gdf_mesh = meshes[0]

    prob_analyser = ProbabilisticSpatialAnalysis(ensemble_selected_prediction, ensemble_predicted_volumes, prob_test_dataset,
                                                    scalers, base_DEM, base_mesh, **temporal_test_dataset_parameters)
        
    ARME_threshold = trainer_options.get("ARME_threshold", 0.25)
    plausible_runs = prob_analyser.get_plausible_runs(ARME_threshold=ARME_threshold)
    plausible_runs_per_location = np.bincount(plausible_runs // num_scenarios_per_location, minlength=num_breaches)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 2]})

    prob_analyser.plot_runs_per_threshold(with_hist=True, ARME_thresholds=np.linspace(0, 1.5, 50), ax=axs[0],
                                                bins=20, alpha=0.4, color='b', edgecolor='k', linewidth=0.5)
    # axs[0].axvline(x=0.2, color='g', linestyle='--', linewidth=1.5)
    axs[0].axvline(x=ARME_threshold, color='k', linestyle='--', linewidth=1.5)
    # axs[0].axvline(x=1, color='r', linestyle='--', linewidth=1.5)
    axs[0].set_ylim(0, len(prob_test_dataset)*1.05)

    plot_valid_runs_breach_distribution(base_datas, plausible_runs_per_location/num_scenarios_per_location*100, 
                                        cmap='YlGn', edgecolor='k', linewidth=0.3, ax=axs[1])
    axs[1].set_aspect('auto')

    panel_labels = [f'({chr(97+i)})' for i in range(len(axs))]
    for i, label in enumerate(panel_labels):
        axs[i].text(0.02, 0.98, label, transform=axs[i].transAxes,
                    fontsize=20, va='top', ha='left')
        
    plt.tight_layout()

    run_thresh_file = os.path.join(save_folder, f"valid_runs_breach_distribution lt {ARME_threshold}.png")
    plt.savefig(run_thresh_file)
    img = PIL.Image.open(run_thresh_file).convert("RGB")
    image = wandb.Image(img, caption=f"valid_runs_breach_distribution lt {ARME_threshold}")
    wandb.log({"valid_runs_breach_distribution": image})
    plt.close(fig)

    plot_percentage_plausible_volumes_vs_ARME(prob_analyser.ARME, prob_test_dataset, ARME_thresholds=[1.0, 0.8, 0.6, 0.4, 0.2])

    perc_plausible_file = os.path.join(save_folder, f"percentage_plausible_volumes_vs_ARME.png")
    plt.savefig(perc_plausible_file)
    img = PIL.Image.open(perc_plausible_file).convert("RGB")
    image = wandb.Image(img, caption="percentage_plausible_volumes_vs_ARME")
    wandb.log({"percentage_plausible_volumes_vs_ARME": image})
    plt.close(fig)

    fig, axs = plot_ARME_thresholds_analysis(prob_analyser, ensemble_selected_prediction, ensemble_predicted_volumes, prob_test_dataset, scalers, 
                                 temporal_test_dataset_parameters, ARME_thresholds=[0.2, 0.4, 0.6, 1, 2])

    ARME_file = os.path.join(save_folder, 'ARME_thresholds_analysis.png')
    plt.savefig(ARME_file, dpi=300, bbox_inches='tight')
    img = PIL.Image.open(ARME_file).convert("RGB")
    image = wandb.Image(img, caption="ARME thresholds analysis")
    wandb.log({"ARME thresholds analysis": image})
    plt.close(fig)
    
    train_coords = np.stack([data.mesh.face_xy[data.node_BC] for data in train_dataset])
    train_discharges = np.concatenate([data.BC[data.type_BC == 2] * data.edge_BC_length[data.type_BC == 2] for data in train_dataset])

    fig = analyze_training_vs_good_tests(train_discharges, train_coords, all_hydrographs[:,:time_stop], BC_loc,
                                         prob_analyser.ARME, gdf_mesh, ARME_threshold=ARME_threshold, check_bads=True, k_neighbors=3)
    training_vs_good_tests_file = os.path.join(save_folder, f"training_vs_good_tests.png")
    plt.savefig(training_vs_good_tests_file)
    img = PIL.Image.open(training_vs_good_tests_file).convert("RGB")
    image = wandb.Image(img, caption=f"training_vs_good_tests")
    wandb.log({"training_vs_good_tests": image})
    plt.close(fig)

    # Only selected runs    
    mask_volume = prob_analyser.ARME < ARME_threshold

    selected_prob_dataset = [prob_test_dataset[i] for i in np.where(mask_volume)[0]]
    prob_analyser = ProbabilisticSpatialAnalysis(ensemble_selected_prediction[mask_volume], ensemble_predicted_volumes[mask_volume],
                                                selected_prob_dataset, scalers, base_DEM, base_mesh, **temporal_test_dataset_parameters)

    # plot percentiles WD
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    if test_dataset_id is not None:
        fig, axs = plot_percentile_diff(ensemble_selected_prediction, test_dataset[test_dataset_id], quantiles=quantiles, 
                                    variable='WD_max', water_threshold=0.05, temporal_res=temporal_res, diff_value=0.5, vmax=3, **prob_analyser.default_plot_kwargs)
    else:
        fig, axs = plot_percentiles(ensemble_selected_prediction, quantiles=quantiles, variable='WD_max', 
                                    temporal_res=temporal_res, **prob_analyser.default_plot_kwargs)
        
    WD_quantiles_file = os.path.join(save_folder, "WD_max_quantiles.png")
    plt.savefig(WD_quantiles_file)
    img = PIL.Image.open(WD_quantiles_file).convert("RGB")
    image = wandb.Image(img, caption="WD max quantiles")
    wandb.log({"WD max quantiles": image})
    plt.close(fig)

    # plot percentiles V
    if test_dataset_id is not None:
        fig, axs = plot_percentile_diff(ensemble_selected_prediction, test_dataset[test_dataset_id], quantiles=quantiles, 
                                    variable='V_max', water_threshold=0.05, temporal_res=temporal_res, diff_value=0.5, vmax=3, **prob_analyser.default_plot_kwargs)
    else:
        fig, axs = plot_percentiles(ensemble_selected_prediction, quantiles=quantiles, variable='V_max', 
                                    temporal_res=temporal_res, **prob_analyser.default_plot_kwargs)

    V_quantiles_file = os.path.join(save_folder, "V_max_quantiles.png")
    plt.savefig(V_quantiles_file)
    img = PIL.Image.open(V_quantiles_file).convert("RGB")
    image = wandb.Image(img, caption="V max quantiles")
    wandb.log({"V max quantiles": image})
    plt.close(fig)

    fig, axs = plot_WD_max_probability(ensemble_selected_prediction, wd_thresholds=[0.05, 0.3, 1], **prob_analyser.default_plot_kwargs)

    WD_max_prob_file = os.path.join(save_folder, "WD_max_prob.png")
    plt.savefig(WD_max_prob_file)
    img = PIL.Image.open(WD_max_prob_file).convert("RGB")
    image = wandb.Image(img, caption="WD max probability")
    wandb.log({"WD max probability": image})
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config_test.yaml', help='Path to the config file')
    main(config_file=parser.parse_args().config)