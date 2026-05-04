# Libraries
import torch
import wandb
import PIL
import os
import argparse
import time
import matplotlib.pyplot as plt
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import matplotlib as mpl
from lightning.pytorch.strategies import DDPStrategy

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['figure.figsize'] = [7, 5]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.bbox'] = 'tight'

mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.titlesize'] = 'small'

mpl.rcParams['font.family'] = 'serif'

from utils.visualization import PlotRollout
from utils.miscellaneous import get_model, SpatialAnalysis, set_cpu_affinity_LUMI
from training.train import LightningTrainer
from utils.dataset import create_model_dataset, get_temporal_test_dataset_parameters, NUM_WATER_VARS, TemporalFloodDataset
from utils.miscellaneous import read_config
from utils.miscellaneous import get_model, set_cpu_affinity_LUMI
from training.train import LightningTrainer

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

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
               tags=["testing"]
               )

    wandb_logger = WandbLogger(log_model='all')

    config = wandb.config
    
    L.seed_everything(config.models['seed'])

    dataset_parameters = config.dataset_parameters
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    save_folder = config.get('save_folder', 'results')
    os.makedirs(save_folder, exist_ok=True)

    # Create test dataset
    _, _, test_dataset, scalers = create_model_dataset(
        scalers=scalers, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )

    test_graph_files = test_dataset.graph_files

    # info for testing dataset
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(config, config.temporal_dataset_parameters)

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
                        strategy=DDPStrategy() if num_GPUs > 1 else 'auto',
                        precision='bf16-mixed',
                        enable_progress_bar=True,
                        logger=wandb_logger)
    
    # Load the best model checkpoint
    plmodel = LightningTrainer.load_from_checkpoint(config.get('saved_model', None), **plmodule_kwargs)
    model = plmodel.model

    # Rollout error and time
    temporal_test_dataset = TemporalFloodDataset(test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
    
    test_dataloader = DataLoader(temporal_test_dataset, batch_size=trainer_options['batch_size'], shuffle=False, 
                                 num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))

    start_time = time.time()
    predicted_rollout = trainer.predict(plmodel, dataloaders=test_dataloader)
    prediction_times = time.time() - start_time
    prediction_times = prediction_times/len(temporal_test_dataset)
    predicted_rollout = [item for roll in predicted_rollout for item in roll]

    spatial_analyser = SpatialAnalysis(predicted_rollout, prediction_times, 
                                       test_dataset, **temporal_test_dataset_parameters)
    
    rollout_loss = spatial_analyser._get_rollout_loss(type_loss='MAE')
                                        
    print('test roll loss WD:',rollout_loss.mean(0)[0].item())
    print('test roll loss V:',rollout_loss.mean(0)[1:].mean().item())

    print(f'test CSI_005: {spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item()}')
    print(f'test CSI_03: {spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()}')

    wandb.log({"mean_prediction_times": prediction_times,
               "test roll loss WD":rollout_loss.mean(0)[0].item(),
               "test roll loss V":rollout_loss.mean(0)[1:].mean().item(),
               "test CSI_005": spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item(),
                "test CSI_03": spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()
                })

    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    plt.savefig("results/temp_CSI.png")
    img = PIL.Image.open("results/temp_CSI.png").convert("RGB")
    image = wandb.Image(img, caption="CSI scores")
    wandb.log({"CSI scores": image})

    sorted_ids = spatial_analyser.plot_loss_per_simulation(type_loss='MAE', ranking='combined', only_where_water=False, water_thresholds=[0.05, 0.3, 1])
    plt.savefig("results/temp_ranking.png")
    img = PIL.Image.open("results/temp_ranking.png").convert("RGB")
    image = wandb.Image(img, caption="Simulation ranking")
    wandb.log({"Simulation ranking": image}) 

    best_id = sorted_ids[0]
    worst_id = sorted_ids[-1]

    plot_times = [0,1,2,5,10]

    for id_dataset, name in zip([best_id, worst_id],['best', 'worst']):
        test_dataset.graph_files = [test_graph_files[id_dataset]]

        rollout_plotter = PlotRollout(plmodel, trainer, test_dataset, scalers=scalers, 
            type_loss='MAE', **temporal_test_dataset_parameters)
    
        rollout_plotter.mesh_scale_plot(scale=0)

        rollout_plotter.real_WD.kwargs['vmax'] = 2.5
        rollout_plotter.predicted_WD.kwargs['vmax'] = 2.5
        rollout_plotter.difference_WD.kwargs['vmax'] = 0.5
        rollout_plotter.difference_WD.kwargs['vmin'] = -0.5

        rollout_plotter.compare_h_rollout(plot_times, scale=None)
        plt.savefig("results/temp_summary.png")
        img = PIL.Image.open("results/temp_summary.png").convert("RGB")
        image = wandb.Image(img, caption=f"id: {id_dataset}")
        wandb.log({f"water depths ({name})": image})
        
        rollout_plotter.difference_V.kwargs['vmax'] = 0.1
        rollout_plotter.difference_V.kwargs['vmin'] = -0.1

        rollout_plotter.compare_v_rollout(plot_times, scale=None, logscale=True)
        plt.savefig("results/temp_summary.png")
        img = PIL.Image.open("results/temp_summary.png").convert("RGB")
        image = wandb.Image(img, caption=f"id: {id_dataset}")
        wandb.log({f"discharges ({name})": image})

        rollout_plotter.compare_FAT(water_threshold=0.05, scale=0)
        plt.savefig("results/temp_summary.png")
        img = PIL.Image.open("results/temp_summary.png").convert("RGB")
        image = wandb.Image(img, caption=f"id: {id_dataset}")
        wandb.log({f"FAT ({name})": image})
    
    print('Testing finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config_test.yaml', help='Path to the config file')
    main(config_file=parser.parse_args().config)