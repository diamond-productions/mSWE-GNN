# Libraries
import torch
import wandb
import PIL
import os
import time
import matplotlib.pyplot as plt
import argparse
import lightning as L
from copy import copy
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from utils.dataset import create_model_dataset, TemporalFloodDataset
from utils.dataset import get_temporal_test_dataset_parameters, NUM_WATER_VARS
from utils.miscellaneous import read_config
from utils.visualization import PlotRollout
from utils.miscellaneous import get_model, SpatialAnalysis, set_cpu_affinity_LUMI
from training.train import LightningTrainer, DataModule, CurriculumLearning

torch.set_float32_matmul_precision('high')

def main(config_file='config.yaml'):
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
               #tags=["mass_val"]
               )

    wandb_logger = WandbLogger(log_model='all')

    config = wandb.config
    
    L.seed_everything(config.models['seed'])

    dataset_parameters = config.dataset_parameters
    scalers = copy(config.scalers)
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    train_dataset, val_dataset, test_dataset, scalers = create_model_dataset(
        scalers=scalers, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )
    
    test_graph_files = test_dataset.graph_files

    temporal_dataset_parameters = config.temporal_dataset_parameters
    temporal_train_dataset = TemporalFloodDataset(train_dataset, **temporal_dataset_parameters)

    if temporal_dataset_parameters.get('save_on_gpu', False):
        temporal_train_dataset = temporal_train_dataset._save_on_gpu()
    # else:
    #     temporal_train_dataset.cache_files()

    previous_t = temporal_dataset_parameters['previous_t']
    future_t = temporal_dataset_parameters['future_t']
    temporal_res = dataset_parameters['temporal_res']
    max_rollout_steps = temporal_dataset_parameters['rollout_steps']
    time_stop = test_dataset[0].WD.shape[1]

    num_node_features = NUM_WATER_VARS*previous_t + sum(selected_node_features.values())
    num_edge_features = sum(selected_edge_features.values())
    
    num_GPUs = torch.cuda.device_count()

    if local_rank == 0:
        print('Number of training simulations:\t', len(train_dataset))
        print('Number of training samples:\t', len(temporal_train_dataset))
        print('Number of node features:\t', temporal_train_dataset[0].x.shape[-1])
        print('Number of rollout steps:\t', temporal_train_dataset[0].y.shape[-1])
        print('Temporal resolution:\t', temporal_res, 'min')
        print('Number of GPUs:\t', num_GPUs)

    model_parameters = config.models
    model_type = model_parameters.pop('model_type')

    if model_type == 'MSGNN':
        num_scales = train_dataset[0].mesh.num_meshes
        model_parameters['num_scales'] = num_scales

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=previous_t,
        future_t=future_t,
        **model_parameters)

    trainer_options = config.trainer_options
    type_loss = trainer_options['type_loss']
    lr_info = config['lr_info']

    # info for val and testing dataset
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(config, temporal_dataset_parameters)
    temporal_val_dataset = TemporalFloodDataset(val_dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
    if temporal_dataset_parameters.get('save_on_gpu', False):
        temporal_val_dataset = temporal_val_dataset._save_on_gpu()
    else:
        temporal_val_dataset.cache_files()
    
    pldatamodule = DataModule(temporal_train_dataset, temporal_val_dataset,
                              batch_size=trainer_options['batch_size'])
    
    plmodel = LightningTrainer(model, lr_info, trainer_options, 
                               temporal_test_dataset_parameters, pldatamodule)

    # Number of parameters
    total_parameters = sum(p.numel() for p in model.parameters())

    wandb.log({"total parameters": total_parameters})

    # Training
    # Define callbacks
    monitor_metric = "num_valid_simulations" if trainer_options.get('validation_method', 'normal') == 'mass_val' else "val_loss_CSI"
    mode = 'min' if monitor_metric == "val_loss_CSI" else 'max'

    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/models', monitor=monitor_metric,
                                        mode=mode, save_top_k=2, save_last=True)
    early_stopping = EarlyStopping(monitor_metric, mode=mode, patience=trainer_options['patience'])
    curriculum_callback = CurriculumLearning(max_rollout_steps, mode=trainer_options['curriculum_mode'], patience=5)
    wandb_logger.watch(model, log="all", log_graph=False)

    # Load trained model
    plmodule_kwargs = {'model': model, 
                       'lr_info': lr_info, 
                       'trainer_options': trainer_options, 
                       'temporal_test_dataset_parameters': temporal_test_dataset_parameters,
                       'datamodule': pldatamodule
                   }

    accelerator="gpu" if torch.cuda.is_available() else 'auto'

    # Define trainer
    trainer = L.Trainer(accelerator=accelerator, 
                        devices=num_GPUs if num_GPUs > 0 else 'auto',
                        strategy=DDPStrategy() if num_GPUs > 1 else 'auto',
                        max_epochs=trainer_options['max_epochs'],
                        gradient_clip_val=2, 
                        # log_every_n_steps=10,
                        # accumulate_grad_batches=1,
                        # profiler="simple",
                        precision='bf16-mixed',
                        enable_progress_bar=True,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, 
                                curriculum_callback, 
                                early_stopping, 
                                ])
    
    # Train and get trained model
    trainer.fit(plmodel, pldatamodule, ckpt_path=config.get('saved_model', None))

    # Load the best model checkpoint
    plmodel = LightningTrainer.load_from_checkpoint(checkpoint_callback.best_model_path, **plmodule_kwargs)
    model = plmodel.model

    # validate with trained model
    trainer.validate(plmodel, pldatamodule)

    # Rollout error and time
    temporal_test_dataset = TemporalFloodDataset(test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
    
    test_dataloader = DataLoader(temporal_test_dataset, batch_size=trainer_options['batch_size'], shuffle=False, 
                                 num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))

    start_time = time.time()
    predicted_rollout = trainer.predict(plmodel, dataloaders=test_dataloader)
    prediction_times = time.time() - start_time
    prediction_times = prediction_times/len(temporal_test_dataset)
    predicted_rollout = [item for roll in predicted_rollout for item in roll]

    wandb.log({"mean_prediction_times": prediction_times})

    spatial_analyser = SpatialAnalysis(predicted_rollout, prediction_times, test_dataset, **temporal_test_dataset_parameters)
    
    rollout_loss = spatial_analyser._get_rollout_loss(type_loss=type_loss)
                                        
    print('test roll loss WD:',rollout_loss.mean(0)[0].item())
    print('test roll loss V:',rollout_loss.mean(0)[1:].mean().item())

    print(f'test CSI_005: {spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item()}')
    print(f'test CSI_03: {spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()}')

    wandb.log({"test roll loss WD":rollout_loss.mean(0)[0].item(),
               "test roll loss V":rollout_loss.mean(0)[1:].mean().item(),
               "test CSI_005": spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item(),
                "test CSI_03": spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()
                })

    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    plt.savefig("results/temp_CSI.png")
    img = PIL.Image.open("results/temp_CSI.png").convert("RGB")
    image = wandb.Image(img, caption="CSI scores")
    wandb.log({"CSI scores": image})

    if partition == 'standard-g':
        CSIs = torch.stack([spatial_analyser._get_CSI(wt) for wt in [0.05, 0.3, 1]], 1).nanmean(2)
        loss_CSI = (1-CSIs.mean(1))*rollout_loss.mean(1)
        sorted_ids = loss_CSI.cpu().numpy().argsort()
    else:
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
            type_loss=type_loss, **temporal_test_dataset_parameters)
    
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
    
    print('Training and testing finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    main(config_file=parser.parse_args().config)
