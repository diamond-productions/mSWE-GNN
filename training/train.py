# Libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import lightning as L
import os
from lightning.pytorch.callbacks import Callback
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

from training.loss import loss_function, conservation_loss
from utils.miscellaneous import get_rollout_loss, get_CSI_rollout, correct_rollout_shape_future_t, calculate_volumes, get_average_relative_mass_error
from utils.dataset import use_prediction, apply_boundary_condition, NUM_WATER_VARS
num_GPUs = torch.cuda.device_count()
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
sync_dist = True if num_GPUs >1 else False
# sync_dist = False

def adapt_batch_training(batch):
    """Normalise batch attributes to reduce conditional branching during training.

    Args:
        batch (Batch): raw batched graph from DataLoader

    Returns:
        Batch: modified batch with flattened scalars and corrected node_BC indices
    """
    assert isinstance(batch, Batch), "This function requires a torch_geometric.data.batch.Batch object as input"
    if batch.temporal_res.size(0) > 0:
        temp = batch.clone()
        temp.node_BC = torch.cat([temp.ptr[i]+temp[i].node_BC for i in range(temp.num_graphs)])
        temp.temporal_res = temp.temporal_res[0]
        temp.previous_t = temp.previous_t[0]
        temp.future_t = temp.future_t[0]
        if 'edge_ptr' in temp.keys():
            update_batch_multiscale(temp)    
            temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.node_ptr[:,0] <= node, node <= temp.node_ptr[:,-1]))[0] 
                                                                        for node in temp.node_BC.cpu()])
        else:
            temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.ptr[0] <= node, node <= temp.ptr[-1]))[0] 
                                                                        for node in temp.node_BC.cpu()])
    return temp

def _accumulate_ptr(ptr_rows):
    """Offset each pointer row by the cumulative max of all preceding rows.

    Args:
        ptr_rows (Tensor): pointer rows stacked per graph

    Returns:
        Tensor: accumulated pointer rows with corrected offsets
    """
    result = [ptr_rows[0]]
    for row in ptr_rows[1:]:
        result.append(row + result[-1].max())
    return torch.stack(result)

def update_batch_multiscale(batch):
    """Rebuild multiscale edge and node pointers for a batched MSGNN graph.

    Args:
        batch (Batch): batched graph with edge_ptr, intra_edge_ptr, and node_ptr attributes
    """
    batch_edge_ptr = batch.edge_ptr.reshape(batch.num_graphs, -1)
    batch_intra_edge_ptr = batch.intra_edge_ptr.reshape(batch.num_graphs, -1)
    batch_node_ptr = batch.node_ptr.reshape(batch.num_graphs, -1)
    num_scales = batch_intra_edge_ptr.shape[1]

    updated_batch_edge_ptr = _accumulate_ptr(batch_edge_ptr)
    updated_batch_intra_edge_ptr = _accumulate_ptr(batch_intra_edge_ptr)
    updated_batch_node_ptr = _accumulate_ptr(batch_node_ptr)

    intra_mesh_edge_index = [torch.cat([batch.intra_mesh_edge_index[:,j[0]:j[1]] for j in updated_batch_intra_edge_ptr[:,i:i+2]], 1)
                             for i in range(num_scales-1)]
    edge_index = [torch.cat([batch.edge_index[:,j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]], 1) for i in range(num_scales)]
    edge_attr = [torch.cat([batch.edge_attr[j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]]) for i in range(num_scales)]

    batch.node_ptr = updated_batch_node_ptr
    batch.edge_index = torch.cat(edge_index, 1)
    batch.edge_attr = torch.cat(edge_attr)
    batch.edge_ptr = torch.tensor(np.cumsum([0] + [edge.shape[1] for edge in edge_index]), device=batch.edge_attr.device).long()
    batch.intra_edge_ptr = torch.tensor(np.cumsum([0] + [edge.shape[1] for edge in intra_mesh_edge_index]), device=batch.edge_attr.device).long()
    batch.intra_mesh_edge_index = torch.cat(intra_mesh_edge_index,1)

@torch.no_grad()
def rollout_test(model, batch):
    """Run autoregressive rollout inference and return full predicted trajectory.

    Args:
        model (nn.Module): trained GNN model
        batch (Batch): single or multiple graphs stacked in a batched fashion

    Returns:
        Tensor: predicted rollout with shape corrected for future_t
    """
    if isinstance(batch, Batch):
        temp = adapt_batch_training(batch)
    else:
        temp = batch.clone()
        
    dynamic_vars = model.previous_t*NUM_WATER_VARS
    assert temp.x.shape[-1] >= dynamic_vars, "The number of dynamic variables is greater than the number of node features"
    final_step = batch.y.shape[-1]
    predicted_rollout = []

    for time_step in range(final_step):
        temp.x[:,-dynamic_vars:] = apply_boundary_condition(temp.x[:,-dynamic_vars:], 
                                                            temp.BC[:,:,time_step], 
                                                            temp.node_BC, type_BC=temp.type_BC)
        pred = model(temp)
        temp.x = use_prediction(temp.x, pred, model.previous_t, model.future_t)
        predicted_rollout.append(pred)
    
    return correct_rollout_shape_future_t(torch.stack(predicted_rollout, -1))

def force_mass_conservation(preds, BC, data):
    """Adjust water depth predictions by a constant offset to enforce mass conservation.

    Args:
        preds (Tensor, shape [num_nodes, NUM_WATER_VARS]): model predictions
        BC (Tensor, shape [num_nodes_BC]): boundary conditions
        data (Batch): single or multiple graphs stacked in a batched fashion

    Returns:
        Tensor: predictions with water depth corrected for mass conservation
    """
    assert data.future_t == 1, "Only works for future_t=1 for now"

    input_WD = data.x[:,-2:-1] #[m] (only water depth)
    pred_WD = preds[:,:1] #[m] (only water depth)
    mask = pred_WD > 0

    added_WD = torch.linspace(-0.5, 0.5, 201, device=preds.device) # remove or add maximum 0.5m of water depth

    conservation_term = torch.tensor([conservation_loss(nn.ReLU()(pred_WD+mask*i), input_WD, data, BC).abs() for i in added_WD])

    preds = torch.cat((pred_WD + mask*added_WD[conservation_term.argmin()], preds[:,-1:]), dim=-1)

    return preds

class LightningTrainer(L.LightningModule):
    """Lightning training wrapper for GNN flood models with curriculum learning support.

    Args:
        model (nn.Module): GNN model to train
        lr_info (dict): keys: learning_rate, weight_decay, step_size, gamma
        trainer_options (dict): keys: gamma, curriculum_epoch, velocity_scaler,
            conservation, type_loss, only_where_water
        temporal_test_dataset_parameters (dict): keys: previous_t, future_t
        datamodule (LightningDataModule, optional): data module
    """

    def __init__(self, model, lr_info, trainer_options, temporal_test_dataset_parameters, datamodule=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.previous_t = temporal_test_dataset_parameters['previous_t']
        self.future_t = temporal_test_dataset_parameters['future_t']
        self.dynamic_vars = self.previous_t * NUM_WATER_VARS
        self.lr_info = lr_info
        self.learning_rate = lr_info.get('learning_rate', 1e-3)
        self.gamma = trainer_options.get('gamma', 1)
        self.batch_size = trainer_options.get('batch_size', 4)
        self.only_where_water = trainer_options.get('only_where_water', True)
        self.conservation = trainer_options.get('conservation', 0)
        self.CSI_loss = trainer_options.get('CSI_loss', 0)
        self.CSI_threshold = trainer_options.get('CSI_threshold', 0.01)
        self.velocity_scaler = trainer_options.get('velocity_scaler', 1)
        self.type_loss = trainer_options.get('type_loss', 'RMSE')
        self.multiscale_loss_scaler = trainer_options.get('multiscale_loss_scaler', None)
        self.validation_method = trainer_options.get('validation_method', 'normal')
        self.ARME_threshold = trainer_options.get('ARME_threshold', 0.25)
        self.temporal_test_dataset_parameters = temporal_test_dataset_parameters
        self.rollout_steps = 1
        
        self.curriculum_mode = trainer_options.get('curriculum_mode', 'epoch')
        if self.curriculum_mode == 'epoch':
            self.curriculum_epoch = trainer_options.get('curriculum_epoch', 0)
        elif self.curriculum_mode == 'loss':
            self.curriculum_loss = trainer_options.get('curriculum_loss', 0.2)
        elif self.curriculum_mode == 'plateau':
            self.curriculum_plateau = trainer_options.get('curriculum_plateau', 5)
        
    def training_step(self, batch):
        temp = adapt_batch_training(batch)
        roll_loss = []

        for i in range(self.rollout_steps):
            temp.x[:,-self.dynamic_vars:] = apply_boundary_condition(temp.x[:,-self.dynamic_vars:], 
                                                                temp.BC[:,:,i], temp.node_BC, type_BC=temp.type_BC)
            # Model prediction
            preds = self.model(temp)
            temp.x = use_prediction(temp.x, preds, self.model.previous_t, self.model.future_t)

            loss = loss_function(preds, temp.y[:,:,i], temp, temp.BC[:,-2:,i].mean(1), type_loss=self.type_loss, 
                           only_where_water=self.only_where_water, CSI_loss=self.CSI_loss, CSI_threshold=self.CSI_threshold,
                           conservation=self.conservation, multiscale_loss_scaler=self.multiscale_loss_scaler,
                           velocity_scaler=self.velocity_scaler)
            roll_loss.append(loss*self.gamma**i)

        loss = torch.stack(roll_loss).mean()
        self.log("train_loss", loss.detach().to(self.device, dtype=torch.float32),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=sync_dist, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), eps=1e-7,
                                lr=self.lr_info['learning_rate'], 
                                weight_decay=self.lr_info['weight_decay'])
        
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                 step_size=self.lr_info['step_size'], 
                                                 gamma=self.lr_info['gamma'])
        return [optimizer], [lr_scheduler]
        
    def validation_step(self, batch, batch_idx):
        predicted_rollout = rollout_test(self.model, batch)
    
        # Masking the output to only consider finest scale
        if 'node_ptr' in batch.keys():
            batch = adapt_batch_training(batch)
            
        if self.validation_method == 'normal':
            real_rollout = correct_rollout_shape_future_t(batch.y)

            if 'node_ptr' in batch.keys():
                mask = self.model._create_scale_mask(batch) == 0
                predicted_rollout = predicted_rollout[mask]
                real_rollout = real_rollout[mask]
            
            assert real_rollout.shape == predicted_rollout.shape, "Real and predicted rollout must have the same dimensions\n"\
                                                    f"Intead there is {real_rollout.shape} == {predicted_rollout.shape}"

            val_loss = get_rollout_loss(predicted_rollout, real_rollout, type_loss=self.type_loss, 
                                    only_where_water=self.only_where_water).mean().detach().to(self.device, dtype=torch.float32)

            # CSI validation
            CSI_005 = get_CSI_rollout(predicted_rollout, real_rollout, water_threshold=0.05).nanmean().detach().to(self.device, dtype=torch.float32)
            CSI_03 = get_CSI_rollout(predicted_rollout, real_rollout, water_threshold=0.3).nanmean().detach().to(self.device, dtype=torch.float32)
            
            self.log("val_loss", val_loss, prog_bar=True, sync_dist=sync_dist, batch_size=batch.num_graphs)
            self.log("val_CSI_005", CSI_005, prog_bar=True, sync_dist=sync_dist, batch_size=batch.num_graphs)
            self.log("val_CSI_03", CSI_03, prog_bar=False, sync_dist=sync_dist, batch_size=batch.num_graphs)
            self.log("val_loss_CSI", val_loss*(1-CSI_005), prog_bar=False, sync_dist=sync_dist, batch_size=batch.num_graphs)
            
        elif self.validation_method == 'mass_val':
            init_volume, real_volumes = calculate_volumes(batch, self.temporal_test_dataset_parameters.get('time_start', 0))

            predicted_volumes = torch.stack([torch.matmul(predicted_rollout[batch.node_ptr[i,0]:batch.node_ptr[i,1],0].T, 
                                                          batch.area[batch.node_ptr[i,0]:batch.node_ptr[i,1]])
                                            for i in range(batch.num_graphs)]).float().cpu().numpy() - init_volume.reshape(-1,1)
            final_time = predicted_volumes.shape[-1]

            # Compute mean absolute error in delta volumes over all samples and timesteps
            ARME_index = get_average_relative_mass_error(real_volumes[:final_time].T, predicted_volumes)
            valid_mask = np.isfinite(ARME_index)

            num_valid_simulations = (ARME_index[valid_mask] < self.ARME_threshold).sum()
            percent_valid_simulations = (num_valid_simulations / predicted_volumes.shape[0])
            mass_loss = ARME_index[valid_mask].mean()*valid_mask.mean() if np.any(valid_mask) else 100.0

            self.log("val_mass_conservation_loss", torch.tensor(mass_loss, device=self.device, dtype=torch.float32), 
                     prog_bar=False, sync_dist=sync_dist, batch_size=batch.num_graphs)
            self.log("num_valid_simulations", torch.tensor(num_valid_simulations, device=self.device, dtype=torch.float32), 
                     prog_bar=False, sync_dist=sync_dist, batch_size=batch.num_graphs, reduce_fx="sum")            
            self.log("percent_valid_simulations", torch.tensor(percent_valid_simulations, device=self.device, dtype=torch.float32), 
                     prog_bar=False, sync_dist=sync_dist, batch_size=batch.num_graphs)
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}. Please choose 'normal' or 'mass_val'.")

    def predict_step(self, batch, batch_idx):
        predicted_rollout = rollout_test(self.model, batch)
        return [predicted_rollout[batch.ptr[i]:batch.ptr[i+1]] 
                for i in range(batch.num_graphs)]

class DataModule(L.LightningDataModule):
    """Lightning data module wrapping train and validation temporal datasets.

    Args:
        temporal_train_dataset (Dataset): training dataset
        temporal_val_dataset (Dataset): validation dataset
        batch_size (int): number of graphs per batch
    """

    def __init__(self, temporal_train_dataset, temporal_val_dataset, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.temporal_train_dataset = temporal_train_dataset
        self.temporal_val_dataset = temporal_val_dataset

    def train_dataloader(self):
        return DataLoader(self.temporal_train_dataset, batch_size=self.batch_size, 
                          num_workers=num_workers, 
                          persistent_workers=torch.cuda.is_available() and num_workers>0, 
                          pin_memory=torch.cuda.is_available(),
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.temporal_val_dataset, batch_size=self.batch_size,
                          num_workers=num_workers,
                          persistent_workers=torch.cuda.is_available() and num_workers>0,
                          pin_memory=torch.cuda.is_available(),
                          shuffle=False)

class CurriculumLearning(Callback):
    """Callback that progressively increases rollout steps during training.

    Args:
        max_rollout_steps (int): upper limit on rollout steps
        mode (str): 'epoch', 'loss', or 'plateau'
        patience (int): epochs without improvement before expanding rollout
        patience_buffer (int): epochs of marginal improvement exempt from patience counter
    """

    def __init__(self, max_rollout_steps, mode='epoch', patience=5, patience_buffer=1) -> None:
        super().__init__()
        self.max_rollout_steps = max_rollout_steps
        self.mode = mode
        assert self.mode in ['epoch', 'loss', 'plateau'], "Invalid curriculum learning mode. Please choose between 'epoch', 'loss', or 'plateau'"
        self.patience = patience
        self.patience_counter = 0
        self.patience_buffer = patience_buffer
        self.patience_buffer_counter = 0

    def on_train_epoch_start(self, trainer, pl_module):
        if self.mode == 'epoch':
            if pl_module.curriculum_epoch == 0:
                rollout_steps = self.max_rollout_steps
            else:
                rollout_steps = trainer.current_epoch//pl_module.curriculum_epoch+1
            self._check_steps(pl_module, rollout_steps)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.mode == 'loss':
            rollout_steps = pl_module.rollout_steps
            if trainer.callback_metrics['train_loss'] < pl_module.curriculum_loss:
                rollout_steps += 1
            self._check_steps(pl_module, rollout_steps)

        elif self.mode == 'plateau':
            rollout_steps = pl_module.rollout_steps
            if trainer.current_epoch == 0:
                self.old_loss = trainer.callback_metrics['train_loss']
            else:
                if trainer.callback_metrics['train_loss'] > self.old_loss:
                    self.patience_counter += 1
                else:
                    self.patience_buffer_counter += 1
                    if self.patience_buffer_counter > self.patience_buffer:
                        self.patience_counter = 0
                    
                if self.patience_counter > self.patience:
                    rollout_steps += 1
                    self.patience_counter = 0
                self.old_loss = trainer.callback_metrics['train_loss']
                self._check_steps(pl_module, rollout_steps)

    def _check_steps(self, pl_module, rollout_steps):
        if rollout_steps > self.max_rollout_steps:
            rollout_steps = self.max_rollout_steps
        pl_module.rollout_steps = rollout_steps