# Libraries
import torch
from torch_geometric.data import Batch
from utils.dataset import get_inflow_volume, NUM_WATER_VARS
from utils.miscellaneous import get_mean_error, mask_on_water

def loss_function(preds, real, data, BC, type_loss='RMSE', only_where_water=False,
            multiscale_loss_scaler=[1,1,1,1], conservation=0, velocity_scaler=1,
            BC_loss=0, CSI_loss=0, CSI_threshold=0.05):
    """Compute weighted loss between predictions and targets, with optional auxiliary terms.

    Args:
        preds (Tensor, shape [num_nodes, num_variables]): model predictions
        real (Tensor, shape [num_nodes, num_variables]): ground truth values
        data (Data): graph data object
        BC (Tensor): boundary conditions
        type_loss (str): loss type, 'RMSE' or 'MAE'
        only_where_water (bool): if True, restrict loss to wet nodes
        multiscale_loss_scaler (list): per-scale loss weights
        conservation (float): coefficient for mass conservation loss
        velocity_scaler (float): loss weight for velocity terms
        BC_loss (float): coefficient for boundary conditions loss
        CSI_loss (float): coefficient for CSI loss
        CSI_threshold (float): water depth threshold for CSI

    Returns:
        Tensor: scalar loss value
    """
    diff = preds - real

    if 'node_ptr' in data.keys() and multiscale_loss_scaler is not None:
        loss = get_multiscale_loss_scaler(diff, data, multiscale_loss_scaler, only_where_water, type_loss, nodes_dim=0)
    else:
        if only_where_water:
            where_water = mask_on_water(diff)
            diff = diff[where_water]
        loss = get_mean_error(diff, type_loss, nodes_dim=0)

    if BC_loss != 0:
        loss = loss + BC_loss*boundary_conditions_loss(preds, real, data)

    loss_scaler = get_loss_variable_scaler(data.future_t, velocity_scaler=velocity_scaler, device=preds.device)
    loss = torch.dot(loss, loss_scaler)/loss_scaler.sum()

    if CSI_loss != 0:
        y_true_class = (real[:,0] > CSI_threshold).float()
        y_pred_class = torch.sigmoid((preds[:,0] - CSI_threshold)*50)
        loss = loss*torch.nn.BCEWithLogitsLoss()(y_pred_class, y_true_class)
        
    if conservation != 0:
        WD_index = NUM_WATER_VARS
        input_WD = data.x[:,-WD_index*2*data.future_t::WD_index][:,:1] #[m] (only last water depth)
        pred_WD = preds[:,0::WD_index] #[m] (only water depth)
        loss = loss + conservation*conservation_loss(pred_WD, input_WD, data, BC).abs()

    return loss

def get_loss_variable_scaler(future_t, velocity_scaler=1, device='cpu'):
    """Build a per-variable loss scaler vector, down-weighting velocity terms.

    Args:
        future_t (int): number of predicted future time steps
        velocity_scaler (float): weight applied to velocity loss terms
        device (str): torch device string

    Returns:
        Tensor, shape [NUM_WATER_VARS * future_t]: per-variable loss weights
    """
    loss_scaler = torch.ones(NUM_WATER_VARS*future_t, device=device)
    loss_scaler[1::NUM_WATER_VARS] = velocity_scaler
        
    return loss_scaler

def get_multiscale_loss_scaler(diff, data, multiscale_loss_scaler=[1,1,1,1],
                               only_where_water=True, type_loss='RMSE', nodes_dim=0):
    """Compute a weighted average of per-scale losses for multiscale models.

    Args:
        diff (Tensor): element-wise difference between predictions and targets
        data (Data or Batch): graph data object with node_ptr
        multiscale_loss_scaler (list): per-scale weights
        only_where_water (bool): if True, restrict loss to wet nodes
        type_loss (str): loss type, 'RMSE' or 'MAE'
        nodes_dim (int): dimension along which nodes are arranged

    Returns:
        Tensor: scalar weighted multiscale loss
    """
    assert sum(multiscale_loss_scaler) != 0, "Multiscale loss scalers cannot be all zeros"

    node_ptr = data.node_ptr
    if only_where_water:
        where_water = mask_on_water(diff)
    else:
        where_water = torch.ones(diff.shape[0]).bool()
    
    if isinstance(data, Batch):
        losses_multiscale = torch.stack([get_mean_error(torch.cat([diff[data.node_ptr[i,s]:data.node_ptr[i,s+1]][where_water[node_ptr[i,s]:node_ptr[i,s+1]]] 
                                                  for i in range(data.num_graphs)]), type_loss, nodes_dim) 
                                                  for s in range(node_ptr.shape[-1]-1)])
    else:
        losses_multiscale = torch.stack([get_mean_error(diff[node_ptr[i]:node_ptr[i+1]][where_water[node_ptr[i]:node_ptr[i+1]]], type_loss, nodes_dim) 
                                         for i in range(node_ptr.shape[-1]-1)])
        
    multiscale_loss_scaler = torch.tensor(multiscale_loss_scaler, device=diff.device).float()
    assert len(multiscale_loss_scaler) == len(losses_multiscale), f"Multiscale loss scalers have wrong dimensions ({len(multiscale_loss_scaler)} != {len(losses_multiscale)})"
    multiscale_loss = multiscale_loss_scaler@losses_multiscale/sum(multiscale_loss_scaler)

    return multiscale_loss

def conservation_loss(pred_WD, input_WD, data, BC):
    """Compute mass conservation residual as difference between predicted and theoretical inflow volume.

    Args:
        pred_WD (Tensor, shape [num_nodes, future_t]): predicted water depth at time t+1
        input_WD (Tensor, shape [num_nodes, future_t]): input water depth at time t
        data (Data or Batch): graph data object with area and node_ptr
        BC (Tensor, shape [num_BCs]): boundary conditions at time t

    Returns:
        Tensor: scalar conservation residual in units of 1e6 m^3
    """
    # Calculate delta_WD
    assert pred_WD.shape == input_WD.shape, f"Input or predictions have wrong dimensions ({pred_WD.shape} != {input_WD.shape})"
    delta_WD = pred_WD - input_WD #[m]
    assert delta_WD.dim() == 2, f"Input or predictions have wrong dimensions ({delta_WD.dim()})"
    assert BC.dim() == 1, f"Boundary conditions have wrong dimensions ({BC.dim()})"

    # Calculate predicted volume
    area = data.area if data.area.dim() ==2 else data.area.unsqueeze(1) #[m^2]

    # Multiscale (select only the finest scale)
    if 'node_ptr' in data.keys():
        num_scales = data.node_ptr.shape[-1] - 1
        if isinstance(data, Batch):
            predicted_inflow_volume = torch.cat([(area*delta_WD)[data.node_ptr[i,0]:data.node_ptr[i,1]] 
                                                 for i in range(data.num_graphs)]).sum() #[m^3]
        else:
            predicted_inflow_volume = ((area*delta_WD)[data.node_ptr[0]:data.node_ptr[1]]).sum()

    # Single scale
    else: 
        predicted_inflow_volume = (area*delta_WD).sum() #[m^3]
    
    # Theoretical inflow volume
    inflow_volume = get_inflow_volume(data, BC) #[m^3]
    boundary_correction = ((area*delta_WD)[data.node_BC]).sum() #[m^3] # remove values at ghost cells
        
    # Mass conservation 
    conservation_loss = (predicted_inflow_volume - inflow_volume - boundary_correction)/1e6 #[m^3 * 1e6]

    if isinstance(data, Batch):
        conservation_loss = conservation_loss/data.num_graphs
    
    return conservation_loss

def boundary_conditions_loss(preds, real, data, type_loss='RMSE', nodes_dim=0):
    """Compute loss restricted to boundary condition nodes.

    Args:
        preds (Tensor, shape [num_nodes, num_variables]): model predictions
        real (Tensor, shape [num_nodes, num_variables]): ground truth values
        data (Data): graph data object with node_BC indices
        type_loss (str): loss type, 'RMSE' or 'MAE'
        nodes_dim (int): dimension along which nodes are arranged

    Returns:
        Tensor: scalar loss at boundary nodes
    """
    diff = preds - real
    BC_loss = get_mean_error(diff[data.node_BC], type_loss, nodes_dim=nodes_dim)
    return BC_loss