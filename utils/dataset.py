# Libraries
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch.utils.data import Subset
from torch_geometric.utils import scatter
import numpy as np
import os
import random
from copy import copy
import json
import hashlib

from database.graph_creation import MultiscaleMesh, rotate_mesh, remove_ghost_cells_multiscale, get_boundary_edges
from database.graph_creation import interpolate_BC_location_multiscale, add_ghost_cells_mesh, convert_mesh_to_pyg
from database.graph_creation import copy_face_BC_attributes_to_ghost_cell, add_ghost_cells_attributes, pool_multiscale_attributes
from utils.scaling import get_scalers, get_scalers_type

NUM_WATER_VARS = 2 # water depth (WD) and unit discharge (V)

def process_attr(attribute, scaler=None, to_min=False, device='cpu'):
    """Reshape, optionally shift to zero-minimum, and scale a tensor attribute.

    Args:
        attribute (torch.Tensor): input attribute, shape (N,) or (N, C)
        scaler (sklearn scaler, optional): fitted scaler to apply column-wise
        to_min (bool): if True, subtracts the minimum before scaling
        device (str): target device

    Returns:
        torch.Tensor: processed attribute, same shape as input
    """
    assert isinstance(attribute, torch.Tensor), "Input attribute is not a tensor"
    
    if attribute.dim() == 1:
        attribute = attribute.reshape(-1,1)

    attr = attribute.clone()

    if to_min:
        attr -= attr.min()

    if scaler is not None:
        attr = torch.cat([torch.FloatTensor(scaler.transform(attr[:,col:col+1])) for col in range(attr.shape[1])], dim=1)

    assert attribute.shape == attr.shape, "Shape has changed during processing: \n"\
        f"Before it was {attribute.shape}, now it is {attr.shape}"
        
    return attr.to(device)

def slopes_from_DEM_grid(DEM):
    """Compute x and y slope components from a DEM defined on a regular grid.

    Args:
        DEM (torch.Tensor): digital elevation model on a grid

    Returns:
        tuple of torch.Tensor: (slope_x, slope_y), each of shape (N,)
    """
    slope_x, slope_y = torch.gradient(DEM)
    return slope_x.reshape(-1), slope_y.reshape(-1)

def slopes_from_DEM_mesh(mesh, edge_index):
    """Compute node-level slope components from DEM differences along mesh edges.

    Args:
        mesh (Mesh): mesh object with face_xy, DEM, and edge attributes
        edge_index (torch.Tensor): edge connectivity, shape (2, num_edges)

    Returns:
        tuple of torch.Tensor: (slope_x, slope_y), each of shape (num_nodes,)
    """
    edge_relative_distance = mesh.face_xy[edge_index[1]] - mesh.face_xy[edge_index[0]]
    edge_slope = (mesh.DEM[edge_index[0]] - mesh.DEM[edge_index[1]])/np.linalg.norm(edge_relative_distance, axis=1)
    directed_slope = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3].T) * edge_slope * 1000
    
    slopex = scatter(directed_slope[1,:], edge_index[0], reduce='mean')
    slopey = scatter(directed_slope[0,:], edge_index[0], reduce='mean')

    return slopex, slopey

def get_temporal_res(matrix, temporal_res=60, original_temporal_res=60*8):
    """Subsample a temporal matrix to the desired time resolution.

    Args:
        matrix (torch.Tensor): temporal matrix of shape (N, T)
        temporal_res (int): desired resolution in minutes
        original_temporal_res (int): resolution of the input matrix in minutes

    Returns:
        torch.Tensor: subsampled matrix of shape (N, T')
    """
    selected_times = torch.arange(0, matrix.shape[-1], temporal_res/original_temporal_res, dtype=int)
        
    return matrix[:, selected_times]

def get_node_features(data, scalers=None, area=False, DEM=False, roughness=False,
                      canal_mask=False, lakes_mask=False, device='cpu'):
    """Return the static node feature tensor for a data sample.

    Args:
        data (Data): dataset sample containing numerical simulation
        scalers (dict, optional): fitted scalers for DEM, roughness, and area
        area (bool): include mesh cell area
        DEM (bool): include digital elevation model
        roughness (bool): include Manning's roughness coefficient
        canal_mask (bool): include canal cell mask
        lakes_mask (bool): include lakes cell mask
        device (str): target device

    Returns:
        torch.Tensor: node features of shape (num_nodes, num_features)
    """
    node_features = {}
    
    if scalers is None:
        scalers = {
            'DEM_scaler'        : None, 
            'roughness_scaler'  : None,
            'area_scaler'       : None,
        }

    if canal_mask:
        node_features['canal_mask'] = process_attr(data.canal_mask, device=device)

    if lakes_mask:
        node_features['lakes_mask'] = process_attr(data.lakes_mask, device=device)
    
    # mesh cell area
    if area:
        if isinstance(data.mesh, MultiscaleMesh):            
            if scalers['area_scaler'] is not None:
                node_features['area'] = torch.cat([process_attr(data.area[data.node_ptr[i]:data.node_ptr[i+1]], device=device, scaler=scaler) 
                                               for i, scaler in enumerate(scalers['area_scaler'])])
            else:
                node_features['area'] = torch.cat([process_attr(data.area[data.node_ptr[i]:data.node_ptr[i+1]], device=device, scaler=None) 
                                               for i in range(data.mesh.num_meshes)])
        else:
            node_features['area'] = process_attr(data.area, device=device, scaler=scalers['area_scaler'])
    
    # roughness coefficient
    if roughness:
        node_features['roughness'] = process_attr(data.roughness, scaler=scalers['roughness_scaler'], device=device)

    # digital elevation model (DEM)
    if DEM:
        node_features['DEM'] = process_attr(data.DEM, scaler=scalers['DEM_scaler'], to_min=True, device=device)

    # Make sure DEM is the last feature
    selected_node_features = locals()
    reordered_dict = {key: selected_node_features[key] for key in selected_node_features if key != 'DEM'}
    reordered_dict['DEM'] = selected_node_features['DEM']

    selected_nodes = [node_features[key] for key, value in reordered_dict.items() if value==True]
    
    if len(selected_nodes) == 0:
        node_features = torch.ones(data.num_nodes, 1, device=device)
    else:
        node_features = torch.cat(selected_nodes, 1)
    
    return node_features

def get_edge_features(data, scalers=None, edge_length=False, edge_slope=False,
                      edge_weir=False, device='cpu'):
    """Return the static edge feature tensor for a data sample.

    Args:
        data (Data): dataset sample containing numerical simulation
        scalers (dict, optional): fitted scalers for edge_length and edge_slope
        edge_length (bool): include distance between cell centres
        edge_slope (bool): include slope across neighbouring cells
        edge_weir (bool): include weir height above DEM for weir edges
        device (str): target device

    Returns:
        torch.Tensor: edge features of shape (num_edges, num_features)
    """
    if scalers is None:
        scalers = {'edge_length_scaler' : None,
                   'edge_slope_scaler'  : None}
        
    edge_features = {}

    if edge_length:
        if isinstance(data.mesh, MultiscaleMesh):
            if scalers['edge_length_scaler'] is not None:
                edge_features['edge_length'] = torch.cat(
                    [process_attr(data.face_distance[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device, scaler=scaler) 
                     for i, scaler in enumerate(scalers['edge_length_scaler'])])
            else:
                edge_features['edge_length'] = torch.cat(
                    [process_attr(data.face_distance[data.edge_ptr[i]:data.edge_ptr[i+1]], device=device) 
                     for i in range(data.mesh.num_meshes)])
        else:
            edge_features['edge_length'] = process_attr(data.face_distance, scaler=scalers['edge_length_scaler'], device=device)
        
    if edge_slope:
        edge_features['edge_slope'] = process_attr((data.DEM[data.edge_index[0]] - data.DEM[data.edge_index[1]])/data.face_distance, device=device)

    if edge_weir:
        edge_features['edge_weir'] = torch.zeros_like(data.edge_weir, device=device).reshape(-1,1)
        edge_features['edge_weir'][data.edge_weir != 0] = process_attr(data.edge_weir[data.edge_weir != 0] - data.DEM[data.edge_index[0, data.edge_weir != 0]], device=device)

    selected_edge_features = locals()

    selected_edges = [edge_features[key].float() for key, value in selected_edge_features.items() if value==True]
    
    if len(selected_edges) == 0:
        edge_features = torch.ones(data.num_edges, 1, device=device)
    else:
        edge_features = torch.cat(selected_edges, 1)
    
    return edge_features

def process_WD_VX_VY(data, temporal_res=60, scalers=None, device='cpu'):
    """Scale and subsample water depth and velocity magnitude from a data sample.

    Args:
        data (Data): dataset sample containing WD, VX, VY time series
        temporal_res (int): desired temporal resolution in minutes
        scalers (dict, optional): fitted scalers for WD and velocity
        device (str): target device

    Returns:
        Data: temporary object with WD (shape (N, T)) and V (shape (N, T))
    """
    if scalers is None:
        scalers = {
            'WD_scaler' : None, 
            'V_scaler' : None
        }

    temp = Data()

    WD = process_attr(data.WD, scaler=scalers['WD_scaler'], device=device)
    temp.WD = get_temporal_res(WD, temporal_res=temporal_res)

    VX = process_attr(data.VX, scaler=scalers['V_scaler'], device=device)*WD
    VY = process_attr(data.VY, scaler=scalers['V_scaler'], device=device)*WD
    V = torch.sqrt(VX**2 + VY**2)
    temp.V = get_temporal_res(V, temporal_res=temporal_res)
    
    return temp

def remove_scale_from_data(data, scale:int=-1):
    """Remove nodes, edges, and intra-scale edges associated with a given scale (last scale only).

    Args:
        data (Data): multiscale data object
        scale (int): scale index to remove (-1 means the last scale)

    Returns:
        Data: data object with the specified scale removed
    """
    scale = scale % len(data.mesh.meshes)
    assert scale == len(data.mesh.meshes)-1, "Only the last scale can be removed for now."

    temp = data.clone()

    temp.mesh.meshes = data.mesh.meshes[:scale] + data.mesh.meshes[scale+1:]
    temp.mesh.num_meshes = len(temp.mesh.meshes)
    temp.edge_index = data.edge_index[:,:data.edge_ptr[scale]]
    temp.edge_attr = data.edge_attr[:data.edge_ptr[scale]]
    temp.edge_ptr = torch.cat([data.edge_ptr[:scale+1], data.edge_ptr[scale+2:]])

    temp.WD = data.WD[:data.node_ptr[scale]]
    temp.V = data.V[:data.node_ptr[scale]]
    temp.x = data.x[:data.node_ptr[scale]]
    temp.DEM = data.DEM[:data.node_ptr[scale]]
    temp.area = data.area[:data.node_ptr[scale]]
    temp.node_ptr = torch.cat([data.node_ptr[:scale+1], data.node_ptr[scale+2:]])

    temp.intra_mesh_edge_index = data.intra_mesh_edge_index[:,:data.intra_edge_ptr[scale-1]]
    temp.intra_edge_ptr = torch.cat([data.intra_edge_ptr[:scale], data.intra_edge_ptr[scale+1:]])

    return temp

def create_data_attr(data, scalers=None, temporal_res=60, device='cpu',
                     area=True, DEM=True, roughness=True, canal_mask=False, lakes_mask=False,
                     edge_length=True, edge_slope=True, edge_weir=False):
    """Build x, edge_attr, WD, and V on a Data object from raw simulation attributes.

    Args:
        data (Data): raw dataset sample containing numerical simulation
        scalers (dict, optional): fitted scalers for normalizing node and edge attributes
        temporal_res (int): desired temporal resolution in minutes
        device (str): target device
        area (bool): include cell area in node features
        DEM (bool): include DEM in node features
        roughness (bool): include roughness in node features
        canal_mask (bool): include canal mask in node features
        lakes_mask (bool): include lakes mask in node features
        edge_length (bool): include edge length in edge features
        edge_slope (bool): include edge slope in edge features
        edge_weir (bool): include weir height in edge features

    Returns:
        Data: processed data object with x, edge_attr, WD, V, and metadata
    """
    temp = process_WD_VX_VY(data, temporal_res=temporal_res, scalers=scalers, device=device)
    temp.edge_index = data.edge_index
    temp.edge_attr = get_edge_features(data, scalers=scalers, edge_length=edge_length, edge_slope=edge_slope, 
                                       edge_weir=edge_weir, device=device)
    temp.x = get_node_features(data, area=area, DEM=DEM, roughness=roughness, 
                               canal_mask=canal_mask, lakes_mask=lakes_mask, scalers=scalers, device=device)
    temp.DEM = data.DEM
    temp.roughness = data.roughness
    temp.temporal_res = temporal_res
    temp.area = data.area
    if 'BC' in data.keys():
        if data.BC.dim() > 2:   #hydrograph BC
            temp.BC = get_temporal_res(data.BC[:,:,1], temporal_res=temporal_res)
        else:                   #constant BC
            time_steps = temp.WD.shape[1]
            temp.BC = torch.ones(time_steps)*data.BC
        temp.BC = temp.BC
        temp.node_BC = data.node_BC
        temp.type_BC = data.type_BC.unsqueeze(0) if data.type_BC.dim()==0 else data.type_BC
        temp.edge_BC_length = data.edge_BC_length
        if (temp.type_BC == 2).sum() > 0: #discharge BC
            discharge_BC_mask = temp.type_BC == 2
            temp.BC[discharge_BC_mask] = (temp.BC[discharge_BC_mask].T/temp.edge_BC_length[discharge_BC_mask]).T
        elif (temp.type_BC == 1).sum() > 0: #water level BC
            water_level_BC_mask = temp.type_BC == 1
            temp.BC[water_level_BC_mask] = (temp.BC[water_level_BC_mask].T - temp.DEM[temp.node_BC[water_level_BC_mask]]).T
    
    if 'mesh' in data.keys():
        temp.mesh = data.mesh
        if isinstance(temp.mesh, MultiscaleMesh):
            temp.node_ptr = data.node_ptr
            temp.edge_ptr = data.edge_ptr
            temp.intra_edge_ptr = data.intra_edge_ptr
            temp.intra_mesh_edge_index = data.intra_mesh_edge_index
    else:
        temp.pos = data.pos
    
    return temp

class FloodDataset(Dataset):
    """PyG Dataset loading .pt graph files for flood simulations.

    Args:
        root (str): directory containing saved raw .pt graph files
        sim_name (str or list): prefix(es) used to filter graph files (e.g. 'DK')
        scalers (dict, optional): fitted scalers passed to create_data_attr
        temporal_res (int): temporal resolution in minutes
        selected_node_features (dict): node feature flags forwarded to create_data_attr
        selected_edge_features (dict): edge feature flags forwarded to create_data_attr
        save_on_gpu (bool): if True, preprocess and cache all graphs on the GPU
    """

    def __init__(self, root, sim_name='DK', scalers=None, temporal_res=60,
                 selected_node_features={}, selected_edge_features={}, save_on_gpu=False):
        super().__init__(root, transform=None, pre_transform=None)
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset folder {root} does not exist. Please check the path.")
        if isinstance(sim_name, str):
            self.graph_files = sorted([f for f in os.listdir(root) 
                                    if f.endswith('.pt') and not f.endswith('_.pt') and f.startswith(sim_name)])
        elif isinstance(sim_name, list):
            self.graph_files = sorted([f for f in os.listdir(root) 
                                    if f.endswith('.pt') and not f.endswith('_.pt') and any(f.startswith(name) for name in sim_name)])
        else:
            raise ValueError("sim_name must be a string or a list of strings")
        assert len(self.graph_files) > 0, f"No graph files found in {root} with the name {sim_name}. Please check the path and name."
        self.root = root
        self.scalers = scalers
        self.selected_node_features = selected_node_features
        self.selected_edge_features = selected_edge_features
        self.temporal_res = temporal_res
        self.sim_name = sim_name
        self.graph_path = None
        self.save_on_gpu = save_on_gpu
        self.device = 'cuda' if torch.cuda.is_available() and save_on_gpu else 'cpu'

    def len(self):
        return len(self.graph_files)
    
    def _save_on_gpu(self):
        """Preprocess all graphs and return them as a list on the target device.

        Returns:
            list of Data: processed graph objects
        """
        data_list = []
        for idx in range(len(self)):
            if isinstance(self, FloodSubset):
                graph_path = os.path.join(self.root, self.dataset[idx])
            else: #FloodDataset
                graph_path = os.path.join(self.root, self.graph_files[idx])

            # Load raw graph
            data = torch.load(graph_path, weights_only=False)
            data = create_data_attr(data, scalers=self.scalers, temporal_res=self.temporal_res, 
                                            **self.selected_node_features, **self.selected_edge_features,
                                            device=self.device)
            data_list.append(data)
        return data_list

    def get(self, idx, process_data=True):
        graph_path = os.path.join(self.root, self.graph_files[idx])
        data = torch.load(graph_path, weights_only=False)
        if process_data:
            data = create_data_attr(data, scalers=self.scalers, temporal_res=self.temporal_res, 
                                            **self.selected_node_features, **self.selected_edge_features,
                                            device=self.device)

        return data
        
    def to_list(self):
        return [self.get(idx, process_data=False) for idx in range(len(self))]
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.sim_name}, n_sim={self.len()})'
    
class FloodSubset(Subset):
    """Subset of a FloodDataset that mirrors all dataset attributes and filters graph_files.

    Args:
        dataset (FloodDataset): source dataset
        indices (list of int): indices of samples to include
    """

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        for attr in dir(dataset):
            if not attr.startswith("__") and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(dataset, attr))
                except Exception:
                    pass
        self.graph_files = [dataset.graph_files[i] for i in indices]
            
    def __len__(self):
        return len(self.indices)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.sim_name}, n_sim={len(self)})'

def clear_cache(dataset_folder='database/datasets'):
    """Delete all cached temporal graph files (files ending with '_.pt') in the dataset folder.

    Args:
        dataset_folder (str): path to the folder containing cached files
    """
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('_.pt'):
                os.remove(os.path.join(root, file))

def create_model_dataset(train_dataset_name=None, val_size=0.3,
                         test_dataset_name=None, dataset_folder='database/datasets',
                         scalers=None, seed=42, save_on_gpu=False, **dataset_parameters):
    """Build scaled train, validation, and test FloodDataset objects.

    Args:
        train_dataset_name (str): sim_name prefix for the training dataset
        val_size (float, str, or list): fraction of training data for validation,
            or sim_name(s) for a separate validation dataset; 0 reuses training data
        test_dataset_name (str): sim_name prefix for the test dataset
        dataset_folder (str): path to the root folder containing train/val/test subfolders
        scalers (dict, optional): scaler type per attribute; fitted on the training split
        seed (int): random seed for reproducible dataset splits
        save_on_gpu (bool): if True, preprocess and cache all graphs on the GPU
        **dataset_parameters: node/edge feature flags and temporal_res passed to FloodDataset

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, scalers)
    """
    selected_node_features = {
        key:dataset_parameters[key] for key in 
        ['area', 'DEM', 'roughness', 'canal_mask', 'lakes_mask']}
    
    selected_edge_features = {
        key:dataset_parameters[key] for key in 
        ['edge_length', 'edge_slope', 'edge_weir']}
    
    train_dataset = FloodDataset(
        root=os.path.join(dataset_folder, 'train'),
        sim_name=train_dataset_name,
        scalers=scalers,
        temporal_res=dataset_parameters.get('temporal_res'),
        selected_node_features=selected_node_features,
        selected_edge_features=selected_edge_features,
        save_on_gpu=save_on_gpu,
    )

    # Create validation dataset from training
    if val_size != 0 and not (isinstance(val_size, str) or isinstance(val_size, list)):
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        val_dataset = FloodSubset(train_dataset, val_indices)
        train_dataset = FloodSubset(train_dataset, train_indices)
        
    # Update scalers based on the true training dataset
    scalers = get_scalers(train_dataset.to_list(), scalers)
    
    if isinstance(val_size, str) or isinstance(val_size, list):
        val_dataset = FloodDataset(
            root=os.path.join(dataset_folder, 'val'),
            sim_name=val_size,
            scalers=scalers,
            temporal_res=dataset_parameters.get('temporal_res'),
            selected_node_features=selected_node_features,
            selected_edge_features=selected_edge_features,
            save_on_gpu=save_on_gpu,
        )
    elif val_size == 0:
        print("The validation dataset you are using is the training one. Careful!")
        val_dataset = copy(train_dataset)

    # Update datasets with the new scalers
    train_dataset.scalers = scalers
    val_dataset.scalers = scalers

    if train_dataset.save_on_gpu:
        train_dataset = train_dataset._save_on_gpu()
        val_dataset = val_dataset._save_on_gpu()

    test_dataset = FloodDataset(
        root=os.path.join(dataset_folder, 'test'),
        sim_name=test_dataset_name,
        scalers=scalers,
        temporal_res=dataset_parameters.get('temporal_res'),
        selected_node_features=selected_node_features,
        selected_edge_features=selected_edge_features,
        save_on_gpu=save_on_gpu,
    )

    return train_dataset, val_dataset, test_dataset, scalers

def aggregate_WD_VX_VY(WD, VX, VY, init_time):
    """Concatenate water depth and velocity components at a single time step.

    Args:
        WD (torch.Tensor): water depth, shape (num_nodes, T)
        VX (torch.Tensor): x-velocity, shape (num_nodes, T)
        VY (torch.Tensor): y-velocity, shape (num_nodes, T)
        init_time (int): time index to extract

    Returns:
        torch.Tensor: shape (num_nodes, 3)
    """
    return torch.cat((WD[:,init_time:init_time+1], 
                      VX[:,init_time:init_time+1], 
                      VY[:,init_time:init_time+1]), 1)

def aggregate_WD_V(WD, V, init_time):
    """Concatenate water depth and velocity magnitude at a single time step.

    Args:
        WD (torch.Tensor): water depth, shape (num_nodes, T)
        V (torch.Tensor): velocity magnitude, shape (num_nodes, T)
        init_time (int): time index to extract

    Returns:
        torch.Tensor: shape (num_nodes, 2)
    """
    return torch.cat((WD[:,init_time:init_time+1], 
                      V[:,init_time:init_time+1]), 1)

def aggregate_WD(WD, init_time):
    """Return water depth at a single time step.

    Args:
        WD (torch.Tensor): water depth, shape (num_nodes, T)
        init_time (int): time index to extract

    Returns:
        torch.Tensor: shape (num_nodes, 1)
    """
    return WD[:,init_time:init_time+1]

def aggregate_BC(BC, previous_t, init_time):
    """Return boundary condition values over a window of previous_t steps.

    Args:
        BC (torch.Tensor): boundary conditions, shape (num_BC, T)
        previous_t (int): window size
        init_time (int): start index of the window

    Returns:
        torch.Tensor: shape (num_BC, previous_t)
    """
    return BC[:,init_time:init_time+previous_t]

def get_previous_steps(aggregate_function, init_time, previous_t, *water_variables_args):
    """Concatenate previous_t consecutive time steps into a single input tensor.

    Args:
        aggregate_function (callable): function to extract variables at one time step
        init_time (int): starting time index
        previous_t (int): number of time steps to concatenate
        *water_variables_args: positional arguments passed to aggregate_function before the time index

    Returns:
        torch.Tensor: concatenated tensor over previous_t steps
    """
    prev_steps = torch.cat([aggregate_function(*water_variables_args, step) for step in range(init_time,init_time+previous_t)], -1)
    return prev_steps

def get_next_steps(aggregate_function, init_time, rollout_steps, future_t, *water_variables_args):
    """Stack future output time steps into a tensor for rollout training.

    Args:
        aggregate_function (callable): function to extract variables at one time step
        init_time (int): starting time index for the output window
        rollout_steps (int): number of rollout steps
        future_t (int): number of time steps per rollout step
        *water_variables_args: positional arguments passed to aggregate_function before the time index

    Returns:
        torch.Tensor: output tensor of shape (num_nodes, num_vars*future_t, rollout_steps)
    """
    next_steps = torch.stack([torch.cat([aggregate_function(*water_variables_args, step_f+step_r*future_t) 
                            for step_f in range(init_time,init_time+future_t)], -1) 
                            for step_r in range(0,rollout_steps)], -1)
    assert next_steps.shape[-1] == rollout_steps, f"The output dimension is wrong: {next_steps.shape}"
    return next_steps

def add_dry_bed_condition(variable, previous_t):
    """Prepend previous_t-1 zero-valued steps to a variable (dry bed initial condition).

    Args:
        variable (torch.Tensor): 1D or 2D tensor
        previous_t (int): number of initial time steps (one is already present)

    Returns:
        torch.Tensor: padded tensor with previous_t-1 zero steps prepended
    """
    device = variable.device
    if variable.dim() == 1:
        return torch.cat((torch.zeros(previous_t-1, device=device), variable))
    elif variable.dim() == 2:
        num_nodes = variable.shape[0]
        return torch.cat((torch.zeros(num_nodes, previous_t-1, device=device), variable), 1)
    else:
        raise ValueError("Something wrong with the dimensions when adding dry bed conditions")
    
def replicate_initial_condition(variable, previous_t):
    """Prepend previous_t-1 copies of the first time step to a 2D variable.

    Args:
        variable (torch.Tensor): tensor of shape (num_nodes, T)
        previous_t (int): number of initial time steps (one is already present)

    Returns:
        torch.Tensor: tensor with previous_t-1 copies of the first column prepended
    """
    if variable.dim() == 2:
        return torch.cat((variable[:,:1].repeat(1, previous_t-1), variable), 1)
    else:
        raise ValueError("Something wrong with the dimensions when adding initial conditions")

def get_rollout_steps(maximum_time:int=None, time_start:int=0, time_stop:int=-1,
                      rollout_steps:int=1, future_t:int=1):
    """Resolve the effective number of rollout steps from temporal parameters.

    Args:
        maximum_time (int): total number of time steps in the simulation
        time_start (int): first time step index
        time_stop (int): last time step index (-1 means all)
        rollout_steps (int): requested rollout steps (-1 means all)
        future_t (int): time steps per rollout step

    Returns:
        int: resolved number of rollout steps
    """
    assert time_start >= 0, "time_start must be greater than or equal to 0"
    assert maximum_time is not None, "maximum_time must be provided"
    return (rollout_steps%(time_stop%maximum_time-time_start+1))//future_t if rollout_steps < 0 else rollout_steps

def get_temporal_samples_size(maximum_time:int=None, time_start:int=0, time_stop:int=-1,
                              rollout_steps:int=1, future_t:int=1):
    """Return the number of temporal samples produced by a simulation.

    Args:
        maximum_time (int): total number of time steps in the simulation
        time_start (int): first time step index
        time_stop (int): last time step index (-1 means all)
        rollout_steps (int): number of output steps per sample (-1 means all)
        future_t (int): number of time steps per rollout step

    Returns:
        int: number of temporal samples
    """
    assert maximum_time > 0, 'The temporal size of the dataset is zero'
    assert time_stop <= maximum_time, 'time_stop cannot be higher than the temporal size of the dataset'
    if time_stop != maximum_time:
        time_stop = time_stop % maximum_time - time_start + 1  # add 1 because rollout_steps starts from 1

    assert time_start <= time_stop, 'time_start cannot be higher than the last selected time'
    assert rollout_steps <= time_stop, 'Number of rollout_steps is too high'
    assert future_t > 0, 'future_t must be higher than 0'
    assert future_t < maximum_time, 'future_t must be lower than the maximum time'
    assert rollout_steps*future_t <= time_stop, 'Number of rollout_steps and future_t is too high'

    # if rollout_steps is -1, it takes all the simulation (indepenent of future_t)
    temporal_sample_size = time_stop - rollout_steps * future_t if rollout_steps > 0 else -rollout_steps

    assert temporal_sample_size >= 0, f'Something went wrong here: the temporal sample size is {temporal_sample_size}'

    return temporal_sample_size

def config_hash_temporal_dataset(previous_t, future_t, time_start, rollout_steps, graph_files=[], **other_kwds):
    """Compute a short MD5 hash identifying a temporal dataset configuration.

    Args:
        previous_t (int): number of input time steps
        future_t (int): number of output time steps per rollout step
        time_start (int): starting time index
        rollout_steps (int): number of rollout steps
        graph_files (list of str): graph file names included in the dataset
        **other_kwds: additional config entries to include in the hash

    Returns:
        str: 8-character hex hash string
    """
    config = {
        'graph_files':graph_files,
        'previous_t':previous_t,
        'future_t': future_t,
        'time_start': time_start,
        'rollout_steps': rollout_steps
    }
    if other_kwds:
        config.update(other_kwds)
    config_str = json.dumps(config, sort_keys=True)  # Consistent ordering
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]  # small 8-char hash

def to_temporal(data, init_time:int, previous_t:int=2, future_t:int=1, rollout_steps:int=1):
    """Convert a simulation graph to a temporal training instance with input and output windows.

    Args:
        data (Data): processed simulation graph with WD, V, BC, and static features
        init_time (int): starting time index for the input window
        previous_t (int): number of input time steps
        future_t (int): number of output time steps per rollout step
        rollout_steps (int): number of rollout steps in the output

    Returns:
        Data: temporal instance with x (input), y (output), BC, and metadata
    """
    # Load raw graph
    WD = replicate_initial_condition(data.WD, previous_t)
    BC = torch.cat((add_dry_bed_condition(data.BC, previous_t), data.BC[:,-1:]), 1) # Also add the last BC because of mass conservation
    V = replicate_initial_condition(data.V, previous_t)

    temp = Data()
    
    temp.edge_index = data.edge_index
    temp.edge_attr = data.edge_attr
    temp.pos = data.pos
    temp.area = data.area
    temp.temporal_res = data.temporal_res
    
    prev_steps = get_previous_steps(aggregate_WD_V, init_time, previous_t, WD, V)
    next_steps = get_next_steps(aggregate_WD_V, init_time+previous_t, rollout_steps, future_t, WD, V)

    assert prev_steps.shape[1] == NUM_WATER_VARS*previous_t, f"The output dimension is wrong: {prev_steps.shape}"
    assert next_steps.shape[1] == NUM_WATER_VARS*future_t , f"The output dimension is wrong: {next_steps.shape}"
    assert next_steps.shape[2] == rollout_steps, f"The output dimension is wrong: {next_steps.shape}"
    if (prev_steps[:,-NUM_WATER_VARS:] != 0).all(): # Except when everything is zero, then no problem
        assert ~torch.isclose(prev_steps[:,-NUM_WATER_VARS:], next_steps[:,:,0]).all(), "You're copying last time step and output"
    
    # current_time = (init_time+previous_t)*data.temporal_res/60
    temp.x = torch.cat((data.x, prev_steps), 1)
    temp.y = next_steps
    
    temp.BC = get_next_steps(aggregate_BC, init_time, rollout_steps+1, future_t, BC, previous_t)[:,::future_t]
    temp.time = init_time
    temp.edge_BC_length = data.edge_BC_length
    temp.previous_t = previous_t
    temp.future_t = future_t
    temp.node_BC = data.node_BC
    temp.type_BC = data.type_BC
    
    if 'mesh' in data.keys() and isinstance(data.mesh, MultiscaleMesh):
        temp.node_ptr = data.node_ptr
        temp.edge_ptr = data.edge_ptr
        temp.intra_edge_ptr = data.intra_edge_ptr
        temp.intra_mesh_edge_index = data.intra_mesh_edge_index

    return temp

class TemporalFloodDataset(Dataset):
    """PyG Dataset that wraps a FloodDataset and yields temporal training instances.

    Args:
        dataset (FloodDataset or FloodSubset): source simulation dataset
        previous_t (int): number of input time steps per sample
        future_t (int): number of output time steps per rollout step
        time_start (int): first time step index to use
        time_stop (int): last time step index (-1 means all)
        rollout_steps (int): number of rollout steps per sample
        save_on_gpu (bool): if True, preprocess and cache all samples on the GPU
    """

    def __init__(self, dataset, previous_t=2, future_t=1, time_start=0,
                 time_stop=-1, rollout_steps=1, save_on_gpu=False):
        super().__init__()
        if isinstance(dataset, FloodDataset) or isinstance(dataset, FloodSubset):
            self.root = dataset.root
            self.graph_files = dataset.graph_files
            self.selected_node_features = dataset.selected_node_features
            self.selected_edge_features = dataset.selected_edge_features
            self.scalers = dataset.scalers

        self.previous_t = previous_t
        self.future_t = future_t
        self.time_start = time_start
        self.time_stop = time_stop
        self.dataset = dataset
        self.num_sim = len(dataset)
        self.save_on_gpu = save_on_gpu
        self.device = 'cuda' if torch.cuda.is_available() and save_on_gpu else 'cpu'
                
        self.temporal_indices = []  # (simulation_idx, timestep_idx)

        for idx, data in enumerate(dataset):
            maximum_time = data.WD.shape[1]
            temporal_samples_size = get_temporal_samples_size(maximum_time, time_start, time_stop, rollout_steps, future_t)
            rollout_steps = get_rollout_steps(maximum_time, time_start, time_stop, rollout_steps, future_t)
            self.temporal_indices.extend([(idx, t) for t in range(time_start, time_start+temporal_samples_size, future_t)])

        self.temporal_samples_size = temporal_samples_size
        self.rollout_steps = rollout_steps
        self.cfg_hash = None

    def len(self):
        return len(self.temporal_indices)
    
    def _save_on_gpu(self):
        """Preprocess all temporal instances and return them as a list on the target device.

        Returns:
            list of Data: processed temporal instances
        """
        data_list = []
        for sim_idx in range(self.num_sim):
            data = self.dataset[sim_idx]
            for init_time in range(self.time_start, self.time_start+self.temporal_samples_size, self.future_t):
                temp = to_temporal(data, init_time, self.previous_t, self.future_t, self.rollout_steps).to(self.device)
                data_list.append(temp)

        return data_list
    
    def cache_files(self):
        """Preprocess and save each temporal instance as a .pt file in the source folder."""
        self.cfg_hash = config_hash_temporal_dataset(self.previous_t, self.future_t, self.time_start, self.rollout_steps, self.graph_files, 
                                                     **self.selected_node_features, **self.selected_edge_features,
                                                     **get_scalers_type(self.scalers))
        
        for idx in range(len(self)):
            sim_idx, init_time = self.temporal_indices[idx]

            graph_path = os.path.join(self.root, self.graph_files[sim_idx])
            cache_graph_path = graph_path.replace('.pt', f'_{self.cfg_hash}_{init_time}_.pt')

            if not os.path.exists(cache_graph_path):
                data = self.dataset[sim_idx]
                temp = to_temporal(data, init_time, self.previous_t, self.future_t, self.rollout_steps)
                
                print(f"Caching graph {idx+1}/{len(self)}: {cache_graph_path}")
                torch.save(temp, cache_graph_path)
    
    def get(self, idx):
        sim_idx, init_time = self.temporal_indices[idx]
        if self.cfg_hash is not None:
            graph_path = os.path.join(self.root, self.graph_files[sim_idx])
            cache_graph_path = graph_path.replace('.pt', f'_{self.cfg_hash}_{init_time}_.pt')
            if os.path.exists(cache_graph_path):
                return torch.load(cache_graph_path, weights_only=False)

        data = self.dataset[sim_idx]
        data = to_temporal(data, init_time, self.previous_t, self.future_t, self.rollout_steps)
            
        return data

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __len__(self):
        return self.len()
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_samples={self.len()}, n_sim={self.num_sim}, n_timesteps={self.temporal_samples_size})'

def to_temporal_predictions(dataset, predicted_rollout, previous_t=2, future_t=1, time_start=0, time_stop=-1, rollout_steps=1):
    """Convert a dataset and its model predictions into temporal instances for finetuning.

    Args:
        dataset (list of Data): processed simulation graphs
        predicted_rollout (list of torch.Tensor): model predictions aligned with dataset
        previous_t (int): number of input time steps
        future_t (int): number of output time steps per rollout step
        time_start (int): first time step index
        time_stop (int): last time step index (-1 means all)
        rollout_steps (int): number of rollout steps (-1 means all)

    Returns:
        list of Data: temporal instances with updated x and BC
    """
    assert len(dataset) == len(predicted_rollout), "The dataset and the predictions must have the same length"

    temporal_data = []
    dynamic_vars = previous_t*NUM_WATER_VARS
    device = dataset[0].x.device

    for data, preds in zip(dataset, predicted_rollout):
        maximum_time = preds.shape[-1]
        time_stop = -1
        temporal_samples_size = get_temporal_samples_size(maximum_time, time_start, time_stop, rollout_steps, future_t)
        rollout_steps = (rollout_steps%(time_stop%maximum_time-time_start+1))//future_t if rollout_steps < 0 else rollout_steps

        init_water = data.x[:, -dynamic_vars:].view(-1, previous_t, NUM_WATER_VARS).permute(0, 2, 1)
        static_variables = data.x[:, :-dynamic_vars]

        all_water = torch.cat((init_water, preds), -1)

        for init_time in range(time_start, time_start+temporal_samples_size, future_t):
            temp = data.clone()
            
            prev_steps = all_water[:,:,init_time:init_time+previous_t].permute(0, 2, 1).reshape(-1, dynamic_vars)        
            temp.x = torch.cat((static_variables, prev_steps), 1)
            
            temp.BC = data.BC[:, :, init_time:init_time+rollout_steps]
            temp.time = init_time

            temporal_data.append(temp)
        
    return temporal_data

def get_edge_BC(node_BC, edge_index):
    """Return the edge indices where boundary conditions are applied.

    Args:
        node_BC (torch.Tensor): boundary condition node indices, shape (num_BC,)
        edge_index (torch.Tensor): graph edge index, shape (2, num_edges)

    Returns:
        torch.Tensor: edge indices corresponding to boundary condition nodes
    """
    edge_BC = torch.cat([torch.where(node == edge_index)[1] for node in node_BC])
    return edge_BC

def apply_boundary_condition(x_d, BC, node_BC, type_BC=2):
    """Apply inflow boundary conditions to the dynamic node feature tensor.

    Args:
        x_d (torch.Tensor): dynamic node features, shape (num_nodes, dynamic_features)
        BC (torch.Tensor): boundary condition values, shape (num_BC, previous_t)
        node_BC (torch.Tensor): boundary node indices, shape (num_BC,)
        type_BC (int or torch.Tensor): 1 for water depth, 2 for discharge

    Returns:
        torch.Tensor: updated dynamic node features, same shape as x_d
    """
    new_x_d = x_d.clone()
    
    assert node_BC.shape[0] == BC.shape[0], "The number of boundary conditions must be equal to the boundary nodes"\
        f"but i got {node_BC.shape[0]} nodes and {BC.shape[0]} BCs"

    if isinstance(type_BC, int):
        check_type_BC(type_BC)
        type_BC = torch.tensor([type_BC]*node_BC.shape[0])
    elif isinstance(type_BC, torch.Tensor):
        assert type_BC.shape[0] == BC.shape[0], "The number of boundary conditions must be equal to the type_BC"
        [check_type_BC(t) for t in type_BC]
    else:
        raise ValueError("type_BC must be an integer or a tensor")

    if (type_BC == type_BC[0]).all():
        type_BC = type_BC[0]    
        new_x_d[node_BC, (type_BC-1)::NUM_WATER_VARS] = BC
    else:
        for bc, node_bc, t_bc in zip(BC, node_BC, type_BC):
            new_x_d[node_bc, (t_bc-1)::NUM_WATER_VARS] = bc

    return new_x_d

def check_type_BC(type_BC):
    """Assert that type_BC is a valid boundary condition type (1 or 2).

    Args:
        type_BC (int): boundary condition type
    """
    if type_BC == 1 or type_BC == 2:
        assert type_BC <= NUM_WATER_VARS, \
            "The boundary conditions are not compatible with the data format you are using."
    elif type_BC == 3:
        raise ValueError("Vector boundary conditions are not yet implemented. Please desist from convincing me to implement them.")
    else:
        raise ValueError(f"BC_type={type_BC} is not a valid input. Please select either:\n1: Inflow water depth\n2: Inflow discharge")

def use_prediction(x, pred, previous_t, future_t):
    """Shift the dynamic input window forward by appending model predictions.

    Args:
        x (torch.Tensor): input tensor, shape (num_nodes, static_features+dynamic_features)
        pred (torch.Tensor): model predictions, shape (num_nodes, num_vars*future_t)
        previous_t (int): number of previous time steps in the input
        future_t (int): number of future time steps predicted

    Returns:
        torch.Tensor: updated input tensor, same shape as x
    """
    out_dim = NUM_WATER_VARS*future_t
    assert pred.shape[-1] == out_dim, "The number of predictions is not consistent with the number of future time steps"
    dynaminc_vars = previous_t*NUM_WATER_VARS
    static_vars = x.shape[1]-dynaminc_vars

    if previous_t == future_t:
        temp = torch.cat((x[:,:static_vars], pred), 1)
    else:
        temp = torch.cat((x[:,:static_vars], x[:,-dynaminc_vars+out_dim:], pred), 1)
    assert temp.shape == x.shape, f'The shape of the input has changed from {x.shape} to {temp.shape}'

    return temp

def get_real_rollout(dataset, time_start, time_stop):
    """Extract the ground-truth rollout from the dataset for a given time interval.

    Args:
        dataset (Data): batched temporal data with attribute y
        time_start (int): first output time index
        time_stop (int): last output time index (-1 means all)

    Returns:
        torch.Tensor: ground-truth rollout with velocities converted
    """
    if time_stop == -1:
        real_rollout = dataset.y[:,:,time_start+1:].clone()
    else:
        real_rollout = dataset.y[:,:,time_start+1:time_stop+1].clone()
    
    real_rollout = convert_to_velocity(real_rollout)

    return real_rollout

def get_input_water(data):
    """Return the water variables used as model input.

    Args:
        data (Data): temporal data instance

    Returns:
        torch.Tensor: initial water state, shape (num_nodes, num_water_vars)
    """
    out_dim = NUM_WATER_VARS
    input_water = data.init_WD if hasattr(data, 'init_WD') else data.x[:,-out_dim:]

    return input_water

def get_temporal_test_dataset_parameters(config, temporal_dataset_parameters):
    """Resolve temporal test dataset parameters from config or fall back to training parameters.

    Args:
        config (dict): experiment configuration, may contain 'temporal_test_dataset_parameters'
        temporal_dataset_parameters (dict): training temporal dataset parameters

    Returns:
        dict: temporal parameters to use for the test dataset
    """
    try:
        temporal_test_dataset_parameters = config['temporal_test_dataset_parameters']
        temporal_test_dataset_parameters['previous_t'] = temporal_dataset_parameters['previous_t']
        temporal_test_dataset_parameters['future_t'] = temporal_dataset_parameters['future_t']
    except:
        temporal_test_dataset_parameters = temporal_dataset_parameters.copy()
        temporal_test_dataset_parameters.pop('rollout_steps')
        temporal_test_dataset_parameters.pop('save_on_gpu')
        # temporal_test_dataset_parameters.pop('previous_t')

    return temporal_test_dataset_parameters

def velocity_from_discharge(discharge, water_depth): #CURRENTLY NOT ACTIVE
    """Convert discharge to velocity as v = q/h (currently inactive, returns discharge).

    Args:
        discharge (torch.Tensor): unit discharge values
        water_depth (torch.Tensor): water depth values

    Returns:
        torch.Tensor: discharge (unchanged while inactive)
    """
    # velocity = discharge/water_depth
    # velocity[water_depth<0.01] = 0
    # return velocity
    return discharge

def convert_to_velocity(rollout):
    """Convert discharge channels in a rollout tensor to velocity using v = q/h.

    Args:
        rollout (torch.Tensor): rollout tensor, shape (num_nodes, num_vars, T)

    Returns:
        torch.Tensor: rollout with discharge channels replaced by velocity
    """
    if rollout.shape[1] == 2: #scalar
        rollout[:,1,:] = velocity_from_discharge(rollout[:,1,:], rollout[:,0,:])
    if rollout.shape[1] == 3: #vector
        rollout[:,1,:] = velocity_from_discharge(rollout[:,1,:], rollout[:,0,:])
        rollout[:,2,:] = velocity_from_discharge(rollout[:,2,:], rollout[:,0,:])
    return rollout

def get_inflow_volume(data, BC):
    """Compute total inflow volume from unit discharge boundary conditions as V = sum |q| * L_bc.

    Args:
        data (Data): simulation graph with edge_BC_length and temporal_res attributes
        BC (torch.Tensor): pre-summed boundary condition values over the desired interval

    Returns:
        torch.Tensor: total inflow volume in m^3
    """
    sec_in_min = 60 #seconds in a minute
    inflow_nodes = BC * data.edge_BC_length  # [m^2/s * m = m^3/s]

    inflow_volume = inflow_nodes.sum() * (sec_in_min * data.temporal_res) #[m^3]
    return inflow_volume 

def get_breach_coordinates(WD, pos):
    """Return spatial coordinates of breach locations identified by non-zero water depth at t=0.

    Args:
        WD (torch.Tensor): water depth time series, shape (num_nodes, T)
        pos (torch.Tensor): node positions, shape (num_nodes, 2)

    Returns:
        list of torch.Tensor: coordinates of breach nodes
    """
    breach_locations = [loc.item() for loc in torch.where(WD[:,0] != 0)]

    breach_coordinates = [pos[loc] for loc in breach_locations]

    return breach_coordinates

def separate_multiscale_node_features(x, node_ptr):
    """Split a concatenated multiscale node feature tensor into per-scale tensors.

    Args:
        x (torch.Tensor): node features of a multiscale mesh, shape (total_nodes, F)
        node_ptr (torch.Tensor): partition indices into scales

    Returns:
        list of torch.Tensor: node features at each scale
    """

    num_scales = len(node_ptr)-1

    x_scales = [x[node_ptr[scale]:node_ptr[scale+1]] for scale in range(num_scales)]

    return x_scales

def create_scale_mask(num_nodes, num_scales, node_ptr, data_type, device='cpu'):
    """Create an integer mask assigning each node its scale index (e.g. [0,0,0,1,1,1,...]).

    Args:
        num_nodes (int): total number of nodes
        num_scales (int): number of scales
        node_ptr (torch.Tensor): start index of each scale
        data_type (Data or Batch): determines whether batch offsets are applied
        device (str): target device

    Returns:
        torch.Tensor: integer mask of shape (num_nodes,)
    """                
    mask = torch.zeros(num_nodes, dtype=torch.int, device=device)
    for i in range(num_scales):
        if isinstance(data_type, Batch):
            for j in node_ptr[:,i:i+2]:
                mask[j[0]:j[1]] = i
        else:
            mask[node_ptr[i]:node_ptr[i+1]] = i
    return mask

def rotate_data_sample(data, angle):
    """Rotate a data sample's mesh and positions for data augmentation.

    Args:
        data (Data): data sample (use after dataset creation)
        angle (float): rotation angle in degrees

    Returns:
        Data: rotated copy of the data sample
    """
    rotated_data = data.clone()

    rotated_data.mesh = rotate_mesh(rotated_data.mesh, angle)

    if isinstance(data.mesh, MultiscaleMesh):
        rotated_data.mesh.meshes = [rotate_mesh(mesh, angle) for mesh in data.mesh.meshes]

    angle = np.deg2rad(angle)
    rot_matrix = torch.FloatTensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).to(data.x.device)

    return rotated_data

def change_BC_locations(mesh, BC_location='random', starting_BC_id=0, num_BC=4):
    """Remove existing ghost cells and reassign boundary condition locations.

    Args:
        mesh (MultiscaleMesh): input mesh object
        BC_location (str): 'random' or 'fixed'
        starting_BC_id (int): starting edge index for fixed BC placement
        num_BC (int): number of boundary conditions

    Returns:
        MultiscaleMesh: mesh with updated boundary condition locations
    """
    assert isinstance(mesh, MultiscaleMesh), 'Input mesh must be a MultiscaleMesh object.'

    # Remove ghost cells
    new_mesh = remove_ghost_cells_multiscale(copy(mesh))
    mesh = new_mesh.meshes[0]
    boundary_edges = get_boundary_edges(mesh)

    # Select BC locations at random
    if BC_location == 'random':
        num_bnd_edges = mesh.boundary_edges.shape[0]
        edge_BC_mid = mesh.boundary_node_xy[boundary_edges[[random.randint(0, num_bnd_edges-1) for _ in range(num_BC)]]].mean(1)            
    else:
        edge_BC_mid = mesh.boundary_node_xy[boundary_edges[starting_BC_id:starting_BC_id+num_BC]].mean(1)

    # Assign new BC locations
    meshes = interpolate_BC_location_multiscale(new_mesh.meshes, edge_BC_mid)
    meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]

    # Create new multiscale mesh
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)
    
    return mesh

def remove_ghost_cells_attributes(data, *features):
    """Strip ghost cell entries from node feature tensors, operating on the finest scale for multiscale data.

    Args:
        data (Data): data object with mesh attribute
        *features (torch.Tensor): node feature tensors to filter

    Returns:
        list of torch.Tensor: feature tensors with ghost cells removed
    """
    assert isinstance(data, Data), 'Input data must be a Data object.'

    new_features = []
    for feature in features:
        if isinstance(data.mesh, MultiscaleMesh): #extract the finest scale
            feature = separate_multiscale_node_features(feature, data.node_ptr)[0]

        ghost_cells_ids = data.mesh.meshes[0].ghost_cells_ids
        mask_no_ghost = torch.ones(feature.shape[0], dtype=bool)
        mask_no_ghost[ghost_cells_ids] = False

        feature = feature[mask_no_ghost].cpu()
        new_features.append(feature)

    return new_features

def update_data_object(data, mesh, WD, V, scalers, device):
    """Rebuild a data object with a new mesh, water depth, and velocity after BC location changes.

    Args:
        data (Data): original data object to copy metadata from
        mesh (MultiscaleMesh): updated mesh with new BC locations
        WD (array-like): updated water depth
        V (array-like): updated velocity magnitude
        scalers (dict): fitted scalers for node and edge features
        device (str): target device

    Returns:
        Data: updated data object with recomputed x and edge_attr
    """
    new_data = copy(data)

    new_data.node_ptr = torch.LongTensor(mesh.face_ptr, device=device)
    new_data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr, device=device)
    new_data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr, device=device)
    new_data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index, device=device)
    
    new_data.DEM = torch.FloatTensor(mesh.DEM, device=device)
    new_data.WD = torch.FloatTensor(WD, device=device)
    new_data.V = torch.FloatTensor(V, device=device)
    
    # Assign other data properties
    new_data.edge_index = torch.LongTensor(mesh.dual_edge_index, device=device)
    new_data.face_distance = torch.FloatTensor(mesh.dual_edge_length, device=device)
    new_data.num_nodes = mesh.face_x.shape[0]
    new_data.area = torch.FloatTensor(mesh.face_area, device=device)

    new_data.x = get_node_features(new_data, scalers=scalers, area=True, DEM=True, device=device)
    new_data.edge_attr = get_edge_features(new_data, scalers=scalers, edge_length=True, device=device)

    new_data.mesh = mesh
    new_data.node_BC = torch.IntTensor(mesh.ghost_cells_ids, device=device)
    new_data.edge_BC_length = torch.FloatTensor(mesh.edge_length[mesh.edge_BC], device=device)
    
    if isinstance(mesh, MultiscaleMesh):
        number_of_multiscales = mesh.num_meshes
        new_data.node_BC = new_data.node_BC[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC only at the finest scale
        new_data.edge_BC_length = new_data.edge_BC_length[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC+edge only at the finest scale

    return new_data

def adapt_ghost_cells_attributes(mesh, DEM, roughness, WD, V):
    """Add ghost cell entries and pool attributes across scales for a multiscale mesh.

    Args:
        mesh (MultiscaleMesh): target mesh
        DEM (array-like): digital elevation model at finest scale
        roughness (array-like): roughness at finest scale
        WD (array-like): water depth at finest scale
        V (array-like): velocity at finest scale

    Returns:
        tuple: (mesh, WD, V) with ghost cells added and multiscale attributes pooled
    """
    DEM, roughness, WD, V = add_ghost_cells_attributes(mesh.meshes[0], DEM, roughness, WD, V)
    
    # get multiscale attributes
    mesh.DEM, WD, V, mesh.roughness = pool_multiscale_attributes(mesh, DEM, WD, V, roughness, reduce='mean')
    mesh.DEM = copy_face_BC_attributes_to_ghost_cell(mesh, mesh.DEM)[0] #correct ghost cells values after pooling
    mesh.roughness = copy_face_BC_attributes_to_ghost_cell(mesh, mesh.roughness)[0] #correct ghost cells values after pooling
    
    return mesh, WD, V

def add_BC_to_data(meshes, BC_loc, BC, type_BC=2):
    """Build a Data object from meshes with boundary conditions assigned at given locations.

    Args:
        meshes (list of Mesh): multiscale mesh list (finest first or last)
        BC_loc (np.ndarray): BC locations (closest point), shape (n, 2)
        BC (np.ndarray): BC values, shape (T, 2)
        type_BC (int): boundary condition type (2 = inflow discharge)

    Returns:
        Data or bool: processed Data object, or False if ghost cell assignment fails
    """
    # create multiscale meshes
    meshes = meshes[::-1] if meshes[0].face_xy.shape[0] < meshes[-1].face_xy.shape[0] else meshes
    fine_mesh = meshes[0]

    DEM = fine_mesh.DEM
    WD = fine_mesh.init_WD
    V = fine_mesh.init_V
    roughness = fine_mesh.roughness
    canal_mask = fine_mesh.canal_mask
    lakes_mask = fine_mesh.lakes_mask

    meshes = interpolate_BC_location_multiscale(meshes, BC_loc)
    meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
    if any([len(m.ghost_cells_ids) > 1 for m in meshes]):
        return False
    
    WD[fine_mesh.face_BC] = 0.01
    DEM, WD, V, roughness, canal_mask, lakes_mask = add_ghost_cells_attributes(
        meshes[0], DEM, WD, V, roughness, canal_mask, lakes_mask)

    # create multiscale mesh
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)
    mesh.correct_BC(meshes)

    # get multiscale attributes
    DEM, WD, V, roughness, mesh.canal_mask, mesh.lakes_mask = pool_multiscale_attributes(mesh, DEM, WD, V, roughness, canal_mask, lakes_mask, reduce='mean')
    mesh.DEM, mesh.roughness = copy_face_BC_attributes_to_ghost_cell(mesh, DEM, roughness) #correct ghost cells values after pooling

    # Convert mesh to pytorch geometric Data object
    data = convert_mesh_to_pyg(mesh)

    data.WD = torch.FloatTensor(WD)
    data.V = torch.FloatTensor(V)

    fine_mesh_ghost_cells = meshes[0].ghost_cells_ids
    fine_mesh_BC_edges = meshes[0].edge_BC

    assert fine_mesh_BC_edges.shape == fine_mesh_ghost_cells.shape, "There's something wrong with the number of BC faces and edges"

    data.node_BC = torch.IntTensor(fine_mesh_ghost_cells)
    data.edge_BC_length = torch.FloatTensor(meshes[0].edge_length[fine_mesh_BC_edges])

    data.BC = torch.FloatTensor(BC).repeat(len(data.node_BC), 1) # This repeats the same BC
    data.type_BC = torch.tensor(type_BC, dtype=torch.int).repeat(len(fine_mesh_ghost_cells)) # This repeats the same BC type

    return data

def create_prob_test_dataset(base_datas, all_hydrographs, for_execution=True, **temporal_and_dataset_parameters):
    """Build a list of Data objects for probabilistic testing across breach locations and hydrographs.

    Args:
        base_datas (list of Data): base simulation graphs with breach location info
        all_hydrographs (list or np.ndarray): hydrographs for all scenarios
        for_execution (bool): if True, compute node/edge features; otherwise store raw WD
        **temporal_and_dataset_parameters: temporal and feature flags (temporal_res, previous_t,
            future_t, time_start, time_stop, scalers, and node/edge feature flags)

    Returns:
        list of Data: probabilistic test instances
    """
    assert len(all_hydrographs) % len(base_datas) == 0, "Incompatible hydrographs and base data lengths."

    num_scenarios_per_location = len(all_hydrographs) // len(base_datas)
    
    temporal_res = temporal_and_dataset_parameters.get("temporal_res")
    previous_t = temporal_and_dataset_parameters.get("previous_t")
    future_t = temporal_and_dataset_parameters.get("future_t")
    time_start = temporal_and_dataset_parameters.get("time_start")
    time_stop = temporal_and_dataset_parameters.get("time_stop")
    scalers = temporal_and_dataset_parameters.get("scalers")
    if time_stop < 0: time_stop = base_datas[0].BC.shape[1]

    selected_node_features = {"area": temporal_and_dataset_parameters.get("area"),
                              "DEM": temporal_and_dataset_parameters.get("DEM"),
                              "roughness": temporal_and_dataset_parameters.get("roughness"),
                              "canal_mask": temporal_and_dataset_parameters.get("canal_mask"),
                              "lakes_mask": temporal_and_dataset_parameters.get("lakes_mask")}

    selected_edge_features = {"edge_length": temporal_and_dataset_parameters.get("edge_length"),
                              "edge_slope": temporal_and_dataset_parameters.get("edge_slope"),
                              "edge_weir": temporal_and_dataset_parameters.get("edge_weir")}

    prob_test_dataset = []
    for k in range(len(base_datas)):
        for p in range(num_scenarios_per_location):
            new_data = copy(base_datas[k])
            
            # BC = torch.FloatTensor(test_BC_resampled.values[:time_stop].T[i%len(test_BC_resampled.T)]).repeat(len(new_data.node_BC), 1) / new_data.edge_BC_length.unsqueeze(1)
            BC = torch.FloatTensor(all_hydrographs[k*num_scenarios_per_location+p]).repeat(len(new_data.node_BC), 1) / new_data.edge_BC_length.unsqueeze(1)
            BC = torch.cat((add_dry_bed_condition(BC, previous_t), BC[:,-1:]), 1) # Also add the last BC because of mass conservation
            new_data.BC = get_next_steps(aggregate_BC, time_start, time_stop+previous_t-2, future_t, BC, previous_t)[:,::future_t]
            
            new_data.temporal_res = temporal_res
            new_data.previous_t = previous_t
            new_data.future_t = future_t
            new_data.ghost_cells_ids = new_data.mesh.meshes[0].ghost_cells_ids

            if for_execution:
                WD = replicate_initial_condition(new_data.WD.unsqueeze(-1), previous_t)
                V = replicate_initial_condition(new_data.V.unsqueeze(-1) ,previous_t)

                new_data.edge_attr = get_edge_features(new_data, scalers=scalers, **selected_edge_features)
                new_data.x = get_node_features(new_data, **selected_node_features, scalers=scalers)
                new_data.x = torch.cat([new_data.x, get_previous_steps(aggregate_WD_V, time_start, previous_t, WD, V)], dim=-1)                
            else:
                new_data.init_WD = separate_multiscale_node_features(new_data.WD, new_data.node_ptr)[0].numpy()

            new_data.y = torch.zeros(1, NUM_WATER_VARS, time_stop, device=new_data.edge_index.device)

            new_data.breach_coords = np.array([new_data.mesh.face_xy[node.item()] for node in new_data.node_BC])
            new_data.gdf_mesh = new_data.mesh.meshes[-1]

            del new_data.mesh, new_data.DEM, new_data.roughness
            del new_data.face_distance, new_data.lakes_mask, new_data.edge_weir, new_data.canal_mask, new_data.V, new_data.WD
            
            prob_test_dataset.append(new_data)

    return prob_test_dataset