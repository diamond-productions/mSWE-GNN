import pickle
import yaml
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
from scipy import stats
from wandb import Config
import psutil
from torch_geometric.data import Batch

from database.graph_creation import MultiscaleMesh
from utils.dataset import TemporalFloodDataset, create_scale_mask, separate_multiscale_node_features
from utils.dataset import get_input_water, NUM_WATER_VARS
from models.gnn import GNN, MSGNN


def read_config(config_file):
    """Read YAML configuration file and return its contents as a dictionary.

    Args:
        config_file (str): path to the YAML configuration file

    Returns:
        dict: dataset creation configuration
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_dataset(dataset_name, size, seed=42, dataset_folder='database/datasets'):
    """Load a dataset (or list of datasets) from pickle files.

    Args:
        dataset_name (str or list of str): dataset name(s) to load
        size (int or list of int): number of samples to load per dataset
        seed (int): random seed for shuffling
        dataset_folder (str): path to folder containing dataset pickle files

    Returns:
        list: PyTorch Geometric Data objects
    """
    if isinstance(dataset_name, list):
        datasets = []
        for i, name in enumerate(dataset_name):
            path = f"{dataset_folder}/{name}.pkl"
            with open(path, 'rb') as file:
                dataset = pickle.load(file)
            if seed != 0:
                random.seed(seed)
                random.shuffle(dataset)
            if isinstance(size, list):
                datasets += dataset[:size[i]]
            else:
                datasets += dataset
        if isinstance(size, list):
            size = sum(size)
        return datasets[:size]
    else:
        path = f"{dataset_folder}/{dataset_name}.pkl"
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
        if seed != 0:
            random.seed(seed)
            random.shuffle(dataset)
        return dataset[:size]

def get_model(model_name):
    """Return the model class corresponding to the given name.

    Args:
        model_name (str): 'GNN' or 'MSGNN'

    Returns:
        type: model class
    """
    models_dict = {'GNN': GNN,
                   'MSGNN': MSGNN}
    return models_dict[model_name]

def get_time_vector(total_time_steps, temporal_res):
    """Return array of time stamps from 0 to total time in hours.

    Args:
        total_time_steps (int): number of time steps
        temporal_res (float): temporal resolution in minutes

    Returns:
        np.ndarray, shape [total_time_steps+1]: time stamps in hours
    """
    total_hours = total_time_steps*temporal_res/60
    time_vector = np.linspace(0, total_hours, total_time_steps+1)
    return time_vector

def add_null_time_start(time_start, temporal_array):
    """Prepend NaN values to a temporal array to account for simulation start offset.

    Args:
        time_start (int): number of null values to prepend
        temporal_array (np.ndarray, shape [T] or [N, T]): temporal array

    Returns:
        np.ndarray: array with NaN-padded start, same shape except time axis extended by time_start+1
    """
    if temporal_array.ndim == 1: # [T]
        new_temporal_array = np.concatenate((np.nan*np.empty(time_start+1), temporal_array))
    elif temporal_array.ndim == 2: # [N_datasets, T]
        new_temporal_array = np.concatenate((np.nan*np.empty((
            temporal_array.shape[0], time_start+1)), temporal_array), axis=1)
    else:
        raise ValueError("Wrong temporal array dimensions")

    return new_temporal_array

def fisher_confidence_interval(r, n, alpha=0.05):
    """Compute confidence interval for a correlation coefficient via Fisher z-transformation.

    Args:
        r (float): correlation coefficient
        n (int): sample size
        alpha (float): significance level

    Returns:
        tuple: (lower, upper) confidence interval bounds
    """
    if abs(r) == 1:
        return (r, r)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh([z - z_crit * se, z + z_crit * se])
    return lo, hi

def compute_correlations(x, y, alpha=0.05):
    """Compute Pearson and Spearman correlations with confidence intervals.

    Args:
        x (array-like): first variable
        y (array-like): second variable
        alpha (float): significance level for confidence intervals

    Returns:
        tuple: (pearson_r, pearson_ci, spearman_r, spearman_ci)
    """
    n = len(x)
    pearson_r, _ = stats.pearsonr(x, y)
    spearman_r, _ = stats.spearmanr(x, y)
    pearson_ci = fisher_confidence_interval(pearson_r, n, alpha)
    spearman_ci = fisher_confidence_interval(spearman_r, n, alpha)
    return pearson_r, pearson_ci, spearman_r, spearman_ci

def summarize_correlation(name, ARME, CSI005, CSI03, MAEh, MAEq):
    """Print summary line for each split for CSI, MAEh, MAEq."""
    print(f"\n{name.upper()} CORRELATION SUMMARY")
    # CSI
    pearson_r, pearson_ci, spearman_r, spearman_ci = compute_correlations(ARME, CSI005)
    print(f"CSI 0.05: Pearson r = {pearson_r:+.3f} [{pearson_ci[0]:+.3f}, {pearson_ci[1]:+.3f}] | "
          f"Spearman ρ = {spearman_r:+.3f} [{spearman_ci[0]:+.3f}, {spearman_ci[1]:+.3f}]")    
    # CSI
    pearson_r, pearson_ci, spearman_r, spearman_ci = compute_correlations(ARME, CSI03)
    print(f"CSI 0.3:  Pearson r = {pearson_r:+.3f} [{pearson_ci[0]:+.3f}, {pearson_ci[1]:+.3f}] | "
          f"Spearman ρ = {spearman_r:+.3f} [{spearman_ci[0]:+.3f}, {spearman_ci[1]:+.3f}]")
    # MAEh
    pearson_r, pearson_ci, spearman_r, spearman_ci = compute_correlations(ARME, MAEh)
    print(f"MAE h:    Pearson r = {pearson_r:+.3f} [{pearson_ci[0]:+.3f}, {pearson_ci[1]:+.3f}] | "
          f"Spearman ρ = {spearman_r:+.3f} [{spearman_ci[0]:+.3f}, {spearman_ci[1]:+.3f}]")
    # MAEq
    pearson_r, pearson_ci, spearman_r, spearman_ci = compute_correlations(ARME, MAEq)
    print(f"MAE q:    Pearson r = {pearson_r:+.3f} [{pearson_ci[0]:+.3f}, {pearson_ci[1]:+.3f}] | "
          f"Spearman ρ = {spearman_r:+.3f} [{spearman_ci[0]:+.3f}, {spearman_ci[1]:+.3f}]")

# The performance of the CPU mapping needs to be tested
def set_cpu_affinity_LUMI(rank, local_rank):
    """Bind the current process to the CPU cores closest to the given GPU on a LUMI-G node.

    Args:
        rank (int): global process rank
        local_rank (int): local GPU/GCD index (0-7)
    """
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)

def get_velocity(discharge, water_depth, epsilon=0.01):
    """Compute velocity from discharge and water depth, zeroing out shallow cells.

    Args:
        discharge (Tensor): discharge values
        water_depth (Tensor): water depth values
        epsilon (float): threshold below which velocity is set to zero

    Returns:
        Tensor: velocity with dry-cell masking applied
    """
    velocity = discharge/water_depth
    low_water = water_depth<=epsilon
    velocity[low_water] = 0
    return velocity

def correct_rollout_shape_future_t(rollout_tensor: torch.tensor):
    """Reshape rollout tensor from (N, future_t*V, T) to (N, V, T*future_t).

    Args:
        rollout_tensor (Tensor, shape [N, future_t*V, T]): raw rollout output

    Returns:
        Tensor, shape [N, V, T*future_t]: reshaped rollout
    """
    assert rollout_tensor.dim() == 3, 'Input tensor must have 3 dimensions'
    return torch.stack([rollout_tensor[:,i::NUM_WATER_VARS].transpose(1,2).flatten(1) 
                        for i in range(NUM_WATER_VARS)], 1)

def get_Froude(velocity, water_depth):
    """Compute Froude number, zeroing out dry cells.

    Args:
        velocity (Tensor): flow velocity
        water_depth (Tensor): water depth

    Returns:
        Tensor: Froude number with dry-cell masking applied
    """
    g = 9.81
    froude = velocity / torch.sqrt(g*water_depth)
    froude[water_depth <= 0] = 0
    return froude

def WD_to_FAT(WD, temporal_res, water_threshold=0, time_start=0):
    """Convert a water depth sequence to flood arrival times.

    Args:
        WD (Tensor, shape [N, T]): water depth time series
        temporal_res (float): temporal resolution in minutes
        water_threshold (float): depth threshold for flood detection
        time_start (int): initial simulation time step

    Returns:
        Tensor, shape [N]: flood arrival time in hours (NaN for dry cells)
    """
    assert WD.ndim == 2, "WD must be a tensor of dimension [N, T]"
    total_time = time_start + WD.shape[-1]
    
    flooded_areas = WD > water_threshold
    flooded_time = flooded_areas.sum(1)
    FAT = -(flooded_time - total_time)
    FAT_hours = FAT*temporal_res/60
    
    if isinstance(WD, torch.Tensor):
        no_water = WD.max(1)[0] == 0
    else:
        no_water = WD.max(1) == 0
    FAT_hours[no_water] = torch.nan

    return FAT_hours

def get_rise_rate(WD, temporal_res):
    """Compute mean water depth rise rate in mm/hour over all time steps.

    Args:
        WD (Tensor, shape [N, T]): water depth time series
        temporal_res (int): temporal resolution in minutes

    Returns:
        Tensor, shape [N]: mean rise rate per node in mm/hour
    """
    assert WD.dim() == 2, "WD must be a tensor of dimension [N, T]"

    rise_rate = (WD[:,1:] - WD[:,:-1])/(temporal_res/60)

    return rise_rate.mean(1) # average rise rate in mm/hour

def get_numerical_times(dataset_size, temporal_res, maximum_time,
                        overview_file='database/overview.csv',
                        **temporal_test_dataset_parameters):
    """Read and scale numerical simulation computation times from an overview CSV.

    Args:
        dataset_size (int): number of simulations to return
        temporal_res (float): temporal resolution in minutes
        maximum_time (int): maximum simulation time steps
        overview_file (str): path to the overview CSV file
        **temporal_test_dataset_parameters: must include 'time_start' and 'time_stop'

    Returns:
        pd.Series: scaled computation times for the first dataset_size simulations
    """
    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']

    final_time = time_stop%maximum_time + (time_stop==-1)    
    assert final_time != -1, "I'm not sure how to interpret final_time value of -1"

    numerical_simulation_overview = pd.read_csv(overview_file, sep=',')

    computation_time = numerical_simulation_overview['computation_time[s]']

    # in case time start is not 0 or final_time is not maximum_time we scale the computation time
    simulated_times = numerical_simulation_overview['simulation_time[h]']
    model_simulated_times = (final_time-time_start)*temporal_res/60

    time_ratio = model_simulated_times/simulated_times

    return (computation_time*time_ratio).iloc[:dataset_size]

def get_speed_up(numerical_times, model_times):
    """Compute mean and std of speed-up ratio between numerical and DL model times.

    Args:
        numerical_times (array-like): numerical simulation computation times
        model_times (array-like): DL model computation times

    Returns:
        tuple: (mean speed-up, std of speed-up)
    """
    speed_up = numerical_times/model_times

    return speed_up.mean(), speed_up.std()

def get_mean_error(diff_rollout, type_loss, nodes_dim=0, delta=1.0):
    """Compute mean error along the node dimension using RMSE, MAE, or Huber loss.

    Args:
        diff_rollout (Tensor): difference between predictions and ground truth
        type_loss (str): 'RMSE', 'MAE', or 'Huber'
        nodes_dim (int): dimension index corresponding to nodes
        delta (float): threshold parameter for Huber loss

    Returns:
        Tensor: mean error with node dimension reduced
    """
    if type_loss.upper() == 'RMSE':
        average_diff_t = torch.sqrt((diff_rollout**2).mean(nodes_dim))
    elif type_loss.upper() == 'MAE':
        average_diff_t = diff_rollout.abs().mean(nodes_dim)
    elif type_loss.lower() == 'huber':
        abs_diff = diff_rollout.abs()
        mask = abs_diff <= delta
        huber_loss = torch.where(mask, 0.5 * diff_rollout**2, delta * (abs_diff - 0.5 * delta))
        average_diff_t = huber_loss.mean(nodes_dim)
    else:
        raise ValueError(f"Unknown type_loss: {type_loss}")
    return average_diff_t

def mask_on_water(diff, water_axis=1):
    """Return boolean mask selecting nodes where any water is present.

    Args:
        diff (Tensor): difference tensor between predictions and ground truth
        water_axis (int): axis index corresponding to water depth variable

    Returns:
        Tensor: boolean mask of shape with water_axis dimension removed
    """
    where_water = (diff != 0).any(water_axis)
    return where_water

def get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=0):
    """Convert predicted and real rollouts to binary flood maps.

    Args:
        predicted_rollout (Tensor, shape [S, N, V, T] or [N, V, T] or [N, V] or [N]): predicted water depth
        real_rollout (Tensor): ground truth with same shape as predicted_rollout
        water_threshold (float): depth threshold above which a cell is considered flooded

    Returns:
        tuple: (predicted_flood, real_flood) boolean tensors of same shape
    """
    if predicted_rollout.dim() == 4: # shape [S, N, WDQ, T]
        predicted_rollout_flood = predicted_rollout[:,:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,:,0,:]>water_threshold

    elif predicted_rollout.dim() == 3: # shape [N, WDQ, T]
        predicted_rollout_flood = predicted_rollout[:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,0,:]>water_threshold

    elif predicted_rollout.dim() == 2: # shape [N, WDQ]
        predicted_rollout_flood = predicted_rollout[:,0]>water_threshold
        real_roll_flood = real_rollout[:,0]>water_threshold

    elif predicted_rollout.dim() == 1: # shape [N]
        predicted_rollout_flood = predicted_rollout>water_threshold
        real_roll_flood = real_rollout>water_threshold

    else:
        raise ValueError("Wrong dimensions of predicted_rollout")

    return predicted_rollout_flood, real_roll_flood

def get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=0):
    """Compute TP, TN, FP, FN counts over the node dimension for a flood rollout.

    Args:
        predicted_rollout (Tensor): predicted water depth rollout
        real_rollout (Tensor): ground truth water depth rollout
        water_threshold (float): depth threshold for flood classification

    Returns:
        tuple: (TP, TN, FP, FN) tensors with node dimension reduced
    """
    predicted_rollout_flood, real_roll_flood = get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=water_threshold)

    if predicted_rollout.dim() == 4:
        nodes_dim = 1
    elif predicted_rollout.dim() <= 3:
        nodes_dim = 0
    else:
        raise ValueError("Wrong dimensions of predicted_rollout")

    TP = (predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #true positive
    TN = (~predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #true negative
    FP = (predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #false positive
    FN = (~predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #false negative

    return TP, TN, FP, FN

def get_CSI(TP, FP, FN):
    """Compute Critical Success Index: TP / (TP + FN + FP).

    Args:
        TP (Tensor): true positives
        FP (Tensor): false positives
        FN (Tensor): false negatives

    Returns:
        Tensor: CSI score
    """
    CSI = TP / (TP + FN + FP)
    return CSI

def get_F1(TP, FP, FN):
    """Compute F1 score: 2*TP / (2*TP + FN + FP).

    Args:
        TP (Tensor): true positives
        FP (Tensor): false positives
        FN (Tensor): false negatives

    Returns:
        Tensor: F1 score
    """
    F1 = 2*TP / (2*TP + FN + FP)
    return F1

def get_CSI_rollout(predicted_rollout, real_rollout, water_threshold=0):
    """Compute CSI over time for a predicted rollout at a given flood threshold.

    Args:
        predicted_rollout (Tensor): predicted water depth rollout
        real_rollout (Tensor): ground truth water depth rollout
        water_threshold (float): depth threshold for flood classification

    Returns:
        Tensor: CSI values over time
    """
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)
    CSI = get_CSI(TP, FP, FN)
    return CSI

def get_F1_rollout(predicted_rollout, real_rollout, water_threshold=0):
    """Compute F1 score over time for a predicted rollout at a given flood threshold.

    Args:
        predicted_rollout (Tensor): predicted water depth rollout
        real_rollout (Tensor): ground truth water depth rollout
        water_threshold (float): depth threshold for flood classification

    Returns:
        Tensor: F1 values over time
    """
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)
    F1 = get_F1(TP, FP, FN)
    return F1

def get_masked_diff(diff_roll, where_water):
    """Extract rollout differences at wet nodes only.

    Args:
        diff_roll (Tensor, shape [N, V, T]): difference between predicted and real rollout
        where_water (Tensor, shape [N]): boolean mask of wet nodes

    Returns:
        Tensor, shape [V, N_wet, T]: differences at wet nodes per variable
    """
    masked_diff = torch.stack([diff_roll[:,water_variable,:][where_water] 
                                    for water_variable in range(diff_roll.shape[1])])
                                
    return masked_diff

def get_rollout_loss(predicted_rollout, real_rollout, type_loss='RMSE', only_where_water=False):
    """Compute per-variable rollout loss over all time steps.

    Args:
        predicted_rollout (Tensor, shape [N, V, T] or [S, N, V, T]): predicted rollout
        real_rollout (Tensor): ground truth with same shape as predicted_rollout
        type_loss (str): 'RMSE', 'MAE', or 'Huber'
        only_where_water (bool): if True, restrict loss to wet nodes

    Returns:
        Tensor: mean error per variable (and simulation if batched)
    """
    diff_roll = predicted_rollout - real_rollout

    if diff_roll.dim() == 4: #multiple simulations
        nodes_dim = 1
        water_axis = 2
    elif diff_roll.dim() == 3: #single simulation
        nodes_dim = 0
        water_axis = 1

    if only_where_water:
        where_water = mask_on_water(diff_roll, water_axis=water_axis)
        
        if diff_roll.dim() == 4:
            roll_loss = torch.stack([get_mean_error(
                get_masked_diff(diff_roll[id_dataset], where_water[id_dataset]), 
                type_loss, nodes_dim=-1) for id_dataset in range(diff_roll.shape[0])])
        elif diff_roll.dim() == 3:
            roll_loss = get_mean_error(get_masked_diff(diff_roll, where_water), type_loss, nodes_dim=-1)
    else:
        roll_loss = get_mean_error(diff_roll, type_loss, nodes_dim=nodes_dim).mean(-1)
    
    return roll_loss

def stack_rollout_different_BC(predicted_rollout, dataset, scale=0):
    """Stack rollouts from different boundary conditions into a single tensor, removing ghost cells.

    Args:
        predicted_rollout (list of Tensor): predicted rollouts per simulation
        dataset (list): dataset objects with mesh and node_ptr attributes
        scale (int): multiscale index to extract

    Returns:
        Tensor: stacked rollout with ghost cells removed
    """
    assert len(predicted_rollout) == len(dataset), "The length of the predicted rollout and the dataset must be equal"
    assert isinstance(predicted_rollout, list), "The predicted rollout must be a list"

    stacked_rollout = []
    for i in range(len(predicted_rollout)):
        temp_rollout = separate_multiscale_node_features(predicted_rollout[i], dataset[i].node_ptr)[scale]

        ghost_cells_ids = dataset[i].mesh.meshes[0].ghost_cells_ids if hasattr(dataset[i], 'mesh') else dataset[i].ghost_cells_ids
        mask_no_ghost = torch.ones(temp_rollout.shape[0], dtype=bool)
        mask_no_ghost[ghost_cells_ids] = False

        stacked_rollout.append(temp_rollout[mask_no_ghost])
    
    return torch.stack(stacked_rollout)

def plot_line_with_deviation(time_vector, variable, with_minmax=False, ax=None, **plt_kwargs):
    """Plot mean line with shaded std band, optionally adding min/max envelope.

    Args:
        time_vector (np.ndarray): time stamps in hours
        variable (np.ndarray): values to plot, shape [N_series, T]
        with_minmax (bool): if True, also plot min/max dashed lines
        ax (Axes, optional): matplotlib axes to plot on

    Returns:
        list: matplotlib Line2D objects
    """
    ax = ax or plt.gca()

    df = pd.DataFrame(np.vstack((time_vector, variable))).T
    df = df.rename(columns={0: "time"})
    df = df.set_index('time')

    mean = df.mean(1)
    std = df.std(1)
    under_line = (mean - std)
    over_line = (mean + std)

    p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
    color = p[0].get_color()
    ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
    if with_minmax:
        ax.plot(df.min(1), color=color, linestyle='--', alpha=.5)
        ax.plot(df.max(1), color=color, linestyle='--', alpha=.5)
    return p

def fix_dict_in_config(wandb):
    """Unflatten dot-separated keys in a wandb config into nested dicts.

    Args:
        wandb: wandb run object whose config may contain 'parent.child' style keys
    """
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]
    
    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v

def get_pareto_front(df, objective_function1, objective_function2, ascending=False):
    """Extract the Pareto front for two objective functions from a DataFrame.

    Args:
        df (pd.DataFrame): data containing the objective columns
        objective_function1 (str): name of the first objective column
        objective_function2 (str): name of the second objective column
        ascending (bool): if True, find Pareto front for minimization

    Returns:
        np.ndarray, shape [P, 2]: Pareto-optimal (obj1, obj2) pairs
    """
    sorted_df = df.sort_values(by=[objective_function1, objective_function2], ascending=ascending)[[objective_function1, objective_function2]]

    pareto_front = sorted_df.values[0].reshape(1,-1)
    for var1, var2 in sorted_df.values[1:]:
        if var2 >= pareto_front[-1,1]:
            pareto_front = np.concatenate((pareto_front, np.array([[var1, var2]])), axis=0)
    
    return pareto_front

def get_sufficient_k_hops(edge_index, WD, Q):
    """Determine the minimum number of GNN message-passing hops to cover flood-front changes in one time step.

    Args:
        edge_index (Tensor, shape [2, E]): graph edge indices
        WD (Tensor, shape [N, T]): water depth simulation
        Q (Tensor, shape [N, T]): discharge simulation

    Returns:
        int: minimum k-hops required
    """
    assert WD.dim() == 2, "The input WD matrix should contain the full original simulation [NxT]"

    row = edge_index[0]
    col = edge_index[1]

    num_nodes = WD.shape[0]
    time_steps = WD.shape[1]

    WD_threshold = 0
    Q_threshold = 0.001

    masked_water = torch.stack([(WD[:,t]>WD_threshold)*(Q[:,t]>Q_threshold) for t in range(time_steps)]).T
    water_diff_t = torch.stack([masked_water[:,t+1] ^ masked_water[:,t] for t in range(time_steps-1)]).T

    fake_water = torch.zeros_like(WD)
    fake_water[WD>WD_threshold] = 1
    fake_water[Q<Q_threshold] = 0
    fake_water = fake_water[:,:-1]

    changes_fully_covered = (fake_water[torch.where(water_diff_t[:,0])[0]] == 1).all()

    k=0
    while not changes_fully_covered:
        fake_water = scatter(fake_water[row], col, reduce='sum', dim=0, dim_size=num_nodes) + fake_water
        fake_water[fake_water>0] = 1
        changes_fully_covered = torch.tensor([(fake_water[torch.where(water_diff_t[:,t])[0], t] == 1).all() for t in range(time_steps-2)]).all()
        if k>100:
            print(f'The number of k-hops is higher than {k}')
            break

    return k

def get_sufficient_k_hops_per_scale(edge_index, WD, Q, edge_ptr, node_ptr):
    """Compute minimum k-hops required at each scale of a multiscale graph.

    Args:
        edge_index (Tensor, shape [2, E]): graph edge indices across all scales
        WD (Tensor, shape [N, T]): water depth simulation across all scales
        Q (Tensor, shape [N, T]): discharge simulation across all scales
        edge_ptr (list): edge pointer boundaries per scale
        node_ptr (list): node pointer boundaries per scale

    Returns:
        list: minimum k-hops per scale
    """
    
    khop_per_scale = [get_sufficient_k_hops(edge_index[:,edge_ptr[i]:edge_ptr[i+1]]-node_ptr[i], 
                                            WD[node_ptr[i]:node_ptr[i+1]], Q[node_ptr[i]:node_ptr[i+1]])
                                            for i in range(len(node_ptr)-1)]
    return khop_per_scale

def calculate_volumes(data, time_start=0, scale=0):
    """Compute the initial volume and cumulative volume time series for a simulation.

    Args:
        data: PyTorch Geometric Data or Batch object with WD or BC attributes
        time_start (int): starting time step index
        scale (int): multiscale index to use

    Returns:
        tuple: (init_volume, cumulative_volume) as numpy arrays
    """
    # compute real volume depending on available attributes
    if hasattr(data, 'WD'):
        input_water_depth = data.WD[:, 0].cpu().numpy()
        real_volume = separate_multiscale_node_features(data.WD[:, 1:].cpu(), data.node_ptr)[scale].numpy().T @ data.mesh.meshes[scale].face_area
    elif hasattr(data, 'BC'):
        if hasattr(data, 'init_WD'):
            input_water_depth = data.init_WD if isinstance(data.init_WD, np.ndarray) else data.init_WD.numpy()
        else:
            input_water_depth = get_input_water(data)[:, 0]
        if data.BC.dim() == 3:
            BC = data.BC[:, -1]
        else:
            BC = data.BC
        real_volume = torch.cumsum(BC.T * data.edge_BC_length * data.temporal_res * 60, 0).T.cpu().numpy()  # [m^2/s * m * s] = [m^3]
    else:
        raise ValueError("Data must have either 'WD' or 'BC' attribute to calculate volumes.")

    # ensure numpy arrays
    real_volume = np.asarray(real_volume)
    if real_volume.ndim == 2:
        real_volume = real_volume.T

    # compute initial volume and ensure numpy type
    if isinstance(data, Batch):
        init_volume = torch.tensor(
            [(data.area * input_water_depth)[data.node_ptr[i, 0]:data.node_ptr[i, 1]].sum()
             for i in range(data.num_graphs)]).cpu().numpy()
    else:
        input_water_depth_ms = separate_multiscale_node_features(input_water_depth, data.node_ptr)[scale]
        area = separate_multiscale_node_features(data.area, data.node_ptr)[scale].numpy()
        # if input_water_depth_ms is a torch tensor convert to numpy
        if hasattr(input_water_depth_ms, "cpu") and not isinstance(input_water_depth_ms, np.ndarray):
            input_water_depth_ms = input_water_depth_ms.cpu().numpy()
        init_volume = np.asarray(input_water_depth_ms) @ np.asarray(area)  # [m^3]

    # ensure init_volume is numpy array (covers any remaining torch tensors)
    if hasattr(init_volume, "cpu") and not isinstance(init_volume, np.ndarray):
        init_volume = init_volume.cpu().numpy()
    init_volume = np.asarray(init_volume)

    # boolean flag -> int for arithmetic, subtract initial volume only if WD exists
    subtract_init = 1 if hasattr(data, 'WD') else 0

    return init_volume, real_volume[time_start:] - init_volume * subtract_init

def plot_BCs(type_BCs, BCs, dataset, ax=None, highlight_ids=None):
    """Plot discharge and water level boundary conditions over time.

    Args:
        type_BCs (Tensor): BC type flags per node (1=water level, 2=discharge)
        BCs (Tensor, shape [N_bc, T]): boundary condition values
        dataset (list): dataset objects with edge_BC_length and DEM attributes
        ax (Axes, optional): matplotlib axes to plot on
        highlight_ids (list, optional): simulation ids to highlight in red

    Returns:
        Axes: matplotlib axes with the plot
    """
    if ax is None: ax = plt.gca()

    discharge_BC_nodes = type_BCs == 2
    water_depth_BC_nodes = type_BCs == 1
    cell_lengths = torch.cat([data.edge_BC_length.cpu() for data in dataset]).cpu()
    water_level_DEM = torch.cat([data.DEM[data.node_BC].cpu() for data in dataset])[water_depth_BC_nodes]

    discharges = (BCs[discharge_BC_nodes].T * cell_lengths[discharge_BC_nodes]).T.cpu()
    water_levels = (BCs[water_depth_BC_nodes].T + water_level_DEM).T.cpu()

    time_vector = get_time_vector(BCs.shape[1]-1, dataset[0].temporal_res)

    # plot both BC types
    if discharges.shape[0] != 0 and water_levels.shape[0] != 0:
        ax2 = ax.twinx()
        ax2.set_ylabel('Water depth [m]', color='purple')
        ax.set_ylabel('Discharge [$m^3$/s]', color='royalblue')

        plot_line_with_deviation(time_vector, discharges, with_minmax=True, ax=ax, c='royalblue')
        plot_line_with_deviation(time_vector, water_levels, with_minmax=True, ax=ax2, c='purple')

        if highlight_ids is not None:
            for i in highlight_ids:
                ax.plot(time_vector, discharges[i], c='red', linestyle='--', label='Valid simulations')
                ax2.plot(time_vector, water_levels[i], c='red', linestyle='--', label='Valid simulations')

    elif discharges.shape[0] != 0:
        ax.set_ylabel('Discharge [$m^3$/s]')
        plot_line_with_deviation(time_vector, discharges, with_minmax=True, ax=ax, c='royalblue')

        if highlight_ids is not None:
            for i in highlight_ids:
                ax.plot(time_vector, discharges[i], c='red', linestyle='--', label='Valid simulations')

    elif water_levels.shape[0] != 0:
        ax.set_ylabel('Water depth [m]')
        plot_line_with_deviation(time_vector, water_levels, with_minmax=True, ax=ax, c='purple')

        if highlight_ids is not None:
            for i in highlight_ids:
                ax.plot(time_vector, water_levels[i], c='red', linestyle='--', label='Valid simulations')

    ax.set_xlabel('Time [h]')
    ax.set_title('Boundary conditions')
    
    return ax

def get_average_relative_mass_error(V_true, V_pred, epsilon=1e-6):
    """Compute Average Relative Mass Error (ARME) between true and predicted volumes.

    Args:
        V_true (np.ndarray, shape [N, T]): ground truth volume time series
        V_pred (np.ndarray, shape [N, T]): predicted volume time series
        epsilon (float): small constant to avoid division by zero

    Returns:
        float or np.ndarray: ARME score per simulation (values < 0.25 are good)
    """
    assert V_true.shape == V_pred.shape, "V_true and V_pred must have the same shape"

    if V_true.ndim == 1:
        V_true = V_true[None, :]
        V_pred = V_pred[None, :]
    
    ARME = np.abs((V_pred - V_true) / (V_true + epsilon)).mean(1)

    if ARME.shape[0] == 1:
        return ARME[0]
    return ARME

class SpatialAnalysis():
    def __init__(self, predicted_rollout, prediction_times, dataset, **temporal_test_dataset_parameters):
        self.dataset = dataset
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.temporal_res = dataset[0].temporal_res
        self.breach_coords = np.stack([data.mesh.face_xy[data.node_BC] for data in self.dataset])

        temporal_dataset = TemporalFloodDataset(dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
        if temporal_test_dataset_parameters.get('save_on_gpu', False):
            temporal_dataset = temporal_dataset._save_on_gpu()

        self.real_rollout = [correct_rollout_shape_future_t(data.y) for data in temporal_dataset]
        self.predicted_rollout = predicted_rollout
        self.prediction_times = prediction_times        
        self.BCs = torch.cat([data.BC for data in self.dataset])
        self.type_BCs = torch.cat([data.type_BC.cpu() for data in self.dataset])
        total_time_steps = self.real_rollout[0].shape[-1]+self.time_start
        self.time_vector = get_time_vector(total_time_steps, self.temporal_res)

        if isinstance(self.dataset[0].mesh, MultiscaleMesh):
            masks = [create_scale_mask(data.num_nodes, data.mesh.num_meshes, data.node_ptr, data) == 0 for data in self.dataset]
            self.real_rollout = [real[masks[i]] for i, real in enumerate(self.real_rollout)]
            self.predicted_rollout = [pred[masks[i]] for i, pred in enumerate(self.predicted_rollout)]
        
    def _plot_metric_rollouts(self, metric_name, metric_function, water_thresholds=[0.05, 0.3], ax=None):
        """Plot a classification metric over time for multiple flood thresholds.

        Args:
            metric_name (str): label for the metric (e.g. 'CSI', 'F1')
            metric_function (callable): function accepting (pred, real, water_threshold)
            water_thresholds (list): flood depth thresholds to evaluate
            ax (Axes, optional): matplotlib axes to plot on

        Returns:
            tuple: (Axes, np.ndarray) axes and metric values array
        """
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        all_metric = []
        for wt in water_thresholds:
            metric = torch.stack([metric_function(pred, real, water_threshold=wt) 
                                  for pred, real in zip(self.predicted_rollout, self.real_rollout)]).to('cpu').numpy()
            all_metric.append(metric)
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(self.time_vector, metric, ax=ax, label=f'{metric_name}$_{{{wt}m}}$')
            # plt.legend()
            
        ax.set_xlabel('Time [h]')
        ax.set_title(f'{metric_name} score')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=4)
        
        return ax, np.array(all_metric)
    
    def _plot_rollouts(self, type_loss, ax=None):
        """Plot rollout loss over time for each water variable on dual y-axes.

        Args:
            type_loss (str): loss type, e.g. 'RMSE' or 'MAE'
            ax (Axes, optional): matplotlib axes to plot on

        Returns:
            Axes: matplotlib axes with the plot
        """
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        water_labels = ['h [m]', '|q| [$m^2$/s]']
        var_colors = ['royalblue', 'purple']
        lines = []

        ax2 = ax.twinx()
        axx = ax

        diff_rollout = torch.stack([get_mean_error(pred - real, type_loss, nodes_dim=0) for pred, real 
                     in zip(self.predicted_rollout, self.real_rollout)])

        for var in range(diff_rollout.shape[1]):
            average_diff_t = diff_rollout[:,var,:].to('cpu').numpy()
            average_diff_t = add_null_time_start(self.time_start, average_diff_t)
            lines.append(plot_line_with_deviation(self.time_vector, average_diff_t, ax=axx,
                                        label=water_labels[var], c=var_colors[var])[0])
            axx = ax2

        ax.tick_params(axis='y', colors='royalblue')
        ax2.tick_params(axis='y', colors=lines[var].get_color())            
        axx = ax
        ax.set_xlabel('Time [h]')
        ax.set_title(type_loss)

        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc=2)
        
        return ax
    
    def _plot_BCs(self, ax=None, highlight_ids=None):
        """Plot boundary conditions over time.

        Args:
            ax (Axes, optional): matplotlib axes to plot on
            highlight_ids (list, optional): simulation ids to highlight in red

        Returns:
            Axes: matplotlib axes with the plot
        """
        return plot_BCs(self.type_BCs, self.BCs, self.dataset, ax=ax, highlight_ids=highlight_ids)

    def _get_CSI(self, water_threshold=0):
        """Compute stacked CSI over all simulations for a given flood threshold.

        Args:
            water_threshold (float): depth threshold for flood classification

        Returns:
            Tensor: CSI values stacked across simulations
        """
        CSI = [get_CSI_rollout(pred, real, water_threshold=water_threshold) for pred, real in zip(self.predicted_rollout, self.real_rollout)]
        return torch.stack(CSI)
        
    def _get_F1(self, water_threshold=0):
        """Compute stacked F1 score over all simulations for a given flood threshold.

        Args:
            water_threshold (float): depth threshold for flood classification

        Returns:
            Tensor: F1 values stacked across simulations
        """
        F1 = [get_F1_rollout(pred, real, water_threshold=water_threshold) for pred, real in zip(self.predicted_rollout, self.real_rollout)]
        return torch.stack(F1)

    def plot_CSI_rollouts(self, water_thresholds=[0.05, 0.3], ax=None):
        """Plot CSI over time for multiple flood thresholds.

        Args:
            water_thresholds (list): flood depth thresholds to evaluate
            ax (Axes, optional): matplotlib axes to plot on

        Returns:
            tuple: (Axes, np.ndarray) axes and CSI values array
        """
        return self._plot_metric_rollouts('CSI', get_CSI_rollout, water_thresholds=water_thresholds, ax=ax)

    def plot_F1_rollouts(self, water_thresholds=[0.05, 0.3], ax=None):
        """Plot F1 score over time for multiple flood thresholds.

        Args:
            water_thresholds (list): flood depth thresholds to evaluate
            ax (Axes, optional): matplotlib axes to plot on

        Returns:
            tuple: (Axes, np.ndarray) axes and F1 values array
        """
        return self._plot_metric_rollouts('F1', get_F1_rollout, water_thresholds=water_thresholds, ax=ax)
    
    def _get_rollout_loss(self, type_loss='RMSE', only_where_water=False):
        """Compute stacked rollout loss over all simulations.

        Args:
            type_loss (str): loss function, e.g. 'RMSE' or 'MAE'
            only_where_water (bool): if True, restrict loss to wet nodes

        Returns:
            Tensor: loss values stacked across simulations
        """
        rollout_losses = [get_rollout_loss(pred, real, type_loss=type_loss, only_where_water=only_where_water)
                          for pred, real in zip(self.predicted_rollout, self.real_rollout)]

        return torch.stack(rollout_losses)

    def plot_loss_per_simulation(self, type_loss='RMSE', water_thresholds=[0.05, 0.3],
                                 ranking='loss', only_where_water=False):
        """Plot loss and CSI for each simulation, sorted by a chosen ranking criterion.

        Args:
            type_loss (str): loss function, e.g. 'RMSE' or 'MAE'
            water_thresholds (list): flood thresholds for CSI computation
            ranking (str): sort criterion; one of 'loss', 'CSI', 'volume', 'ARME', 'combined'
            only_where_water (bool): if True, restrict loss to wet nodes

        Returns:
            np.ndarray: sorted simulation indices
        """
        rollout_loss = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water).cpu().numpy()
        CSIs = torch.stack([self._get_CSI(wt) for wt in water_thresholds], 1).nanmean(2).cpu().numpy()

        assert rollout_loss.ndim == 2, "rollout_loss should have dimension [S, O]"\
            "where S is the number of simulations and O is the output dimension"
        if rollout_loss.shape[0] == 1:
            raise ValueError("This plot works only for multiple simulations")

        fig, axs = plt.subplots(4, 1, figsize=(18,12), sharex='col')
        _ = self.get_plausible_runs()

        if ranking == 'loss':
            sorted_ids = rollout_loss.mean(1).argsort()
        elif ranking == 'CSI':
            sorted_ids = CSIs.mean(1).argsort().flip(-1)
        elif ranking == 'combined':
            loss_CSI = (1-CSIs.mean(1))*rollout_loss.mean(1)
            sorted_ids = loss_CSI.argsort()
        elif ranking == 'volume':
            sorted_ids = self.real_volumes[:,-1].argsort().flip(-1).numpy()
        elif ranking == 'ARME':
            sorted_ids = self.ARME.argsort()[::-1]
        else:
            raise ValueError("ranking can only be either 'loss' or 'CSI', 'volume', 'ARME', 'combined'")

        axs[0].set_title(f'{ranking} ranking for test simulations')
        n_x_ticks = range(len(sorted_ids))

        # total flood volume
        axs[0].plot(self.real_volumes[:,-1][sorted_ids]/10e6, 'o--')
        axs[0].set_ylabel('Volume [$10^6 m^3$]')
        
        # Error
        axs[2].plot(rollout_loss[sorted_ids, 0], 'o--', label='h [m]', c='royalblue')
        ax2 = axs[2].twinx()
        ax2.plot(rollout_loss[sorted_ids, 1], 'o--', label='|q| [$m^2$/s]', c='purple')
        axs[2].plot([], [], 'o--', label='|q| [$m^2$/s]', c='purple')
        axs[2].set_ylabel(type_loss)
        axs[2].set_yscale('log')
        ax2.set_yscale('log')
        axs[2].legend()
        
        # CSI
        axs[1].set_xticks(n_x_ticks)
        axs[1].set_xticklabels(sorted_ids)
        
        [axs[1].plot(CSIs[sorted_ids, i], 'o--', label=f'CSI$_{{{wt}}}$') for i, wt in enumerate(water_thresholds)]
        axs[1].set_ylim(0,1)
        axs[1].set_xlabel('Simulation id')
        axs[1].set_ylabel('CSI')
        axs[1].legend()

        # ARME volumes
        axs[3].plot(self.ARME[sorted_ids], 'ok--')
        axs[3].set_ylim(0, 1)
        axs[3].set_ylabel('ARME [-]')

        for y in [1, 0, -1, -10]:
            axs[3].axhline(y, color='gray', linestyle=':', linewidth=0.8)

        fig.subplots_adjust(wspace=0, hspace=0.05)

        return sorted_ids

    def plot_summary(self, numerical_times, type_loss='RMSE', water_thresholds=[0.05, 0.3],
                     only_where_water=False, figsize=(10,5)):
        """Plot summary boxplots of CSI, loss, and execution times across all simulations.

        Args:
            numerical_times (array-like): numerical model computation times in seconds
            type_loss (str): loss function, e.g. 'RMSE' or 'MAE'
            water_thresholds (list): flood thresholds for CSI computation
            only_where_water (bool): if True, restrict loss to wet nodes
            figsize (tuple): figure size

        Returns:
            Figure: matplotlib figure with summary plots
        """
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        RMSE = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water).cpu()
        CSIs = [self._get_CSI(wt).nanmean(1).cpu() for wt in water_thresholds]

        axs[0].boxplot(CSIs)
        axs[0].set_ylim(0,1)
        axs[0].set_xticklabels([r'$\tau$'f'={wt}m' for wt in water_thresholds])
        axs[0].set_title(r'CSI$_\tau$ [-]')

        axs[1].boxplot((RMSE[:,0], RMSE[:,1:].mean(1)))
        axs[1].set_xticklabels(('h [m]', '|q| [$m^2$/s]'))
        axs[1].set_title(f'{type_loss}')
        axs[1].set_yscale('log')

        axs[2].boxplot((self.prediction_times, numerical_times))
        axs[2].set_title('Execution times [sec]')
        axs[2].set_xticklabels(('DL', 'Numerical'))
        axs[2].set_ylim(0)

        plt.tight_layout()

        return fig

    def get_plausible_runs(self, ARME_threshold=0.25):
        """Return indices of simulations with ARME below threshold, computing volumes if needed.

        Args:
            ARME_threshold (float): maximum acceptable ARME value

        Returns:
            np.ndarray: indices of plausible simulations
        """
        if not hasattr(self, 'real_volumes'):
            volumes = [calculate_volumes(data, self.time_start) for data in self.dataset]
            self.init_volume = np.stack([v[0] for v in volumes])
            self.real_volumes = np.stack([v[1] for v in volumes])
            
            predicted_volumes = torch.stack([(self.predicted_rollout[i][:, 0, :].T.to(torch.float32) @ torch.as_tensor(self.dataset[i].mesh.meshes[0].face_area, 
                                                                                                                    device=self.predicted_rollout[i].device, dtype=torch.float32))
                                                   for i in range(len(self.predicted_rollout))]).cpu().numpy()
            self.predicted_volumes = predicted_volumes - self.init_volume.reshape(-1,1)
            self.diff_volumes = self.predicted_volumes - self.real_volumes

            mask_WD_nan = np.isnan(self.predicted_volumes)
            self.predicted_volumes[mask_WD_nan] = 0

            self.ARME = np.array([get_average_relative_mass_error(self.real_volumes[i], self.predicted_volumes[i]) for i in range(len(self.real_volumes))])

        plausible_ids = np.where(self.ARME < ARME_threshold)[0]

        return plausible_ids