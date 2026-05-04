## Libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from copy import copy
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, LogNorm, ListedColormap, BoundaryNorm
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import scipy
import sys
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

from database.graph_creation import MultiscaleMesh, remove_ghost_cells, plot_faces, plot_mesh, plot_mesh_and_dual, plot_edges, find_closest_nodes
from utils.miscellaneous import add_null_time_start, WD_to_FAT, plot_line_with_deviation, get_CSI_rollout, get_F1_rollout, get_rollout_loss, plot_BCs
from utils.miscellaneous import get_velocity, get_Froude, correct_rollout_shape_future_t, get_mean_error, get_time_vector, calculate_volumes, get_average_relative_mass_error
from utils.dataset import get_input_water, separate_multiscale_node_features, TemporalFloodDataset
from utils.scaling import get_none_scalers

WD_color = LinearSegmentedColormap.from_list('', ['white', 'MediumBlue'])
V_color = LinearSegmentedColormap.from_list('', ['white', 'darkviolet'])
diff_color_positive = LinearSegmentedColormap.from_list('', ['white', '#5D3A9B'])
diff_color_negative = LinearSegmentedColormap.from_list('', ['white', '#E66100'])
probability_color = LinearSegmentedColormap.from_list('', ['white', '#a50f15'], N=8)
probability_color = plt.get_cmap('autumn_r')

def add_transparency_colormap(cmap):
    rgba = cmap(np.arange(cmap.N))
    rgba[:, -1] = np.linspace(0, 1, cmap.N)**0.2
    cmap = ListedColormap(rgba)
    cmap.set_bad(color='none')
    return cmap

probability_color = add_transparency_colormap(probability_color)
WD_color = add_transparency_colormap(WD_color)
V_color = add_transparency_colormap(V_color)
diff_color_positive = add_transparency_colormap(diff_color_positive)
diff_color_negative = add_transparency_colormap(diff_color_negative).reversed()
diff_color = LinearSegmentedColormap.from_list('diff_color', 
                                               np.vstack((diff_color_negative(np.linspace(0., 1, 128)), 
                                                          diff_color_positive(np.linspace(0., 1, 128)))))

# Values specifc to Dike ring 41 
FAT_bounds = [0, 8, 16, 24, 48, 72, 168, 504]
FAT_norm = BoundaryNorm(FAT_bounds, ncolors=len(FAT_bounds))
FAT_color = mpl.cm.get_cmap('magma_r', len(FAT_bounds)).reversed()
FAT_color.set_bad(color='none')

def get_coords(pos):
    """Return array of x and y coordinates of each node.

    Args:
        pos (dict or array): node positions; if dict, keys are (x,y) indices and values are spatial coordinates

    Returns:
        np.ndarray: shape [N, 2]
    """
    if isinstance(pos, dict):
        coordinates = np.array([xy for xy in pos.values()])
    else:
        coordinates = pos
    return coordinates

def get_corners(pos):
    """Return the corner coordinates of a grid.

    Args:
        pos (dict): node positions; keys are (x,y) indices and values are spatial coordinates

    Returns:
        tuple: (BL, TR, BR, TL) corner coordinates
    """
    BL = min(pos.values()) #bottom-left
    TR = max(pos.values()) #top-right
    BR = (BL[0], TR[1]) #bottom-right
    TL = (TR[0], BL[1]) #top-left
    
    return BL, TR, BR, TL

def plot_loss(train_losses, val_losses=None, scale='log'):
    """Plot training (and optionally validation) losses after training.

    Args:
        train_losses (list): training losses per epoch
        val_losses (list, optional): validation losses per epoch
        scale (str): y-axis scale, e.g. 'linear', 'log'
    """
    plt.plot(train_losses, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale(scale)
    
    if val_losses is not None:
        plt.plot(val_losses, 'r-')
        plt.legend(['Training', 'Validation'], loc='upper right')
    plt.title('Loss vs. No. of epochs')
        
    return None

class BasePlotMap(object):
    """Base class for plotting a map defined by either a graph or an unstructured mesh.

    Args:
        map (np.ndarray or Tensor, shape [N], [N, 1] or [N_x, N_y]): single feature per domain point
        pos (dict, optional): node positions; keys are (x,y) indices and values are spatial coordinates
        graph (networkx.Graph, optional): graph with nodes and edges
        mesh (Mesh, optional): unstructured mesh object
        scaler (sklearn scaler, optional): scaler for inverse-transforming the map
        edge_index (Tensor, optional): edge connectivity
        difference_plot (bool): if True, uses diverging colormap
    """
    def __init__(self, map, pos=None, graph=None, mesh=None, scaler=None, edge_index=None, 
                 difference_plot=False, **kwargs):
        self.map = map
        self.scaler = scaler
        self.pos = pos
        self.graph = graph
        self.mesh = mesh
        self.edge_index = edge_index
        self.kwargs = {**kwargs}
        self.difference_plot = difference_plot
    
        self.map = self._check_device(self.map)
        self._check_map_type()

    def _check_map_dimension(self, map):
        '''map must be of dimension [N] when plotting'''
        if len(map.shape)>1:
            map = map.reshape(-1)
        return map

    def _scale_map(self, map):
        '''Scales back map, given scaler'''
        if self.scaler is not None:
            if len(map.shape)==1:
                map = map.reshape(-1, 1)
            map = self.scaler.inverse_transform(map)
        map = self._check_map_dimension(map)
        return map

    def _check_device(self, map):
        '''Convert map to cpu'''
        if isinstance(map, torch.Tensor):
            if map.device.type != 'cpu':
                map = map.to('cpu')
            map = map.float().numpy()
        return map

    def _check_map_type(self):
        if self.graph is None and self.mesh is None:
            raise AttributeError("BasePlotMap must receive either a graph 'graph' or a Mesh 'mesh'")

    def _get_vmin(self, map):
        if 'vmin' not in self.kwargs:
            self.kwargs['vmin'] = np.nanmin(map)
            
    def _get_vmax(self, map):
        if 'vmax' not in self.kwargs:
            self.kwargs['vmax'] = np.nanmax(map)

    def _create_axes(self, ax=None):
        if ax is None:
            ax = plt.gca()
        return ax

    def _get_cmap(self):
        if self.difference_plot:
            if self.kwargs['vmin'] >= 0:
                self.kwargs['vmin'] = 0
                self.kwargs['cmap'] = diff_color_positive
            elif self.kwargs['vmax'] <= 0:
                self.kwargs['vmax'] = 0
                self.kwargs['cmap'] = diff_color_negative
            else:
                self.kwargs['cmap'] = diff_color
        elif 'cmap' not in self.kwargs:
            self.kwargs['cmap'] = plt.cm.plasma

    def _add_colorbar(self, ax=None, colorbar=True, logscale=False):
        ax = self._create_axes(ax=ax)
        self.kwargs['vmax'] = self._check_device(self.kwargs['vmax'])
        self.kwargs['vmin'] = self._check_device(self.kwargs['vmin'])
        if self.difference_plot:
            if self.kwargs['vmin'] >= 0:
                ticks_interval = np.linspace(0, self.kwargs['vmax'], 5, endpoint=True)
                norm = plt.Normalize(vmin = 0, vmax=self.kwargs['vmax'])
            elif self.kwargs['vmax'] <= 0:
                ticks_interval = np.linspace(self.kwargs['vmin'], 0, 5, endpoint=True)
                norm = plt.Normalize(vmin=self.kwargs['vmin'], vmax=0)
            else:
                ticks_interval = np.linspace(self.kwargs['vmin'], self.kwargs['vmax'], 5, endpoint=True)
                norm = TwoSlopeNorm(vmin=self.kwargs['vmin'], vcenter=0, vmax=self.kwargs['vmax'])
        elif logscale and self.kwargs['vmin'] >= 0:
                ticks_interval = np.logspace(-3, -1, 3, endpoint=True)
                norm = LogNorm(vmin=1e-3, vmax=self.kwargs['vmax'], clip=True)
        else:                        
            ticks_interval = np.linspace(self.kwargs['vmin'], self.kwargs['vmax'], 5, endpoint=True)
            norm = plt.Normalize(vmin = self.kwargs['vmin'], vmax=self.kwargs['vmax'])

        norm = self.kwargs.pop('norm', norm)

        if colorbar:
            decimals = 2
            if logscale:
                self.clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.kwargs['cmap']), 
                            ticks=ticks_interval, fraction=0.05, shrink=0.9, ax=ax)
            else:
                self.clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.kwargs['cmap']), 
                        #  ticks=np.sign(ticks_interval)*np.floor(np.abs(ticks_interval)*10**decimals)/10**decimals,
                         fraction=0.05, shrink=0.9, ax=ax)
                
        return norm
    
    def plot_map(self, ax=None, colorbar=True, logscale=False):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar, logscale=logscale)
        
        if self.graph is not None:
            '''Plot map as graph'''
            nx.draw_networkx_nodes(self.graph, pos=self.pos, node_color=self.map, node_shape='s', node_size=20, 
                                   ax=ax, **self.kwargs)
        if self.mesh is not None:
            '''Plot map as mesh'''
            vmin = self.kwargs.pop('vmin')
            vmax = self.kwargs.pop('vmax')
            plot_faces(self.mesh, ax=ax, face_value=self.map, norm=norm, **self.kwargs)
            
        return ax

    def plot_edge_map(self, ax=None, colorbar=True):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar)
        self.kwargs['edge_vmin'] = self.kwargs.pop('vmin')
        self.kwargs['edge_vmax'] = self.kwargs.pop('vmax')
        self.kwargs['edge_cmap'] = self.kwargs.pop('cmap')
        edge_list = self.edge_index.T.numpy()
        
        if self.graph is None:
            raise NotImplementedError("This function only works with graphs as input")
        else:
            '''Plot edges of a graph'''
            nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=edge_list, 
                edge_color=self.map, ax=ax, **self.kwargs)
            
        return ax

class TemporalPlotMap(BasePlotMap):
    """Plot class for maps with temporal attributes.

    Args:
        map (np.ndarray or Tensor, shape [N, T] or [N, 1]): temporal map to plot
        temporal_res (int): temporal resolution in minutes
        time_start (int): index of the first time step
    """
    def __init__(self, map, temporal_res, time_start=0, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.temporal_res = temporal_res
        self.time_start = time_start
        self.total_time = self.map.shape[1]

    def _get_map_at_time_step(self, map):
        if self.total_time > 1:
            map = map[:, self.time_step]
        return map

    def _get_current_time_step(self):
        # Take function out -> pytest
        self.time_in_minutes = (self.time_start + 1 + self.time_step%self.total_time)*self.temporal_res
        self.time_in_hours = int(self.time_in_minutes/60)
    
    def plot_map(self, time_step, ax=None, colorbar=True, logscale=False):
        self.time_step = time_step
        self._get_current_time_step()
        map = self._get_map_at_time_step(self.map)
        
        map = self._scale_map(map)
        self._get_vmin(map)
        self._get_vmax(map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar, logscale=logscale)
        
        if self.graph is not None:
            '''Plot map as graph'''
            nx.draw_networkx_nodes(self.graph, pos=self.pos, node_color=map, node_shape='s', node_size=20, 
                                   ax=ax, **self.kwargs)
        if self.mesh is not None:
            '''Plot map as mesh'''
            mesh_kwargs = copy(self.kwargs)
            vmin = mesh_kwargs.pop('vmin')
            vmax = mesh_kwargs.pop('vmax')
            plot_faces(self.mesh, ax=ax, face_value=map, norm=norm, **mesh_kwargs)

        return ax
    
def correct_plt_units(ax, pos, x_label=True, y_label=True):
    """Rescale axis tick labels from m to km if coordinates exceed 1000 m.

    Args:
        ax (Axes): matplotlib axes to correct
        pos (Tensor or np.ndarray, shape [N, 2]): node positions
        x_label (bool): whether to add x-axis label
        y_label (bool): whether to add y-axis label

    Returns:
        Axes: updated matplotlib axes
    """
    exp_size = int(f'{pos.mean():.2e}'[-2:])
    distance_unit = 'm' if exp_size < 3 else 'km'
    if distance_unit == 'km':
        m2km = lambda x, _: f'{x/1000:g}'
        ax.xaxis.set_major_formatter(m2km)
        ax.yaxis.set_major_formatter(m2km)

    if x_label:
        ax.set_xlabel(f'x distance [{distance_unit}]')
    if y_label:
        ax.set_ylabel(f'y distance [{distance_unit}]')

    return ax

class DEMPlotMap(BasePlotMap):
    """Plot digital elevation model (DEM) using terrain colormap."""
    def __init__(self, map, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.kwargs['cmap'] = 'terrain'
        
    def _add_axes_info(self, ax, title=True, x_label=True, y_label=True):
        if title:
            ax.set_title('DEM (m)')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        correct_plt_units(ax, self.pos, x_label, y_label)

        # ax.set_xticks(np.linspace(self.pos.min(0)[0].round(-3), self.pos.max(0)[0].round(-3), num=5).round(0))
        # ax.set_yticks(np.linspace(self.pos.min(0)[1].round(-3), self.pos.max(0)[1].round(-3), num=5).round(0))

    def _add_breach_location(self, ax, breach_coordinates, type_BC=None):
        """Plot breach locations as scatter markers on the axes.

        Args:
            ax (Axes): matplotlib axes
            breach_coordinates (list or Tensor): x,y coordinates of each breach
            type_BC (Tensor, optional): boundary condition type per breach node
        """
        for i, breach in enumerate(breach_coordinates):
            if type_BC is not None:
                c = 'purple' if type_BC[i] == 1 else 'royalblue'
            ax.scatter(breach[0], breach[1], s=200, c=c, marker='X', zorder=3, edgecolor='k')

def plot_rollout_diff_in_time_all(diff_rollout, temporal_res, type_loss='RMSE',
                              time_start=0, ax=None):
    """Plot average node error for water depth and discharge across time.

    Args:
        diff_rollout (Tensor, shape [N, V, T]): difference between predicted and real rollout
        temporal_res (int): temporal resolution in minutes
        type_loss (str): error metric name, e.g. 'RMSE'
        time_start (int): index of the first time step
        ax (Axes, optional): matplotlib axes

    Returns:
        tuple: (ax, ax2) primary and secondary axes
    """
    ax = ax or plt.gca()

    V_unit = "$m^2$/s"

    # WD plot
    lns = plot_rollout_diff_in_time_var(
        diff_rollout, temporal_res, type_loss, dim=0, 
        time_start=time_start, ax=ax, label='h', c='royalblue')

    ax.set_ylabel(f'h {type_loss} [m]')
    ax.set_xlabel('Time [h]')
    ax.set_xlim(0)

    ax2 = ax.twinx()
    V_symbol = "|q|"
    # V
    lin_V = plot_rollout_diff_in_time_var(
        diff_rollout, temporal_res, type_loss, dim=1, 
        time_start=time_start, ax=ax2, label=V_symbol, c='purple')
    lns = lns + lin_V
    ax2.set_ylabel(f'{V_symbol} {type_loss} [{V_unit}]')

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    
    return ax, ax2

def plot_rollout_diff_in_time_var(diff_rollout, temporal_res, type_loss='RMSE', dim=0,
                                  time_start=0, ax=None, **plot_kwargs):
    """Plot average node error for one variable across time (dim: 0=WD, 1=Vx/V, 2=Vy).

    Args:
        diff_rollout (Tensor, shape [N, V, T]): difference between predicted and real rollout
        temporal_res (int): temporal resolution in minutes
        type_loss (str): error metric name, e.g. 'RMSE'
        dim (int): variable dimension index
        time_start (int): index of the first time step
        ax (Axes, optional): matplotlib axes

    Returns:
        list: matplotlib line objects
    """
    diff_rollout = diff_rollout[:,dim,:].to('cpu')

    ax = ax or plt.gca()
    
    time_stop = diff_rollout.shape[-1]
    time_vector = np.linspace(0, (time_start+time_stop)*temporal_res/60, time_stop+time_start+1)
        
    average_diff_t = get_mean_error(diff_rollout, type_loss).numpy()

    average_diff_t = add_null_time_start(time_start, average_diff_t)
    
    return ax.plot(time_vector, average_diff_t, marker='.', **plot_kwargs)

def plot_breach_distribution(breach_coords, discharges, gdf_mesh, ax=None, with_label=True, discharge_intervals=None, volume_intervals=None, temporal_res=480, **plt_kwargs):
    """Plot breach locations colored by discharge interval and shaped by volume interval.

    Args:
        breach_coords (np.ndarray, shape [N, 2]): x,y coordinates of breaches
        discharges (np.ndarray, shape [N, T]): discharge time series per breach
        gdf_mesh (Mesh): mesh object for boundary plotting
        ax (Axes, optional): matplotlib axes
        with_label (bool): whether to annotate breach indices
        discharge_intervals (list, optional): bin edges for max discharge coloring
        volume_intervals (list, optional): bin edges for total volume marker shape
        temporal_res (int): temporal resolution in seconds

    Returns:
        Axes: matplotlib axes
    """
    ax = ax or plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    gdf_mesh.plot_boundary(ax=ax)
    
    if discharge_intervals is not None:
        total_volumes = np.cumsum(discharges * temporal_res * 60, 1)[:, -1]
        max_discharges = discharges.max(1)
        color_indices = np.digitize(max_discharges, discharge_intervals)

        # create volume-based bins (tertiles by default) and marker mapping
        if volume_intervals is None:
            volume_intervals = np.quantile(total_volumes, np.linspace(0, 1, 4)[1:-1])  # two thresholds -> 3 bins
        shape_indices = np.digitize(total_volumes, volume_intervals)

        n_color_bins = len(discharge_intervals) + 1
        cmap = plt.get_cmap(plt_kwargs.pop('cmap', 'cividis'), n_color_bins)

        # available markers (extend if you expect more bins)
        markers_list = plt_kwargs.pop('markers', ['o', 's', '^', 'D', 'v', 'P', 'X'])
        markers = markers_list[:len(volume_intervals) + 1]

        # build color legend handles
        handles = []
        for i in range(n_color_bins):
            if i == 0:
                label = f"<{discharge_intervals[0]}"
            elif i == n_color_bins - 1:
                label = f">{discharge_intervals[-1]}"
            else:
                label = f"{discharge_intervals[i-1]}–{discharge_intervals[i]}"
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=label))

        # build shape legend handles (show volumes in 10^6 m^3)
        shape_handles = []
        for j in range(len(volume_intervals) + 1):
            if j == 0:
                lab = f"<{volume_intervals[0]/1e6:.0f}"
            elif j == len(volume_intervals):
                lab = f">{volume_intervals[-1]/1e6:.0f}"
            else:
                lab = f"{volume_intervals[j-1]/1e6:.0f}–{volume_intervals[j]/1e6:.0f}"
            shape_handles.append(Line2D([0], [0], marker=markers[j], color='k', linestyle='None', markersize=8, label=lab))

        # scatter points per volume-bin so each point gets a marker, color still encodes discharge-bin
        for j, mk in enumerate(markers):
            mask = shape_indices == j
            if not np.any(mask):
                continue
            ax.scatter(breach_coords[mask, 0], breach_coords[mask, 1],
                   s=150, marker=mk, zorder=3, c=color_indices[mask], cmap=cmap, 
                   norm=mpl.colors.BoundaryNorm(np.arange(n_color_bins+1)-0.5, n_color_bins),**plt_kwargs)

        # add two legends: color (discharge) and shape (volume)
        leg1 = ax.legend(handles=handles, title="Max discharge \n[m$^3$/s]", fontsize=12, loc='upper left')
        ax.add_artist(leg1)
        leg1.get_title().set_ha("center")  # Center-align the title
        leg2 = ax.legend(shape_handles, [h.get_label() for h in shape_handles],
              title="Total volume \n[$10^6$ m$^3$]", fontsize=12, loc='lower left')
        leg2.get_title().set_ha("center")  # Center-align the title
    else:
        ax.scatter(*breach_coords.T, s=150, marker='X', zorder=3, **plt_kwargs)

    if with_label:
        for i, breach in enumerate(breach_coords):
            plt.annotate(i, (breach[0], breach[1]), ha='right', va='bottom')

    correct_plt_units(ax, gdf_mesh.face_xy)

    ax.set_aspect('equal')
    plt.tight_layout()

    return ax

def plot_discharge_breach_distribution(dict_breach_coords_discharges, gdf_mesh, ax=None, **plt_kwargs):
    """Plot breach locations across datasets; shape encodes dataset, color encodes max discharge.

    Args:
        dict_breach_coords_discharges (dict): mapping dataset name to (coords, volumes) tuples
        gdf_mesh (Mesh): mesh object for boundary plotting
        ax (Axes, optional): matplotlib axes

    Returns:
        Axes: matplotlib axes
    """
    ax = ax or plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    gdf_mesh.plot_boundary(ax=ax)

    # Assign a marker for each dataset
    markers_list = ['s', 'X', 'o', '^', 'D', 'v', 'P', '*', 'H', '8']
    datasets = list(dict_breach_coords_discharges.keys())
    marker_map = {ds: markers_list[i % len(markers_list)] for i, ds in enumerate(datasets)}

    # Gather all volumes for color normalization
    all_volumes = np.concatenate([v for _, v in dict_breach_coords_discharges.values() if v is not None])
    vmin, vmax = all_volumes.min(), all_volumes.max()
    cmap = plt.get_cmap(plt_kwargs.pop('cmap', 'viridis'))

    # Plot each dataset with its marker, color by volume
    handles = []
    for ds in datasets:
        coords, volumes = dict_breach_coords_discharges[ds]
        if volumes is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=150, marker=marker_map[ds], 
                       c='red', zorder=3)
            handles.append(Line2D([0], [0], marker=marker_map[ds], color='w', markerfacecolor='red', markersize=10, label=ds))
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], s=150, marker=marker_map[ds], 
                            c=volumes, cmap=cmap, vmin=vmin, vmax=vmax, zorder=3, **plt_kwargs)
            handles.append(Line2D([0], [0], marker=marker_map[ds], color='w', markerfacecolor='gray', markersize=10, label=ds))

    # Legend for dataset (shape)
    leg1 = ax.legend(handles=handles, title="Dataset", fontsize=16, loc='upper left')
    ax.add_artist(leg1)
    leg1.get_title().set_ha("center")

    # Colorbar for volume
    cbar = plt.colorbar(sc, ax=ax, pad=0.01, label="Max discharge [m$^3$/s]")
    cbar.ax.ticklabel_format(style='sci', scilimits=(0,0))

    correct_plt_units(ax, gdf_mesh.face_xy)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return ax

def plot_percentage_plausible_volumes_vs_ARME(ARME, prob_test_dataset, ARME_thresholds=[1.0, 0.8, 0.6, 0.4, 0.2], time_start=0, ax=None):
    """Plot percentage of plausible runs per flood volume bin for each ARME threshold.

    Args:
        ARME (np.ndarray, shape [N]): ARME scores per simulation
        prob_test_dataset (list): list of dataset objects
        ARME_thresholds (list): ARME threshold values to compare
        time_start (int): index of the first time step
        ax (Axes, optional): matplotlib axes

    Returns:
        Axes: matplotlib axes
    """

    ax = ax or plt.gca()

    volumes = [calculate_volumes(data, time_start) for data in prob_test_dataset]
    real_volumes = np.concatenate([v[1] for v in volumes], 1)
    all_volumes = real_volumes[-1] / 1e6

    volume_ranges = np.linspace(all_volumes.min(), all_volumes.max(), 21)    
    plausible_counts = np.zeros((len(ARME_thresholds) + 1, len(volume_ranges)-1), dtype=int)

    for i in range(len(volume_ranges)-1):
        mask = (all_volumes >= volume_ranges[i]) & (all_volumes < volume_ranges[i+1])
        for j, threshold in enumerate(ARME_thresholds):
            if mask.sum() == 0:
                plausible = 0
            else:
                plausible = np.sum((ARME[mask] < threshold))*100 / mask.sum()
            plausible_counts[j, i] = plausible
        # Add the "ARME >= max(ARME_thresholds)" bar
        if mask.sum() == 0:
            not_plausible = 0
        else:
            not_plausible = np.sum((ARME[mask] >= 0))*100 / mask.sum()
        plausible_counts[-1, i] = not_plausible

    cmap = mpl.colormaps.get_cmap('YlGnBu_r')
    colors = cmap(np.linspace(0, 1, len(ARME_thresholds)))
    color_not_plausible = 'lightgray'

    bin_centers = 0.5 * (volume_ranges[:-1] + volume_ranges[1:])
    bin_widths = volume_ranges[1:] - volume_ranges[:-1]

    # Plot for ARME >= max(ARME_thresholds)
    ax.bar(bin_centers, plausible_counts[-1], width=bin_widths, align='center', alpha=0.8, label=f'ARME≥{max(ARME_thresholds)}', color=color_not_plausible)
    for j, threshold in enumerate(ARME_thresholds):
        ax.bar(bin_centers, plausible_counts[j], width=bin_widths, align='center', alpha=0.8, label=f'ARME<{threshold}', color=colors[j])

    ax.set_xlabel('Total flood volume [$10^6 m^3$]')
    ax.set_ylabel('Plausible runs [%]')
    ax.set_ylim(0, 105)
    ax.grid(False)

    # Move legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax

def plot_valid_runs_breach_distribution(base_datas, plausible_runs_per_location, ax=None, **plt_kwargs):
    """Plot breach locations colored by percentage of plausible simulations.

    Args:
        base_datas (list): list of dataset objects with breach location info
        plausible_runs_per_location (np.ndarray, shape [N]): percentage of plausible runs per location
        ax (Axes, optional): matplotlib axes

    Returns:
        Axes: matplotlib axes
    """
    ax = ax or plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    breach_coordinates = [[data.mesh.face_xy[node.item()] for node in data.node_BC] for data in base_datas]
    breach_coordinates = np.array(breach_coordinates).squeeze()

    scat = ax.scatter(*breach_coordinates.T, c=plausible_runs_per_location, s=120, marker='X', zorder=3, **plt_kwargs)
    cb = plt.colorbar(scat, fraction=0.05, shrink=0.8, ax=ax)
    cb.set_label('Percentage of plausible \nsimulations [%]', fontsize=20)
        
    base_datas[0].mesh.meshes[-1].plot_boundary(ax=ax)
    correct_plt_units(ax, base_datas[0].mesh.face_xy)

    ax.set_aspect('equal')
    plt.tight_layout()

    return ax

def plot_breach_distribution_and_quantiles(ensemble_predicted_volumes, prob_test_dataset, num_breach_groups=None,
                                           q_ranges=[0, 25, 50, 75, 100], max_ARME=2, show_percentage_runs=False, time_start=0):
    """Plot breach distribution, flood volume histogram, and plausible-run curves per volume quantile.

    Args:
        ensemble_predicted_volumes (np.ndarray, shape [N, T]): ensemble predicted volumes
        prob_test_dataset (list): list of test dataset objects
        num_breach_groups (int, optional): number of breach groups for coloring
        q_ranges (list of int): quantile boundaries, e.g. [0, 25, 50, 75, 100]
        max_ARME (float): x-axis upper limit for ARME threshold plots
        show_percentage_runs (bool): if True, y-axis shows percentage instead of count
        time_start (int): index of the first time step

    Returns:
        Figure: matplotlib figure
    """
    num_breaches = len(prob_test_dataset)
    if num_breach_groups is None:
        num_breach_groups = num_breaches

    volumes = [calculate_volumes(data, time_start) for data in prob_test_dataset]
    init_volume = np.stack([v[0] for v in volumes])
    real_volumes = np.concatenate([v[1] for v in volumes], 1)
    all_volumes = real_volumes[-1] / 1e6
    
    quantiles = np.percentile(all_volumes, q_ranges)
    cmap = plt.get_cmap('nipy_spectral', num_breach_groups) if num_breaches > 1 else plt.get_cmap('Reds', 1)
    colors = [cmap(i) for i in range(num_breach_groups)]

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, len(quantiles)-1, height_ratios=[1, 0.5])
        
    # Top left: breach distribution
    n_cols_breach = int(gs.ncols//1.3)
    ax_breach = fig.add_subplot(gs[0, :n_cols_breach])
    ax_breach.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    breach_coordinates = np.array([data.breach_coords for data in prob_test_dataset]).squeeze()
    # if num_breaches > 1:
    #     change_index = np.where(np.array([data.CODE[:2] for data in prob_test_dataset]) != np.array([data.CODE[:2] for data in prob_test_dataset])[0])[0][0]
    #     breach_coordinates[change_index:] = breach_coordinates[change_index:][::-1]
    group_indices = np.repeat(np.arange(num_breach_groups), num_breaches // num_breach_groups)
    if len(group_indices) < num_breaches:
        group_indices = np.concatenate([group_indices, group_indices[-(num_breaches - len(group_indices)):]])

    ax_breach.scatter(*breach_coordinates.T, s=120, marker='X', zorder=3, c=group_indices, cmap=cmap, linewidths=0.25, edgecolors='black')
    prob_test_dataset[0].gdf_mesh.plot_boundary(ax=ax_breach)
    correct_plt_units(ax_breach, prob_test_dataset[0].gdf_mesh.face_xy)
    ax_breach.set_aspect('equal')

    # Top right: Density distribution
    ax_density = fig.add_subplot(gs[0, n_cols_breach:])
    ax_density.hist(all_volumes, bins=20, alpha=0.5, label='Real volumes', color='C0')

    # Fit a Weibull distribution
    dist_params = scipy.stats.weibull_min.fit(all_volumes)  # fit only positive values
    x = np.linspace(all_volumes.min(), all_volumes.max(), 1000)
    pdf = scipy.stats.weibull_min.pdf(x, *dist_params)
    # ax_density.plot(x, pdf, 'r-', label='Weibull')

    # Quartiles
    for q in q_ranges[1:-1]:
        quantile = np.percentile(all_volumes, q)
        plt.axvline(quantile, color='k', linestyle='--', linewidth=1)
        plt.text(quantile, ax_density.get_ylim()[1]*0.9, f'$Q_{{{q}}}$', rotation=90, va='top', ha='right', fontsize=20)

    # ax_density.legend(loc='center right')
    ax_density.set_xlabel('Total flood volume [$10^6$ m$^3$]', fontsize=22)
    ax_density.set_ylabel('Frequency', fontsize=22)
    ax_density.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_density.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    num_scenarios = all_volumes.shape[0]
    num_scenarios_per_location = num_scenarios // num_breaches

    # Bottom: Threshold plots for quantiles
    max_plausible = 0
    for i in range(len(quantiles) - 1):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f"$Q_{{{q_ranges[i]}}}$ - $Q_{{{q_ranges[i + 1]}}}$", fontsize=22)

        if i == 0:            
            if show_percentage_runs:
                ax.set_ylabel('% plausible runs', fontsize=22)
            else:
                ax.set_ylabel('number of plausible runs', fontsize=22)
        mask_volume = (all_volumes >= quantiles[i]) & (all_volumes < quantiles[i + 1]) #& (full_prob_analyser.ARME > 0)
        mask_breach = np.arange(num_scenarios) // num_scenarios_per_location

        for breach_idx in range(num_breach_groups):
            group_mask = np.isin(mask_breach, np.where(group_indices == breach_idx)[0])
            combined_mask = mask_volume & group_mask
            if not combined_mask.any():
                continue
            selected_prob_dataset = [prob_test_dataset[i] for i in np.where(combined_mask)[0]]

            # Calculate the volumes
            volumes = [calculate_volumes(data, time_start) for data in selected_prob_dataset]
            init_volume = np.stack([v[0] for v in volumes])
            real_volumes = np.concatenate([v[1] for v in volumes], 1)
            predicted_volumes = ensemble_predicted_volumes[combined_mask] - init_volume.reshape(-1,1)

            final_time = predicted_volumes.shape[-1]
            real_volumes = real_volumes[:final_time].T

            if np.isnan(predicted_volumes).any():
                predicted_volumes[np.isnan(predicted_volumes)] = 0

            ARME = get_average_relative_mass_error(real_volumes, predicted_volumes).reshape(-1,1)
            
            thresholds = np.linspace(0, max_ARME, 50)
            all_runs = []
            for threshold_val in thresholds:
                plausible = np.where(ARME < threshold_val)[0]
                all_runs.append(len(plausible))
            all_runs = np.array(all_runs)
            if show_percentage_runs:
                all_runs = all_runs / len(selected_prob_dataset) * 100
            ax.plot(thresholds, all_runs, color=colors[breach_idx], alpha=1)

            max_plausible = max(max_plausible, all_runs.max())
            ax.set_xlim(0, thresholds[-1])
            ax.set_xticks(np.append(np.arange(0, thresholds[-1], 0.5), thresholds[-1]))
            ax.set_xlabel('ARME threshold [-]', fontsize=20)

    # set y-axis limits for all plots in the bottom of the figure
    for i, ax in enumerate(fig.axes):
        if i > 1:
            if show_percentage_runs:
                ax.set_ylim(0, 105)
            elif i > 1:
                ax.set_ylim(0, max_plausible*1.05)
    
    panel_labels = [f'({chr(97+i)})' for i in range(len(fig.axes))]
    for i, label in enumerate(panel_labels):
        if i == 0:
            x_space = 0.01
            ha='left'
        elif i == 1:
            x_space = 0.98
            ha='right'
        else:
            x_space = 0.02
            ha='left'
        fig.axes[i].text(x_space, 0.98, label, transform=fig.axes[i].transAxes,
                    fontsize=20, va='top', ha=ha)

    return fig

def plot_percentiles(ensemble_selected_prediction, test_dataset=None, quantiles=[0.05, 0.5, 0.95],
                     variable='FAT', water_threshold=0.05, temporal_res=480, **default_plot_kwargs):
    """Plot spatial percentile maps for a variable from an ensemble prediction.

    Args:
        ensemble_selected_prediction (np.ndarray, shape [n_samples, n_nodes, n_vars]): ensemble predictions
        test_dataset (Dataset, optional): dataset for deterministic reference plot
        quantiles (list of float): quantile levels to plot
        variable (str): variable to plot; one of 'FAT', 'WD', 'WD_max', 'V_max'
        water_threshold (float): water depth threshold for FAT; must be 0.05 or 0.3
        temporal_res (int): temporal resolution in minutes
        **default_plot_kwargs: must include 'pos' and 'mesh'

    Returns:
        tuple: (Figure, np.ndarray of Axes)
    """
    assert variable in ['FAT', 'WD', 'WD_max', 'V_max'], "Variable must be either 'FAT', 'WD', 'WD_max' or 'V_max'."
    assert water_threshold in [0.05, 0.3], "Water threshold must be either 0.05 or 0.3."
    
    dict_variable_index = {'FAT': [0, 1], 'WD': 2, 'WD_max': 4, 'V_max': 3}
    dict_index_FAT = {0.05: 0, 0.3: 1}

    if test_dataset is None:
        fig, axs = plt.subplots(len(quantiles), 1, figsize=(10, len(quantiles)*4))
    else:
        assert len(quantiles) == 3, "If test_dataset is provided, quantiles must be of length 3."
        fig, axs = plt.subplots(2, 2, figsize=(15, 7))

    axs = axs.flatten()

    if variable == 'FAT':
        FAT = ensemble_selected_prediction[:,:,dict_index_FAT[water_threshold]] # shape (n_samples, n_nodes)
        
        FAT_quantiles = np.nanquantile(np.where(np.isnan(FAT), 9999, FAT), quantiles, axis=0).astype(np.float32)
        FAT_quantiles = np.where(FAT_quantiles > 999, np.nan, FAT_quantiles)

        FAT_kwargs = dict(**default_plot_kwargs, norm=FAT_norm, cmap=FAT_color)

        for ax, arr in zip(axs, FAT_quantiles):
            bp = BasePlotMap(arr, **FAT_kwargs)
            bp.plot_map(ax=ax)

        if test_dataset is not None:
            FAT = WD_to_FAT(test_dataset.WD, temporal_res, water_threshold)
            base_FAT = separate_multiscale_node_features(FAT, test_dataset.node_ptr)[0].numpy()[:-1]

            FAT_quantile_plot = BasePlotMap(base_FAT, **FAT_kwargs)
            FAT_quantile_plot.plot_map(ax=axs[-1])

    else:
        WD = ensemble_selected_prediction[:,:,dict_variable_index[variable]]

        WD_quantiles = np.nanquantile(WD, quantiles, axis=0).astype(np.float32)
        vmax = default_plot_kwargs.pop('vmax', np.nanquantile(WD_quantiles, 0.95))
        cmap = WD_color if variable in ['WD', 'WD_max'] else V_color
        WD_kwargs = dict(**default_plot_kwargs, cmap=cmap, vmax=vmax)

        for ax, arr in zip(axs, WD_quantiles):
            arr = np.nan_to_num(arr, nan=0.0)
            bp = BasePlotMap(arr, **WD_kwargs)
            bp.plot_map(ax=ax)

        if test_dataset is not None:
            if variable == 'WD_max':
                WD = test_dataset.WD.max(1)[0]
            elif variable == 'V_max':
                WD = test_dataset.V.max(1)[0]
            else:
                WD = test_dataset.WD[:,-1]
            base_WD = separate_multiscale_node_features(WD, test_dataset.node_ptr)[0].numpy()
            WD_quantile_plot = BasePlotMap(base_WD, **WD_kwargs)
            WD_quantile_plot.plot_map(ax=axs[-1])

    xlims = axs[0].get_xlim()
    ylims = axs[0].get_ylim()
    for ax in axs:
        correct_plt_units(ax, default_plot_kwargs['mesh'].face_xy)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])

    # panel_labels = [f'({chr(97+i)})' for i in range(len(axs))]
    # for i, label in enumerate(panel_labels):
    #     axs[i].text(0.02, 0.98, label, transform=axs[i].transAxes,
    #                 fontsize=18, va='top', ha='left')

    for ax, q in zip(axs, quantiles):
        ax.set_title(f'{q*100:.0f}th Percentile', fontsize=22)
    if test_dataset is not None:
        axs[-1].set_title('Deterministic')

    return fig, axs

def plot_percentile_diff(ensemble_selected_prediction, test_dataset, quantiles=[0.05, 0.5, 0.95], deterministic_prediction=None,
                         variable='FAT', water_threshold=0.05, temporal_res=480, **default_plot_kwargs):
    """Plot ensemble percentile maps alongside their difference from the ground truth.

    Args:
        ensemble_selected_prediction (np.ndarray, shape [n_samples, n_nodes, n_vars]): ensemble predictions
        test_dataset (Dataset): dataset providing ground-truth values
        quantiles (list of float): quantile levels to plot
        deterministic_prediction (np.ndarray, optional): deterministic prediction to show in extra row
        variable (str): variable to plot; one of 'FAT', 'WD', 'WD_max'
        water_threshold (float): water depth threshold for FAT; must be 0.05 or 0.3
        temporal_res (int): temporal resolution in minutes
        **default_plot_kwargs: must include 'pos' and 'mesh'; accepts 'diff_value', 'vmax'

    Returns:
        tuple: (Figure, np.ndarray of Axes)
    """
    assert variable in ['FAT', 'WD', 'WD_max'], "Variable must be either 'FAT', 'WD' or 'WD_max'."
    assert water_threshold in [0.05, 0.3], "Water threshold must be either 0.05 or 0.3."

    dict_index_FAT = {0.05: 0, 0.3: 1}
    show_deterministic = deterministic_prediction is not None
    unit = 'h' if variable == 'FAT' else 'm'
    n_rows = len(quantiles) + show_deterministic*2
    
    fig = plt.figure(figsize=(15, n_rows * 3))
    gs = fig.add_gridspec(n_rows, 4, wspace=0.3, hspace=0.3)

    axs = np.empty((n_rows, 2), dtype=object)

    # Next rows: quantiles and deterministic
    for i in range(show_deterministic, n_rows):
        axs[i, 0] = fig.add_subplot(gs[i, :2])
        axs[i, 1] = fig.add_subplot(gs[i, 2:])
    if show_deterministic:
        ax_gt = fig.add_subplot(gs[0, 1:3])
        ax_gt.set_aspect('equal')

    if variable == 'FAT':
        FAT = WD_to_FAT(test_dataset.WD, temporal_res, water_threshold)
        base_FAT = separate_multiscale_node_features(FAT, test_dataset.node_ptr)[0].numpy()[:-1]
        FAT = ensemble_selected_prediction[:, :, dict_index_FAT[water_threshold]]  # shape (n_samples, n_nodes)
        FAT_quantiles = np.nanquantile(np.where(np.isnan(FAT), 9999, FAT), quantiles, axis=0).astype(np.float32)
        FAT_quantiles = np.where(FAT_quantiles > 999, np.nan, FAT_quantiles)

        diff_value = default_plot_kwargs.pop('diff_value', 48)
        FAT_kwargs = dict(**default_plot_kwargs, norm=FAT_norm, cmap=FAT_color)
        diff_kwargs = dict(**default_plot_kwargs, cmap=diff_color, vmin=-diff_value, vmax=diff_value)

        for ax, arr in zip(axs[show_deterministic:len(quantiles)+1], FAT_quantiles):
            bp = BasePlotMap(arr, **FAT_kwargs)
            bp.plot_map(ax=ax[0])

        for ax, arr in zip(axs[show_deterministic:len(quantiles)+1], FAT_quantiles):
            bp = BasePlotMap(np.nan_to_num(arr, nan=0.0) - np.nan_to_num(base_FAT, nan=0.0), **diff_kwargs)
            bp.plot_map(ax=ax[1])

        if show_deterministic:
            # Top row: left and right empty, center for real
            bp_real = BasePlotMap(base_FAT, **FAT_kwargs)
            bp_real.plot_map(ax=ax_gt)

            # Plot deterministic prediction vs real
            bp_pred = BasePlotMap(deterministic_prediction, **FAT_kwargs)
            bp_pred.plot_map(ax=axs[-1, 0])
            bp_pred = BasePlotMap(np.nan_to_num(deterministic_prediction, nan=0.0) - np.nan_to_num(base_FAT, nan=0.0), **diff_kwargs)
            bp_pred.plot_map(ax=axs[-1, 1])

    elif variable == 'WD' or variable == 'WD_max':
        WD = ensemble_selected_prediction[:, :, 2] if variable == 'WD' else ensemble_selected_prediction[:, :, -1]
        base_WD = separate_multiscale_node_features(
            test_dataset.WD.max(1)[0] if variable == 'WD' else test_dataset.WD[:, -1],
            test_dataset.node_ptr
        )[0].numpy()[:-1]
        WD_quantiles = np.nanquantile(WD, quantiles, axis=0).astype(np.float32)

        diff_value = default_plot_kwargs.pop('diff_value', 1)
        vmax = default_plot_kwargs.pop('vmax', 3)
        WD_kwargs = dict(**default_plot_kwargs, cmap=WD_color, vmax=vmax)
        diff_kwargs = dict(**default_plot_kwargs, cmap=diff_color, vmin=-diff_value, vmax=diff_value)

        for ax, arr in zip(axs[show_deterministic:len(quantiles)+1], WD_quantiles):
            bp = BasePlotMap(arr, **WD_kwargs)
            bp.plot_map(ax=ax[0])

        for ax, arr in zip(axs[show_deterministic:len(quantiles)+1], WD_quantiles):
            bp = BasePlotMap(arr - base_WD, **diff_kwargs)
            bp.plot_map(ax=ax[1])

        if show_deterministic:
            # Top row: left and right empty, center for real
            bp_real = BasePlotMap(base_WD, **WD_kwargs)
            bp_real.plot_map(ax=ax_gt)

            # Plot deterministic prediction vs real
            bp_pred = BasePlotMap(deterministic_prediction, **WD_kwargs)
            bp_pred.plot_map(ax=axs[-1, 0])
            bp_pred = BasePlotMap(deterministic_prediction - base_WD, **diff_kwargs)
            bp_pred.plot_map(ax=axs[-1, 1])

    if show_deterministic:    
        ax_gt.set_title(f'Real [{unit}]')
        axs[-1, 0].set_title(f'Deterministic prediction [{unit}]')
        axs[-1, 1].set_title(f'Deterministic prediction - real [{unit}]')

    # xlims = axs[0].get_xlim()
    # ylims = axs[0].get_ylim()
    axs = axs.flatten()[show_deterministic*2:]
    
    for ax in axs:
        # correct_plt_units(ax, default_plot_kwargs['mesh'].face_xy)
        # ax.set_xlim(xlims)
        # ax.set_ylim(ylims)
        ax.set_aspect('equal')

    # # Hide x labels for all but the last two axes
    # for ax in axs[:-2]:
    #     ax.set_xlabel('')
    #     ax.set_xticklabels([])

    # # Hide y labels for every odd axis (right column)
    # for i in range(1, len(axs), 2):
    #     axs[i].set_ylabel('')
    #     axs[i].set_yticklabels([])

    # # # Panel labels: (a), (b), (c), ...
    # # panel_labels = [f'({chr(97 + i)})' for i in range(len(axs))]
    # # for i, label in enumerate(panel_labels):
    # #     axs[i].text(0.02, 0.98, label, transform=axs[i].transAxes,
    # #                 fontsize=18, va='top', ha='left')

    for ax, q in zip(axs[::2][:len(quantiles)], quantiles):
        ax.set_title(f'${q * 100:.0f}^{{th}}$ Percentile [{unit}]')
    for ax, q in zip(axs[1::2][:len(quantiles)], quantiles):
        ax.set_title(f'${q * 100:.0f}^{{th}}$ Percentile - real [{unit}]')
    
    return fig, axs

def plot_WD_max_probability(ensemble_selected_prediction, wd_thresholds=[0.05, 0.3, 1], **default_plot_kwargs):
    """Plot probability maps of max water depth exceeding given thresholds.

    Args:
        ensemble_selected_prediction (np.ndarray, shape [n_samples, n_nodes, n_vars]): ensemble predictions
        wd_thresholds (list of float): water depth thresholds in metres
        **default_plot_kwargs: must include 'pos' and 'mesh'

    Returns:
        tuple: (Figure, np.ndarray of Axes)
    """
    assert len(wd_thresholds) > 0, "Provide at least one threshold."
    # layout: if deterministic dataset provided, require 3 thresholds to keep 2x2 layout (same logic as plot_percentiles)
    fig, axs = plt.subplots(len(wd_thresholds), 1, figsize=(10, len(wd_thresholds) * 4))

    axs = axs.flatten()

    WD = ensemble_selected_prediction[:,:,2] # shape (n_samples, n_nodes)

    for i, water_threshold in enumerate(wd_thresholds):
        flooded_areas = WD > water_threshold
        WD_prob = flooded_areas.sum(0)/len(WD)

        WD_kwargs = dict(**default_plot_kwargs, cmap=probability_color, vmin=0, vmax=1)

        WD_prob = BasePlotMap(WD_prob, **WD_kwargs)
        WD_prob.plot_map(ax=axs[i])
        axs[i].set_title(f'P (WD$_{{max}}$ > {water_threshold} m)')

    # unify axes extents/aspect and tidy ticks/labels
    xlims = axs[0].get_xlim()
    ylims = axs[0].get_ylim()
    for ax in axs:
        correct_plt_units(ax, default_plot_kwargs['mesh'].face_xy)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal')

        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.set_ylabel('')
        ax.set_yticklabels([])

    # panel labels (a,b,c,...)
    panel_labels = [f'({chr(97+i)})' for i in range(len(axs))]
    for i, label in enumerate(panel_labels):
        axs[i].text(0.02, 0.98, label, transform=axs[i].transAxes,
                    fontsize=18, va='top', ha='left')

    return fig, axs

def add_inset_distribution_to_ax(var, x, y, meshes, ax=None, **bin_kwargs):
    """Add a histogram inset at (x, y) if the point lies inside the mesh boundary.

    Args:
        var (np.ndarray, shape [N_samples, N_nodes]): variable values across ensemble
        x (float): x-coordinate of the inset location
        y (float): y-coordinate of the inset location
        meshes (list): multiscale mesh objects
        ax (Axes, optional): matplotlib axes

    Returns:
        Axes: matplotlib axes
    """
    ax = ax or plt.gca()

    meshes[0].plot_boundary(ax=ax)

    # Check if the point is within the boundary polygon
    if not meshes[0].gdf.contains(Point(x, y)).any():
        print("The point is outside the boundary polygon. No inset added.")
        return ax

    point_id = find_closest_nodes(meshes[-1].face_xy, [x, y], top_n=1)

    inset_ax = inset_axes(ax, width=0.8, height=0.8, loc='lower left',
                          bbox_to_anchor=(x, y, 1000, 1000),
                          bbox_transform=ax.transData)

    inset_ax.hist(var[:, point_id], **bin_kwargs)
    inset_ax.patch.set_alpha(0)  # Set background transparent
    xticks = np.linspace(np.floor(var[:, point_id].min() * 10) / 10, np.ceil(var[:, point_id].max() * 10) / 10, 4).round(2)
    inset_ax.set_xticks(xticks)
    inset_ax.set_xticklabels(xticks, fontsize=8)
    inset_ax.set_yticks([])
    inset_ax.set_xlabel("h [m]", fontsize=8)
    inset_ax.set_ylabel("Count", fontsize=8)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)

    ax.scatter(x, y, zorder=-1, marker='x', c='k')

    return ax

def plot_ARME_thresholds_analysis(full_prob_analyser, ensemble_selected_prediction, ensemble_predicted_volumes, prob_test_dataset, scalers,
                                 temporal_test_dataset_parameters, ARME_thresholds=[0.25, 0.5, 1, 2, 3]):
    """Plot predicted vs real volume curves filtered by successive ARME thresholds.

    Args:
        full_prob_analyser (ProbabilisticSpatialAnalysis): full-ensemble analyser object
        ensemble_selected_prediction (np.ndarray, shape [N, n_nodes, n_vars]): ensemble predictions
        ensemble_predicted_volumes (np.ndarray, shape [N, T]): predicted flood volumes
        prob_test_dataset (list): list of test dataset objects
        scalers (dict): scalers used during training
        temporal_test_dataset_parameters (dict): parameters for the temporal dataset
        ARME_thresholds (list of float): ARME cut-off values to evaluate

    Returns:
        tuple: (Figure, np.ndarray of Axes)
    """
    fig, axs = plt.subplots(1, len(ARME_thresholds), figsize=(len(ARME_thresholds)*5, 6))

    max_volume = max(ensemble_predicted_volumes.max(), full_prob_analyser.real_volumes.max())

    for i, ARME_threshold in enumerate(ARME_thresholds):
        mask_volume = full_prob_analyser.ARME < ARME_threshold

        selected_prob_dataset = [prob_test_dataset[j] for j in np.where(mask_volume)[0]]
        prob_analyser = ProbabilisticSpatialAnalysis(ensemble_selected_prediction[mask_volume],
                                                     ensemble_predicted_volumes[mask_volume],
                                                     selected_prob_dataset, scalers, full_prob_analyser.DEM, 
                                                     full_prob_analyser.mesh, **temporal_test_dataset_parameters)

        prob_analyser.plot_volumes_in_time(with_difference=False, ax=axs[i])
        axs[i].set_title(f'ARME < {ARME_threshold:.2f}', fontsize=26)
        axs[i].legend_.remove()

        if i != 0:
            axs[i].set_ylabel('')

        axs[i].text(0.15, 0.95, f'% Runs:\n{mask_volume.sum()/len(mask_volume)*100:.1f}%', 
                                          fontsize=18, ha='center', va='top', transform=axs[i].transAxes)
        axs[i].set_aspect('auto')
        axs[i].set_ylim(0, max_volume*1.1)
    axs[0].legend(loc='upper right', fontsize=18)
    plt.tight_layout()

    return fig, axs

def analyze_training_vs_good_tests(train_hydrographs, train_coords, test_hydrographs, test_coords,
                                   ARME_test, gdf_mesh, ARME_threshold=0.4, check_bads=False, k_neighbors=5):
    """Map test breach locations colored by hydrograph similarity to nearest training locations.

    Args:
        train_hydrographs (np.ndarray, shape [N_train, T]): training boundary condition hydrographs
        train_coords (np.ndarray, shape [N_train, 2]): training breach x,y coordinates
        test_hydrographs (np.ndarray, shape [N_test, T]): test boundary condition hydrographs
        test_coords (np.ndarray, shape [N_test, 2]): test breach x,y coordinates
        ARME_test (np.ndarray, shape [N_test]): ARME scores for each test run
        gdf_mesh (Mesh): mesh object for boundary plotting
        ARME_threshold (float): threshold to classify plausible runs
        check_bads (bool): if True, also plot runs with ARME >= threshold
        k_neighbors (int): number of nearest training locations to search

    Returns:
        Figure: matplotlib figure
    """
    if not check_bads:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        axes = [ax]
        masks = [ARME_test < ARME_threshold]
        labels = ['Plausible runs']
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        masks = [ARME_test < ARME_threshold, ARME_test >= ARME_threshold]
        labels = ['Plausible runs', 'Not plausible runs']

    num_scenarios_per_location = test_hydrographs.shape[0] // test_coords.shape[0]
    test_hydrographs = test_hydrographs.reshape(test_coords.shape[0], num_scenarios_per_location, -1)

    for plot_idx, ax, mask, label in zip(range(len(labels)), axes, masks, labels):
        # plot base mesh boundary
        gdf_mesh.plot_boundary(ax=ax, color='k', linewidth=0.5)

        # Build spatial tree of training locations
        tree = cKDTree(train_coords)
        nearest_train_idxs = tree.query(test_coords, k=k_neighbors)[1]  # (n_test, k) or (n_test,)

        best_errs_per_location = np.full((len(test_coords), num_scenarios_per_location), np.nan)
        mask_matrix = mask.reshape(test_coords.shape[0], num_scenarios_per_location)

        for i in range(len(test_coords)):
            target = test_hydrographs[i][mask_matrix[i]]
            if target.size == 0:
                continue
            for j in range(target.shape[0]):
                eps = 1e-8
                errs = np.linalg.norm(train_hydrographs[nearest_train_idxs[i]] - target[j], axis=1) / (np.linalg.norm(target[j]) + eps)
                best_errs_per_location[i][j] = errs[int(np.nanargmin(errs))]

        with np.errstate(invalid='ignore'):
            mean_errs = np.nanmean(best_errs_per_location, axis=1)

        valid = ~np.isnan(mean_errs)
        if valid.any():
            sc = ax.scatter(test_coords[valid, 0], test_coords[valid, 1], c=mean_errs[valid], label=label, 
                            cmap='Reds', edgecolors='k', linewidths=0.25, s=80, zorder=3)
            plt.colorbar(sc, ax=ax, label='mean relative L2 distance to\n{} closest training breaches'.format(k_neighbors), shrink=0.7)
        if (~valid).any():
            ax.scatter(test_coords[~valid, 0], test_coords[~valid, 1], label='No valid runs', 
                       c='black', marker='X', edgecolors='k', s=100, zorder=3, linewidths=0.25)
        ax.scatter(train_coords[:, 0], train_coords[:, 1], label='Training locations', 
                   c='green', s=60, alpha=0.8, marker='x', linewidth=1.5, zorder=4)
        ax.legend(fontsize=18)
        correct_plt_units(ax, gdf_mesh.face_xy)
        ax.set_aspect('equal')

        ax.set_title(f'{label} (ARME {["<", "≥"][plot_idx]} {ARME_threshold})', fontsize=26)

    if check_bads:
        axes[0].set_xlabel('')
        axes[0].set_xticklabels([])
    
    plt.tight_layout()

    return fig

def plot_correlation_ARME(spatial_analyser_train, spatial_analyser_val, spatial_analyser_test):

    water_thresholds = [0.05]
    markers = ['o', 's', '^', 'D'][:len(water_thresholds)]
    cmap = 'nipy_spectral'
    fontsize_r2 = 18

    fig = plt.figure(figsize=(21, 16))  # Wider for 3 columns
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1.2, 1.2, 1.2], width_ratios=[1, 1, 1])

    # Get unique breach coordinates and assign IDs as a dictionary
    unique_coords, _ = np.unique(spatial_analyser_val.breach_coords, axis=0, return_inverse=True)
    breach_dict = {i: coord for i, coord in enumerate(unique_coords)}
    breach_colors_val = list(breach_dict.keys())

    unique_coords, breach_colors_test = np.unique(spatial_analyser_test.breach_coords, axis=0, return_inverse=True)
    breach_dict = {i: coord for i, coord in enumerate(unique_coords)}

    unique_coords, _ = np.unique(spatial_analyser_train.breach_coords, axis=0, return_inverse=True)
    breach_dict = {i: coord for i, coord in enumerate(unique_coords)}
    breach_colors_train = np.array(list(breach_dict.keys()))

    sizes = [160, 80, 30]*3

    # Row 1: Map
    ax_map_train = fig.add_subplot(gs[0, 0])
    ax_map_val = fig.add_subplot(gs[0, 1])
    ax_map_test = fig.add_subplot(gs[0, 2])

    spatial_analyser_train.dataset[0].mesh.meshes[-1].plot_boundary(ax=ax_map_train, color='k', linewidth=0.5)
    ax_map_train.scatter(*spatial_analyser_train.breach_coords.T, c=breach_colors_train, cmap=cmap, s=100, zorder=5, marker='X', edgecolor='k', linewidth=0.5)
    ax_map_train.set_xticks([])
    ax_map_train.set_yticks([])

    spatial_analyser_val.dataset[0].mesh.meshes[-1].plot_boundary(ax=ax_map_val, color='k', linewidth=0.5)
    ax_map_val.scatter(*spatial_analyser_val.breach_coords.T, c=breach_colors_val, cmap=cmap, s=100, zorder=5, marker='X', edgecolor='k', linewidth=0.5)
    ax_map_val.set_xticks([])
    ax_map_val.set_yticks([])

    spatial_analyser_test.dataset[0].mesh.meshes[-1].plot_boundary(ax=ax_map_test, color='k', linewidth=0.5)
    ax_map_test.scatter(*spatial_analyser_test.breach_coords.T, c=breach_colors_test, cmap=cmap, s=100, zorder=5, marker='X', edgecolor='k', linewidth=0.5)
    ax_map_test.set_xticks([])
    ax_map_test.set_yticks([])

    ax_map_train.set_ylabel('Breach locations')
    ax_map_train.set_title('Train')
    ax_map_val.set_title('Val')
    ax_map_test.set_title('Test')

    # Row 2: CSI vs ARME (with regression lines)
    CSIs_train = torch.stack([spatial_analyser_train._get_CSI(wt) for wt in water_thresholds], 1).nanmean(2).cpu().numpy()
    CSIs_val = torch.stack([spatial_analyser_val._get_CSI(wt) for wt in water_thresholds], 1).nanmean(2).cpu().numpy()
    CSIs_test = torch.stack([spatial_analyser_test._get_CSI(wt) for wt in water_thresholds], 1).nanmean(2).cpu().numpy()

    rollout_loss_train = spatial_analyser_train._get_rollout_loss(type_loss='MAE').cpu().numpy()
    rollout_loss_val = spatial_analyser_val._get_rollout_loss(type_loss='MAE').cpu().numpy()
    rollout_loss_test = spatial_analyser_test._get_rollout_loss(type_loss='MAE').cpu().numpy()

    for i, (CSI_train, CSI_val, CSI_test) in enumerate(zip(CSIs_train.T, CSIs_val.T, CSIs_test.T), start=1):
        ax_train = fig.add_subplot(gs[i, 0])
        ax_val = fig.add_subplot(gs[i, 1])
        ax_test = fig.add_subplot(gs[i, 2])
        x_train = np.asarray(spatial_analyser_train.ARME)
        y_train = CSI_train

        selected_train = x_train < 5
        y_train = y_train[selected_train]
        x_train = x_train[selected_train]

        ax_train.scatter(x_train, y_train, c=breach_colors_train[selected_train], cmap=cmap, edgecolor='k', s=80, facecolors='none')
        ax_train.set_ylabel(f'CSI$_{{{water_thresholds[i-1]}m}}$')

        mask = np.isfinite(x_train) & np.isfinite(y_train)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x_train[mask], y_train[mask], 1)
            p = np.poly1d(coeffs)
            xs = np.linspace(x_train[mask].min(), x_train[mask].max(), 200)
            ax_train.plot(xs, p(xs), color='k', linestyle='--', linewidth=1)
            # r = np.corrcoef(x_train[mask], y_train[mask])[0, 1]
            spearman_r, _ = scipy.stats.spearmanr(x_train[mask], y_train[mask])
            ax_train.text(0.75, 0.45, f"$\\rho$={spearman_r:.2f}", transform=ax_train.transAxes,
                        va='top', ha='left', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

        x_val = np.asarray(spatial_analyser_val.ARME)
        y_val = CSI_val

        ax_val.scatter(x_val, y_val, c=breach_colors_val, cmap=cmap, edgecolor='k', s=80, facecolors='none')


        mask = np.isfinite(x_val) & np.isfinite(y_val)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x_val[mask], y_val[mask], 1)
            p = np.poly1d(coeffs)
            xs = np.linspace(x_val[mask].min(), x_val[mask].max(), 200)
            ax_val.plot(xs, p(xs), color='k', linestyle='--', linewidth=1)
            # r = np.corrcoef(x_val[mask], y_val[mask])[0, 1]
            spearman_r, _ = scipy.stats.spearmanr(x_val[mask], y_val[mask])
            ax_val.text(0.75, 0.45, f"$\\rho$={spearman_r:.2f}", transform=ax_val.transAxes,
                        va='top', ha='left', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

        x_test = np.asarray(spatial_analyser_test.ARME)
        y_test = CSI_test

        ax_test.scatter(x_test, y_test, c=breach_colors_test, cmap=cmap, edgecolor='k', s=sizes, facecolors='none')

        mask = np.isfinite(x_test) & np.isfinite(y_test)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x_test[mask], y_test[mask], 1)
            p = np.poly1d(coeffs)
            xs = np.linspace(x_test[mask].min(), x_test[mask].max(), 200)
            ax_test.plot(xs, p(xs), color='k', linestyle='--', linewidth=1)
            # r = np.corrcoef(x_test[mask], y_test[mask])[0, 1]
            spearman_r, _ = scipy.stats.spearmanr(x_test[mask], y_test[mask])
            ax_test.text(0.7, 0.3, f"$\\rho$={spearman_r:.2f}", transform=ax_test.transAxes,
                        va='top', ha='left', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

        min_y = min(np.nanmin(y_test), np.nanmin(y_val), np.nanmin(y_train)) * 0.9
        ax_train.set_ylim(min_y, 1)
        ax_val.set_ylim(min_y, 1)
        ax_test.set_ylim(min_y, 1)

        # for ax in [ax_train, ax_val, ax_test]:
        #     ax.axvline(0.4, color='grey', linestyle=':', linewidth=1.5, zorder=0)

    # Row 3: MAE h vs ARME (with regression)
    ax3_train = fig.add_subplot(gs[2, 0])
    ax3_val = fig.add_subplot(gs[2, 1])
    ax3_test = fig.add_subplot(gs[2, 2])

    y_train_mae = rollout_loss_train[:,0][selected_train]
    y_val_mae = rollout_loss_val[:,0]
    y_test_mae = rollout_loss_test[:,0]

    ax3_train.scatter(x_train, y_train_mae, c=breach_colors_train, cmap=cmap, edgecolor='k', s=80, facecolors='none')
    ax3_val.scatter(x_val, y_val_mae, c=breach_colors_val, cmap=cmap, edgecolor='k', s=80, facecolors='none')
    ax3_test.scatter(x_test, y_test_mae, c=breach_colors_test, cmap=cmap, edgecolor='k', s=sizes, facecolors='none')
    ax3_train.set_ylabel('MAE h [m]')

    mask = np.isfinite(x_train) & np.isfinite(y_train_mae)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_train[mask], y_train_mae[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_train[mask].min(), x_train[mask].max(), 200)
        ax3_train.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_train[mask], y_train_mae[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_train[mask], y_train_mae[mask])
        ax3_train.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax3_train.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    mask = np.isfinite(x_val) & np.isfinite(y_val_mae)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_val[mask], y_val_mae[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_val[mask].min(), x_val[mask].max(), 200)
        ax3_val.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_val[mask], y_val_mae[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_val[mask], y_val_mae[mask])
        ax3_val.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax3_val.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    mask = np.isfinite(x_test) & np.isfinite(y_test_mae)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_test[mask], y_test_mae[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_test[mask].min(), x_test[mask].max(), 200)
        ax3_test.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_test[mask], y_test_mae[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_test[mask], y_test_mae[mask])
        ax3_test.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax3_test.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    max_y = max(y_test_mae.max(), y_val_mae.max(), y_train_mae.max())*1.1
    min_y = min(y_test_mae.min(), y_val_mae.min(), y_train_mae.min())*0.9
    ax3_train.set_ylim(min_y, max_y)
    ax3_val.set_ylim(min_y, max_y)
    ax3_test.set_ylim(min_y, max_y)

    # Row 4: MAE q vs ARME (with regression)
    ax4_train = fig.add_subplot(gs[3, 0])
    ax4_val = fig.add_subplot(gs[3, 1])
    ax4_test = fig.add_subplot(gs[3, 2])

    y_train_mae_q = rollout_loss_train[:,1][selected_train]
    y_val_mae_q = rollout_loss_val[:,1]
    y_test_mae_q = rollout_loss_test[:,1]

    ax4_train.scatter(x_train, y_train_mae_q, c=breach_colors_train, cmap=cmap, edgecolor='k', s=80, facecolors='none')
    ax4_val.scatter(x_val, y_val_mae_q, c=breach_colors_val, cmap=cmap, edgecolor='k', s=80, facecolors='none')
    ax4_test.scatter(x_test, y_test_mae_q, c=breach_colors_test, cmap=cmap, edgecolor='k', s=sizes, facecolors='none')
    ax4_train.set_ylabel('MAE q [$m^2$/s]')

    mask = np.isfinite(x_train) & np.isfinite(y_train_mae_q)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_train[mask], y_train_mae_q[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_train[mask].min(), x_train[mask].max(), 200)
        ax4_train.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_train[mask], y_train_mae_q[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_train[mask], y_train_mae_q[mask])
        ax4_train.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax4_train.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    mask = np.isfinite(x_val) & np.isfinite(y_val_mae_q)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_val[mask], y_val_mae_q[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_val[mask].min(), x_val[mask].max(), 200)
        ax4_val.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_val[mask], y_val_mae_q[mask])[0, 1]    
        spearman_r, _ = scipy.stats.spearmanr(x_val[mask], y_val_mae_q[mask])
        ax4_val.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax4_val.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    mask = np.isfinite(x_test) & np.isfinite(y_test_mae_q)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_test[mask], y_test_mae_q[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_test[mask].min(), x_test[mask].max(), 200)
        ax4_test.plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_test[mask], y_test_mae_q[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_test[mask], y_test_mae_q[mask])
        ax4_test.text(0.85, 0.4, f"$\\rho$={spearman_r:.2f}", transform=ax4_test.transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    max_y = max(y_test_mae_q.max(), y_val_mae_q.max(), y_train_mae_q.max())*1.1
    min_y = min(y_test_mae_q.min(), y_val_mae_q.min(), y_train_mae_q.min())*0.9
    ax4_train.set_ylim(min_y, max_y)
    ax4_val.set_ylim(min_y, max_y)
    ax4_test.set_ylim(min_y, max_y)

    ax4_train.set_xlabel('ARME')
    ax4_val.set_xlabel('ARME')
    ax4_test.set_xlabel('ARME')

    # Custom legend for sizes (only for test column)
    size_handles = [Line2D([0], [0], marker='o', color='w', label='RP 10000', markerfacecolor='none', markeredgecolor='k', markersize=np.sqrt(sizes[0])),
                    Line2D([0], [0], marker='o', color='w', label='RP 1000', markerfacecolor='none', markeredgecolor='k', markersize=np.sqrt(sizes[1])),
                    Line2D([0], [0], marker='o', color='w', label='RP 100', markerfacecolor='none', markeredgecolor='k', markersize=np.sqrt(sizes[2]))]

    handles, labels = ax_test.get_legend_handles_labels()
    handles = handles[:3] + size_handles
    labels = labels[:3] + [h.get_label() for h in size_handles]
    ax_test.legend(handles, labels)

    plt.tight_layout()
    return fig

def plot_CSI_MAE(spatial_analyser_train, spatial_analyser_val, spatial_analyser_test, 
                 rollout_loss_train, rollout_loss_val, rollout_loss_test,
                   x_train, x_val, x_test, selected_train):    

    # Colors for datasets
    colors = {'train': '#1f77b4', 
            'val': '#ff7f0e', 
            'test': '#2ca02c'}
    water_thresholds = [0.05]
    markers = ['o', 's', '^', 'D'][:len(water_thresholds)]
    alpha = 0.8
    size = 80
    fontsize_r2 = 15

    # --- CSI Plot ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Compute CSIs
    for i, wt in enumerate(water_thresholds):
        CSIs_train = spatial_analyser_train._get_CSI(wt).nanmean(1).cpu().numpy()[selected_train]
        CSIs_val = spatial_analyser_val._get_CSI(wt).nanmean(1).cpu().numpy()
        CSIs_test = spatial_analyser_test._get_CSI(wt).nanmean(1).cpu().numpy()

        # Plot for threshold
        axs[0].scatter(x_train, CSIs_train, color=colors['train'], s=size, marker=markers[i], label='Train', alpha=alpha)
        axs[0].scatter(x_val, CSIs_val, color=colors['val'], s=size, marker=markers[i], label='Val', alpha=alpha)
        axs[0].scatter(x_test, CSIs_test, color=colors['test'], s=size, marker=markers[i], label='Test', alpha=alpha)

        # Regression line for all datasets
        x_all = np.concatenate([x_train, x_val, x_test])
        y_all = np.concatenate([CSIs_train, CSIs_val, CSIs_test])
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x_all[mask], y_all[mask], 1)
            p = np.poly1d(coeffs)
            xs_reg = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
            axs[0].plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
            # r = np.corrcoef(x_all[mask], y_all[mask])[0, 1]
            spearman_r, _ = scipy.stats.spearmanr(x_all[mask], y_all[mask])
            axs[0].text(0.85, 0.5, f"$\\rho$={spearman_r:.2f}", transform=axs[0].transAxes,
                        va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    axs[0].set_xlabel('ARME')
    axs[0].set_ylabel('CSI')

    # --- MAE Plot (water depth and discharge) ---
    # Water depth MAE (circle marker)
    axs[1].scatter(x_train, rollout_loss_train[:, 0][selected_train], s=size, color=colors['train'], marker='o', label='Train', alpha=alpha)
    axs[1].scatter(x_val, rollout_loss_val[:, 0], s=size, color=colors['val'], marker='o', label='Val', alpha=alpha)
    axs[1].scatter(x_test, rollout_loss_test[:, 0], s=size, color=colors['test'], marker='o', label='Test', alpha=alpha)

    # Regression line for all datasets (MAE Water Depth)
    x_all_mae_h = np.concatenate([x_train, x_val, x_test])
    y_all_mae_h = np.concatenate([rollout_loss_train[:, 0][selected_train], rollout_loss_val[:, 0], rollout_loss_test[:, 0]])
    mask = np.isfinite(x_all_mae_h) & np.isfinite(y_all_mae_h)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_all_mae_h[mask], y_all_mae_h[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_all_mae_h[mask].min(), x_all_mae_h[mask].max(), 200)
        axs[1].plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_all_mae_h[mask], y_all_mae_h[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_all_mae_h[mask], y_all_mae_h[mask])
        axs[1].text(0.85, 0.35, f"$\\rho$={spearman_r:.2f}", transform=axs[1].transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    axs[1].set_xlabel('ARME')
    axs[1].set_ylabel('MAE Water Depth')

    # Discharge MAE (square marker)
    axs[2].scatter(x_train, rollout_loss_train[:, 1][selected_train], s=size, color=colors['train'], label='MAE Q', alpha=alpha)
    axs[2].scatter(x_val, rollout_loss_val[:, 1], s=size, color=colors['val'], label='MAE Q', alpha=alpha)
    axs[2].scatter(x_test, rollout_loss_test[:, 1], s=size, color=colors['test'], label='MAE Q', alpha=alpha)

    # Regression line for all datasets (MAE Discharge)
    x_all_mae_q = np.concatenate([x_train, x_val, x_test])
    y_all_mae_q = np.concatenate([rollout_loss_train[:, 1][selected_train], rollout_loss_val[:, 1], rollout_loss_test[:, 1]])
    mask = np.isfinite(x_all_mae_q) & np.isfinite(y_all_mae_q)
    if mask.sum() >= 2:
        coeffs = np.polyfit(x_all_mae_q[mask], y_all_mae_q[mask], 1)
        p = np.poly1d(coeffs)
        xs_reg = np.linspace(x_all_mae_q[mask].min(), x_all_mae_q[mask].max(), 200)
        axs[2].plot(xs_reg, p(xs_reg), color='k', linestyle='--', linewidth=1)
        # r = np.corrcoef(x_all_mae_q[mask], y_all_mae_q[mask])[0, 1]
        spearman_r, _ = scipy.stats.spearmanr(x_all_mae_q[mask], y_all_mae_q[mask])
        axs[2].text(0.85, 0.35, f"$\\rho$={spearman_r:.2f}", transform=axs[2].transAxes,
                    va='center', ha='center', fontsize=fontsize_r2, bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))

    axs[2].set_ylabel('MAE Discharge')
    axs[2].set_xlabel('ARME')

    # Custom legend for color (dataset) and shape (metric)
    legend_elements_mae = [
        Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor=colors['train'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Val', markerfacecolor=colors['val'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test', markerfacecolor=colors['test'], markersize=10),
    ]
    axs[2].legend(handles=legend_elements_mae, loc='best', fontsize=20)

    plt.tight_layout()

    return axs

class PlotRollout():
    """Explore predictions vs real simulations for DEM, temporal losses, water depth, discharge, and differences.

    Args:
        plmodel (LightningModule): trained lightning model
        trainer (Trainer): lightning trainer used for prediction
        dataset (Dataset): dataset object with simulation data
        scalers (dict, optional): scalers used during training
        type_loss (str): loss metric name, e.g. 'RMSE'
        **temporal_test_dataset_parameters: parameters for the temporal test dataset
    """
    def __init__(self, plmodel, trainer, dataset, scalers=None, type_loss='RMSE', **temporal_test_dataset_parameters):
        super().__init__()
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.V_unit = "$m^2$/s" 
        self.V_label = "Discharge" 
        self.V_symbol = "q" 
        self.type_loss = type_loss
        self.water_threshold = 0

        self.data = dataset[0]
        self.DEM = self.data.DEM
        self.temporal_res = self.data.temporal_res
        self.num_scales = self.data.mesh.num_meshes if isinstance(self.data.mesh, MultiscaleMesh) else 1

        # convert to temporal dataset for predictions
        temporal_dataset = TemporalFloodDataset(dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
        test_dataloader = DataLoader(temporal_dataset, batch_size=1, shuffle=False)

        # scalers
        self.scalers = scalers if scalers is not None else get_none_scalers()
        
        # plotting info        
        self.pos = self.data.mesh.face_xy
        self.mesh = self.data.mesh
        self.default_plot_kwargs = {'pos':self.pos, 'mesh':self.mesh}
        self.default_temporal_plot_kwargs = self.default_plot_kwargs|\
            {'time_start': self.time_start, 'temporal_res':self.temporal_res}
        
        self.breach_coordinates = np.array([self.pos[node.item()] for node in self.data.node_BC])
        br = self.breach_coordinates[0]
        self.around_breach = np.array([[br[0] - 3000, br[0] + 3000], [br[1] - 3000, br[1] + 3000]])

        # get rollouts
        self.predicted_rollout = trainer.predict(plmodel, dataloaders=test_dataloader)[0][0]
        self.real_rollout = correct_rollout_shape_future_t(temporal_dataset[0].y).detach()
        self.diff_rollout = self.predicted_rollout - self.real_rollout
        self.input_water = get_input_water(temporal_dataset[0]).unsqueeze(-1)

        # get maps
        self._get_maps(self.real_rollout, self.predicted_rollout, self.diff_rollout, self.input_water, self.DEM)

        # time vector        
        self.total_time_steps = self.real_rollout.shape[-1]+self.time_start
        self.time_vector = get_time_vector(self.total_time_steps, self.temporal_res)

    def mesh_scale_plot(self, scale):
        """Switch all plot objects to data from the given mesh scale.

        Args:
            scale (int): mesh scale index; 0 is the finest scale
        """
        mesh = self.data.mesh
        assert isinstance(mesh, MultiscaleMesh), "This function only works for multiscale meshes"
        self.default_plot_kwargs['mesh'] = mesh.meshes[scale]
        self.default_temporal_plot_kwargs['mesh'] = mesh.meshes[scale]

        predicted_rollout = separate_multiscale_node_features(self.predicted_rollout, self.data.node_ptr)[scale]
        real_rollout = separate_multiscale_node_features(self.real_rollout, self.data.node_ptr)[scale]
        diff_rollout = separate_multiscale_node_features(self.diff_rollout, self.data.node_ptr)[scale]
        input_water = separate_multiscale_node_features(self.input_water, self.data.node_ptr)[scale]
        DEM = separate_multiscale_node_features(self.DEM, self.data.node_ptr)[scale]

        self._get_maps(real_rollout, predicted_rollout, diff_rollout, input_water, DEM)
            
    def _get_maps(self, real_rollout, predicted_rollout, diff_rollout, input_water, DEM):
        self._get_maxs(real_rollout, predicted_rollout, diff_rollout)
        self.DEMPlot = DEMPlotMap(DEM, **self.default_plot_kwargs)
        self._get_WDPlots(real_rollout, predicted_rollout, diff_rollout, input_water)
        self._get_FATPlots(real_rollout, predicted_rollout)
        self._get_VPlots(real_rollout, predicted_rollout, diff_rollout, input_water)

    def _get_maxs(self, real_rollout, predicted_rollout, diff_rollout):
        self.WD_max = max(real_rollout[:,0,:].max(), predicted_rollout[:,0,:].max())

        self.max_diff_WD = diff_rollout[:,0,:].max()
        self.min_diff_WD = diff_rollout[:,0,:].min()
        
        self.V_max = max(abs(predicted_rollout[:,1:,:]).max(), abs(real_rollout[:,1:,:]).max())
        self.max_diff_V = diff_rollout[:,1:,:].max()
        self.min_diff_V = diff_rollout[:,1:,:].min()

    def _reset_maxs(self, *plotmap):
        for plot in plotmap:
            plot.kwargs.pop('vmax')
        
    def _plot_temporal_errors(self, diff_rollout, ax):
        axs = plot_rollout_diff_in_time_all(diff_rollout, ax=ax, 
            type_loss=self.type_loss, temporal_res=self.temporal_res, 
            time_start=self.time_start)
        return axs

    def _plot_DEM(self, ax):
        ax = ax or plt.gca()
        self.DEMPlot.plot_map(ax=ax)
        self.DEMPlot._add_axes_info(ax=ax)
        self.DEMPlot._add_breach_location(ax, self.breach_coordinates, self.data.type_BC)
        ax.set_aspect('equal')

    def _get_WDPlots(self, real_rollout, predicted_rollout, diff_rollout, input_water):
        # Water depth
        self.real_WD = TemporalPlotMap(real_rollout[:,0,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'], 
            cmap=WD_color, vmax=self.WD_max)

        self.predicted_WD = TemporalPlotMap(predicted_rollout[:,0,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'], 
            cmap=WD_color, vmin=0, vmax=self.WD_max)

        self.difference_WD = TemporalPlotMap(diff_rollout[:,0,:],  
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'],
            difference_plot=True, vmin=self.min_diff_WD, vmax=self.max_diff_WD)
            
        self.init_WD = TemporalPlotMap(input_water[:,0,:],
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=WD_color, vmax=self.WD_max)
                
    def _get_VPlots(self, real_rollout, predicted_rollout, diff_rollout, input_water):
        # Scalar velocity
        self.real_V = TemporalPlotMap(real_rollout[:,1,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'], 
            cmap=V_color, vmax=self.V_max)

        self.predicted_V = TemporalPlotMap(predicted_rollout[:,1,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'], 
            cmap=V_color, vmin=0, vmax=self.V_max)

        self.difference_V = TemporalPlotMap(diff_rollout[:,1,:],  
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'],
            difference_plot=True, vmin=self.min_diff_V, vmax=self.max_diff_V)
            
        self.init_V = TemporalPlotMap(input_water[:,1,:], 
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=V_color, vmax=self.V_max)
        
    def _get_FATPlots(self, real_rollout, predicted_rollout):
        # Flood arrival times
        real_FAT = WD_to_FAT(real_rollout[:,0,:], self.temporal_res, self.water_threshold, self.time_start)
        predicted_FAT = WD_to_FAT(predicted_rollout[:,0,:], self.temporal_res, self.water_threshold, self.time_start)
        diff_FAT = np.nan_to_num(predicted_FAT) - np.nan_to_num(real_FAT)
        max_diff_FAT = np.nanmax(np.abs(diff_FAT))
        
        self.pred_FATPlot = BasePlotMap(predicted_FAT, **self.default_plot_kwargs, norm=FAT_norm, cmap=FAT_color)
        self.real_FATPlot = BasePlotMap(real_FAT, **self.default_plot_kwargs, norm=FAT_norm, cmap=FAT_color)
        self.diff_FATPlot = BasePlotMap(diff_FAT, **self.default_plot_kwargs, difference_plot=True,
            vmin=-max_diff_FAT, vmax=max_diff_FAT)

    def plot_BC(self, ax=None):
        """Plot boundary condition time series for discharge and water level nodes.

        Args:
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        discharge_BC_nodes = self.data.type_BC == 2
        water_depth_BC_nodes = self.data.type_BC == 1

        discharges = (self.data.BC[discharge_BC_nodes].T * self.data.edge_BC_length[discharge_BC_nodes])
        water_levels = self.data.BC[water_depth_BC_nodes].T + self.data.DEM[self.data.node_BC[water_depth_BC_nodes]]
        
        time_vector = get_time_vector(self.data.BC.shape[1]-1, self.temporal_res)

        # plot both BC types
        if discharges.shape[1] != 0 and water_levels.shape[1] != 0:
            ax2 = ax.twinx()
            ax2.set_ylabel('Water level [m]', color='purple')
            ax.set_ylabel('Discharge [$m^3$/s]', color='royalblue')

            ax.plot(time_vector, discharges.cpu(), c='royalblue', marker='.')
            ax2.plot(time_vector, water_levels.cpu(), c='purple', marker='.')
        elif discharges.shape[1] != 0:
            ax.set_ylabel('Discharge [$m^3$/s]')
            ax.plot(time_vector, discharges.cpu(), c='royalblue', marker='.')
        elif water_levels.shape[1] != 0:
            ax.set_ylabel('Water level [m]')
            ax.plot(time_vector, water_levels.cpu(), c='purple', marker='.')

        ax.set_xlabel('Time [h]')
        ax.set_title('Boundary conditions')
        
        return ax
            
    def explore_rollout(self, time_step=-1, scale=None, logscale=False, zoom_extent=None):
        """Plot DEM, ground-truth, prediction and difference maps for water depth and discharge.

        Args:
            time_step (int): time step index to plot
            scale (int, optional): mesh scale index for multiscale meshes
            logscale (bool): if True, plot discharge in log scale
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region

        Returns:
            Figure: matplotlib figure
        """
        fig, axs = plt.subplots(2, 4, figsize=(6*4, 11), facecolor='white',
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        if scale is not None: self.mesh_scale_plot(scale=scale)

        self._plot_DEM(ax=axs[0,0])

        # water depth
        self._reset_maxs(self.real_WD, self.predicted_WD, self.difference_WD)
        self.real_WD.plot_map(time_step=time_step, ax=axs[0,1])
        self.predicted_WD.plot_map(time_step=time_step, ax=axs[0,2])
        self.difference_WD.plot_map(time_step=time_step, ax=axs[0,3])

        axs[0,1].set_ylabel('Water depth [m]')
        axs[0,1].set_title('Ground-truth')
        axs[0,2].set_title('Predicted')
        axs[0,3].set_title('Difference')

        self._plot_temporal_errors(self.diff_rollout, ax=axs[1,0])

        # velocities
        axs[1,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')
        self._reset_maxs(self.real_V, self.predicted_V, self.difference_V)
        self.real_V.plot_map(time_step=time_step, ax=axs[1,1], logscale=logscale)
        self.predicted_V.plot_map(time_step=time_step, ax=axs[1,2], logscale=logscale)
        self.difference_V.plot_map(time_step=time_step, ax=axs[1,3])

        if zoom_extent is not None:
            for i,ax in enumerate(axs.flatten()):
                if i != 4:
                    ax.set_xlim(zoom_extent[0])
                    ax.set_ylim(zoom_extent[1])
                            
        return fig
    
    def explore_multiscale_rollout(self, time_step=-1, variable='WD', logscale=False):
        """Plot ground-truth, prediction, and difference maps at every mesh scale.

        Args:
            time_step (int): time step index to plot
            variable (str): variable to plot; 'WD' or 'V'
            logscale (bool): if True, plot variable in log scale

        Returns:
            Figure: matplotlib figure
        """
        assert isinstance(self.data.mesh, MultiscaleMesh), "This function only works for multiscale meshes"
        fig, axs = plt.subplots(self.num_scales, 4, figsize=(4*4, self.num_scales*4), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        for i in range(self.num_scales):
            self.mesh_scale_plot(scale=i)

            self._plot_DEM(ax=axs[i,0])

            # water depth
            if variable == 'WD':
                self.real_WD.plot_map(time_step=time_step, ax=axs[i,1], colorbar=False)
                self.predicted_WD.plot_map(time_step=time_step, ax=axs[i,2])
                self.difference_WD.plot_map(time_step=time_step, ax=axs[i,3])
                axs[i,1].set_ylabel('Water depth [m]')
            # velocities
            elif variable == 'V':
                axs[i,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')
                self.real_V.plot_map(time_step=time_step, ax=axs[i,1], colorbar=False, logscale=logscale)
                self.predicted_V.plot_map(time_step=time_step, ax=axs[i,2], logscale=logscale)
                self.difference_V.plot_map(time_step=time_step, ax=axs[i,3])

        axs[0,1].set_title('Ground-truth')
        axs[0,2].set_title('Predicted')
        axs[0,3].set_title('Difference')
                        
        return fig
    
    def compare_h_rollout(self, plot_times=[1,6,24,40], scale=None, zoom_extent=None):
        """Plot ground-truth, predicted, and difference water depth maps at multiple time steps.

        Args:
            plot_times (list of int): time step indices to include
            scale (int, optional): mesh scale index for multiscale meshes
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        plot_times = plot_times + [-1] #add final time step
        if scale is not None: self.mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        width_ratios = [0.9]*(n_plots*2-2) + [1,1]
        fig = plt.figure(figsize=(n_plots*4, 17), facecolor='white')
        spec = mpl.gridspec.GridSpec(ncols=2*len(plot_times), nrows=9, 
                                    height_ratios=[1,1,0.9,1,1,1,1,1,1],
                                    width_ratios=width_ratios)

        ax01 = fig.add_subplot(spec[0:2,1:3]) 
        ax02 = fig.add_subplot(spec[0:2,n_plots*2-3:n_plots*2-1]) 

        self._plot_DEM(ax=ax01)
        self.plot_BC(ax02)
        # ax02.set_title('')
        # ax01.set_title('')

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1:
                colorbar = True
            ax1 = fig.add_subplot(spec[3:5,i*2:i*2+2]) 
            ax2 = fig.add_subplot(spec[5:7,i*2:i*2+2])
            ax3 = fig.add_subplot(spec[7:9,i*2:i*2+2])
            if i==0:
                ax1.set_ylabel(f'Ground-truth [m]')
                ax2.set_ylabel(f'Predictions [m]')
                ax3.set_ylabel(f'Difference [m]')
            self.real_WD.plot_map(time_step=time_step, ax=ax1, colorbar=colorbar)
            self.predicted_WD.plot_map(time_step=time_step, ax=ax2, colorbar=colorbar)
            self.difference_WD.plot_map(time_step=time_step, ax=ax3, colorbar=colorbar)
            ax1.set_title(f'time: {self.real_WD.time_in_hours} h')

            if zoom_extent is not None:
                for i in range(3):
                    ax1.set_xlim(zoom_extent[0])
                    ax1.set_ylim(zoom_extent[1])
                    ax2.set_xlim(zoom_extent[0])
                    ax2.set_ylim(zoom_extent[1])
                    ax3.set_xlim(zoom_extent[0])
                    ax3.set_ylim(zoom_extent[1])
                                 
        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None
    
    def compare_v_rollout(self, plot_times=[1,6,24,40], scale=None, logscale=False, zoom_extent=None):
        """Plot ground-truth, predicted, and difference discharge maps at multiple time steps.

        Args:
            plot_times (list of int): time step indices to include
            scale (int, optional): mesh scale index for multiscale meshes
            logscale (bool): if True, plot discharge in log scale
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        plot_times = plot_times + [-1] #add final time step
        if scale is not None: self.mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        width_ratios = [0.9]*(n_plots*2-2) + [1,1]
        fig = plt.figure(figsize=(n_plots*4, 17), facecolor='white')
        spec = mpl.gridspec.GridSpec(ncols=2*len(plot_times), nrows=9, 
                                    height_ratios=[1,1,0.9,1,1,1,1,1,1],
                                    width_ratios=width_ratios)

        ax01 = fig.add_subplot(spec[0:2,1:3]) 
        ax02 = fig.add_subplot(spec[0:2,n_plots*2-3:n_plots*2-1]) 

        self._plot_DEM(ax=ax01)
        self.plot_BC(ax02)

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1:
                colorbar = True
            ax1 = fig.add_subplot(spec[3:5,i*2:i*2+2]) 
            ax2 = fig.add_subplot(spec[5:7,i*2:i*2+2])
            ax3 = fig.add_subplot(spec[7:9,i*2:i*2+2])
            if i==0:
                ax1.set_ylabel(f'Ground-truth [{self.V_unit}]')
                ax2.set_ylabel(f'Predictions [{self.V_unit}]')
                ax3.set_ylabel(f'Difference [{self.V_unit}]')
            self.real_V.plot_map(time_step=time_step, ax=ax1, colorbar=colorbar, logscale=logscale)
            self.predicted_V.plot_map(time_step=time_step, ax=ax2, colorbar=colorbar, logscale=logscale)
            self.difference_V.plot_map(time_step=time_step, ax=ax3, colorbar=colorbar)
            ax1.set_title(f'time: {self.real_V.time_in_hours} h')

            if zoom_extent is not None:
                for i in range(3):
                    ax1.set_xlim(zoom_extent[0])
                    ax1.set_ylim(zoom_extent[1])
                    ax2.set_xlim(zoom_extent[0])
                    ax2.set_ylim(zoom_extent[1])
                    ax3.set_xlim(zoom_extent[0])
                    ax3.set_ylim(zoom_extent[1])

        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None
    
    def compare_FAT(self, water_threshold=0, scale=None, zoom_extent=None):
        """Plot ground-truth, predicted, and difference flood arrival time maps.

        Args:
            water_threshold (float): water depth threshold for FAT calculation
            scale (int, optional): mesh scale index for multiscale meshes
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        self.water_threshold = water_threshold
        if scale is not None: self.mesh_scale_plot(scale=scale)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

        self.real_FATPlot.plot_map(ax=axs[0])
        self.pred_FATPlot.plot_map(ax=axs[1])
        self.diff_FATPlot.plot_map(ax=axs[2])
        
        axs[0].set_title('FAT Ground-truth [h]')
        axs[1].set_title('FAT Predictions [h]')
        axs[2].set_title('FAT Difference [h]')

        if zoom_extent is not None:
            for ax in axs:
                ax.set_xlim(zoom_extent[0])
                ax.set_ylim(zoom_extent[1])

        fig.tight_layout()

        return None
    
    def compare_Froude(self, time_step, scale=None, logscale=False):
        """Plot ground-truth, predicted, and difference Froude number maps.

        Args:
            time_step (int): time step index to plot
            scale (int, optional): mesh scale index for multiscale meshes
            logscale (bool): if True, use log scale
        """
        if scale is not None: self.mesh_scale_plot(scale=scale)

        real_vel = get_velocity(torch.norm(self.real_rollout[:,1:,:], dim=1), self.real_rollout[:,0,:])
        predicted_vel = get_velocity(torch.norm(self.predicted_rollout[:,1:,:], dim=1), self.predicted_rollout[:,0,:])

        self.real_fr = get_Froude(real_vel, self.real_rollout[:,0,:])
        self.predicted_fr = get_Froude(predicted_vel, self.predicted_rollout[:,0,:])
        self.diff_fr = self.real_fr - self.predicted_fr
        max_diff_fr = self.diff_fr[:,time_step].max()
        max_fr = max(self.real_fr.max(), self.predicted_fr.max())

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

        real_FrPlot = TemporalPlotMap(self.real_fr, 
            **self.default_temporal_plot_kwargs, cmap=V_color, vmin=0, vmax=max_fr)

        pred_FrPlot = TemporalPlotMap(self.predicted_fr, 
            **self.default_temporal_plot_kwargs, cmap=V_color, vmin=0, vmax=max_fr)

        diff_FrPlot = TemporalPlotMap(self.diff_fr, 
            **self.default_temporal_plot_kwargs, difference_plot=True, 
            vmin=-max_diff_fr, vmax=max_diff_fr)

        real_FrPlot.plot_map(time_step=time_step, ax=axs[0], logscale=logscale)
        pred_FrPlot.plot_map(time_step=time_step, ax=axs[1], logscale=logscale)
        diff_FrPlot.plot_map(time_step=time_step, ax=axs[2])
        
        # axs[0].set_title('Froude Ground-truth [-]')
        # axs[1].set_title('Froude Predictions [-]')
        # axs[2].set_title('Froude Difference [-]')

        fig.tight_layout()

        return None

    def create_video(self, logscale=False, interval=200, blit=False, **anim_kwargs):
        """Create an animated video of the simulation rollout (works in Jupyter only).

        Args:
            logscale (bool): if True, plot discharge in log scale
            interval (int): delay between frames in milliseconds
            blit (bool): whether to use blitting for faster rendering
        """
        from IPython.display import clear_output
        from matplotlib.animation import FuncAnimation

        fig, axs = plt.subplots(2, 4, figsize=(6*4, 11), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        self._plot_DEM(ax=axs[0,0])

        axs[1,0].set_ylabel(self.type_loss)
        axs[1,0].set_xlabel('Time [h]')
        average_diff_t = get_mean_error(self.diff_rollout, self.type_loss).cpu().numpy()
        max_avg_WD = average_diff_t[0].max()
        max_avg_V = average_diff_t[1].max()

        self.add_initial_colorbars(axs, logscale=logscale)

        def animate(time_step):
            for axx in axs:
                for ax in axx[1:]:
                    ax.cla()

            axs[1,0].cla()

            ax, axv = self._plot_temporal_errors(self.diff_rollout[:,:,:time_step], ax=axs[1,0])

            ax.set_xlim(0, (self.real_WD.total_time+self.time_start)*self.temporal_res/60)
            ax.set_ylim(0, max_avg_WD*1.1)
            axv.set_ylim(0, max_avg_V*1.1)
            axv.ticklabel_format(style='sci', scilimits=(-1,3), useMathText=True)
            ax.ticklabel_format(style='sci', scilimits=(-1,3), useMathText=True)

            # water depth
            self.real_WD.plot_map(time_step=time_step, ax=axs[0,1], colorbar=False)
            self.predicted_WD.plot_map(time_step=time_step, ax=axs[0,2], colorbar=False)
            self.difference_WD.plot_map(time_step=time_step, ax=axs[0,3], colorbar=False)

            current_time = self.real_WD.time_in_hours
            axs[0,1].set_title(f'Ground-truth h [m]\ntime {current_time} h')
            axs[0,2].set_title(f'Predicted h [m]\ntime {current_time} h')
            axs[0,3].set_title(f'Difference h [m]\ntime {current_time} h')

            # velocities
            self.real_V.plot_map(time_step=time_step, ax=axs[1,1], colorbar=False, logscale=logscale)
            self.predicted_V.plot_map(time_step=time_step, ax=axs[1,2], colorbar=False, logscale=logscale)
            self.difference_V.plot_map(time_step=time_step, ax=axs[1,3], colorbar=False)
            axs[1,1].set_title(f'Ground-truth |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
            axs[1,2].set_title(f'Predicted |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
            axs[1,3].set_title(f'Difference |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')

            fig.subplots_adjust(wspace=0.4, hspace=0.3)

            clear_output(wait=True)
            print ('It: %i'%time_step)
            sys.stdout.flush()
            return (fig)
        
        frames = self.real_WD.total_time
        self.anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, **anim_kwargs)
        plt.close()

    def create_multiscale_video(self, variable='WD', interval=200, blit=False, **anim_kwargs):
        """Create animated video of a hydraulic variable at all mesh scales (multiscale meshes only).

        Args:
            variable (str): variable to animate; 'WD' or 'V'
            interval (int): delay between frames in milliseconds
            blit (bool): whether to use blitting for faster rendering
        """
        from IPython.display import clear_output
        from matplotlib.animation import FuncAnimation

        assert isinstance(self.data.mesh, MultiscaleMesh), "This function only works for multiscale meshes"
        fig, axs = plt.subplots(self.num_scales, 4, figsize=(5*4, self.num_scales*4), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        axs[0,1].set_title('Ground-truth')
        axs[0,2].set_title('Predicted')
        axs[0,3].set_title('Difference')

        for i in range(self.num_scales):
            self.mesh_scale_plot(scale=i)
            
            self.DEMPlot.plot_map(ax=axs[i,0])
            self.DEMPlot._add_axes_info(ax=axs[i,0], title=False, x_label=i//(self.num_scales-1))
            if i==0:
                axs[i,0].set_title('DEM (m)')
                self.DEMPlot._add_breach_location(axs[i,0], self.breach_coordinates, self.data.type_BC)
        
            # water depth
            if variable == 'WD':
                self.predicted_WD.kwargs['vmin'] = 0
                self.predicted_WD.kwargs['vmax'] = self.WD_max
                self.predicted_WD._get_cmap()
                self.predicted_WD._add_colorbar(ax=axs[i,2], colorbar=True)
                
                self.difference_WD._get_cmap()
                self.difference_WD._add_colorbar(ax=axs[i,3], colorbar=True)
            # velocities
            elif variable == 'V':
                self.predicted_V.kwargs['vmin'] = 0
                self.predicted_V.kwargs['vmax'] = self.V_max
                self.predicted_V._get_cmap()
                self.predicted_V._add_colorbar(ax=axs[i,2], colorbar=True)
                
                self.difference_V._get_cmap()
                self.difference_V._add_colorbar(ax=axs[i,3], colorbar=True)

        def animate(time_step):
            for axx in axs:
                for ax in axx[1:]:
                    ax.cla()


            for i in range(self.num_scales):
                self.mesh_scale_plot(scale=i)
                # water depth
                if variable == 'WD':
                    self.real_WD.plot_map(time_step=time_step, ax=axs[i,1], colorbar=False)
                    self.predicted_WD.plot_map(time_step=time_step, ax=axs[i,2], colorbar=False)
                    self.difference_WD.plot_map(time_step=time_step, ax=axs[i,3], colorbar=False)
                    current_time = self.real_WD.time_in_hours
                    axs[0,1].set_title(f'Ground-truth h [m]\ntime {current_time} h')
                    axs[0,2].set_title(f'Predicted h [m]\ntime {current_time} h')
                    axs[0,3].set_title(f'Difference h [m]\ntime {current_time} h')
                # velocities
                elif variable == 'V':
                    axs[i,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')
                    self.real_V.plot_map(time_step=time_step, ax=axs[i,1], colorbar=False)
                    self.predicted_V.plot_map(time_step=time_step, ax=axs[i,2], colorbar=False)
                    self.difference_V.plot_map(time_step=time_step, ax=axs[i,3], colorbar=False)
                    current_time = self.real_V.time_in_hours
                    axs[1,1].set_title(f'Ground-truth |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
                    axs[1,2].set_title(f'Predicted |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
                    axs[1,3].set_title(f'Difference |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')                
                else:        
                    for ax in axs[1,1:]:
                        ax.axis('off')
                    
            fig.subplots_adjust(wspace=0.4, hspace=0.3)

            clear_output(wait=True)
            print ('It: %i'%time_step)
            sys.stdout.flush()
            return (fig)
        
        if variable == 'WD':
            frames = self.real_WD.total_time
        elif variable == 'V':
            frames = self.real_V.total_time
        self.anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, **anim_kwargs)
        plt.close()

    def add_initial_colorbars(self, axs, logscale=False):
        """Add static colorbars to axes before animation begins.

        Args:
            axs (np.ndarray): 2D array of matplotlib axes
            logscale (bool): if True, use log scale for discharge colorbar
        """
        self.predicted_WD._get_cmap()
        self.predicted_WD._add_colorbar(ax=axs[0,2], colorbar=True)
        
        self.difference_WD._get_cmap()
        self.difference_WD._add_colorbar(ax=axs[0,3], colorbar=True)

        self.predicted_V._get_cmap()
        self.predicted_V._add_colorbar(ax=axs[1,2], colorbar=True, logscale=logscale)
        
        self.difference_V._get_cmap()
        self.difference_V._add_colorbar(ax=axs[1,3], colorbar=True)

    def save_video(self, path, fps=5, dpi=250, **save_kwargs):
        """Save the animation to a GIF file.

        Args:
            path (str): output file path without extension
            fps (int): frames per second
            dpi (int): resolution in dots per inch
        """
        from matplotlib.animation import PillowWriter
        writergif = PillowWriter(fps=fps, metadata={
            'title':'test_dataset', 'artist':'Roberto Bentivoglio'}, **save_kwargs)

        self.anim.save(f'{path}.gif', writer=writergif)

    def HTML_plot(self):
        """Render the animation as an HTML5 video in a Jupyter notebook."""
        from IPython.display import HTML
        HTML(self.anim.to_html5_video())

    def _get_CSI(self, water_threshold=0):
        return get_CSI_rollout(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)
        
    def _get_F1(self, water_threshold=0):
        return get_F1_rollout(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)

    def _plot_metric(self, metric_name='CSI', water_thresholds=[0.05, 0.3], ax=None):
        """Plot a skill score in time for multiple water depth thresholds.

        Args:
            metric_name (str): metric to plot; 'CSI' or 'F1'
            water_thresholds (list of float): water depth thresholds in metres
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        metrics_dict = {'CSI': self._get_CSI,
                        'F1': self._get_F1}
        metric_function = metrics_dict[metric_name]

        ax = ax or plt.gca()

        for wt in water_thresholds:
            metric = metric_function(water_threshold=wt).to('cpu').numpy()
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(self.time_vector, metric, label=f'{metric_name}$_{{{wt}}}$')
            
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(f'{metric_name} score')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=4)
        
        return ax
    
    def _plot_volumes(self, ax=None):
        """Plot real, predicted, and difference flood volumes over time.

        Args:
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()

        # total volumes (only at finest scale)
        if isinstance(self.mesh, MultiscaleMesh): self.mesh_scale_plot(scale=0)
        self.init_volume, self.real_volume = calculate_volumes(self.data, self.time_start)
        self.predicted_volume = (self.predicted_WD.map.T @ self.predicted_WD.mesh.face_area) - self.init_volume
        self.diff_volume = self.predicted_volume - self.real_volume

        ax.plot(self.time_vector[self.time_start+1:], self.predicted_volume, label='Predicted Volume')
        ax.plot(self.time_vector[self.time_start+1:], self.real_volume, label='Real Volume')
        ax.plot(self.time_vector[self.time_start+1:], self.diff_volume, label='Difference Volume')

        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Volume [$m^3$]')
        ax.legend()
        ax.grid()
        return ax
    
    def get_volumes_r2(self):
        """Return R2 score between real and predicted volumes.

        Returns:
            float: R2 score
        """
        return r2_score(self.real_volume, self.predicted_volume)

    def get_volumes_ARME(self):
        """Return average relative mass error between real and predicted volumes.

        Returns:
            float: ARME value
        """
        return get_average_relative_mass_error(self.real_volume, self.predicted_volume)

    def _get_rollout_loss(self, type_loss='RMSE', only_where_water=False):
        """Compute rollout loss between predicted and real outputs.

        Args:
            type_loss (str): loss metric name, e.g. 'RMSE'
            only_where_water (bool): if True, compute loss only at wet nodes

        Returns:
            Tensor: loss values per variable and time step
        """
        return get_rollout_loss(self.predicted_rollout, self.real_rollout,
                                type_loss=type_loss, only_where_water=only_where_water)


class SinglePlotRollout():
    """Explore predicted simulations (no ground truth) for DEM, water depth, and discharge.

    Args:
        plmodel (LightningModule): trained lightning model
        trainer (Trainer): lightning trainer used for prediction
        dataset (Data): single dataset object with simulation data
        scalers (dict, optional): scalers used during training
        **temporal_test_dataset_parameters: parameters for the temporal test dataset
    """
    def __init__(self, plmodel, trainer, dataset, scalers=None, **temporal_test_dataset_parameters):
        super().__init__()
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.temporal_res = dataset.temporal_res
        self.V_unit = "$m^2$/s"
        self.V_label = "Discharge"
        self.V_symbol = "q"
        self.dataset = dataset
        self.DEM = dataset.DEM
        self.water_threshold = 0
        self.BC = self.dataset.BC[:,-1,:]
        self.num_scales = dataset.mesh.num_meshes if isinstance(dataset.mesh, MultiscaleMesh) else 1
        
        # scalers
        self.scalers = scalers if scalers is not None else get_none_scalers()
        
        # plotting info        
        self.pos = dataset.mesh.face_xy
        self.mesh = dataset.mesh
        self.default_plot_kwargs = {'pos':self.pos, 'mesh':self.mesh}
        self.default_temporal_plot_kwargs = self.default_plot_kwargs|\
            {'time_start': self.time_start, 'temporal_res':self.temporal_res}
        
        self.breach_coordinates = np.array([self.pos[node.item()] for node in self.dataset.node_BC])

        br = self.breach_coordinates[0]
        self.zoom_size = 1000
        self.around_breach = np.array([[br[0] - self.zoom_size, br[0] + self.zoom_size], [br[1] - self.zoom_size, br[1] + self.zoom_size]])

        # get rollout
        # convert to temporal dataset for predictions
        test_dataloader = DataLoader([dataset], batch_size=1, shuffle=False)
        self.predicted_rollout = trainer.predict(plmodel, dataloaders=test_dataloader)[0][0]
        self.input_water = get_input_water(dataset).unsqueeze(-1)

        # get maps
        self._get_maps(self.predicted_rollout, self.input_water, self.DEM)

        # time vector        
        self.total_time_steps = self.predicted_rollout.shape[-1]+self.time_start
        self.time_vector = get_time_vector(self.total_time_steps, self.temporal_res)

        # total volumes (only at finest scale)
        if isinstance(self.mesh, MultiscaleMesh): self._mesh_scale_plot(scale=0)
        self.init_volume = (self.init_WD.map[:,0] * self.init_WD.mesh.face_area).sum()
        self.real_volume = (torch.cumsum(self.BC[:,:-1] * self.dataset.edge_BC_length.unsqueeze(1) * self.temporal_res * 60, dim=1).mean(0)).cpu().numpy()
        self.predicted_volume = (self.predicted_WD.map.T * self.predicted_WD.mesh.face_area).sum(1) - self.init_volume
        self.diff_volume = self.predicted_volume - self.real_volume

    def _mesh_scale_plot(self, scale):
        mesh = self.dataset.mesh
        assert isinstance(mesh, MultiscaleMesh), "This function only works for multiscale meshes"
        self.default_plot_kwargs['mesh'] = mesh.meshes[scale]
        self.default_temporal_plot_kwargs['mesh'] = mesh.meshes[scale]

        predicted_rollout = separate_multiscale_node_features(self.predicted_rollout, self.dataset.node_ptr)[scale]
        input_water = separate_multiscale_node_features(self.input_water, self.dataset.node_ptr)[scale]
        DEM = separate_multiscale_node_features(self.DEM, self.dataset.node_ptr)[scale]

        self._get_maps(predicted_rollout, input_water, DEM)
            
    def _get_maps(self, predicted_rollout, input_water, DEM):
        self._get_maxs(predicted_rollout)
        self.DEMPlot = DEMPlotMap(DEM, **self.default_plot_kwargs)
        self._get_WDPlots(predicted_rollout, input_water)
        self._get_FATPlots(predicted_rollout)
        self._get_VPlots(predicted_rollout, input_water)

    def _get_maxs(self, predicted_rollout):
        self.WD_max = predicted_rollout[:,0,:].max()    
        self.V_max = abs(predicted_rollout[:,1:,:]).max()

    def _plot_DEM(self, ax=None):
        ax = ax or plt.gca()
        self.DEMPlot.plot_map(ax=ax)
        self.DEMPlot._add_axes_info(ax=ax)
        self.DEMPlot._add_breach_location(ax, self.breach_coordinates, self.dataset.type_BC)

    def _get_WDPlots(self, predicted_rollout, input_water):
        # Water depth
        self.predicted_WD = TemporalPlotMap(predicted_rollout[:,0,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'], 
            cmap=WD_color, vmin=0, vmax=self.WD_max)
            
        self.init_WD = TemporalPlotMap(input_water[:,0,:],
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=WD_color, vmax=self.WD_max)
                
    def _get_VPlots(self, predicted_rollout, input_water):
        # Scalar velocity
        self.predicted_V = TemporalPlotMap(predicted_rollout[:,1,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'], 
            cmap=V_color, vmin=0, vmax=self.V_max)

        self.init_V = TemporalPlotMap(input_water[:,1,:], 
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=V_color, vmax=self.V_max)
        
    def _get_FATPlots(self, predicted_rollout):
        # Flood arrival times
        predicted_FAT = WD_to_FAT(predicted_rollout[:,0,:], self.temporal_res, self.water_threshold, self.time_start)
        self.pred_FATPlot = BasePlotMap(predicted_FAT, **self.default_plot_kwargs, cmap=FAT_color, norm=FAT_norm)

    def plot_BC(self, ax=None):
        """Plot boundary condition time series for discharge and water level nodes.

        Args:
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        discharge_BC_nodes = self.dataset.type_BC == 2
        water_depth_BC_nodes = self.dataset.type_BC == 1

        discharges = (self.BC[discharge_BC_nodes].T * self.dataset.edge_BC_length[discharge_BC_nodes])
        water_levels = self.BC[water_depth_BC_nodes].T + self.dataset.DEM[self.dataset.node_BC[water_depth_BC_nodes]]
        
        time_vector = get_time_vector(self.BC.shape[1]-1, self.temporal_res)

        # plot both BC types
        if discharges.shape[1] != 0 and water_levels.shape[1] != 0:
            ax2 = ax.twinx()
            ax2.set_ylabel('Water level [m]', color='purple')
            ax.set_ylabel('Discharge [$m^3$/s]', color='royalblue')

            ax.plot(time_vector, discharges.cpu(), c='royalblue', marker='.')
            ax2.plot(time_vector, water_levels.cpu(), c='purple', marker='.')
        elif discharges.shape[1] != 0:
            ax.set_ylabel('Discharge [$m^3$/s]')
            ax.plot(time_vector, discharges.cpu(), c='royalblue', marker='.')
        elif water_levels.shape[1] != 0:
            ax.set_ylabel('Water level [m]')
            ax.plot(time_vector, water_levels.cpu(), c='purple', marker='.')

        ax.set_xlabel('Time [h]')
        ax.set_title('Boundary conditions')
        
        return ax
            
    def explore_rollout(self, time_step=-1, scale=None, logscale=False, zoom_extent=None):
        """Plot DEM, boundary conditions, predicted water depth, and discharge at one time step.

        Args:
            time_step (int): time step index to plot
            scale (int, optional): mesh scale index for multiscale meshes
            logscale (bool): if True, plot discharge in log scale
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region

        Returns:
            np.ndarray: array of Axes
        """
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), facecolor='white',
                            gridspec_kw={'width_ratios': [1, 1]},
                            constrained_layout = True)

        if scale is not None: self._mesh_scale_plot(scale=scale)

        # DEM
        self._plot_DEM(ax=axs[0,0])

        # BC
        self.plot_BC(ax=axs[0,1])

        # water depth
        self.predicted_WD.plot_map(time_step=time_step, ax=axs[1,0])
        axs[1,0].set_ylabel('Water depth [m]')

        # velocities
        axs[1,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')
        self.predicted_V.plot_map(time_step=time_step, ax=axs[1,1], logscale=logscale)

        if zoom_extent is not None:
            for i,ax in enumerate(axs.flatten()):
                if i == 0:
                    # draw rectangle in red
                    rect = mpl.patches.Rectangle((zoom_extent[0][0], zoom_extent[1][0]), 
                                                zoom_extent[0][1]-zoom_extent[0][0], 
                                                zoom_extent[1][1]-zoom_extent[1][0], 
                                                linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                if i > 1:
                    ax.set_xlim(zoom_extent[0])
                    ax.set_ylim(zoom_extent[1])
                            
        return axs
    
    def show_h_rollout(self, plot_times=[1,6,24,40], scale=None, zoom_extent=None):
        """Plot predicted water depth maps at multiple time steps.

        Args:
            plot_times (list of int): time step indices to include
            scale (int, optional): mesh scale index for multiscale meshes
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        plot_times = plot_times + [-1] #add final time step
        if scale is not None: self._mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots*4, 5), facecolor='white',
                            constrained_layout=True)

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1:
                colorbar = True
            self.predicted_WD.plot_map(time_step=time_step, ax=axs[i], colorbar=colorbar)
            axs[i].set_title(f'time: {self.predicted_WD.time_in_hours} h')

            if zoom_extent is not None:
                for i in range(n_plots):
                    axs[i].set_xlim(zoom_extent[0])
                    axs[i].set_ylim(zoom_extent[1])
            
        return None
    
    def show_v_rollout(self, plot_times=[1,6,24,40], scale=None, logscale=False, zoom_extent=None):
        """Plot predicted discharge maps at multiple time steps.

        Args:
            plot_times (list of int): time step indices to include
            scale (int, optional): mesh scale index for multiscale meshes
            logscale (bool): if True, use log scale
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        plot_times = plot_times + [-1] #add final time step
        if scale is not None: self._mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots*4, 5), facecolor='white',
                            constrained_layout=True)

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1: colorbar = True
            self.predicted_V.plot_map(time_step=time_step, ax=axs[i], colorbar=colorbar, logscale=logscale)
            axs[i].set_title(f'time: {self.predicted_V.time_in_hours} h')

            if zoom_extent is not None:
                for i in range(n_plots):
                    axs[i].set_xlim(zoom_extent[0])
                    axs[i].set_ylim(zoom_extent[1])
                                             
        return None
    
    def show_FAT(self, ax=None, water_threshold=0, scale=None, zoom_extent=None):
        """Plot predicted flood arrival time map.

        Args:
            ax (Axes, optional): matplotlib axes
            water_threshold (float): water depth threshold for FAT calculation
            scale (int, optional): mesh scale index for multiscale meshes
            zoom_extent (list, optional): [[xmin, xmax], [ymin, ymax]] zoom region
        """
        self.water_threshold = water_threshold
        if scale is not None: self._mesh_scale_plot(scale=scale)

        ax = ax or plt.gca()

        self.pred_FATPlot.plot_map(ax=ax)
        ax.set_title('FAT Predictions [h]')

        if zoom_extent is not None:
            ax.set_xlim(zoom_extent[0])
            ax.set_ylim(zoom_extent[1])

        return None

    def _plot_volumes(self, ax=None):
        """Plot predicted, real, and difference flood volumes over time.

        Args:
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()
        ax.plot(self.time_vector[self.time_start+1:], self.predicted_volume, label='Predicted Volume')
        ax.plot(self.time_vector[self.time_start+1:], self.real_volume, label='Real Volume')
        ax.plot(self.time_vector[self.time_start+1:], self.diff_volume, label='Difference Volume')

        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Volume [$m^3$]')
        ax.legend()
        ax.grid()
        return ax

    def get_volumes_r2(self):
        """Return R2 score between real and predicted volumes.

        Returns:
            float: R2 score
        """
        return r2_score(self.real_volume, self.predicted_volume)

    def get_volumes_ARME(self):
        """Return average relative mass error between real and predicted volumes.

        Returns:
            float: ARME value
        """
        return get_average_relative_mass_error(self.real_volume, self.predicted_volume)
      

class ProbabilisticSpatialAnalysis():
    """Analyse and plot spatial probabilistic outputs from an ensemble of flood simulations.

    Args:
        ensemble_selected_prediction (np.ndarray, shape [n_samples, n_nodes, n_vars]): ensemble predictions at finest scale
        predicted_volumes (np.ndarray, shape [n_samples, T]): predicted flood volumes
        dataset (list): list of torch_geometric.data.Data objects
        scalers (dict): scalers used during training
        base_DEM (Tensor or np.ndarray): digital elevation model at finest scale
        base_mesh (Mesh): base mesh object
        **temporal_test_dataset_parameters: parameters for the temporal test dataset
    """
    
    def __init__(self, ensemble_selected_prediction, predicted_volumes, dataset, scalers, base_DEM, base_mesh, **temporal_test_dataset_parameters):
        assert len(ensemble_selected_prediction) == len(dataset), "Number of simulations should be the same"
        
        self.dataset = dataset
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.previous_t = temporal_test_dataset_parameters['previous_t']
        self.temporal_res = dataset[0].temporal_res
        self.DEM = base_DEM

        self.breach_coordinates = np.concatenate([data.breach_coords for data in dataset])
        self.breach_coordinates_dict = {tuple(coord) : i for i, coord in enumerate(np.unique(self.breach_coordinates, axis=0))}
        self.sim_breach_ids = np.array([self.breach_coordinates_dict[tuple(coord)] for coord in self.breach_coordinates])

        self.ensemble_selected_prediction = ensemble_selected_prediction

        self.BCs = torch.cat([data.BC for data in self.dataset])
        self.type_BCs = torch.cat([data.type_BC for data in self.dataset])
        self.edge_BC_lengths = torch.cat([data.edge_BC_length for data in self.dataset])

        total_time_steps = predicted_volumes[0].shape[-1]+self.time_start
        self.time_vector = get_time_vector(total_time_steps, self.temporal_res)

        # scalers
        self.scalers = scalers if scalers is not None else get_none_scalers()

        # plotting info
        self.mesh = remove_ghost_cells(copy(base_mesh))  # assume all meshes are the same
        self.pos = self.mesh.face_xy
        self.default_plot_kwargs = {'pos':self.pos, 'mesh':self.mesh}
        self.default_temporal_plot_kwargs = self.default_plot_kwargs|\
            {'time_start': self.time_start, 'temporal_res':self.temporal_res}
        
        self.DEMPlot = DEMPlotMap(self.DEM, **self.default_plot_kwargs)

        # Calculate the volumes
        volumes = [calculate_volumes(data, self.time_start) for data in self.dataset]
        self.init_volume = np.stack([v[0] for v in volumes])
        self.real_volumes = np.concatenate([v[1] for v in volumes], 1)
        self.predicted_volumes = predicted_volumes - self.init_volume.reshape(-1,1)

        self.final_time = self.predicted_volumes.shape[-1]
        self.real_volumes = self.real_volumes[:self.final_time].T
        self.diff_volumes = self.predicted_volumes - self.real_volumes

        if np.isnan(self.predicted_volumes).any():
            self.predicted_volumes[np.isnan(self.predicted_volumes)] = 0

        self.ARME = get_average_relative_mass_error(self.real_volumes, self.predicted_volumes)
        
    def plot_max_WD_probability(self, water_threshold=0.1, with_DEM=True,
                                show_contributing_breaches=False, ax=None,
                                show_volume_distribution=False, time_step=None,
                                ):
        """Plot probability map of max water depth exceeding a threshold across the ensemble.

        Args:
            water_threshold (float): water depth threshold in metres
            with_DEM (bool): if True, overlay the DEM map
            show_contributing_breaches (bool): if True, scatter breach locations
            ax (Axes, optional): matplotlib axes
            show_volume_distribution (bool): if True, add inset volume scatter plots
            time_step (int, optional): time step for volume distribution insets

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()

        WD = self.ensemble_selected_prediction[:,:,2] # shape (n_samples, n_nodes)
        if self.scalers['WD_scaler'] is not None:
            WD = self.scalers['WD_scaler'].inverse_transform(WD)
        flooded_areas = WD > water_threshold
        WD_prob = flooded_areas.sum(0)/len(WD)
        
        if with_DEM:
            self.DEMPlot.plot_map()

        self.WD_prob = BasePlotMap(WD_prob, **self.default_plot_kwargs, cmap=probability_color, vmin=0, vmax=1)
        self.WD_prob.plot_map()

        if show_contributing_breaches:
            plt.scatter(*self.breach_coordinates.T, s=50, edgecolor='k', linewidth=0.1, marker='X', zorder=3)
            
        if show_volume_distribution:
            ax = self._show_volume_distribution(time_step=time_step, ax=ax)

        return ax
    
    
    def plot_FAT_quantile(self, water_threshold=0.05, quantile=0.5, ax=None,
                        with_DEM=False, show_volume_distribution=False,
                        ):
        """Plot a quantile map of flood arrival time across the ensemble.

        Args:
            water_threshold (float): water depth threshold; must be 0.05 or 0.3
            quantile (float): quantile level of the FAT distribution
            ax (Axes, optional): matplotlib axes
            with_DEM (bool): if True, overlay the DEM map
            show_volume_distribution (bool): if True, add inset volume scatter plots

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()

        assert water_threshold == 0.05 or water_threshold == 0.3, "Only water thresholds of 0.05 and 0.3 are supported for FAT quantile plots."
        dict_index_FAT = {0.05: 0, 0.3: 1}

        FAT = self.ensemble_selected_prediction[:,:,dict_index_FAT[water_threshold]] # shape (n_samples, n_nodes)
        FAT_quantile = np.nanquantile(np.where(np.isnan(FAT), 9999, FAT), quantile, axis=0)
        FAT_quantile = np.where(FAT_quantile > 999, np.nan, FAT_quantile)

        self.FAT_prob = BasePlotMap(FAT_quantile, **self.default_plot_kwargs, norm=FAT_norm, cmap=FAT_color)
        self.FAT_prob.plot_map()
        
        if with_DEM:
            self.DEMPlot.plot_map()
            
        if show_volume_distribution:
            ax = self._show_volume_distribution(ax=ax)

        return ax
    
    def _show_volume_distribution(self, time_step=None, ax=None):
        """Add inset scatter plots of predicted vs real volumes at each breach location.

        Args:
            time_step (int, optional): if given, plot volumes at that time step only
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """

        ax = ax or plt.gca()

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        # Add mini regression plots at each point
        for xy, id_ in self.breach_coordinates_dict.items():
            x, y = xy

            if time_step is None:
                real = self.real_volumes[self.sim_breach_ids==id_]
                pred = self.predicted_volumes[self.sim_breach_ids==id_]
            else:
                real = self.real_volumes[self.sim_breach_ids==id_, time_step]
                pred = self.predicted_volumes[self.sim_breach_ids==id_, time_step]

            inset_ax = inset_axes(ax, width=0.4, height=0.4, loc='center',
                                bbox_to_anchor=(x, y, 0, 0),
                                bbox_transform=ax.transData)
            
            inset_ax.scatter(real, pred, s=4, zorder=1, color='blue', alpha=3/np.log(self.real_volumes.size))
            inset_ax.patch.set_alpha(0)  # Set background transparent

            # Optional: add bisector line
            line = np.linspace(0.75*min(real.min(), pred.min()), 1.2*max(real.max(), pred.max()), 100)
            inset_ax.plot(line, line, color='k', lw=0.5, linestyle='--', zorder=0)
            
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.spines['top'].set_visible(False)
            inset_ax.spines['right'].set_visible(False)

        return ax
    
    def _get_breach_distribution(self, ax=None):
        """Plot the distribution of breach locations across the dataset.

        Args:
            ax (Axes, optional): matplotlib axes

        Returns:
            Axes: matplotlib axes
        """
        return plot_breach_distribution(self.dataset, edgecolor='k', linewidth=0.2, with_label=True, ax=ax)
    
    def _plot_BCs(self, ax=None, highlight_ids=None):
        """Plot boundary condition time series for all simulations.

        Args:
            ax (Axes, optional): matplotlib axes
            highlight_ids (list, optional): indices of simulations to highlight

        Returns:
            Axes: matplotlib axes
        """
        return plot_BCs(self.type_BCs, self.BCs[:,-1,:self.final_time], self.dataset, ax=ax, highlight_ids=highlight_ids)
    
    def get_plausible_runs(self, ARME_threshold=0.25):
        """Return indices of simulations with ARME below the given threshold.

        Args:
            ARME_threshold (float): maximum allowable ARME

        Returns:
            np.ndarray: indices of plausible simulations
        """
        plausible_ids = np.where(self.ARME < ARME_threshold)[0]

        return plausible_ids
    
    def plot_runs_per_threshold(self, ARME_thresholds=None, ax=None, with_hist=False, **kwargs):
        """Plot the number of plausible runs as a function of ARME threshold.

        Args:
            ARME_thresholds (np.ndarray, optional): threshold values to evaluate
            ax (Axes, optional): matplotlib axes
            with_hist (bool): if True, overlay a histogram of ARME values

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()
        if ARME_thresholds is None:
            ARME_thresholds = np.linspace(0, 2, 50)
        all_runs = []
        all_thresholds = []

        for threshold in ARME_thresholds:
            plausible_runs = self.get_plausible_runs(ARME_threshold=threshold)
            all_runs.append(len(plausible_runs))
            all_thresholds.append(threshold)

        all_runs = np.array(all_runs)
        all_thresholds = np.array(all_thresholds)

        ax.plot(all_thresholds, all_runs)
        ax.set_xlabel('ARME threshold')
        ax.set_ylabel('Number of plausible runs')

        if with_hist:
            hist_ax = ax.twinx()
            hist_ax.set_ylabel('Frequency')
            hist_ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            hist_ax.hist(self.ARME[self.ARME < ARME_thresholds[-1]], **kwargs)

        return ax
    
    def plot_volume_distribution(self, ax=None, time_step=None, log_scale=False, with_bisector=True):
        """Scatter plot of predicted vs real flood volumes.

        Args:
            ax (Axes, optional): matplotlib axes
            time_step (int, optional): if given, plot volumes at that time step only
            log_scale (bool): if True, use log scale on both axes
            with_bisector (bool): if True, draw the 1:1 reference line

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()
        if time_step is None:
            ax.scatter(self.real_volumes, self.predicted_volumes, color='blue', alpha=1/(1+np.log(self.real_volumes.size)))
            ARME_volumes = np.mean([get_average_relative_mass_error(real, pred) for real, pred in zip (self.real_volumes, self.predicted_volumes)])

            min_val = min(self.predicted_volumes.min(), self.real_volumes.min())*0.9
            max_val = max(self.predicted_volumes.max(), self.real_volumes.max())*1.1

            ax.set_title(f'Average ARME: {ARME_volumes:.2f}')
        else:
            ax.scatter(self.real_volumes[:,time_step], self.predicted_volumes[:,time_step], color='blue', alpha=1/(1+np.log(self.real_volumes[:,time_step].size)))
            r2_volumes = r2_score(self.real_volumes[:,time_step], self.predicted_volumes[:,time_step])

            min_val = min(self.predicted_volumes[:,time_step].min(), self.real_volumes[:,time_step].min())*0.9
            max_val = max(self.predicted_volumes[:,time_step].max(), self.real_volumes[:,time_step].max())*1.1
            ax.set_title(f'R2: {r2_volumes:.2f}')

        if with_bisector:
            ax.plot([min_val, max_val], [min_val, max_val], 'k--')

        ax.set_ylabel('Predicted volume [m$^3$]')
        ax.set_xlabel('Real volume [m$^3$]')

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.axis('equal')

        return ax
    
    def plot_volumes_in_time(self, ax=None, with_difference=True):
        """Plot real and predicted flood volumes over time with spread bands.

        Args:
            ax (Axes, optional): matplotlib axes
            with_difference (bool): if True, also plot the volume difference

        Returns:
            Axes: matplotlib axes
        """
        ax = ax or plt.gca()

        plot_line_with_deviation(self.time_vector[1:], self.real_volumes, with_minmax=True, ax=ax, label='Real Volume')
        plot_line_with_deviation(self.time_vector[1:], self.predicted_volumes, with_minmax=True, ax=ax, label='Predicted Volume')
        if with_difference:
            plot_line_with_deviation(self.time_vector[1:], self.diff_volumes, with_minmax=True, ax=ax, label='Difference Volume')

        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Volume [$m^3$]')
        ax.legend()

        return ax