import numpy as np
import pandas as pd
import os
import shutil
import geopandas as gpd
from scipy.stats import truncnorm

from dhydro_utils import modify_pli_files, get_boundary_condition_files, read_boundary_condition_file_dhydro
from dhydro_utils import get_mdu_as_dict, replace_boundary_condition, run_simulation, generate_realistic_hydrograph, save_hydrograph
from graph_creation import Mesh, find_closest_nodes

project_folder = 'D:\\DK41'
model_folder = os.path.join(project_folder, 'model', 'DflowFm')
save_folder = "raw_datasets_dk41"
# save_folder = os.path.join(project_folder, 'simulations')

dijkpalen_file = os.path.join(project_folder, "Dijkpalen", "Dijkpalen.shp")
polygon_file = os.path.join(project_folder, "Dijkpalen", "boundary_41.pol")

boundary_node_xy = np.loadtxt(polygon_file, skiprows=2)
dijkpalen = gpd.read_file(dijkpalen_file)
dijkpalen.sort_values(by='CODE', inplace=True)
dijkpalen = dijkpalen[dijkpalen.WS_DIJKRIN == 'DR41']
selected_dijkpalen_HD = dijkpalen[dijkpalen.CODE.str.startswith(('HD'))][5::28]
selected_dijkpalen_ND = dijkpalen[dijkpalen.CODE.str.startswith(('ND'))][10::30][::-1]

selected_dijkpalen = pd.concat([selected_dijkpalen_HD, selected_dijkpalen_ND])
points = selected_dijkpalen.geometry.apply(lambda geom: (geom.x, geom.y))
points = np.array([list(point) for point in points], dtype=np.float32)

# Finest scale (original mesh)
output_map = os.path.join(model_folder, "output", "base_map.nc")
mesh = Mesh()
mesh._import_from_map_netcdf(output_map, import_BC=False)

# Find boundary edges for breach location
boundary_edges = np.where(mesh.edge_type > 1)[0]
boundary_nodes = mesh.node_xy[mesh.edge_index.T[boundary_edges]].reshape(-1,2)

mdu_file = os.path.join(model_folder, 'LMW.mdu')
BC_folder = model_folder
forcing_file = os.path.join(model_folder, 'LMW.ext')
output_folder = os.path.join(model_folder, 'output')
map_file = os.path.join(output_folder, 'LMW_map.nc')
clm_file = os.path.join(output_folder, 'LMW_clm.nc')
fou_file = os.path.join(output_folder, 'LMW_fou.nc')
his_file = os.path.join(output_folder, 'LMW_his.nc')
dia_file = os.path.join(output_folder, 'LMW.dia')

BC_values_file, BC_loc_file = get_boundary_condition_files(forcing_file)

loc, scale=(600, 500) # loc, scale of the truncated normal distribution
min_peak, max_peak = (0, 2000)
a, b = (min_peak - loc) / scale, (max_peak - loc) / scale  # Set bounds

min_discharge=0
param1=(1, 0.4)
param2=(0., 0.2)
param3=(5,15)
param4=0.6
x_window=(0.2, 1.6)

# you must modify the boundary condition files before running the simulations
for sim, point in enumerate(points):
    np.random.seed(sim + 288)

    type_BCs, _ = read_boundary_condition_file_dhydro(model_folder, forcing_file)
    boundary_edge = find_closest_nodes(boundary_nodes, point, top_n=3)
    coords = np.unique(boundary_nodes[boundary_edge], axis=0)[:2]

    modify_pli_files(BC_folder, BC_loc_file, [coords], type_BCs, name="LMW")
    
    # change BC
    boundary_polygon_file = polygon_file
    config = get_mdu_as_dict(mdu_file)

    total_time = int(config['TStop'].split('.')[0]) # total time of the simulation in seconds
    time_resolution = 3600 # time resolution of the simulation in seconds
    init_time = pd.to_datetime(config['RefDate']) # initial time of the simulation

    hydrograph_time_series = generate_realistic_hydrograph(total_time, time_resolution, 
                                                           truncnorm.rvs(a, b, loc=loc, scale=scale),
                                                           min_discharge, 
                                                           np.random.lognormal(*param1), 
                                                           np.random.uniform(*param2), 
                                                           np.random.uniform(*param3), 
                                                           param4,
                                                           x_window)
    hydrograph_time_series = np.stack(hydrograph_time_series, -1)
    replace_boundary_condition(os.path.join(BC_folder, BC_values_file[0]), hydrograph_time_series, init_time, BC_type=2, BC_name="LMW")
    save_hydrograph(hydrograph_time_series, f'{save_folder}\\Boundary_conditions\\Hydrograph_{sim}.txt')

    # calculate total flood volume from discahrge time series
    total_flood_volume = np.trapezoid(hydrograph_time_series[:,1], dx=time_resolution)/1e6

    # run simulation
    print(f'Running simulation {sim}...')
    computation_time = run_simulation(model_folder)

    # save results in overview folder   
    shutil.copy(os.path.join(model_folder, map_file), f'{save_folder}\\Simulations\\simulation_{sim}_map.nc')
    shutil.copy(os.path.join(model_folder, clm_file), f'{save_folder}\\Simulations\\Misc\\simulation_{sim}_clm.nc')
    shutil.copy(os.path.join(model_folder, fou_file), f'{save_folder}\\Simulations\\Misc\\simulation_{sim}_fou.nc')
    shutil.copy(os.path.join(model_folder, his_file), f'{save_folder}\\Simulations\\Misc\\simulation_{sim}_his.nc')
    shutil.copy(os.path.join(model_folder, dia_file), f'{save_folder}\\Simulations\\Misc\\simulation_{sim}.dia')

    # concatenate simulation stas to df and save it at every iteration
    df = pd.DataFrame([{'seed': sim, 'total_flood_volume [1e6 m3]': np.round(total_flood_volume, 3), 'computation_time[s]': computation_time}])

    df.to_csv(f'{save_folder}\\overview.csv', mode='a', sep = ',', index = False, header=not os.path.exists(f'{save_folder}\\overview.csv'))