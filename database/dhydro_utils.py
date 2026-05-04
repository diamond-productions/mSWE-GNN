import shutil
import numpy as np 
from perlin_noise import PerlinNoise
import random
import subprocess
import psutil
import time
import os
import pandas as pd
import pickle
import xarray as xr
from shapely.geometry import Polygon
from tqdm import tqdm
from scipy.stats import weibull_max

from graph_creation import center_grid_graph, interpolate_variable, get_coords, create_mesh_dhydro, generate_random_polygon, save_mesh
from graph_creation import create_polygon_meshes, get_outside_elements, remove_elements_outside_boundary, find_face_BC, Mesh

def structured_field_to_mesh(field_2d, mesh, scale):
    """Interpolate a regular 2D field onto mesh face centers.

    Args:
        field_2d (np.ndarray): Field on a regular grid with shape (ypix, xpix).
        mesh (Mesh): Mesh object with face and node coordinates.
        scale (float): Pixel-to-coordinate scaling factor.

    Returns:
        np.ndarray: Interpolated values at mesh face centers, shape (num_faces,).
    """
    ypix, xpix = field_2d.shape
    _, grid_nodes = center_grid_graph(xpix, ypix)
    grid_nodes = get_coords(grid_nodes)
    grid_nodes[:, 0] = grid_nodes[:, 0] * scale + mesh.node_x.min()
    grid_nodes[:, 1] = grid_nodes[:, 1] * scale + mesh.node_y.min()

    return interpolate_variable(mesh.face_xy, grid_nodes, field_2d.T.reshape(-1), method='nearest')

def create_raw_dataset_folder(folder_name):
    """Create a folder for the raw datasets which contains 
    "DEM", "Geometry", "Hydrograph", "Roughness" and "Simulations" folders.
    If the folder exists, it skips the rest.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        for subfolder in ["Boundary_conditions", "Simulations", "Geometry", "Mesh"]:
            os.makedirs(os.path.join(folder_name, subfolder), exist_ok=True)
    else:
        print("Folder already exists")
    return None

def pliz_to_list(pliz_path):
    """Convert a .pliz file to a list of geometries."""
    with open(pliz_path, 'r') as f:
        lines = f.readlines()

    id_locations = [index for index, line in enumerate(lines) if len(line.split()) == 1]
    num_points = [int(lines[i+1].split()[0]) for i in id_locations]

    line_geoms = []
    for i in range(len(id_locations)):
        coords = [list(map(float, line.split())) for line in lines[id_locations[i]+2:id_locations[i]+2+num_points[i]]]
        line_geoms.append(np.array(coords))

    return line_geoms

def export_dhydro_dk41_folder_data(model_folder, save_folder, _id):
    """Export the data from a D-Hydro model folder to a raw dataset folder

    Args:
        model_folder (str): the folder where the model files are stored
        save_folder (str): the folder where the raw dataset will be stored
        _id (int): the id of the simulation
    """
    sim_name = os.path.basename(os.path.dirname(model_folder))

    # Extract and save boundary polygon
    polygon_file = os.path.join(model_folder, "Begrenzing", "Gridenclosure", "LMW_enclosure.pol")
    new_polygon_file = f'{save_folder}\\Geometry\\boundary_dk41.pol'

    # remove nodes that are too close to each other
    boundary_node_xy = np.loadtxt(polygon_file, skiprows=2)
    distances = np.sqrt(np.sum(np.diff(np.concatenate([boundary_node_xy, boundary_node_xy[:1]], axis=0), axis=0)**2, axis=1))
    boundary_node_xy = boundary_node_xy[distances > 1]

    np.savetxt(new_polygon_file, boundary_node_xy, fmt='%.2f', delimiter=' ')

    # Create and save meshes
    polygon_folder = f'{save_folder}\\Geometry'
    meshes = []

    # Create and append meshes for dense polygons
    dense_polygons_file = polygon_folder + f"\\simplified_polygons.gpkg"
    meshes.extend(create_polygon_meshes(dense_polygons_file, include_polygon_mesh=True))

    # Finest scale (original mesh)
    sim_name = os.path.basename(os.path.dirname(model_folder))
    output_map = os.path.join(model_folder, "Mdu", f"DFM_OUTPUT_{sim_name}/{sim_name}_map.nc")
    shutil.copy(output_map, f'{save_folder}\\Simulations\\output_{_id}.nc')

    mesh = Mesh()
    mesh._import_from_map_netcdf(output_map)

    # Remove elements outside the boundary (use a small buffer to exclude elements on the boundary)
    boundary_polygon = Polygon(boundary_node_xy)
    _,_,_, outside_faces = get_outside_elements(mesh, boundary_polygon.buffer(20))
    inside_faces = np.delete(np.arange(len(mesh.face_xy)), outside_faces)
    mesh = remove_elements_outside_boundary(mesh, boundary_polygon)
    mesh.inside_faces = inside_faces

    # Change boundary conditions location
    breach_file = model_folder + f"\\Bres\\{sim_name}_bres.ini"
    config_breach = get_mdu_as_dict(breach_file)

    breach_xy = np.array((float(config_breach['StartLocationx']), float(config_breach['StartLocationy'])))
    # breach_DEM = float(config_breach['crestlevelmin']) # breach_crest = float(config_breach['crestlevelini'])

    mesh.edge_BC = np.where(np.sqrt(((mesh.node_xy[mesh.edge_index].mean(0) - breach_xy)**2).sum(1)) < 30)[0]
    if len(mesh.edge_BC) == 0: mesh.edge_BC = np.where(np.sqrt(((mesh.node_xy[mesh.edge_index].mean(0) - breach_xy)**2).sum(1)) < 60)[0]
    if len(mesh.edge_BC) == 0: raise ValueError("No boundary condition found")
    mesh.edge_index_BC = mesh.edge_index[:,mesh.edge_BC].T
    mesh.face_BC = find_face_BC(mesh)

    meshes.append(mesh)

    # Save meshes as pickle file
    mesh_file = f"{save_folder}\\Mesh\\meshes_{_id}.pkl"

    with open(mesh_file, 'wb') as f:
        pickle.dump(meshes, f)

    # Load hydrograph boundary conditions
    # resample the history file to match the output temporal resolution
    his_file = os.path.join(model_folder, "Mdu", f"DFM_OUTPUT_{sim_name}/{sim_name}_his.nc")
    map_dataset = xr.open_dataset(output_map)
    his_dataset = xr.open_dataset(his_file)
    
    output_time_resolution = pd.to_timedelta(map_dataset.time.diff(dim='time').mean().values).components.hours # [h]

    resampled_his_dataset = his_dataset.resample(time=f'{output_time_resolution}h').mean()
    inflow_discharge = resampled_his_dataset.dambreak_discharge.values
    inflow_discharge[inflow_discharge < 0] = 0 # remove negative values
    time_vector = (resampled_his_dataset.time - resampled_his_dataset.time[0]).dt.total_seconds().values

    BCs = np.concatenate([time_vector.reshape(-1,1), inflow_discharge.reshape(-1,1)], axis=1)

    save_hydrograph(BCs, os.path.join(save_folder, 'Boundary_conditions', f"BC_{_id}.txt"))

    return None

def get_extent_in_pixels(node_x, node_y):
    """Return the extent of some coordinates in pixels (needed for noise generation).
    The extent gets scaled so that the maximum dimension is 1000 pixels.
    
    Args:
        node_x (np.array): x-coordinates of the nodes
        node_y (np.array): y-coordinates of the nodes
    """
    
    xpix = int(node_x.max() - node_x.min())
    ypix = int(node_y.max() - node_y.min())

    assert xpix > 0 and ypix > 0, "Invalid mesh extent"

    scale = 10**(abs(int(np.log10(xpix))-2))

    xpix = int(xpix/scale)
    ypix = int(ypix/scale)
    
    assert ypix > 0, "Invalid mesh extent. X and Y dimensions may be too different."
    assert np.log10(ypix/xpix) < 1.5, "Invalid mesh extent. X and Y dimensions are too different."

    return xpix, ypix, scale

def generate_2D_perlin_noise(noise, xpix, ypix, squishing=1):
    """Generate 2D perlin noise as array of shape (xpix, ypix)
    
    Args:
        noise (PerlinNoise): PerlinNoise object
        xpix (int): x pixels
        ypix (int): y pixels
        squishing (float): how much to squish the noise in the x direction (squishing<1) or stretch it (squishing>1)
    """
    assert squishing > 0, "Squishing must be positive"
    assert xpix > 0 and isinstance(xpix, int), "xpix must be a positive integer"
    assert ypix > 0 and isinstance(ypix, int), "ypix must be a positive integer"
    assert isinstance(noise, PerlinNoise), "noise must be a PerlinNoise object"

    return np.array([[noise([squishing*i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])

def generate_mesh_DEM(seed, mesh, noise_octave=(1,8), DEM_multiplier=(1,7), 
                       slope_multiplier=(0,0.01)):
    '''
    Generates a random digital elevation model (DEM) based on Perlin noise
    
    Args:
        seed: int, replicable randomness for Perlin noise and magnitude randomizer
        mesh: meshkernel.py_structures.Mesh2d or Mesh object (contains mesh nodes and faces coordinates)
        noise_octave: tuple, range of octaves for Perlin noise
        DEM_multiplier: tuple, range of multipliers for the Perlin noise
        slope_multiplier: tuple, range of multipliers for the slope
    '''
    np.random.seed(seed)

    octaves = np.random.uniform(noise_octave[0], noise_octave[1])
    noise = PerlinNoise(octaves=octaves, seed=seed)
    DEM_multiplier = np.random.uniform(DEM_multiplier[0], DEM_multiplier[1])
    slope_multiplier = np.random.uniform(slope_multiplier[0], slope_multiplier[1])
    slope_direction = np.random.uniform(0, 2*np.pi)
    
    xpix, ypix, scale = get_extent_in_pixels(mesh.node_x, mesh.node_y)
    squishing = np.random.uniform(0.5, 2) # how much to squish the noise in the x direction (squishing<1) or stretch it (squishing>1)

    DEM = generate_2D_perlin_noise(noise, xpix, ypix, squishing)
    slope = np.array([[i*np.cos(slope_direction) + j*np.sin(slope_direction) for j in range(xpix)] for i in range(ypix)])

    DEM = DEM*DEM_multiplier + slope*slope_multiplier

    mesh_DEM = structured_field_to_mesh(DEM, mesh, scale)

    return mesh_DEM

def generate_mesh_roughness(seed, mesh, noise_octave=(3,7), roughness_range=(0.023, 0.2), discretized=False):
    '''
    Generates a random roughness map based on Perlin noise
    
    Args:
        seed: int, replicable randomness for Perlin noise and magnitude randomizer
        mesh: meshkernel.py_structures.Mesh2d or Mesh object (contains mesh nodes and faces coordinates)
        noise_octave: tuple, range of octaves for Perlin noise
        roughness_range: tuple, range of roughness values
        discretized: bool, if True, the roughness is discretized into 10 classes
    '''
    np.random.seed(seed)

    octaves = np.random.uniform(noise_octave[0], noise_octave[1])
    noise = PerlinNoise(octaves=octaves, seed=seed)
    
    xpix, ypix, scale = get_extent_in_pixels(mesh.node_x, mesh.node_y)
    squishing = random.uniform(0.5, 2) # how much to squish the noise in the x direction (squishing<1) or stretch it (squishing>1)

    roughness = generate_2D_perlin_noise(noise, xpix, ypix, squishing)

    mesh_roughness = structured_field_to_mesh(roughness, mesh, scale)

    # scale roughness between 0 and 1
    mesh_roughness = (mesh_roughness - mesh_roughness.min()) / (mesh_roughness.max() - mesh_roughness.min())

    # scale roughness between min_roughness and max_roughness
    mesh_roughness = roughness_range[0] + mesh_roughness * (roughness_range[1] - roughness_range[0])

    if discretized:
        bins = np.linspace(mesh_roughness.min(), mesh_roughness.max(), 11)
        mesh_roughness = bins[np.digitize(mesh_roughness, bins)-1]

    return mesh_roughness

def get_weir_discharge(base, height, C_d=0.6, g=9.81):
    """
    Calculate the discharge across a rectangular section with surface free flow.

    Args:
        base (float): Base width of the rectangular section (m)
        height (float): Water height (m)
        C_d (float): Discharge coefficient (default is 0.6)
        g (float): Acceleration due to gravity (default is 9.81 m/s²)

    Returns:
        float: Discharge (m³/s)
    """
    Q = C_d * base * height * (2 * g * height) ** 0.5
    return Q

def get_breach_growth_rate_verheij_VDKnaap(delta_water, delta_t, cs_b=0.5, f1=1.3, f2=0.04, g=9.81):
    """
    Calculate the erosion rate using the Verheij and Van Der Knaap (1998) formula.

    Args:
        delta_water (float): The difference between the height of the water columns on either side of the breach at time t.
        delta_t (float): Computational timestep.
        cs_b (float): The critical breach speed of the breach (e.g. 0.2 for sand and 0.5 for clay).
        f1 (float): Material factor, set to 1.3 (average for sand and clay levees)
        f2 (float): Constant, set to 0.04
        g (float): Acceleration due to gravity (m/s²)

    Returns:
        float: Erosion rate (m/s)
    """
    return (f1 * f2 / np.log(10)) * ((g * delta_water) ** 1.5 / cs_b**2) * 1/(1+f2*g*delta_t/cs_b/3600)

def get_breach_width_verheij_VDKnaap(delta_water, time_step, f1=1.3):
    """
    Calculate the erosion rate using the Verheij and Van Der Knaap (1998) formula.

    Args:
        delta_water (float): The difference between the height of the water columns on either side of the breach at time t.
        time_step (float): The time step at which the breach width is calculated.
        
    Returns:
        float: The width of the breach at time t.
    """    
    f2=0.04
    g=9.81
    critical_vel=0.2

    breach_width = f1 * g**0.5 * delta_water**1.5 / critical_vel * np.log(1 + f2 * g * time_step/ critical_vel)
    return breach_width

def change_nc_file(grid_file, varname, new_value):
    '''
    Change NETCDF variable 'varname' with 'new_value'
    '''
    from netCDF4 import Dataset
    ncfile = Dataset(grid_file,mode='r+') 

    ncfile.variables[varname][:] = new_value

    ncfile.close()
    return None

def save_map(map, dst_file, node_coords=None):
    '''Saves map as .xyz or .txt file
    
    Args:
        map: np.array, map representing some value in x and y directions
        dst_file: str (path-like), destination file for saving DEM
        node_coords: np.array, coordinates of the nodes (default=None, uses mesh nodes)
    '''
    if node_coords is None:
        number_grids = map.shape[0] #int(len(pos)**0.5)
        y_grid = np.array([[(i+0.5) for j in range(number_grids)] for i in range(number_grids)])
        x_grid = y_grid.T
        xyz = np.array([[x, y, z] for x, y, z in zip(x_grid.reshape(-1,1), y_grid.reshape(-1,1), map.reshape(-1,1))]).squeeze()
    else:
        xyz = np.array([[x, y, z] for (x, y), z in zip(node_coords, map)])
    
    #creating xyz file for DEM with proper number of decimals
    np.savetxt(dst_file, xyz, fmt = ('%1.1f', '%1.1f', '%1.5f'))

    return None

def modify_pli_files(BC_folder, BC_loc_file, breach_coords, type_BCs, verbose=False, name=None):
    """Modify the boundary condition location files
    
    Args:
        BC_folder (str): the folder containing the boundary condition location files
        BC_loc_file (list): the list of boundary condition location files
        breach_coords (list): the coordinates of the breach locations
        type_BCs (list): the type of boundary conditions
    """
    assert len(breach_coords) == len(type_BCs), 'The number of breach locations and boundary conditions must be the same'
    assert all([type_BC in [1,2] for type_BC in type_BCs]), 'The type of boundary condition must be either 1 or 2'
    # assert all([len(coords) == 2 for coords in breach_coords]), 'The coordinates of the breach locations must be a list of 2D coordinates'

    # assert len(BC_loc_file) == len(breach_coords), f'The number of boundary condition location files {len(BC_loc_file)} \
    #                                                  must be the same as the number of breach locations {len(breach_coords)}'
    
    if verbose: print('Modifying the boundary condition location files...')
    for BC_loc, coords, type_BC in zip(BC_loc_file, breach_coords, type_BCs):
        if verbose: print(f'Modifying {BC_loc}...')
        breach_polygon_file = os.path.join(BC_folder, BC_loc)
        replace_boundary_location(breach_polygon_file, coords, type_BC, name)
    
    return None

def replace_boundary_location(breach_polygon_file, coords, type_BC=2, name=None):
    """Replace the boundary condition polygon file according to the new breach location's coordinates
    
    Args:
        
    """
    assert type_BC in [1,2], "Invalid boundary condition type"

    name = 'WaterlevelH' if type_BC == 1 else 'HydrographQ' if name is None else name

    replacement = f'{name}\n'\
              f'    {len(coords)}    2\n'\
              f'{coords[0,0]}    {coords[0,1]}\n'\
              f'{coords[1,0]}    {coords[1,1]}'

    with open(breach_polygon_file, "w") as f:
        f.write(replacement)
        
    return None

def select_random_boundary_location(mesh, seed, num_BC=1):
    """Selects a random edge on the boundary of the domain for the breach boundary condition
    
    Args:        
        mesh: meshkernel.py_structures.Mesh2d object
        seed: int, seed for random selection of boundary edge
        num_BC: int, number of boundary conditions to select

    Returns: x and y coordinates of the breach location (np.array)
    """    
    np.random.seed(seed)

    boundary_edges = np.where((mesh.edge_faces.reshape(-1,2) == -1).sum(1) == 1)[0]
    breach_edge_id = np.random.randint(len(boundary_edges)-1, size=num_BC)
    boundary_edge = mesh.edge_nodes.reshape(-1,2)[boundary_edges[breach_edge_id]]

    coords = mesh.mesh_nodes[boundary_edge]

    assert coords.shape == (num_BC, 2, 2), 'The coordinates of the breach location must be a list of 2D coordinates'

    return coords

def generate_weibull_hydrograph(total_time:float, time_resolution:float, 
                                peak_value:float, min_discharge:float=0, shape:float=0):
    """Generates a hydrograph based on a Weibull distribution shape

    Args:
        shape: float, shape parameter of the Weibull distribution
            high values correspond to right-skewed hydrographs
            small values correspond to left-skewed hydrographs
        total_time: float, total time of the hydrograph [seconds]
        time_resolution: float, time resolution of the hydrograph [seconds]
        peak_value: float, peak value of the hydrograph [m3/s]
        min_discharge: float, minimum discharge value (default=0) [m3/s]

    Returns: time, y
    """
    assert shape > 0, "Shape parameter must be greater than 0"
    assert total_time > 0, "Total time must be greater than 0"
    assert time_resolution > 0, "Time resolution must be greater than 0"
    assert peak_value > 0, "Peak value must be greater than 0"
    assert min_discharge >= 0, "Minimum discharge must be greater or equal to 0"
    
    shape = 1+5**shape

    time_steps = int(total_time/time_resolution)+1
    time_x = np.linspace(weibull_max.ppf(0.01, shape), weibull_max.ppf(0.999, shape), time_steps)
    time_hydrograph = time_x - time_x.min()
    time_hydrograph = time_hydrograph / time_hydrograph.max() * total_time

    y = weibull_max.pdf(time_x, shape)
    y = y/y.max() * (peak_value-min_discharge) + min_discharge
    
    return time_hydrograph, y

def generate_realistic_hydrograph(total_time:float, time_resolution:float, 
                                  peak_value:float, min_discharge:float=0, 
                                  param1:float=5, param2:float=0.3, param3:float=3, param4:float=0.5,
                                  x_window:tuple=(0, 1)):
    """Generates a hydrograph based on a weird peak equation
    
    Args:
        total_time (float): total time of the hydrograph [s]
        time_resolution (float): time resolution of the hydrograph [s]
        peak_value (float): peak value of the hydrograph [m3/s]
        min_discharge (float): minimum discharge of the hydrograph [m3/s] (default is 0)
        param1 (float): parameter of the peak equation (default is 5)
        param2 (float): parameter of the peak equation (default is 0.3)
        x_window (tuple): window of the x-axis of the peak equation (default is (0, 1))

    Returns: time, y
    """
    assert x_window[0] < x_window[1], 'x_window[0] should be smaller than x_window[1]'
    assert total_time > 0, 'total_time should be positive'
    assert time_resolution > 0, 'time_resolution should be positive'
    assert min_discharge >= 0, 'min_discharge should be positive'
    assert peak_value > min_discharge, 'peak_value should be greater than min_discharge'
    # assert param1 > 1, 'param1 should be greater than 1'
    # assert param2/x_window[1] < 2, 'param2/x_window[1] should be smaller than 2 for a realistic peak equation'
    # assert param1/x_window[1] < 10, 'param1/x_window[1] should be smaller than 10 for a realistic peak equation'

    time_steps = int(total_time / time_resolution) + 1
    time_x = np.linspace(x_window[0], x_window[1], time_steps)
    time_hydrograph = time_x - time_x.min()
    time_hydrograph = time_hydrograph / time_hydrograph.max() * total_time

    F = time_x**2
    x = np.linspace(0, 1, time_steps)
    y = F / np.sqrt((param1 * F)**param3 + F * (F - 1)**2 * (param1 - 1)**2 * param2**2) * np.exp(-(x+param4)**10)
    y = y / y.max() * (peak_value - min_discharge) + min_discharge

    return time_hydrograph, y

def save_hydrograph(hydrograph_time_series, dst_file):
    '''Saves hydrograph file as time series'''
    np.savetxt(dst_file, hydrograph_time_series, fmt = ('%1d', '%4.4f'))
    return None

def replace_boundary_condition(boundary_condition_file, BC_time_series, init_time, BC_type=2, BC_name="HydrographQ"):
    """Changes boundary condition file to include the new boundary conditions (IN PLACE FILE MODIFICATION)
    
    Args:
        boundary_condition_file (str): the boundary condition file to be modified
        time_series (np.array): the time series of the boundary condition (time, value)
        init_time (pd.Timestamp): the initial time of the simulation
        BC_type (int): the type of boundary condition (1: water level, 2: discharge)
    """
    BC_dict = {1: "waterlevelbnd", 2: "dischargebnd"}
    unit_dict = {1: "m", 2: "m3/s"}

    replacement = f'[forcing]\n\
    Name				            = {BC_name}_0001\n\
    Function                        = timeseries\n\
    Time-interpolation              = linear\n\
    Quantity                        = time\n\
    Unit                            = seconds since {init_time}\n\
    Quantity                        = {BC_dict[BC_type]}\n\
    Unit                            = {unit_dict[BC_type]}\n'

    for time, y in BC_time_series:
        replacement += f'{time:.2f}\t {y:.2f}\n'
        
    with open(boundary_condition_file, "w") as f:
        f.write(replacement)

    return None

# Function to kill process and all children
def kill_process_tree(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):  # Get all child processes
            child.terminate()
        parent.terminate()  # Terminate parent after children
    except psutil.NoSuchProcess:
        pass  # Process already terminated

def run_simulation(model_folder):
    '''Run D-Hydro simulation, given model folder location
    Returns computational time
    '''
    # get parent folder where execution file is
    input_folder = os.path.abspath(os.path.join(model_folder, os.pardir))
    execution_file = f'{input_folder}\\run.bat'
    
    start_time = time.time()

    # Run D-Hydro, let Python wait till D-Hydro is done or timeout
    command = subprocess.Popen(execution_file, cwd=input_folder)
    try:
        command.wait()

    except KeyboardInterrupt:
        print("Manual interrupt detected. Stopping simulation...")
        kill_process_tree(command.pid)

    computation_time = round(time.time() - start_time, 4)

    return computation_time

def get_mdu_as_dict(config_file): 
    """Reads the configuration file of D-HYDRO (*.mdu) and returns it as a dictionary"""
    with open(config_file) as f:
        config = {}
        for line in f:
            if '=' in line:
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config

def get_boundary_condition_files(BC_file: str):
    """Reads a boundary condition file and returns the name of the forcing files
    
    Args:
        BC_file (str): path to the model folder (like 'FlowFM_bnd.ext')

    Returns:
        BC_values_file (list): name of the forcing files\n
        BC_loc_file (list): name of the location files
    """
    with open(BC_file, 'r') as file:
        BC_info = file.readlines()
        num_BCs = BC_info.count('[Boundary]\n') + BC_info.count('[boundary]\n')

        BC_values_file = [line.split('=')[1].strip() for line in BC_info if line.strip().startswith('forcingfile')]
        BC_loc_file = [line.split('=')[1].strip() for line in BC_info if line.strip().startswith('locationfile')]
        assert len(BC_values_file) == num_BCs

    return BC_values_file, BC_loc_file

def read_boundary_condition_file_dhydro(model_folder, BC_file):
    """Reads a boundary condition file and returns the type of boundary conditions and the values
    
    Args:
        model_folder (str): path to the model folder
        BC_file (str): name of the boundary condition file

    Returns:
        type_BCs (np.array): type of boundary conditions (1: water level, 2: discharge)\n
        BCs      (np.array): boundary conditions
    """
    BC_values_file, _ = get_boundary_condition_files(BC_file)
        
    type_BCs = []
    BCs = []
    for file in BC_values_file:
        file_folder = os.path.join(model_folder, file.lstrip('../'))
        with open(file_folder, 'r') as f:
            lines = f.readlines()
            type_BC = lines[6].split('=')[1].strip()
            type_BC = 2 if type_BC == 'dischargebnd' else 1 if type_BC == 'waterlevelbnd' else ValueError('Unknown boundary condition type')
            type_BCs.append(type_BC)
        BC = np.loadtxt(file_folder, skiprows=8)
        BCs.append(BC)
        
    return np.array(type_BCs), np.array(BCs)
    
def run_simulations_mesh(n_sim, model_folder, save_folder, start_sim=1, verbose=True, mesh=None,
                         polygon_file=None, num_vertices_polygon=(20,30), number_of_multiscales=4, ellipticality=(0.5,2), grid_size=100,
                         DEM_file=None, DEM_noise_octave=(3,8), DEM_multiplier=(1,7), slope_multiplier=(0,0.001),
                         roughness_file=None, roughness_noise_octave=(3,7), roughness_range=(0.023, 0.2), discretized=False,
                         random_breach=False, random_hydrograph=False, peak_value=(250,800), min_discharge=0, param1=(2, 5), param2=(0.1, 0.3), x_window=(0.35, 0.8)):
    '''
    Run multiple hydraulic simulations using D-Hydro

    Args:
        n_sim: int, number of simulations to run
        model_folder: str, directory containing the dimr_config file used to run the simulations
        save_folder: str, directory in which to store simulations
        start_sim: int, starting simulation id (for replicability)
        verbose: bool, if True, print simulation progress
        Mesh parameters:
            mesh: meshkernel.py_structures.Mesh2d (contains mesh nodes and faces coordinates)
            polygon_file: str, name of the polygon file (default=None, random polygon is generated)
            num_vertices_polygon: tuple of ints, minimum and maximum number of vertices for the polygon
            number_of_multiscales: int, number of mesh scales and refinement iterations
            ellipticality: tuple of floats, minimum and maximum ellipticality of the mesh (1 = circle)
            grid_size: float, multiplier for length of each cell in the mesh
        DEM parameters:
            DEM_file: str, name of the DEM file (default=None, random DEM is generated)
            DEM_noise_octave: tuple of floats, minimum and maximum spatial variation of DEM bumps
            DEM_multiplier: tuple of floats, minimum and maximum multiplier for DEM
            slope_multiplier: tuple of floats, minimum and maximum multiplier for slope
        Roughness parameters:
            roughness_file: str, name of the roughness file (default=None, random roughness is generated)
            roughness_noise_octave: tuple of floats, minimum and maximum spatial variation of roughness
            roughness_range: tuple of floats, minimum and maximum roughness values
            discretized: bool, if True, the roughness is discretized into 10 classes
        Boundary condition parameters:
            random_breach: bool, if True, the breach location is randomly selected
            random_hydrograph: bool, if True, the hydrograph is randomly generated
            peak_value: float or tuple of floats, peak value or peak range of the hydrograph
            min_discharge: float or tuple of floats, minimum discharge or minimum discharge range of the hydrograph
            param1: float or tuple of floats, parameter of the peak equation
            param2: float or tuple of floats, parameter of the peak equation
            x_window: tuple of floats, window of the x-axis of the peak equation
    '''
    boundary_polygon_file = polygon_file
    config = get_mdu_as_dict(f'{model_folder}\\FlowFM.mdu')

    total_time = int(config['TStop']) # total time of the simulation in seconds
    simulated_time_hours = total_time/3600 # total time of the simulation in hours
    time_resolution = int(config['MapInterval']) # time resolution of the simulation in seconds
    init_time = pd.to_datetime(config['RefDate']) # initial time of the simulation

    simulation_stats = []

    existing_mesh = mesh is not None

    for sim in tqdm(range(start_sim, start_sim+n_sim)):
        np.random.seed(sim)

        if polygon_file is None:
            # Generate random polygon for mesh creation
            _ = generate_random_polygon(save_polygon=True, avg_radius=100*grid_size, irregularity=0.2, 
                                        spikiness=0.08, seed=sim, num_vertices=num_vertices_polygon, ellipticality=ellipticality)
            boundary_polygon_file = 'random_polygon.pol'
            shutil.copy(boundary_polygon_file, f'{save_folder}\\Geometry\\polygon_{sim}.pol')
        else:
            shutil.copy(boundary_polygon_file, f'{save_folder}\\Geometry\\polygon_{sim}.pol')

        # Create mesh
        if not existing_mesh:
            if verbose: print('Creating mesh...')
            mesh = create_mesh_dhydro(boundary_polygon_file, number_of_multiscales)

            # Save mesh in model folder
            save_mesh(mesh, f'{model_folder}\\SWE_GNN_mesh_net.nc')

        # Generate DEM 
        if DEM_file is None:
            if verbose: print('Generating DEM...')
            mesh_DEM = generate_mesh_DEM(sim, mesh=mesh, 
                                         noise_octave=DEM_noise_octave, DEM_multiplier=DEM_multiplier, 
                                         slope_multiplier=slope_multiplier)

            # Save DEM in model folder
            new_DEM_file = os.path.join(model_folder, 'DEM.xyz')
            save_map(mesh_DEM, new_DEM_file, mesh.face_xy)
            shutil.copy(new_DEM_file, f'{save_folder}\\DEM\\DEM_{sim}.xyz')
        else:
            if verbose: print('Loading DEM...')
            DEM = np.loadtxt(DEM_file)[:,2].reshape(-1)
            grid_nodes = np.loadtxt(DEM_file)[:,:2]
            mesh_DEM = interpolate_variable(mesh.face_xy, grid_nodes, DEM, method='nearest')
        save_map(mesh_DEM, f'{save_folder}\\DEM\\DEM_{sim}.xyz', mesh.face_xy)

        # Generate roughness map
        if roughness_file is None:
            if verbose: print('Generating roughness map...')
            mesh_roughness = generate_mesh_roughness(sim+1, mesh=mesh, # sim+1 to avoid same seed as DEM
                                                     noise_octave=roughness_noise_octave, roughness_range=roughness_range, 
                                                     discretized=discretized)
            
            # save roughness map in model folder
            new_roughness_file = os.path.join(model_folder, 'roughness.xyz')
            save_map(mesh_roughness, new_roughness_file, mesh.face_xy)
            shutil.copy(new_roughness_file, f'{save_folder}\\Roughness\\roughness_{sim}.xyz')
        else:
            if verbose: print('Loading roughness map...')
            roughness = np.loadtxt(roughness_file)[:,2].reshape(-1)
            grid_nodes = np.loadtxt(roughness_file)[:,:2]
            mesh_roughness = interpolate_variable(mesh.face_xy, grid_nodes, roughness, method='nearest')
        save_map(mesh_roughness, f'{save_folder}\\Roughness\\roughness_{sim}.xyz', mesh.face_xy)

        # Locate breach boundary condition
        if random_breach:
            if verbose: print('Locating breach boundary condition...')
            BC_file = os.path.join(model_folder, 'FlowFM_bnd.ext')
            type_BCs, _ = read_boundary_condition_file_dhydro(model_folder, BC_file)
            breach_coords = select_random_boundary_location(mesh, seed=sim, num_BC=len(type_BCs))
            modify_pli_files(model_folder, breach_coords, type_BCs)
        
        # Generate random hydrograph (for now only works with 1 BC)
        if random_hydrograph:
            if verbose: print('Generating random hydrograph...')
            BC_values_file, _ = get_boundary_condition_files(os.path.join(model_folder, 'FlowFM_bnd.ext'))

            param1 = np.random.uniform(param1[0], param1[1]) if isinstance(param1, tuple) else param1
            param2 = np.random.uniform(param2[0], param2[1]) if isinstance(param2, tuple) else param2
            peak_value = np.random.uniform(peak_value[0], peak_value[1]) if isinstance(peak_value, tuple) else peak_value
            min_discharge = np.random.uniform(min_discharge[0], min_discharge[1]) if isinstance(min_discharge, tuple) else min_discharge

            hydrograph_time_series = generate_realistic_hydrograph(total_time, time_resolution, 
                                                                   peak_value, min_discharge, 
                                                                   param1, param2, x_window)
            hydrograph_time_series = np.stack(hydrograph_time_series, -1)
            replace_boundary_condition(os.path.join(model_folder, BC_values_file[0]), hydrograph_time_series, init_time, BC_type=2)
            save_hydrograph(hydrograph_time_series, f'{save_folder}\\Hydrograph\\Hydrograph_{sim}.txt')

        # run simulation
        if verbose: print(f'Running simulation {sim}...')
        computation_time = run_simulation(model_folder)

        # if simulation failed, skip
        if computation_time < 1:
            print(f'Error in simulation {sim}')
            continue

        # save results in overview folder   
        shutil.copy(os.path.join(model_folder, 'output\\FlowFM_map.nc'), f'{save_folder}\\Simulations\\output_{sim}_map.nc')

        simulation_stats.append([sim, mesh.face_x.shape[0], simulated_time_hours, computation_time])

    df = pd.DataFrame(simulation_stats, columns=['seed', 'mesh_num_faces', 'simulation_time[h]', 'computation_time[s]'])
    df.to_csv(f'{save_folder}\\overview.csv', mode='a', sep = ',', index = False, header=not os.path.exists(f'{save_folder}\\overview.csv'))

    return df

def get_breach_DL_info(model_folder):
    """Extracts water levels and corresponding discharge from a breach simulation.
    Also extract other information related to breach growth, such as crestlevelini, breachwidthini, crestlevelmin, f1."""
    sim_name = os.path.basename(os.path.dirname(model_folder))

    # mdu file
    mdu_file = os.path.join(model_folder, "Mdu", f"{sim_name}.mdu")
    config = get_mdu_as_dict(mdu_file)
    forcing_file = os.path.join(mdu_file, "..", config['ExtForceFileNew'])

    # Load original boundary conditions (water levels)
    _, BCs = read_boundary_condition_file_dhydro(model_folder, forcing_file)
    output_time_resolution = float(config['MapInterval']) #[s]
    output_time_resolution_h = int(output_time_resolution/3600) # [h]
    total_BC_time_steps = int(float(config['TStop'])/3600) # [h]
    BCs = BCs[:,:total_BC_time_steps+output_time_resolution_h][:,::output_time_resolution_h]

    water_level_BC = BCs[0,:,1] # water level

    # get boundary conditions
    nc_dataset = xr.open_dataset(os.path.join(model_folder, "Mdu", f"DFM_OUTPUT_{sim_name}/{sim_name}_his.nc"))
    time_step = np.diff(nc_dataset.time).mean().astype('timedelta64[m]').astype(int) # [min]
    num_time_steps_h = int(60/time_step)

    inflow_discharge_BC = nc_dataset.dambreak_discharge.values[::num_time_steps_h*output_time_resolution_h, 0][:total_BC_time_steps+output_time_resolution_h]
    inflow_discharge_BC = np.nan_to_num(inflow_discharge_BC) # replace nan with 0

    max_breach_width = nc_dataset.dambreak_crest_width.values.max()

    # Load breach information
    breach_file = os.path.join(model_folder, "Bres", f'{sim_name}_bres.ini')
    config_breach = get_mdu_as_dict(breach_file)
    crestlevelmin = float(config_breach['crestlevelmin'])
    breachwidthini = float(config_breach['breachwidthini'])
    f1 = float(config_breach['f1'])

    water_level_0 = water_level_BC[0]
    
    extra_info = np.array([water_level_0, crestlevelmin, breachwidthini, max_breach_width, f1])

    return water_level_BC, inflow_discharge_BC, extra_info