# Libraries
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from database.graph_creation import MultiscaleMesh

def get_none_scalers():
    none_scalers = {'DEM_scaler'        : None, 
                    'roughness_scaler'  : None,
                    'slope_scaler'      : None, 
                    'area_scaler'       : None,
                    'edge_length_scaler': None,
                    'edge_slope_scaler' : None, 
                    'WD_scaler'         : None,  
                    'V_scaler'          : None}
    return none_scalers

def stack_attributes(dataset, attribute, inverse=False, to_min=False):
    """Stack a single attribute across all samples in the dataset into a column vector.

    Args:
        dataset (list of Data): dataset samples
        attribute (str): name of the attribute to stack
        inverse (bool): if True, stacks 1/attribute
        to_min (bool): if True, subtracts the per-sample minimum before stacking

    Returns:
        torch.Tensor: stacked attribute of shape (N, 1)
    """
    if inverse:
        stacked_map = torch.cat([1/data[attribute] for data in dataset])
    else:
        stacked_map = torch.cat([data[attribute] - data[attribute].min()*to_min for data in dataset])
    
    return stacked_map.reshape(-1,1)

def scaler(train_database, attribute, type_scaler='minmax', inverse=False, to_min=False):
    """Fit a scaler on a dataset attribute.

    Args:
        train_database (list of Data): training samples
        attribute (str or list or tuple): attribute name to scale; list scales multiple
            attributes jointly, tuple computes vector norm before scaling
        type_scaler (str): 'minmax', 'minmax_neg', or 'standard'
        inverse (bool): if True, fits on 1/attribute
        to_min (bool): if True, subtracts per-sample minimum before fitting

    Returns:
        MinMaxScaler or StandardScaler: fitted scaler
    """    
    assert isinstance(train_database, list), 'train_database must be a list of torch_geometric.data.data.Data objects'
    
    if type_scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
    elif type_scaler == 'minmax_neg':
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif type_scaler == 'standard':
        scaler = StandardScaler()
    elif type_scaler is None:
        return None
    else:
        raise ValueError('type_scaler can be only "minmax", "minmax_neg", or "standard"')
    
    assert hasattr(train_database[0], attribute), 'train_database must contain the attribute {}'.format(attribute)

    if isinstance(attribute, list):
        all_attrs = torch.cat([stack_attributes(train_database, attr, inverse=inverse, to_min=to_min) for attr in attribute])
    elif isinstance(attribute, tuple):
        all_attrs = torch.cat([stack_attributes(train_database, attr, inverse=inverse, to_min=to_min)**2 for attr in attribute], 1)
        all_attrs = all_attrs.sum(1).sqrt().reshape(-1,1)
    else:
        all_attrs = stack_attributes(train_database, attribute, inverse=inverse, to_min=to_min)

    scaler.fit(all_attrs)
        
    return scaler

def multiscale_scaler(train_database, attribute: str, type_feature:str, type_scaler='minmax'):
    """Fit one scaler per scale for a multiscale dataset attribute.

    Args:
        train_database (list of Data): training samples (must be multiscale)
        attribute (str): name of the attribute to scale
        type_feature (str): 'node' or 'edge'
        type_scaler (str): 'minmax', 'minmax_neg', or 'standard'

    Returns:
        list of MinMaxScaler or StandardScaler: one fitted scaler per scale
    """
    assert isinstance(train_database, list), 'train_database must be a list of torch_geometric.data.data.Data objects'
    assert isinstance(train_database[0].mesh, MultiscaleMesh), 'train_database must be a list of multiscale datasets'

    num_scales = train_database[0].mesh.num_meshes

    if type_scaler == 'minmax':
        scalers = [MinMaxScaler(feature_range=(0,1)) for _ in range(num_scales)]
    elif type_scaler == 'minmax_neg':
        scalers = [MinMaxScaler(feature_range=(-1,1)) for _ in range(num_scales)]
    elif type_scaler == 'standard':
        scalers = [StandardScaler() for _ in range(num_scales)]
    elif type_scaler is None:
        return None
    else:
        raise ValueError('type_scaler can be only "minmax", "minmax_neg", or "standard", instead got {}'.format(type_scaler) + ' for multiscale_scaler()')
    
    if type_feature == 'node':
        all_attrs = [torch.cat([data[attribute][data.node_ptr[i]:data.node_ptr[i+1]] for data in train_database]).reshape(-1,1) for i in range(num_scales)]
    elif type_feature == 'edge':
        all_attrs = [torch.cat([data[attribute][data.edge_ptr[i]:data.edge_ptr[i+1]] for data in train_database]).reshape(-1,1) for i in range(num_scales)]
    else:
        raise ValueError('type_feature can be only "node" or "edge"')
    
    for attr, scaler in zip(all_attrs, scalers):
        scaler.fit(attr)
        
    return scalers

def get_scalers(dataset, scalers: dict):
    """Fit all scalers for the standard set of node and edge attributes.

    Args:
        dataset (list of Data): training samples used to fit the scalers
        scalers (dict): scaler type per attribute key ('minmax', 'minmax_neg', 'standard', or None)

    Returns:
        dict: same keys as input with fitted scaler objects as values
    """
    if scalers is None:
        scalers = get_none_scalers()

    scalers['roughness_scaler'] = scaler(dataset, 'roughness', type_scaler=scalers['roughness_scaler'])
    scalers['DEM_scaler'] = scaler(dataset, 'DEM', type_scaler=scalers['DEM_scaler'], to_min=True)
    scalers['WD_scaler'] = scaler(dataset, 'WD', type_scaler=scalers['WD_scaler'])
    # scalers['edge_slope_scaler'] = scaler(dataset, 'edge_slope', type_scaler=scalers['edge_slope_scaler'])

    # for multiscale datasets use different scalers for area and edges
    if isinstance(dataset[0].mesh, MultiscaleMesh):
        scalers['area_scaler'] = multiscale_scaler(dataset, 'area', type_feature='node', type_scaler=scalers['area_scaler'])
        scalers['edge_length_scaler'] = multiscale_scaler(dataset, 'face_distance', type_feature='edge', type_scaler=scalers['edge_length_scaler'])
    else:
        scalers['area_scaler'] = scaler(dataset, 'area', type_scaler=scalers['area_scaler'], inverse=False)
        scalers['edge_length_scaler'] = scaler(dataset, 'face_distance', type_scaler=scalers['edge_length_scaler'])
    
    scalers['V_scaler'] = scaler(dataset, ('VX', 'VY'), type_scaler=scalers['V_scaler'])

    return scalers

def get_scalers_type(scalers):
    """Return the scaler type string for each entry in a fitted scalers dict (inverse of get_scalers).

    Args:
        scalers (dict): fitted scalers keyed by attribute name

    Returns:
        dict: same keys with scaler type strings ('minmax', 'minmax_neg', 'standard', or None)
    """
    scaler_types = {}
    for attr_name, scaler in scalers.items():
        if scaler is None:
            scaler_types[attr_name] = None
        else:                
            if isinstance(scaler, list):
                scaler = scaler[0]
            
            if isinstance(scaler, StandardScaler):
                scaler_types[attr_name] = 'standard'
            elif isinstance(scaler, MinMaxScaler):
                if scaler.feature_range[0] == 0 and scaler.feature_range[1] == 1:
                    scaler_types[attr_name] = 'minmax'
                elif scaler.feature_range[0] == -1 and scaler.feature_range[1] == 1:
                    scaler_types[attr_name] = 'minmax_neg'
            else:
                scaler_types[attr_name] = scaler
                
    return scaler_types