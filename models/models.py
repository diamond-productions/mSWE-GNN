# Libraries
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch.nn import ReLU, PReLU, ELU, SiLU, Sigmoid, Dropout, Tanh, LeakyReLU

from utils.dataset import NUM_WATER_VARS

class BaseFloodModel(nn.Module):
    """Base class for flood inundation modelling with optional learned residual connections.

    Args:
        previous_t (int): number of previous time steps given as input
        future_t (int): number of future time steps to predict
        learned_residuals (bool or str or None): residual connection mode; True, 'all', False, or None
        seed (int): random seed for reproducibility
        residuals_base (int): base of the exponent used when residual_init='exp'
        residual_init (str): residual weights initialisation; 'exp' or 'random'
        with_WL (bool): whether to append water level as static input
        device (str): compute device
    """
    def __init__(self, previous_t=1, future_t=1, learned_residuals=None, seed=42, 
                 residuals_base=2, residual_init='exp', with_WL=False, 
                 device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.previous_t = previous_t
        self.future_t = future_t
        self.with_WL = with_WL
        self.learned_residuals = learned_residuals
        self.device = device
        self.residuals_base = residuals_base
        self.residual_init = residual_init
        assert residual_init == 'exp' or residual_init == 'random', "Argument 'residual_init' can only be either 'exp' or 'random'"
        self.out_dim = NUM_WATER_VARS*future_t
        
        if learned_residuals == True:
            if residual_init == 'exp':
                self.residual_weights = nn.Parameter(init_true_residuals_weights(previous_t, residuals_base, 
                                                                                 repeat=self.future_t, device=device))
            else:
                self.residual_weights = nn.Parameter(torch.Tensor(previous_t, future_t, device=device))
                nn.init.xavier_normal_(self.residual_weights)

        elif learned_residuals == 'all':
            assert future_t == 1, "When using learned_residuals='all', future_t must be 1, because i don't bother to implement it for future_t>1"
            if residual_init == 'exp':
                self.residual_weights = nn.Parameter(init_true_residuals_weights(previous_t, residuals_base,
                                                                                 repeat=self.out_dim, device=device))
            else:
                self.residual_weights = nn.Parameter(torch.Tensor(previous_t,self.out_dim, device=device))
                nn.init.xavier_normal_(self.residual_weights)

    def _add_residual_connection(self, x):
        """Compute residual output from input features based on learned_residuals mode.

        Args:
            x (Tensor, shape [N, F]): full node feature matrix including dynamic history

        Returns:
            Tensor, shape [N, out_dim]: residual term to add to the model output
        """
        residual_output = torch.zeros(x.shape[0], self.out_dim, device=self.device)

        if self.learned_residuals==True:
            x0 = x[:,-self.previous_t*NUM_WATER_VARS:].reshape(-1, self.previous_t, NUM_WATER_VARS)
            residual_output = torch.cat([torch.stack([(x0[:,:,i]@self.residual_weights[:,j])
                                                    for i in range(NUM_WATER_VARS)], -1)
                                                    for j in range(self.future_t)], -1)
            
        elif self.learned_residuals=='all':
            x0 = x[:,-self.previous_t*self.out_dim:].reshape(-1, self.previous_t, self.out_dim)
            residual_output = torch.stack([(x0[:,:,i]@self.residual_weights[:,i]) for i in range(self.out_dim)], -1)
                
        elif self.learned_residuals==False:
            x0 = x[:,-self.out_dim:]
            residual_output = x0

        return residual_output
    
    def _mask_small_WD(self, x, epsilon=0.001):
        """Zero out water depth below epsilon and velocities where water depth is zero.

        Args:
            x (Tensor, shape [N, out_dim]): predicted water variables (depth and velocity)
            epsilon (float): threshold below which water depth is zeroed

        Returns:
            Tensor, shape [N, out_dim]: masked predictions
        """
        wd_index = slice(0, x.shape[1], NUM_WATER_VARS)
        v_index = slice(1, x.shape[1], NUM_WATER_VARS)

        wd = x[:,wd_index] * (x[:,wd_index].abs() > epsilon)

        # Mask velocities where there is no water
        v = x[:,v_index] * (x[:,wd_index] != 0)
        x = torch.cat((wd, v), dim=-1)

        return x

    def _split_features(self, x):
        """Split node features into static and dynamic parts, optionally appending water level.

        Args:
            x (Tensor, shape [N, F]): full node feature matrix

        Returns:
            tuple[Tensor, Tensor]: static features (shape [N, F_s]) and dynamic features (shape [N, F_d])
        """
        x_s = x[:, :self.static_node_features - self.with_WL]
        x_d = x[:, self.static_node_features - self.with_WL:]
        if self.with_WL:
            WL = x_s[:, -1] + x_d[:, -self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)
        return x_s, x_d

def init_true_residuals_weights(previous_t: int, base=2, repeat=1, device='cpu'):
    """Initialise exponential residual weights so that later time steps have higher influence.

    Args:
        previous_t (int): number of previous time steps
        base (int): base of the exponential weighting
        repeat (int): number of output dimensions to repeat the weights for
        device (str): compute device

    Returns:
        Tensor, shape [previous_t, repeat]: normalised residual weight matrix
    """
    residual_weights = (base ** torch.arange(previous_t, device=device).float())
    norm_residual_weights = residual_weights/residual_weights.sum()
    norm_residual_weights = norm_residual_weights.unsqueeze(1).repeat(1, repeat).contiguous()
    return norm_residual_weights

def add_norm_dropout_activation(hidden_size, layer_norm=False, dropout=0, activation='relu',
                                device='cpu'):
    """Build a list of optional LayerNorm, Dropout, and activation layers.

    Args:
        hidden_size (int): feature dimension used for LayerNorm
        layer_norm (bool): whether to include a LayerNorm layer
        dropout (float): dropout probability; 0 disables dropout
        activation (str or None): activation function name
        device (str): compute device

    Returns:
        list[nn.Module]: sequence of norm/dropout/activation layers
    """
    layers = []
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_size, eps=1e-5, device=device))
    if dropout:
        layers.append(Dropout(dropout))
    if activation is not None:
        layers.append(activation_functions(activation, device=device))
    return layers


def init_weights(layer):
    if isinstance(layer, Lin):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)

def make_mlp(input_size, output_size, hidden_size=32, n_layers=2, bias=False,
             activation='relu', dropout=0, layer_norm=False, device='cpu'):
    """Build a fully-connected MLP with optional norm, dropout, and activation.

    Args:
        input_size (int): input feature dimension
        output_size (int): output feature dimension
        hidden_size (int): hidden layer width
        n_layers (int): total number of linear layers
        bias (bool): whether linear layers include a bias term
        activation (str or None): activation function name
        dropout (float): dropout probability; 0 disables dropout
        layer_norm (bool): whether to add LayerNorm after each layer
        device (str): compute device

    Returns:
        nn.Sequential: constructed MLP
    """
    layers = []
    if n_layers==1:
        layers.append(Lin(input_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(output_size, layer_norm=layer_norm, 
                                                      dropout=dropout, activation=activation, device=device)
    else:
        layers.append(Lin(input_size, hidden_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(hidden_size, layer_norm=layer_norm, dropout=dropout, 
                                                      activation=activation, device=device)
            
        for layer in range(n_layers-2):
            layers.append(Lin(hidden_size, hidden_size, bias=bias, device=device))
            layers = layers + add_norm_dropout_activation(hidden_size, layer_norm=layer_norm, dropout=dropout, 
                                                          activation=activation, device=device)

        layers.append(Lin(hidden_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(output_size, layer_norm=layer_norm, dropout=dropout, 
                                                      activation=activation, device=device)

    mlp = Seq(*layers)
    # mlp.apply(init_weights)

    return mlp


def activation_functions(activation_name, device='cpu'):
    """Return an activation module instance for the given name.

    Args:
        activation_name (str or None): one of 'relu', 'prelu', 'leakyrelu', 'elu', 'swish', 'sigmoid', 'tanh', or None
        device (str): compute device (used by PReLU)

    Returns:
        nn.Module or None: the corresponding activation layer
    """
    if activation_name == 'relu':
        return ReLU()
    elif activation_name == 'prelu':
        return PReLU(device=device)
    elif activation_name == 'leakyrelu':
        return LeakyReLU(0.1)
    elif activation_name == 'elu':
        return ELU()
    elif activation_name == 'swish':
        return SiLU()
    elif activation_name == 'sigmoid':
        return Sigmoid()
    elif activation_name == 'tanh':
        return Tanh()
    elif activation_name is None:
        return None
    else:
        raise AttributeError('Please choose one of the following options:\n'\
            '"relu", "prelu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh"')