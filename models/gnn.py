# Libraries
import torch
import torch.nn as nn
from models.models import BaseFloodModel, make_mlp, activation_functions
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch_geometric.utils import scatter
from torch.linalg import vector_norm
from typing import Optional

from utils.dataset import create_scale_mask, NUM_WATER_VARS

class GNN(BaseFloodModel):
    """Single-scale GNN with encoder-processor-decoder architecture.

    Args:
        num_node_features (int): number of input node features
        num_edge_features (int): number of input edge features
        hid_features (int): hidden feature dimension for GNN layers
        K (int): K-hop neighbourhood size
        n_GNN_layers (int): number of GNN processor layers
        type_GNN (str): graph operator type; 'SWEGNN', 'GNN_A', 'GNN_L', or 'GAT'
        mlp_layers (int): number of layers in MLP blocks
        mlp_activation (str): activation function for MLP blocks
        gnn_activation (str): activation function applied after each GNN layer
        dropout (float): dropout probability in the decoder
        with_WL (bool): whether to append water level as static input
        normalize (bool): whether to normalise learned fluxes in SWEGNN
        with_filter_matrix (bool): whether to include a filter matrix H in the graph convolution
        edge_mlp (bool): whether to apply an MLP edge encoder (SWEGNN only)
        with_gradient (bool): whether to use hydraulic gradients in SWEGNN
        **base_model_kwargs: additional arguments forwarded to BaseFloodModel
    """
    def __init__(self, num_node_features, num_edge_features, hid_features=32, K=2, n_GNN_layers=2, type_GNN="SWEGNN", 
                 mlp_layers=1, mlp_activation='prelu', gnn_activation='prelu', dropout=0, 
                 with_WL=True, normalize=True, with_filter_matrix=True, edge_mlp=True,
                 with_gradient=True, **base_model_kwargs):
        super(GNN, self).__init__(**base_model_kwargs)
        self.type_model = "GNN"
        self.hid_features = hid_features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.type_GNN = type_GNN
        self.edge_mlp = edge_mlp
        self.with_WL = with_WL
        self.gnn_activation = gnn_activation
        self.dynamic_node_features = self.previous_t*self.out_dim
        self.static_node_features = num_node_features - self.dynamic_node_features + self.with_WL
        
        # Edge encoder
        if type_GNN == "SWEGNN" and edge_mlp:
            self.num_edge_features = hid_features
            self.edge_encoder = make_mlp(num_edge_features, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                         activation=mlp_activation, device=self.device)
        
        # Node encoder
        if type_GNN == "SWEGNN":
            self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=mlp_layers,
                                        activation=mlp_activation, device=self.device)
    
            self.static_node_encoder = make_mlp(
                self.static_node_features, hid_features, hid_features, n_layers=2, bias=True,
                activation=mlp_activation, device=self.device)
        else:
            self.node_encoder = make_mlp(num_node_features + self.with_WL, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                        activation=mlp_activation, device=self.device)
        
        # GNN
        self.gnn_processor = self._make_gnn(hid_features, K_hops=K, n_GNN_layers=n_GNN_layers, n_layers=mlp_layers, 
                                            activation=mlp_activation, bias=True, type_GNN=type_GNN, 
                                            normalize=normalize, with_filter_matrix=with_filter_matrix,
                                            with_gradient=with_gradient)

        self.gnn_activation = activation_functions(gnn_activation, device=self.device)
        
        # Decoder
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=mlp_layers, dropout=dropout,
                                     activation=mlp_activation, device=self.device)

    def _make_gnn(self, hidden_size, K_hops=1, n_GNN_layers=1, type_GNN='SWEGNN', **swegnn_kwargs):
        """Build a ModuleList of GNN convolution layers.

        Args:
            hidden_size (int): node feature dimension inside the GNN
            K_hops (int): number of hops per convolution
            n_GNN_layers (int): number of stacked convolution layers
            type_GNN (str): graph operator type; 'SWEGNN', 'GNN_A', 'GNN_L', or 'GAT'
            **swegnn_kwargs: additional keyword arguments forwarded to SWEGNN

        Returns:
            nn.ModuleList: list of GNN convolution layers
        """
        convs = nn.ModuleList()
        for l in range(n_GNN_layers):
            if type_GNN == "GNN_L":
                convs.append(ChebConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GNN_A":
                convs.append(TAGConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GAT":
                convs.append(GATConv(hidden_size, hidden_size, heads=1))
            elif type_GNN == "SWEGNN":
                convs.append(SWEGNN(hidden_size, hidden_size, self.num_edge_features, K=K_hops, 
                            device=self.device, **swegnn_kwargs))
            else:
                raise ValueError("Only 'GNN_A', 'GNN_L', 'GAT', and 'SWEGNN' are valid for now")
        return convs
    
    def forward(self, graph):
        """Run encoder-processor-decoder forward pass on a graph batch.

        Args:
            graph (Data): PyG Data object with attributes x, edge_index, edge_attr

        Returns:
            Tensor, shape [N, out_dim]: predicted water variables for each node
        """
        x = graph.x.clone()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        # 1. Node and edge encoder
        if self.type_GNN == "SWEGNN" and self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)

        x0 = x
        x_s, x_d = self._split_features(x)

        if self.type_GNN == "SWEGNN":
            x_s = self.static_node_encoder(x_s)
            x = x_d = self.dynamic_node_encoder(x_d)
        else:
            x = self.node_encoder(torch.cat((x_s, x_d), 1))

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processor):
            if self.type_GNN == "SWEGNN":
                x = conv(x_s, x_d, edge_index, edge_attr)
            else:
                x = conv(x=x, edge_index=edge_index)

            # Add non-linearity
            if self.gnn_activation is not None:
                x = self.gnn_activation(x)

            x_d = x

        # 3. Decoder
        x = self.node_decoder(x)
                    
        # Add residual connections
        x = x + self._add_residual_connection(x0)
        
        # ReLU because of negative water depth or discharge
        x = torch.relu(x)

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.0001)

        return x
    
class MSGNN(BaseFloodModel):
    """Multi-scale GNN where each scale has its own processor and scales are linked by intra-scale edges.

    Args:
        num_node_features (int): number of input node features
        num_edge_features (int): number of input edge features
        num_scales (int): number of resolution scales
        hid_features (int): hidden feature dimension
        K (int or list): K-hop neighbourhood per scale; int applies the same value to all scales
        mlp_layers (int): number of layers in MLP blocks
        mlp_activation (str): activation function for MLP blocks
        gnn_activation (str): activation function applied after the final processor step
        learned_pooling (bool): whether to use a learnable pooling MLP
        skip_connections (bool): whether to add skip connections from fine to coarse during up-pass
        with_WL (bool): whether to append water level as static input
        normalize (bool): whether to normalise learned fluxes in SWEGNN
        with_filter_matrix (bool): whether to include a filter matrix H in the graph convolution
        edge_mlp (bool): whether to apply an MLP edge encoder
        with_gradient (bool): whether to use hydraulic gradients in SWEGNN
        upwind_mode (bool): whether to apply upwind masking to hydraulic gradients
        **base_model_kwargs: additional arguments forwarded to BaseFloodModel
    """
    def __init__(self, num_node_features, num_edge_features, num_scales, hid_features=32, K=2, 
                 mlp_layers=2, mlp_activation='prelu', gnn_activation='tanh', 
                 learned_pooling=False, skip_connections=True,
                 with_WL=False, normalize=True, with_filter_matrix=True, edge_mlp=True,
                 with_gradient=True, upwind_mode=False, **base_model_kwargs):
        super(MSGNN, self).__init__(**base_model_kwargs)
        self.type_model = "MSGNN"
        self.hid_features = hid_features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_scales = num_scales
        self.edge_mlp = edge_mlp
        self.with_WL = with_WL
        self.gnn_activation = gnn_activation
        self.upwind_mode = upwind_mode
        self.with_gradient = with_gradient
        self.dynamic_node_features = self.previous_t*NUM_WATER_VARS
        self.static_node_features = num_node_features - self.dynamic_node_features + self.with_WL
        self.learned_pooling = learned_pooling
        self.skip_connections = skip_connections
        self.K = [K]*num_scales if isinstance(K, int) else K
        if len(self.K) != num_scales*2-1:
            self.K = self.K + self.K[::-1][1:] # add reverse K_hops for the coarse to fine
            assert len(self.K) == num_scales*2-1, "K must be an int or a list of length num_scales or num_scales*2-1"
        
        # Edge encoder
        if edge_mlp:
            self.edge_encoder = make_mlp(num_edge_features, hid_features, hid_features, n_layers=mlp_layers, bias=True,
                                         activation=mlp_activation, device=self.device)
            num_edge_features = hid_features
        
        # Node encoders
        self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=mlp_layers,
                                    activation=mlp_activation, device=self.device)

        self.static_node_encoder = make_mlp(self.static_node_features, hid_features, hid_features, 
                                            n_layers=mlp_layers, bias=True, activation=mlp_activation, device=self.device)
                
        # Intra-scale GNN
        self.intra_scale_gnn = nn.ModuleList([
            SWEGNN(hid_features, hid_features, 0, K=1, n_layers=mlp_layers, 
                    activation=mlp_activation, bias=True, normalize=True, 
                    with_filter_matrix=False, with_gradient=False,
                    device=self.device) for _ in range(num_scales-1)])

        # learnable pooling
        if learned_pooling:
            self.pooling_mlp = make_mlp(hid_features*2, hid_features, hid_features, 
                                        n_layers=mlp_layers, activation=mlp_activation, device=self.device)
        
        # Processor  
        # GNN (1 per each scale)
        self.gnn_processor = nn.ModuleList([
            SWEGNN(hid_features, hid_features, num_edge_features, K=K, 
                    n_layers=mlp_layers, activation=mlp_activation, bias=True, 
                    normalize=normalize, with_filter_matrix=with_filter_matrix, upwind_mode=upwind_mode,
                    with_gradient=with_gradient) if K > 0 else Copy_GNN()
                    for K in self.K])

        self.gnn_activation = activation_functions(gnn_activation, device=self.device)
        
        # Decoder
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=mlp_layers, dropout=0,
                                     activation=mlp_activation, device=self.device)
        
    def _pooling(self, x, row_fine, col_coarse, reduce='mean', learnable=False):
        """Pool node features from a fine scale to a coarse scale.

        Args:
            x (Tensor, shape [N, F]): node feature matrix
            row_fine (Tensor, shape [E]): fine-scale source node indices
            col_coarse (Tensor, shape [E]): coarse-scale target node indices
            reduce (str): scatter reduction; e.g. 'mean' or 'sum'
            learnable (bool): whether to use the learned pooling MLP

        Returns:
            Tensor, shape [N, F]: node features after pooling into coarse positions
        """
        if learnable:
            e_ij = self.pooling_mlp(torch.cat((x[row_fine], x[col_coarse]), -1))
            x = scatter(src=e_ij, index=col_coarse, dim=0, dim_size=x.shape[0], reduce=reduce)
        else:
            x = scatter(src=x[row_fine], index=col_coarse, dim=0, dim_size=x.shape[0], reduce=reduce)
        return x
    
    def _create_scale_mask(self, data):
        """Return a per-node integer mask indicating which scale each node belongs to.

        Args:
            data (Data): PyG Data object containing node_ptr and graph metadata

        Returns:
            Tensor, shape [N]: integer scale index for each node (e.g. [0, 0, 1, 1, 2, ...])
        """
        num_nodes = data.x.size(0)
        return create_scale_mask(num_nodes, self.num_scales, data.node_ptr, data, device=self.device)
    
    def forward(self, graph):
        """Run multiscale encoder-processor-decoder forward pass.

        Args:
            graph (Data): PyG Data object with x, edge_index, edge_attr, edge_ptr, intra_mesh_edge_index, intra_edge_ptr

        Returns:
            Tensor, shape [N, out_dim]: predicted water variables for each node
        """
        x = graph.x.clone()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        edge_ptr = graph.edge_ptr
        intra_mesh_edge_index = graph.intra_mesh_edge_index
        intra_edge_ptr = graph.intra_edge_ptr
        self.device = x.device

        # Create scale mask
        mask = self._create_scale_mask(graph)
        
        # 1. Node and edge encoder
        if self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)

        x0 = x
        x_s, x_d = self._split_features(x)

        x_s = self.static_node_encoder(x_s)
        x_d = self.dynamic_node_encoder(x_d)

        # Create temporary variables to save the features after down and upgoing GNNs
        x_down = torch.zeros_like(x_d, device=self.device)
        x_up = torch.zeros_like(x_d, device=self.device)

        # 2. Processor
        # fine to coarse but skipping the coarsest scale (which is processed in the next loop)
        for i in range(self.num_scales-1):
            # downgoing GNN pass
            x_d = self.gnn_processor[i](x_s, x_d, edge_index[:,edge_ptr[i]:edge_ptr[i+1]], edge_attr[edge_ptr[i]:edge_ptr[i+1]])

            # keep in memory the last operation before pooling (which would be overwritten otherwise)
            x_down = x_down + x_d * (mask==i)[:,None]
                
            # Pool multiscale attributes from finest to coarsest scale
            col_coarse, row_fine = intra_mesh_edge_index[:,intra_edge_ptr[i]:intra_edge_ptr[i+1]]
            x_d = self._pooling(x_d, row_fine, col_coarse, reduce='mean', learnable=self.learned_pooling)

        x_down = x_down + x_d

        # coarse to fine
        for i in range(self.num_scales):
            gnn_id = self.num_scales-1+i
            # upgoing GNN pass
            x_d = self.gnn_processor[gnn_id](x_s, x_d, edge_index[:,edge_ptr[-i-2]:edge_ptr[-i-1]], edge_attr[edge_ptr[-i-2]:edge_ptr[-i-1]])
                
            # save GNN output before pooling
            x_up = x_up + x_d * (mask==self.num_scales-i-1)[:,None]

            # Un-pool multiscale attributes from coarsest to finest scale
            if i < self.num_scales-1:
                intra_scale_edges = intra_mesh_edge_index[:,intra_edge_ptr[-i-2]:intra_edge_ptr[-i-1]]
                x_d = self.intra_scale_gnn[i](x_s, x_d, intra_scale_edges)

                # add skip connection of saved finer scale attributes
                if self.skip_connections:
                    x_d = x_d + x_down * (mask==self.num_scales-i-2)[:,None]
        x = x_up

        # Add non-linearity
        if self.gnn_activation is not None:
            x = self.gnn_activation(x)

        # 3. Decoder
        x = self.node_decoder(x)
                    
        # Add residual connections
        x = x + self._add_residual_connection(x0)
        
        # ReLU because of negative water depth or discharge
        x = torch.relu(x)
        
        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)

        return x

class SWEGNN(nn.Module):
    """Shallow Water Equations-inspired graph convolution layer.

    Args:
        static_node_features (int): dimension of static node features
        dynamic_node_features (int): dimension of dynamic node features
        edge_features (int): dimension of edge features
        K (int): number of message-passing hops
        normalize (bool): whether to L2-normalise learned flux vectors
        with_filter_matrix (bool): whether to apply a linear filter matrix at each hop
        with_gradient (bool): whether to use hydraulic gradients in the node update
        upwind_mode (bool): whether to zero negative hydraulic gradients (upwind scheme)
        device (str): compute device
        **mlp_kwargs: additional keyword arguments forwarded to make_mlp
    """
    def __init__(self, static_node_features: int, dynamic_node_features: int, edge_features: int, 
                 K: int = 2, normalize=True, with_filter_matrix=True, with_gradient=True,
                 upwind_mode=False, device='cpu', **mlp_kwargs):
        super().__init__()
        self.edge_features = edge_features
        self.edge_input_size = edge_features + static_node_features*2 + dynamic_node_features*2
        self.edge_output_size = dynamic_node_features
        hidden_size = self.edge_output_size*2
        self.normalize = normalize
        self.K = K
        self.with_filter_matrix = with_filter_matrix
        self.device = device
        self.with_gradient = with_gradient
        self.upwind_mode = upwind_mode
        
        self.edge_mlp = make_mlp(self.edge_input_size, self.edge_output_size,
                                hidden_size=hidden_size, device=device, **mlp_kwargs)

        if with_filter_matrix:
            self.filter_matrix = torch.nn.ModuleList([
                nn.Linear(dynamic_node_features, dynamic_node_features, bias=False, device=device) for _ in range(K+1)
            ])


    def forward(self, 
                x_s: Tensor, 
                x_d: Tensor, 
                edge_index: Tensor, 
                edge_attr: Optional[Tensor]=None) -> Tensor:
        """Apply K hops of SWE-inspired message passing.

        Args:
            x_s (Tensor, shape [N, F_s]): static node features
            x_d (Tensor, shape [N, F_d]): dynamic node features
            edge_index (Tensor, shape [2, E]): edge connectivity
            edge_attr (Tensor, shape [E, F_e] or None): edge features

        Returns:
            Tensor, shape [N, F_d]: updated dynamic node features
        """
        row = edge_index[0]
        col = edge_index[1]
        num_nodes = x_d.size(0)
        if self.with_filter_matrix:
            out = self.filter_matrix[0].forward(x_d.clone())
        else:
            out = x_d.clone()
        
        for k in range(self.K):
            # Filter out zero values
            mask = out.sum(1) != 0
            mask_row = mask[row]
            mask_col = mask[col]
            edge_index_mask = mask_row + mask_col

            # Edge update
            e_ij = torch.cat([x_s[row][edge_index_mask], 
                                x_s[col][edge_index_mask], 
                                x_d[row][edge_index_mask], 
                                x_d[col][edge_index_mask]], 1)
            
            if self.edge_features > 0:
                e_ij = torch.cat([e_ij, edge_attr[edge_index_mask]], 1)

            s_ij = self.edge_mlp(e_ij)
            
            if self.normalize:
                s_ij = s_ij/vector_norm(s_ij, dim=1, keepdim=True)
                s_ij.masked_fill_(torch.isnan(s_ij), 0)

            # Node update
            if self.with_gradient:
                hydraulic_gradient = out[col][edge_index_mask]-out[row][edge_index_mask]
                if self.upwind_mode:
                    hydraulic_gradient[hydraulic_gradient<0] = 0
                shift_sum = hydraulic_gradient*s_ij
            else:
                shift_sum = s_ij*out[row][edge_index_mask]

            scattered = scatter(shift_sum, col[edge_index_mask], reduce='sum', 
                          dim=0, dim_size=num_nodes)

            if self.with_filter_matrix:
                scattered = self.filter_matrix[k+1].forward(scattered)

            out = out + scattered
        
        return out

    def __repr__(self):
        return '{}(node_features={}, edge_features={}, K={}, with_filter_matrix={}, with_gradient={})'.format(
            self.__class__.__name__, self.edge_output_size, 
            self.edge_features, self.K, self.with_filter_matrix,
            self.with_gradient)
    
class Copy_GNN(nn.Module):
    """Copy GNN layer that does nothing"""
    def __init__(self):
        super().__init__()

    def forward(self,
                x_s: Tensor,
                x_d: Tensor,
                edge_index: Tensor,
                edge_attr: Optional[Tensor]=None) -> Tensor:
        """Return a copy of dynamic node features unchanged.

        Args:
            x_s (Tensor, shape [N, F_s]): static node features (unused)
            x_d (Tensor, shape [N, F_d]): dynamic node features
            edge_index (Tensor, shape [2, E]): edge connectivity (unused)
            edge_attr (Tensor or None): edge features (unused)

        Returns:
            Tensor, shape [N, F_d]: copy of x_d
        """
        return x_d.clone()

    def __repr__(self):
        return '{} (this layer does nothing)'.format(self.__class__.__name__)