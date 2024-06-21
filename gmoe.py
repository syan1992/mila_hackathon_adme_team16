import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
import math
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import pdb
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import GATConv, GENConv
from torch_geometric.nn import DeepGCNLayer
from fairseq.modules import LayerNorm
import torch_scatter

class DeeperGCNLayer(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(DeeperGCNLayer, self).__init__(aggr='add')  # "Add" aggregation.
            self.lin = torch.nn.Linear(in_channels, out_channels)
            self.norm = LayerNorm(out_channels)
            self.residual = (in_channels == out_channels)

        def forward(self, x, edge_index):
            out = self.propagate(edge_index, x=x)
            out = self.lin(out)
            #out = self.lin(torch.cat((out, x0), dim=1))
            out = self.norm(out)
            if self.residual:
                out = out + x
            return F.relu(out)

        def message(self, x_j):
            return x_j

class VirtualNodeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(VirtualNodeGNN, self).__init__()
        self.node_encoder = torch.nn.Linear(9, hidden_channels)
        self.edge_encoder = torch.nn.Linear(3, hidden_channels)
        
        self.num_layers = num_layers
        self.virtual_node_emb = torch.nn.Parameter(torch.zeros(1, hidden_channels))
        
        self.layers = torch.nn.ModuleList()
        conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
        norm = torch.nn.LayerNorm(hidden_channels, elementwise_affine=True)
        act = torch.nn.ReLU(inplace=True)
        self.layers.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1))#DeeperGCNLayer(in_channels, hidden_channels))#DirectedMPNN(in_channels, hidden_channels))#DeeperGCNLayer(in_channels, hidden_channels))
        for i in range(num_layers - 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)
            self.layers.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x.float())
        edge_attr = self.edge_encoder(edge_attr.float())
        
        virtual_node = self.virtual_node_emb.repeat(batch.max().item() + 1, 1)
        i = 0
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            if i != self.num_layers - 1:  # Update virtual node in all but last layer
                virtual_node = virtual_node + torch_scatter.scatter_mean(x, batch, dim=0)
                x = x + virtual_node[batch]
            i = i + 1

        x = self.layers[0].act(self.layers[0].norm(x))
        x = global_mean_pool(x, batch)

        return x

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = torch.nn.Linear(9, hidden_channels)
        self.edge_encoder = torch.nn.Linear(3, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)


    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x.float())
        edge_attr = self.edge_encoder(edge_attr.float())

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        x = global_mean_pool(x, batch)
        return x

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, hop=1):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if hop==1:
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif hop==2:
            self.bond_encoder = TwoHopBondEncoder(emb_dim = emb_dim)
        else:
            raise Exception('Unimplemented hop %d' % hop)  

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, hop=1):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if hop==1:
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif hop==2:
            self.bond_encoder = TwoHopBondEncoder(emb_dim = emb_dim)
        else:
            raise Exception('Unimplemented hop %d' % hop)  

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GNN_MoE_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_experts=3, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', num_experts_1hop=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
        '''

        super(GNN_MoE_node, self).__init__()
        self.num_layer = num_layer
        self.num_experts = num_experts
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            convs_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for expert_idx in range(num_experts):
                if gnn_type == 'gin':
                    convs_list.append(GINConv(emb_dim))
                elif gnn_type == 'gcn':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GCNConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GCNConv(emb_dim, hop=2))  
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                
                bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            self.convs.append(convs_list)
            self.batch_norms.append(bn_list)

        # self.mix_fn = lambda h_expert_list: torch.mean(torch.stack(h_expert_list, dim=0), dim=0)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # edge_index: shape=(2, N_batch)
        # edge_attr: shape=(N_batch, d_attr)

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        #mask_2d = None
        #h_list = [self.atom_feature(batched_data, mask_2d=mask_2d)]

        for layer in range(self.num_layer):

            h_expert_list = []
            for expert in range(self.num_experts):

                h = self.convs[layer][expert](h_list[layer], edge_index, edge_attr) # TODO: use different edge_index and edge_attr for each expert
                h = self.batch_norms[layer][expert](h)
                h_expert_list.append(h)

            h = torch.stack(h_expert_list, dim=0) # shape=[num_experts, num_nodes, d_features]         
            h = torch.mean(h, dim=0) # shape=[num_nodes, d_features]

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

def tensor_to_one_hot(mask, seqlens):
    for i in range(len(seqlens)):
        mask[i, :seqlens[i]] = 1
    return mask
'''
from gmoe_module import MoE

class GNN_SpMoE_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_experts=3, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gcn', k=1, coef=1e-2, num_experts_1hop=None):
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
            k: k value for top-k sparse gating. 
            num_experts: total number of experts in each layer. 
            num_experts_1hop: number of hop-1 experts in each layer. The first num_experts_1hop are hop-1 experts. The rest num_experts-num_experts_1hop are hop-2 experts.

        super(GNN_SpMoE_node, self).__init__()
        self.num_layer = num_layer
        self.num_experts = num_experts
        self.k = k
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.atom_encoder.atom_embedding_list[7] = torch.nn.Embedding(3, 512).cuda()
        self.atom_encoder.atom_embedding_list[8] = torch.nn.Embedding(3, 512).cuda()
        ###List of GNNs
        self.ffns = torch.nn.ModuleList()

        self.atom_feature = AtomFeature(
                num_heads=8,
                num_atoms=512*9,
                num_in_degree=512,
                num_out_degree=512,
                hidden_dim=512,
                n_layers=6,
                no_2d=False,
        )

        self.edge_encoder = torch.nn.Linear(3, emb_dim)

        self.norm_input = torch.nn.BatchNorm1d(emb_dim)
        for layer in range(num_layer):
            convs_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for expert_idx in range(num_experts):
                if gnn_type == 'gin':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GINConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GINConv(emb_dim, hop=2)) 
                elif gnn_type == 'gcn':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GCNConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GCNConv(emb_dim, hop=2))  
                elif gnn_type == 'gen':
                    #convs_list.append(GATConv(emb_dim, emb_dim))
                    convs_list.append(GENConv(emb_dim, emb_dim, aggr='mean', t=1.0, learn_t=True, num_layers=2))
                elif gnn_type == 'gat':
                    convs_list.append(GATConv(emb_dim, emb_dim))
                elif gnn_type == 'gen_agg':
                    if expert_idx % 4 == 0:
                        convs_list.append(GENConv(emb_dim, emb_dim, aggr='add', t=1.0, learn_t=True, num_layers=2, norm='layer'))
                    elif expert_idx % 4 == 1:
                        convs_list.append(GENConv(emb_dim, emb_dim, aggr='mean', t=1.0, learn_t=True, num_layers=2, norm='layer'))
                    elif expert_idx % 4 == 2:
                        convs_list.append(GENConv(emb_dim, emb_dim, aggr='max', t=1.0, learn_t=True, num_layers=2, norm='layer'))
                    elif expert_idx % 4 == 3:
                        convs_list.append(GENConv(emb_dim, emb_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer'))
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))

                bn_list.append(torch.nn.BatchNorm1d(emb_dim))
                
            ffn = MoE(input_size=emb_dim, output_size=emb_dim, num_experts=num_experts, experts_conv=convs_list, experts_bn=bn_list, 
                    k=k, coef=coef, num_experts_1hop=self.num_experts_1hop)

            self.ffns.append(ffn)

        # self.mix_fn = lambda h_expert_list: torch.mean(torch.stack(h_expert_list, dim=0), dim=0)
        self.edge_encoder = BondEncoder(emb_dim = emb_dim)

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, batched_data, h_atom = None, batched_data_2hop=None):
        x, edge_index, edge_attr, batch = batched_data['x'].cuda(), batched_data['edge_index'], batched_data['edge_attr'], batched_data['batch'].cuda()

        edge_index = torch.cat(batched_data['edge_index'], dim=1).cuda()
        edge_attr = torch.cat(batched_data['edge_attr'], dim=0).cuda()
        edge_attr = self.edge_encoder(edge_attr)
        edge_attr = edge_attr.long()

        if h_atom is not None:
            mask = torch.zeros([h_atom.shape[1], h_atom.shape[0]])
            mask = tensor_to_one_hot(mask, batched_data['seqlens'])
        else:
            mask = torch.zeros([x.shape[0], x.shape[1]])
            mask = tensor_to_one_hot(mask, batched_data['seqlens'])
        #x = x[mask.bool()]

        if batched_data_2hop:
            x_2hop, edge_index_2hop, edge_attr_2hop, batch_2hop = batched_data_2hop.x, batched_data_2hop.edge_index, batched_data_2hop.edge_attr, batched_data_2hop.batch

        if h_atom is None:
            h_list = [self.atom_encoder(batched_data['x_raw'].cuda())]
        else:
            h_atom = h_atom.permute(1, 0, 2)
            h_list = [h_atom[mask.bool()]]
        
        ### computing input node embedding
        mask_2d = None
        self.load_balance_loss = 0 # initialize load_balance_loss to 0 at the beginning of each forward pass.
        
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            if batched_data_2hop:
                h, _layer_load_balance_loss = self.ffns[layer](h_list[layer], edge_index, edge_attr, edge_index_2hop, edge_attr_2hop) 
            else:
                h, _layer_load_balance_loss = self.ffns[layer](h_list[layer], edge_index, edge_attr, None, None) 
            self.load_balance_loss += _layer_load_balance_loss

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        self.load_balance_loss /= self.num_layer

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        #if h_atom is not None:
        #node_representation, _ = to_dense_batch(node_representation, torch.tensor(batched_data['batch']))
        #node_representation = node_representation.permute(1, 0, 2)

        return node_representation, self.load_balance_loss
'''
