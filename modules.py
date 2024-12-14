import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

########################
### GraphSAGE (DGL)
########################
class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h):
        g = g.local_var()
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), \
                    fn.sum(msg='m', out='h'))
        ah = g.ndata.pop('h')
        h = self.linear(ah)
        h = F.relu(h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphSAGELayer(in_feats, n_hidden))
        # hidden layers
        self.layers.append(GraphSAGELayer(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphSAGELayer(n_hidden, n_classes))

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = layer(g, h)
        return h
        
########################
### GIN (DGL)
########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP"""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h

class GIN(nn.Module):
    """GIN model"""
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                output_dim, 
                num_layers=3):
        """model parameters setting
        Paramters
        ---------
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.ginlayers = torch.nn.ModuleList()
        # Input Layer
        self.ginlayers.append(GINConv(ApplyNodeFunc(nn.Linear(input_dim, hidden_dim)), 
                                    "sum", init_eps=0, learn_eps=False))
        # Hidden Layer
        self.ginlayers.append(GINConv(ApplyNodeFunc(nn.Linear(hidden_dim, hidden_dim)), 
                                      "sum", init_eps=0, learn_eps=False))
        # Output Layer
        self.ginlayers.append(GINConv(ApplyNodeFunc(nn.Linear(hidden_dim, output_dim)), 
                                      "sum", init_eps=0, learn_eps=False))

    def forward(self, g):
        h = g.ndata['feat']
        for i in range(self.num_layers):
            h = self.ginlayers[i](g, h)
        return h