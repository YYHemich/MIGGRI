from dgl.nn import SAGEConv, GraphConv
from dgl.nn.pytorch.conv import GATConv
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch


# Define a GraphSAGE model
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat, edge_weight=None):
        h = self.conv1(g, in_feat, edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight)
        # h = F.relu(h)
        # h = self.conv2(g, h, edge_weight)
        return h


# Define a GAT model
# build a two-layer GAT model
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GAT, self).__init__()
        num_heads = 4
        self.layer1 = GATConv(in_feats, h_feats, num_heads, feat_drop=0., attn_drop=0.,
                              residual=False, allow_zero_in_degree=True)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(h_feats * num_heads, h_feats, 1, feat_drop=0., attn_drop=0.,
                              residual=False, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.layer1(g, in_feat)
        # Concat last 2 dim (num_heads * out_dim)
        h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = F.elu(h)
        h = self.layer2(g, h)
        # Sueeze the head dim as it's = 1
        h = h.squeeze() # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        return h


# Define a GCN model
# build a two-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# link prediction
class DotPredictor(nn.Module):
    def __init__(self, h_feat):
        super(DotPredictor, self).__init__()
        pass
    
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        self.model = nn.Sequential(
            nn.Linear(in_features=h_feats * 2, out_features=h_feats, bias=False),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=h_feats, out_features=1, bias=False),
        )

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        # return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
        return {'score': self.model(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class LinearPredictor(nn.Module):
    def __init__(self, h_feats):
        super(LinearPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(h_feats * 2),
            nn.Linear(h_feats * 2, 1)
        )

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        # return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
        return {'score': self.fc(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
