import torch
import torch.nn as nn
import numpy as np
import dgl
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx


class ExplainerTags:
    FEAT_NAME = '_nfeat'
    ORIGINAL_ID = '_original_id'


class ExplainerModule(nn.Module):
    def __init__(self, graph, args):
        super(ExplainerModule, self).__init__()
        n_edges = graph.num_edges()
        feat_dim = graph.ndata[ExplainerTags.FEAT_NAME].shape[-1]
        self.params = dict()
        self.params['edge_size'] = args.edge_size
        self.params['edge_ent'] = args.edge_ent
        self.params['eps'] = args.eps
        self.bidirectional_mapping = None
        if args.bidirectional:
            u, v = graph.edges()
            self.bidirectional_mapping = graph.edge_ids(v, u)
        self.edge_mask = self.construct_edge_mask(n_edges)

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = torch.FloatTensor(feat_dim)
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        if self.bidirectional_mapping is not None:
            mask = (mask + mask[self.bidirectional_mapping]) / 2
        return nn.Parameter(mask)

    def construct_edge_mask(self, num_edges, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_edges))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_edges + num_edges)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)
        return mask

    def forward(self, gnn_model, train_sub_graph, feat):
        masked_feat = feat
        edge_mask = self.edge_mask
        if self.bidirectional_mapping is not None:
            edge_mask = (self.edge_mask + self.edge_mask[self.bidirectional_mapping]) / 2
        h = gnn_model(train_sub_graph, masked_feat, torch.sigmoid(edge_mask))
        return h

    def loss(self, pos_pred_prob, pred_label):
        """
        :param pos_pred_prob: probability to be positive
        :param pred_label:
        :return:
        """
        # prediction loss
        loss = -pred_label * torch.log(pos_pred_prob + self.params['eps']) - (1 - pred_label) * torch.log(1 - pos_pred_prob + self.params['eps'])

        # edge loss
        me = torch.square(torch.sigmoid(self.edge_mask))
        loss = loss + torch.sum(me) * self.params['edge_size']  # edge regularization - subgraph size
        entropy = -me * torch.log(me + self.params['eps']) - (1 - me) * torch.log(1 - me + self.params['eps'])
        loss = loss + self.params['edge_ent'] * entropy.mean()  # edge los: entropy + regularization
        return loss

    def get_feature_mask(self):
        return self.feat_mask

    def get_edge_mask(self):
        return self.edge_mask if self.bidirectional_mapping is None else (self.edge_mask + self.edge_mask[self.bidirectional_mapping]) / 2


class GNNExplainer:
    def __init__(self, gnn_model, pred_model, input_g, pred_g, k_hop, graph_features, args):
        self.gnn_model = gnn_model
        self.pred_model = pred_model
        self.input_g = input_g
        self.pred_g = pred_g
        self.k_hop = k_hop
        self.features = graph_features
        self.args = args

    def _get_sub_graph_edges(self, u, v):
        """ get all nodes that contribute to the computation of node's embedding """
        nodes = torch.tensor([u, v])
        eid_list = []
        for _ in range(self.k_hop):
            predecessors, _, eid = self.input_g.in_edges(nodes, form='all')
            eid_list.extend(eid)
            predecessors = torch.flatten(predecessors).unique()
            nodes = torch.cat([nodes, predecessors])
            nodes = torch.unique(nodes)
        eid_list = list(np.unique(np.array(eid_list)))
        if self.args.bidirectional:
            sub_g = dgl.node_subgraph(self.input_g, nodes)
        else:
            sub_g = dgl.edge_subgraph(self.input_g, eid_list)  # TODO - handle heterogeneous graphs
        sub_g.edata[ExplainerTags.ORIGINAL_ID] = sub_g.edata[dgl.EID]
        sub_g.ndata[ExplainerTags.FEAT_NAME] = self.features[sub_g.ndata[dgl.NID]]
        new_u, new_v = np.where(sub_g.ndata[dgl.NID] == u)[0][0], np.where(sub_g.ndata[dgl.NID] == v)[0][0]
        return sub_g, new_u, new_v

    def explain_edge(self, edge_idx, predict_func):
        """ main function - calculate explanation """
        # get prediction label
        self.gnn_model.eval()
        self.pred_model.eval()
        with torch.no_grad():
            h = self.gnn_model(self.input_g, self.features)
            logits = self.pred_model(self.pred_g, h)
            pred_labels = (logits > 0).long()
            pred_label = pred_labels[edge_idx]

        # create initial subgraph (all nodes and edges that contribute to the explanation)
        u, v = self.pred_g.find_edges(edge_idx)
        u, v = u[0].item(), v[0].item()
        subgraph, new_u, new_v = self._get_sub_graph_edges(u, v)
        print(f'new u {new_u} new v {new_v}')
        print(subgraph.num_nodes())

        explainer = ExplainerModule(subgraph, self.args)

        subgraph = subgraph.to(self.args.device)
        explainer.to(self.args.device)
        self.gnn_model.to(self.args.device)

        # start optimizing
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr)
        # optimizer = torch.optim.Adam([self.feature_mask, subgraph.edata[ExplainerTags.EDGE_MASK]], lr=self.lr)

        pbar = tqdm(total=self.args.epoch)
        pbar.set_description('Explaining edge ({}, {})'.format(u, v))
        # training loop
        for epoch in range(1, self.args.epoch + 1):
            h = explainer(self.gnn_model, subgraph, subgraph.ndata[ExplainerTags.FEAT_NAME])
            h_u, h_v = h[new_u], h[new_v]
            pred_prob = torch.sigmoid(predict_func(h_u, h_v))
            # loss = self.__loss__(subgraph, new_node_id, log_logits, pred_label)
            loss = explainer.loss(pred_prob, pred_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.detach())
            pbar.update(1)
        pbar.close()
        # subgraph.__class__ = original_graph_class

        # get hard node feature mask and edge mask
        # node_feat_mask = explainer.get_feature_mask().detach().sigmoid() > self.args.threshold
        node_feat_mask = None
        edge_mask = explainer.get_edge_mask().detach().sigmoid().to('cpu').data.numpy()
        subgraph.edata['mask_intensity'] = explainer.get_edge_mask().detach()

        tmp_edges = sorted(edge_mask, reverse=True)
        save_edges_num = (1 + self.args.bidirectional) * self.args.save_edges_num
        threshold = tmp_edges[save_edges_num] if len(tmp_edges) > save_edges_num else tmp_edges[-1]
        if self.args.clean:
            subgraph.remove_edges(np.where(edge_mask < threshold)[0])

        # remove isolated nodes from subgraph
        isolated_nodes = np.where(((subgraph.in_degrees() == 0) & (subgraph.out_degrees() == 0)).to('cpu').data.numpy())[0]
        # if node_idx is not None:  # node classification
        #     # don't delete our node in any case..
        isolated_nodes = isolated_nodes[(isolated_nodes != new_u) & (isolated_nodes != new_v)]
        if len(isolated_nodes) != subgraph.number_of_nodes():
            subgraph.remove_nodes(isolated_nodes)
        subgraph = subgraph.to('cpu')
        return subgraph, edge_mask, node_feat_mask, u, v
