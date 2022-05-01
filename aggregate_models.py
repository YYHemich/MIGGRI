import torch
import torch.nn as nn


class LSTMAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(LSTMAggregator, self).__init__()
        self.aggregator = nn.LSTM(feature_dim, feature_dim, batch_first=True)

    def forward(self, gene_features: dict, device):
        """
        :param device:
        :param gene_features: key: node idx, value: set of features
        :return:
        """
        n_data_feature = []
        node_seq = sorted(list(gene_features.keys()))
        for i in node_seq:
            feature = gene_features[i]
            feature = feature.to(device)
            r = torch.randperm(feature.size(0))
            _, (h, _) = self.aggregator(feature[r].unsqueeze(0))
            n_data_feature.append(h.squeeze(0))
        n_data_feature = torch.cat(n_data_feature)
        return n_data_feature


class MaxAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(MaxAggregator, self).__init__()

    def forward(self, gene_features: dict, device):
        n_data_feature = []
        node_seq = sorted(list(gene_features.keys()))
        for i in node_seq:
            feature = gene_features[i]
            feature = feature.to(device)
            feature = torch.max(feature, dim=0)[0].unsqueeze(0)
            n_data_feature.append(feature)
        n_data_feature = torch.cat(n_data_feature)
        return n_data_feature


class MeanAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(MeanAggregator, self).__init__()

    def forward(self, gene_features: dict, device):
        n_data_feature = []
        node_seq = sorted(list(gene_features.keys()))
        for i in node_seq:
            feature = gene_features[i]
            feature = feature.to(device)
            feature = torch.mean(feature, dim=0).unsqueeze(0)
            n_data_feature.append(feature)
        n_data_feature = torch.cat(n_data_feature)
        return n_data_feature
