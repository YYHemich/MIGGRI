import torch
import itertools
import pickle as pkl
from graph import GraphCreateK, UnDirectionalGraphCreateK
from model import GraphSAGE, DotPredictor, MLPPredictor, GAT, GCN, LinearPredictor
from aggregate_models import LSTMAggregator, MeanAggregator, MaxAggregator
from utils import evaluation, compute_loss, get_cosine_schedule_with_warmup
import argparse


AGGREGATOR_CONSTRUCTOR = {
    'lstm': LSTMAggregator,
    'mean': MeanAggregator,
    'max': MaxAggregator
}


GNN_CONSTRUCTOR = {
    'sage': GraphSAGE,
    'gat': GAT,
    'gcn': GCN
}

DECISION_MODEL = {
    'dot': DotPredictor,
    'mlp': MLPPredictor,
    'lin': LinearPredictor
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda:0')
    return ap.parse_args()


def set_torch_random_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config.model, map_location=device)
    args = state_dict['args']
    args.valid_ratio = 0.
    set_torch_random_seed(args.torch_random_seed)
    if args.bidirectional:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = UnDirectionalGraphCreateK(args)
    else:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = GraphCreateK(args)

    train_g = train_g.to(device)
    train_pos_g, train_neg_g = train_pos_g.to(device), train_neg_g.to(device)
    valid_pos_g, valid_neg_g = valid_pos_g.to(device), valid_neg_g.to(device)
    test_pos_g, test_neg_g = test_pos_g.to(device), test_neg_g.to(device)

    with open(f'data/{args.CNNFeatureExtractor}_{args.FeatureDim}.pkl', 'rb') as f1:
        gene_feature_dict = pkl.load(f1)

    # Key of gene_number_dict is name of gene. Value of of gene_number_dict is idx of gene
    with open('data/number_gene.pkl', 'rb') as f2:
        gene_number_dict = pkl.load(f2)

    gene_features = dict()
    for gene in gene_feature_dict:
        features = []
        for view in ['lateral', 'ventral', 'dorsal']:
            if view in gene_feature_dict[gene]:
                for embed in gene_feature_dict[gene][view]:
                    features.append(list(embed))
        gene_features[gene_number_dict[gene]] = torch.tensor(features)

    aggregator = AGGREGATOR_CONSTRUCTOR[args.aggr](args.FeatureDim)
    aggregator = aggregator.to(device)

    model = GNN_CONSTRUCTOR[args.gnn](args.FeatureDim, args.HiddenLayerDim)
    model = model.to(device)
    model.train()

    pred = DECISION_MODEL[args.Predictor](args.HiddenLayerDim)
    pred.to(device)
    optimizer = torch.optim.Adam(itertools.chain(aggregator.parameters(), model.parameters(), pred.parameters()),
                                 lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.Epoch * args.warmup_portion), args.Epoch)

    for epoch in range(state_dict['epoch']):
        n_data_feature = aggregator(gene_features, device)
        h = model(train_g, n_data_feature)
        pred.train()
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.warmup:
            lr_scheduler.step()

        with torch.no_grad():
            pred.eval()
            model.eval()
            h = model(train_g, n_data_feature)
            train_acc, train_f1, train_auc = evaluation(torch.sigmoid(pos_score), torch.sigmoid(neg_score))

            if epoch % 2 == 0:
                print(
                    'In epoch {}, train loss: {:.4f} acc: {:.4f} f1: {:.4f} auc: {:.4f}'.format(epoch, loss, train_acc,
                                                                                                train_f1, train_auc))

    torch.save({
        'gnn': model.state_dict(),
        'args': args,
        'pred': pred.state_dict(),
        'aggregator': aggregator,
        'feature': n_data_feature.cpu().detach(),
        'epoch': state_dict['epoch']
    }, config.model)


if __name__ == '__main__':
    main()
