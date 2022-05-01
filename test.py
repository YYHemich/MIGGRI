import torch
from graph import GraphCreateK, UnDirectionalGraphCreateK
from model import GraphSAGE, DotPredictor, MLPPredictor, LinearPredictor, GAT, GCN
from utils import evaluation
from aggregate_models import LSTMAggregator, MeanAggregator, MaxAggregator
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
    ap.add_argument('--model', default='model/mean_sage_dot_testIdx3_best.pkl')
    ap.add_argument('--device', default='cuda:0')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    data = torch.load(args.model, map_location=device)
    args = data['args']
    print(args)
    n_data_feature = data['feature']
    if args.bidirectional:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = UnDirectionalGraphCreateK(
            args)
    else:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = GraphCreateK(args)

    train_g = train_g.to(device)
    train_pos_g, train_neg_g = train_pos_g.to(device), train_neg_g.to(device)
    valid_pos_g, valid_neg_g = valid_pos_g.to(device), valid_neg_g.to(device)
    test_pos_g, test_neg_g = test_pos_g.to(device), test_neg_g.to(device)
    n_data_feature = n_data_feature.to(device)

    model = GNN_CONSTRUCTOR[args.gnn](args.FeatureDim, args.HiddenLayerDim)
    model.load_state_dict(data['gnn'])
    model = model.to(device)
    model.eval()

    pred = DECISION_MODEL[args.Predictor](args.HiddenLayerDim)
    pred.load_state_dict(data['pred'])
    pred.to(device)
    pred.eval()

    with torch.no_grad():
        h = model(train_g, n_data_feature)
        pos_score = torch.sigmoid(pred(test_pos_g, h))
        neg_score = torch.sigmoid(pred(test_neg_g, h))
        test_acc, test_f1, test_auc = evaluation(pos_score, neg_score)
        print('test acc: {:.4f} f1: {:.4f} auc: {:.4f}'.format(test_acc, test_f1, test_auc))


if __name__ == '__main__':
    main()
