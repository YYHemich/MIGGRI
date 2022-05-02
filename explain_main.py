import argparse
import torch
from model import GraphSAGE, MLPPredictor, DotPredictor
from graph import GraphCreateK, UnDirectionalGraphCreateK
from MyGNNExplainer import GNNExplainer


def args_parse():
    ap = argparse.ArgumentParser()

    # model src
    ap.add_argument('--model', default='model/gcn_best_testIdx4.pkl')
    ap.add_argument('--k_hop', type=int, default=2)
    ap.add_argument('--bidirectional', action='store_true')
    ap.add_argument('--edge_idx', type=int, default=2)

    # GNN-Explainer coefficients
    ap.add_argument('--edge_size', type=float, default=0.05)
    ap.add_argument('--edge_ent', type=float, default=0.1)
    ap.add_argument('--eps', type=float, default=1e-15)
    ap.add_argument('--threshold', type=float, default=0.05)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--epoch', type=int, default=50)
    ap.add_argument('--save_edges_num', type=int, default=50)

    # other settings
    ap.add_argument('--device', default="cuda:0")
    ap.add_argument('--random_seed', type=int, default=1024)
    return ap.parse_args()


def set_torch_random_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(pth):
    data = torch.load(pth)
    args = data['args']
    print(args)
    n_data_feature = data['feature']
    if args.bidirectional:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = UnDirectionalGraphCreateK(
            args)
    else:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = GraphCreateK(args)
    # aggregate the feature from different views(multi-instance)

    # ----------- 2. create model -------------- #

    # build a two-layer GraphSAGE model
    model = GraphSAGE(args.FeatureDim, args.HiddenLayerDim)
    model.load_state_dict(data['gnn'])

    if args.Predictor == 'dot':
        # raise Exception('pred method of DotPredictor has not been implemented.')
        pred = DotPredictor(args.HiddenLayerDim)
        predict_func = lambda h_u, h_v: torch.dot(h_u, h_v)
    else:
        pred = MLPPredictor(args.HiddenLayerDim)
        pred.load_state_dict(data['pred'])
        predict_func = lambda h_u, h_v: pred.model(torch.cat([h_u, h_v], dim=-1).reshape(1, -1))[0]

    return model, pred, n_data_feature, predict_func, train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g


def main():
    args = args_parse()
    set_torch_random_seed(args.random_seed)
    gnn_model, pred_model, nodes_feature, predict_func, train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = load_model(args.model)
    explainer = GNNExplainer(gnn_model, pred_model, train_g, test_pos_g, args.k_hop, nodes_feature, args)

    subgraph, edge_mask, node_feat_mask, u, v = explainer.explain_edge(args.edge_idx, predict_func)
    print('Edges of subgraph %s.' % (subgraph.num_edges()))


if __name__ == '__main__':
    main()
