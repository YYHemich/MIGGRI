import argparse


def argument():
    parser = argparse.ArgumentParser(description='GRN predictor')

    parser.add_argument('--CNNFeatureExtractor', type=str, default='con',
                        help='CNN feature extractor e.g. con, vgg16, resnet50, resnrt101.')
    parser.add_argument('--FeatureAggregationMode', type=str, default='mean',
                        help='Aggregation mode (Multi-instance) e.g. mean, max.')
    parser.add_argument('--FeatureDim', type=int, default=128, help='CNN extracted feature dimension e.g. 128')
    parser.add_argument('--HiddenLayerDim', type=int, default=128, help='Hidden Layer feature dimension e.g. 128')
    parser.add_argument('--Epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--warmup', type=bool, default=True, help='warm up learning rate scheduler ')
    parser.add_argument('--warmup_portion', type=float, default=0.2, help='warm up learning rate scheduler ')
    parser.add_argument('--test_idx', type=int, default=3,
                        help='index of the test set. out of the range means all data use for training.')
    parser.add_argument('--mini_batch_len', type=int, default=5, help='number of the mini batch')
    parser.add_argument('--mini_batch_pattern', default='data/rr_mini_batch_%s.pkl', help='pattern of dataset src')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='ratio of valid set proportion to train set')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--torch_random_seed', default=0, type=int)
    parser.add_argument('--bidirectional', action='store_true', help='build bidirectional graph')
    parser.add_argument('--gnn', default='sage', choices=['sage', 'gat', 'gcn'])
    parser.add_argument('--aggr', type=str, default='mean', choices=['lstm', 'max', 'mean'])
    parser.add_argument('--Predictor', type=str, default='dot', choices=['dot', 'mlp', 'lin'],
                        help='edge feature predictor e.g. dot mlp linear')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument()
    print(args.warmup)
