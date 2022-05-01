import dgl
import pickle as pkl
import numpy as np


def GraphCreateK(args):
    # train + valid : test = 4 : 1; train : valid = 9 :  1
    test_idx = args.test_idx
    test_pos_u, test_pos_v = [], []
    train_pos_u, train_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_neg_u, train_neg_v = [], []
    for i in range(args.mini_batch_len):
        with open(args.mini_batch_pattern % i, 'rb') as f:
            mini_edges_dict = pkl.load(f)
        if i == test_idx:
            test_pos_u.append(mini_edges_dict['Src'])
            test_pos_v.append(mini_edges_dict['Dst'])
            test_neg_u.append(mini_edges_dict['NegSrc'])
            test_neg_v.append(mini_edges_dict['NegDst'])
        else:
            train_pos_u.append(mini_edges_dict['Src'])
            train_pos_v.append(mini_edges_dict['Dst'])
            train_neg_u.append(mini_edges_dict['NegSrc'])
            train_neg_v.append(mini_edges_dict['NegDst'])
    train_pos_u = np.hstack(train_pos_u)
    train_pos_v = np.hstack(train_pos_v)
    train_neg_u = np.hstack(train_neg_u)
    train_neg_v = np.hstack(train_neg_v)
    test_pos_u = np.hstack(test_pos_u) if len(test_pos_u) > 0 else np.array([])
    test_pos_v = np.hstack(test_pos_v) if len(test_pos_v) > 0 else np.array([])
    test_neg_u = np.hstack(test_neg_u) if len(test_neg_u) > 0 else np.array([])
    test_neg_v = np.hstack(test_neg_v) if len(test_neg_v) > 0 else np.array([])

    eids = np.arange(len(train_pos_u))
    eids = np.random.permutation(eids)
    valid_size = int((len(train_pos_u)) * args.valid_ratio)

    # split all positive edges for training and testing
    valid_pos_u, valid_pos_v = train_pos_u[eids[:valid_size]], train_pos_v[eids[:valid_size]]
    train_pos_u, train_pos_v = train_pos_u[eids[valid_size:]], train_pos_v[eids[valid_size:]]

    # split all negative edges for training and testing
    valid_neg_u, valid_neg_v = train_neg_u[eids[:valid_size]], train_neg_v[eids[:valid_size]]
    train_neg_u, train_neg_v = train_neg_u[eids[valid_size:]], train_neg_v[eids[valid_size:]]

    raw_dir = f'data/{args.CNNFeatureExtractor}_{args.FeatureDim}.pkl'
    with open(raw_dir, 'rb') as f1:
        gene_feature_dict = pkl.load(f1)

    # create positive and negative sub-graph for training/validation/test set
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=len(gene_feature_dict))
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=len(gene_feature_dict))
    valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=len(gene_feature_dict))
    valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=len(gene_feature_dict))
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=len(gene_feature_dict))
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=len(gene_feature_dict))

    return train_pos_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g


def UnDirectionalGraphCreateK(args):
    # train + valid : test = 4 : 1; train : valid = 9 :  1
    test_idx = args.test_idx
    test_pos_u, test_pos_v = [], []
    train_pos_u, train_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_neg_u, train_neg_v = [], []
    for i in range(args.mini_batch_len):
        with open(args.mini_batch_pattern % i, 'rb') as f:
            mini_edges_dict = pkl.load(f)
        if i == test_idx:
            test_pos_u.append(mini_edges_dict['Src'])
            test_pos_v.append(mini_edges_dict['Dst'])
            test_neg_u.append(mini_edges_dict['NegSrc'])
            test_neg_v.append(mini_edges_dict['NegDst'])
        else:
            train_pos_u.append(mini_edges_dict['Src'])
            train_pos_v.append(mini_edges_dict['Dst'])
            train_neg_u.append(mini_edges_dict['NegSrc'])
            train_neg_v.append(mini_edges_dict['NegDst'])
    train_pos_u = np.hstack(train_pos_u)
    train_pos_v = np.hstack(train_pos_v)
    train_neg_u = np.hstack(train_neg_u)
    train_neg_v = np.hstack(train_neg_v)
    test_pos_u = np.hstack(test_pos_u) if len(test_pos_u) > 0 else np.array([])
    test_pos_v = np.hstack(test_pos_v) if len(test_pos_v) > 0 else np.array([])
    test_neg_u = np.hstack(test_neg_u) if len(test_neg_u) > 0 else np.array([])
    test_neg_v = np.hstack(test_neg_v) if len(test_neg_v) > 0 else np.array([])

    train_pos_set = set(zip(train_pos_u, train_pos_v))
    cnt_pos = 0
    for u, v in zip(test_pos_u, test_pos_v):
        if (v, u) in train_pos_set:
            train_pos_set.remove((v, u))
            cnt_pos += 1

    train_neg_set = set(zip(train_neg_u, train_neg_v))
    cnt_neg = 0
    for u, v in zip(test_neg_u, test_neg_v):
        if (v, u) in train_neg_set:
            train_neg_set.remove((v, u))
            cnt_neg += 1

    print(f"move {cnt_pos} pos and {cnt_neg} neg to test.")
    train_pos_u = np.array([u for u, v in train_pos_set])
    train_pos_v = np.array([v for u, v in train_pos_set])
    train_neg_u = np.array([u for u, v in train_neg_set])
    train_neg_v = np.array([v for u, v in train_neg_set])

    eids = np.arange(len(train_pos_u))
    eids = np.random.permutation(eids)
    valid_size = int((len(train_pos_u)) * args.valid_ratio)

    # split all positive edges for training and testing
    # test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    valid_pos_u, valid_pos_v = train_pos_u[eids[:valid_size]], train_pos_v[eids[:valid_size]]
    train_pos_u, train_pos_v = train_pos_u[eids[valid_size:]], train_pos_v[eids[valid_size:]]

    # split all negative edges for training and testing
    eids = np.arange(len(train_neg_u))
    eids = np.random.permutation(eids)
    valid_size = int((len(train_neg_u)) * args.valid_ratio)
    valid_neg_u, valid_neg_v = train_neg_u[eids[:valid_size]], train_neg_v[eids[:valid_size]]
    train_neg_u, train_neg_v = train_neg_u[eids[valid_size:]], train_neg_v[eids[valid_size:]]

    train_pos_set = set(zip(train_pos_u, train_pos_v))
    cnt_pos = 0
    for u, v in zip(test_pos_u, test_pos_v):
        if (v, u) in train_pos_set:
            train_pos_set.remove((v, u))
            cnt_pos += 1

    train_neg_set = set(zip(train_neg_u, train_neg_v))
    cnt_neg = 0
    for u, v in zip(test_neg_u, test_neg_v):
        if (v, u) in train_neg_set:
            train_neg_set.remove((v, u))
            cnt_neg += 1

    print(f"move {cnt_pos} pos and {cnt_neg} neg to val.")
    train_pos_u = np.array([u for u, v in train_pos_set])
    train_pos_v = np.array([v for u, v in train_pos_set])
    train_neg_u = np.array([u for u, v in train_neg_set])
    train_neg_v = np.array([v for u, v in train_neg_set])

    raw_dir = f'data/{args.CNNFeatureExtractor}_{args.FeatureDim}.pkl'
    with open(raw_dir, 'rb') as f1:
        gene_feature_dict = pkl.load(f1)

    # create positive and negative sub-graph for training/validation/test set
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=len(gene_feature_dict))
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=len(gene_feature_dict))
    valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=len(gene_feature_dict))
    valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=len(gene_feature_dict))
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=len(gene_feature_dict))
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=len(gene_feature_dict))

    train_g, train_pos_g, train_neg_g = list(
        map(dgl.to_bidirected, [train_pos_g, train_pos_g, train_neg_g])
    )

    return train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g
