from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import pickle

def get_gcn_data(train_df, test_df, save_dir):
    with open('{}/gcn_pretrain.pkl'.format(save_dir), 'rb') as f:
        embedding = pickle.load(f)
        node2id = pickle.load(f)
    gcn_train = np.array([[node2id[i]] for i in train_df['query_id'].values])
    gcn_test = np.array([[node2id[i]] for i in test_df['query_id'].values])
    return gcn_train, gcn_test, embedding


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="dataset/cora/", edge_dataset="cora", feature_dataset='feature', feature_less = True):
    if feature_less ==True:
        feature_dataset=None

    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(edge_dataset))
    print('Loading {} dataset...'.format(feature_dataset))


    meta_name = '{}{}.npy'.format(path, feature_dataset)
    edge_name = '{}{}/{}.pkl'.format(path, edge_dataset.split('.')[0], edge_dataset)
    # with open(meta_name,'rb') as f:
    #     idx_features = pickle.load(f)

    with open(edge_name, 'rb') as f:
        edges_unordered = pickle.load(f)
    print(edges_unordered)
    # idx_features = np.load(meta_name)



    # idx_features_labels = np.genfromtxt("{}{}.content.txt".format(path, dataset), dtype=np.dtype(str))
    try:
        idx_features = np.load(meta_name)
        features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)
        print(idx_features)
        idx = np.array(idx_features[:, 0] )

    except:
        print("Feature less")
        features = None
        idx = np.unique(np.vstack([edges_unordered[:, 0], edges_unordered[:, 1]]))
        # idx = edge_id.values

    print('=====')
    print(edges_unordered)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # idx = np.array(idx_features[:, 0] )
    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites.txt".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    try:
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
        return features.todense(), adj, idx_map
    except:
        print('Featureless Dataset has {} nodes, {} edges'.format(adj.shape[0], edges.shape[0]))
        return None, adj, idx_map


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
