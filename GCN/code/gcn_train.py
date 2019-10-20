from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import random
import pickle
import scipy.sparse as sp
import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

random.seed(42)

from GCN.code.utils import load_data
from GCN.code.optimizer import OptimizerAE, OptimizerVAE
from GCN.code.model import GCNModelAE, GCNModelVAE
from GCN.code.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges


def paper_pretrain(data_name, gcn_model, gcn_epochs, gcn_lr, gcn_hidden1, gcn_hidden2, save_dir):

    NOW_DIR = os.getcwd()
    DATA_DIR = os.path.join(NOW_DIR, 'glue', 'ACRS', 'full_context_{}.csv'.format(data_name))

    aan_data = pd.read_csv(DATA_DIR)
    graph_edge = aan_data[['source_id', 'target_id']].drop_duplicates(subset=['target_id', 'source_id'])
    graph_edge.reset_index(inplace=True)
    graph_edge.drop(labels=['index'], axis=1, inplace=True)
    target_info = aan_data[['target_id', 'target_author', 'target_title', 'target_venue', 'target_abstract']]
    source_info = aan_data[['source_id', 'source_author', 'source_title', 'source_venue', 'source_abstract']]
    target_info.drop_duplicates(subset=['target_id'], inplace=True)
    source_info.drop_duplicates(subset=['source_id'], inplace=True)
    target_info.reset_index(inplace=True)
    target_info.drop(labels=['index'], axis=1, inplace=True)
    source_info.reset_index(inplace=True)
    source_info.drop(labels=['index'], axis=1, inplace=True)

    with open('{}/{}/{}.graph_edge_total.pkl'.format(save_dir, data_name, data_name), 'wb') as f:
        pickle.dump(graph_edge.values, f)
    features, adj, idx_map = load_data(path='{}/'.format(save_dir), edge_dataset='{}.graph_edge_total'.format(data_name),
                                       feature_dataset=None, feature_less=True)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_orig_orig = adj_orig.copy()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    features = sp.identity(adj.shape[0])  # featureless
    adj_norm = preprocess_graph(adj)
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    features = sp.coo_matrix(features)
    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    if gcn_model == 'AE':
        model = GCNModelAE(placeholders, num_features, features_nonzero, gcn_hidden1, gcn_hidden2)
    elif gcn_model == 'VAE':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, gcn_hidden1, gcn_hidden2)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if gcn_model == 'AE':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm,
                              learning_rate=gcn_lr)
        elif gcn_model == 'VAE':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               learning_rate=gcn_lr)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_roc_score(edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, preds_all, labels_all

    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    # Train model
    for epoch in range(gcn_epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0.5})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr, preds, labels = get_roc_score(val_edges, val_edges_false)
        val_roc_score.append(roc_curr)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    roc_score, ap_score, _, _ = get_roc_score(test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    origin_adj_norm = preprocess_graph(adj_orig_orig)
    origin_adj_label = adj_orig_orig + sp.eye(adj_orig_orig.shape[0])
    origin_adj_label = sparse_to_tuple(origin_adj_label)

    origin_features = sp.identity(adj_orig_orig.shape[0])  # featureless
    origin_features = sp.coo_matrix(origin_features)

    origin_features = sparse_to_tuple(origin_features)

    origin_feed = construct_feed_dict(origin_adj_norm, origin_adj_label, origin_features, placeholders)
    # feature extraction
    if gcn_model == 'AE':
        encoded = sess.run(model.embeddings, feed_dict=origin_feed)
    elif gcn_model == 'VAE':
        encoded = sess.run(model.z, feed_dict=origin_feed)

    with open('{}/{}/{}_gcn_pretrain.pkl'.format(save_dir, data_name, data_name), 'wb') as f:
        pickle.dump(encoded, f)
        pickle.dump(idx_map, f)
