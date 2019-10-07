import os
import argparse

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices='AAN PeerRead'.split(), help='Name of dataset')
    parser.add_argument('--gpu_id', default="0", type=str)
    parser = gcn_config(parser)

    return parser.parse_args()


def gcn_config(parser):

    parser.add_argument('--gcn_epochs', default=200, type=int, help='Epochs')
    parser.add_argument('--gcn_lr', default=0.01, type=int, help='Learning Rate')
    parser.add_argument('--gcn_model', default='VAE', choices='AE VAE'.split(), help='Name of gcn model')
    parser.add_argument('--gcn_hidden1', default=7071, type=int, help='Number of units in hidden layer 1.')
    parser.add_argument('--gcn_hidden2', default=768, type=int, help='Number of units in hidden layer 2.')
    parser.add_argument('--save_dir', default='./pre_train/gcn', type=str, help='GCN Pretrain save dir')

    return parser
