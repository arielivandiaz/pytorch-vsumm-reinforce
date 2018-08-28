from __future__ import absolute_import
import os
import argparse
import sys
import errno
import shutil
import json
import os.path as osp

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_args():

    parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
    # Dataset options
    parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
    parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
    parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
    parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                        help="evaluation metric ['tvsum', 'summe']")
    # Model options
    parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
    parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
    parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
    parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
    # Optimization options
    parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
    parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
    parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
    parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
    parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
    parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
    parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
    # Misc
    parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
    parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
    parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
    parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
    parser.add_argument('--resume', type=str, default='', help="path to resume file")
    parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
    parser.add_argument('--save-results', action='store_true', help="whether to save output results")

    return parser.parse_args()
