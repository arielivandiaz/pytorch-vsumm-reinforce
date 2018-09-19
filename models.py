import torch
import torch.nn as nn
from torch.nn import functional as F
from dnc import *

__all__ = ['DSN']

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=2048, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            #self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
            """
            rnn = DNC(
              input_size=64,
              hidden_size=128,
              rnn_type='lstm',
              num_layers=4,
              nr_cells=100,
              cell_size=32,
              read_heads=4,
              batch_first=True,
              gpu_id=0
            )
            """
            self.rnn = DNC(input_size=in_dim,
              hidden_size=hid_dim,
              rnn_type='lstm',
              num_layers=4,
              nr_cells=100,
              cell_size=32,
              read_heads=4,
              batch_first=True,
              gpu_id=-1
            )
            #"""
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p
