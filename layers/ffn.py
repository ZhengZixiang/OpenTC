# -*- coding: utf-8 -*-
"""
Position-wise feed forward network.
"""

import torch.nn as nn


class PostionwiseFeedForward(nn.Module):
    """
    A two-feed forward-layer module.
    """
    def __init__(self, hidden_dim, inner_hidden_dim=None, dropout=0):
        super(PostionwiseFeedForward, self).__init__()
        if inner_hidden_dim is None:
            inner_hidden_dim = hidden_dim
        self.w_1 = nn.Conv1d(hidden_dim, inner_hidden_dim, 1)  # position-wise
        self.w_2 = nn.Conv1d(inner_hidden_dim, hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(inner_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.w_1(x.transpose(1, 2)))
        out = self.w_2(out).transpose(2, 1)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out
