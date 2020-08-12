# -*- coding: utf-8 -*-
"""
Implementation of various attention mechanism such as (scaled) dot product, bi-linear, MLP, cosine and multi-head,
 self attention and no-query attention.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, num_heads, model_dim, k_dim=None, v_dim=None, out_dim=None, temperature=None, dropout=0,
                 score_function='scaled_dot_product'):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # Control whether to use multi-head attention mechanism.
        self.model_dim = model_dim
        if k_dim is None:
            self.k_dim = model_dim // num_heads
        else:
            self.k_dim = k_dim

        if v_dim is None:
            self.v_dim = self.k_dim
        else:
            self.v_dim = v_dim

        if out_dim is None:
            self.out_dim = model_dim
        else:
            self.out_dim = out_dim

        self.w_k = nn.Linear(model_dim, num_heads * self.k_dim)
        self.w_q = nn.Linear(model_dim, num_heads * self.k_dim)
        self.w_v = nn.Linear(model_dim, num_heads * self.k_dim)
        self.dense = nn.Linear(num_heads * self.k_dim, self.out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.out_dim)
        self.score_function = score_function
        if temperature is None:
            self.temperature = math.sqrt(model_dim)
        else:
            self.temperature = temperature

        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(self.k_dim * 2))
        elif score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(self.k_dim, self.k_dim))
        else:
            self.register_parameter('weight', None)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.model_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v=None, mask=None):
        if len(q.shape) == 2:  # q_len missing.
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing.
            k = torch.unsqueeze(k, dim=1)
        if v is None:
            v = k
        batch_size = q.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        v_len = v.shape[1]

        # [batch_size, seq_len, num_heads, hidden_dim]
        kx = self.w_k(k).view(batch_size, k_len, self.num_heads, self.k_dim)
        qx = self.w_q(q).view(batch_size, q_len, self.num_heads, self.k_dim)
        vx = self.w_v(v).view(batch_size, v_len, self.num_heads, self.v_dim)
        # [num_heads * batch_size, seq_len, hidden_dim]
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.k_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.k_dim)
        vx = vx.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.v_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)  # [num_heads * batch_size, q_len, k_len]
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
            score = torch.div(score, self.temperature)  # [num_heads * batch_size, q_len, k_len]
        elif self.score_function == 'mlp':
            # [num_heads * batch_size, q_len, k_len, hidden_dim]
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # [num_heads * batch_size, q_len, k_len, hidden_dim * 2]
            score = torch.tanh(torch.matmul(kq, self.weight))  # [num_heads * batch_size, q_len, k_len]
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(torch.tanh(qx), self.weight)  # [num_heads * batch_size, q_len, hidden_dim]
            kt = kx.permute(0, 2, 1)  # [num_heads * batch_size, hidden_dim, k_len]
            score = torch.bmm(qw, kt)  # [num_heads * batch_size, q_len, k_len]
        else:
            raise RuntimeError('Invalid score function.')

        if mask is not None:
            score = score.masked_fill(mask, -np.inf)

        attn = F.softmax(score, dim=-1)
        out = torch.bmm(attn, vx)
        out = self.dropout(out)
        # or like:
        # out = out.view(self.num_heads, batch_size, q_len, self.d_v)
        # out = out.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # [batch_size, q_len, num_heads * hidden_dim]
        out = self.dropout(F.relu(self.dense(out)))  # [batch_size, q_len, out_dim]
        out = self.layer_norm(out)
        return out, attn  # out: [batch_size, q_len, out_dim] attn: [num_heads * batch_size, q_len, k_len]


class NoQueryAttention(Attention):
    def __init__(self, model_dim, num_heads=1, k_dim=None, v_dim=None, q_len=1, out_dim=None, temperature=None, dropout=0,
                 score_function='scaled_dot_product'):
        super(NoQueryAttention, self).__init__(
            num_heads, model_dim, k_dim, v_dim, out_dim, temperature, dropout, score_function)
        self.model_dim = model_dim
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, model_dim))
        self._reset_q()

    def _reset_q(self):
        stdv = 1. / math.sqrt(self.model_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        batch_size = k.shape[0]
        q = self.q.expand(batch_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
