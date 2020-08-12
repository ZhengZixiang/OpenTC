# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn

from layers import DynamicRNN, NoQueryAttention, SqueezeEmbedding


class AttnBiLSTM(nn.Module):

    def __init__(self, opt, embedding_matrix):
        super(AttnBiLSTM, self).__init__()
        self.opt = opt
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        else:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicRNN(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = NoQueryAttention(opt.hidden_dim * 2, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_labels)
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, inputs):
        x = inputs[0]  # [batch_size, seq_len]
        x_len = torch.sum(x != 0, dim=-1)  # [batch_size]
        embedded_x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded_x = self.squeeze_embedding(embedded_x, x_len)  # [batch_size, seq_len, embed_dim]
        embedded_x = self.dropout(embedded_x)
        h, (_, _) = self.lstm(embedded_x, x_len)  # [batch_size, seq_len, hidden_dim * 2]
        _, score = self.attention(h)  # [batch_size, q_len=1, seq_len]
        out = torch.squeeze(torch.bmm(score, h), dim=1)  # [batch_size, hidden_dim * 2]
        out = self.dense(out)  # [batch_size, num_labels]
        return out
