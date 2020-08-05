# -*- coding: utf-8 -*-
"""
Recurrent Neural Network for Text Classification with Multi-Task Learning
Pengfei Liu, Xipeng Qiu, Xuanjing Huang
(IJCAI 2016) https://arxiv.org/abs/1605.05101
"""

import torch
import torch.nn as nn

from layers import DynamicRNN


class TextRNN(nn.Module):

    def __init__(self, opt, embedding_matrix):
        super(TextRNN, self).__init__()
        self.opt = opt
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        else:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.lstm = DynamicRNN(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_labels)
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, inputs):
        x = inputs[0]  # [batch_size, seq_len]
        embedded_x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded_x = self.dropout(embedded_x)
        x_len = torch.sum(x != 0, dim=-1)  # [batch_size]
        _, (h, _) = self.lstm(embedded_x, x_len)  # [num_directions * num_layers, batch_size, hidden_dim]
        out = torch.cat((h[0], h[1]), dim=-1)  # [batch_size, hidden_dim * 2]
        out = self.dense(out)  # [batch_size, num_labels]
        return out
