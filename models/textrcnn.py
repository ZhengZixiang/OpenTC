# -*- coding: utf-8 -*-
"""
Recurrent Convolutional Neural Networks for Text Classification
Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao
(AAAI 2015) http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf
"""

import torch
import torch.nn as nn


class TextRCNN(nn.Module):

    def __init__(self, opt, embedding_matrix):
        super(TextRCNN, self).__init__()
        self.opt = opt
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        else:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.maxpool = nn.MaxPool1d(opt.max_seq_len)
        self.dense = nn.Linear(opt.hidden_dim * 2 + opt.embed_dim, opt.num_labels)
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, inputs):
        x = inputs[0]  # [batch_size, seq_len]
        embedded_x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded_x = self.dropout(embedded_x)
        out, _ = self.lstm(embedded_x)  # [batch_size, seq_len, hidden_dim * 2]
        out = torch.cat((out, embedded_x), 2)  # [batch_size, seq_len, hidden_dim * 2 + embed_dim]
        out = out.permute(0, 2, 1)  # [batch_size, hidden_dim * 2 + embed_dim, seq_len]
        out = self.maxpool(out).squeeze(2)  # [batch_size, hidden_dim * 2 + embed_dim]
        out = self.dropout(out)
        out = self.dense(out)  # [batch_size, num_labels]
        return out
