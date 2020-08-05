# -*- coding: utf-8 -*-
"""
Convolutional Neural Networks for Sentence Classification
Yoon Kim
EMNLP 2014 https://arxiv.org/abs/1408.5882
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, opt, embedding_matrix):
        super(TextCNN, self).__init__()
        self.opt = opt
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        else:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.opt.kernel_sizes:
            self.convs.append(nn.Conv1d(opt.embed_dim, opt.num_kernels, kernel_size, padding=kernel_size - 1))
        self.dropout = nn.Dropout(p=opt.dropout)
        self.dense = nn.Linear(len(opt.kernel_sizes) * opt.num_kernels, opt.num_labels)

    def forward(self, inputs):
        x = inputs[0]  # [batch_size, seq_len]
        embedded_x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded_x = self.dropout(embedded_x)
        embedded_x = embedded_x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        convs_x = []
        for conv in self.convs:
            conv_x = F.relu(conv(embedded_x))  # [batch_size, num_kernels, seq_len_out]
            convs_x.append(F.max_pool1d(conv_x, conv_x.shape[2]).squeeze(2))  # [batch_size, num_kernels]
        out = torch.cat(convs_x, 1)  # [batch_size, num_kernels * len(kernel_sizes)]
        # out = self.dropout(out)
        out = self.dense(out)  # [batch_size, num_labels]
        return out
