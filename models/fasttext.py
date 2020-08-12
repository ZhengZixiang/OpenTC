# -*- coding: utf-8 -*-
"""
Bag of Tricks for Efficient Text Classification
Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
(EACL 2016) https://arxiv.org/abs/1607.01759
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):

    def __init__(self, opt, embedding_matrix):
        super(FastText, self).__init__()
        self.opt = opt
        if embedding_matrix is not None:
            self.token_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        else:
            self.token_embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.ngram_embeds = torch.nn.ModuleList()
        self.ngram_embeds.append(self.token_embedding)  # 1gram
        for i in range(2, opt.ngram + 1):
            self.ngram_embeds.append(nn.Embedding(opt.ngram_vocab_sizes[i - 2], opt.embed_dim))
        self.dense1 = nn.Linear(opt.embed_dim * 3, opt.hidden_dim)
        self.dense2 = nn.Linear(opt.hidden_dim, opt.num_labels)
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, inputs):
        embedded_x = []
        for i in range(self.opt.ngram):
            embedded_x.append(self.dropout(self.ngram_embeds[i](inputs[i])))
        out = torch.cat(embedded_x, -1)  # [batch_size, seq_len, embed_dim * 3]
        out = out.mean(dim=1)  # [batch_size, embed_dim * 3]
        out = self.dense1(out)  # [batch_size, hidden_dim]
        out = F.relu(out)
        out = self.dense2(out)  # [batch_size, num_labels]
        return out
