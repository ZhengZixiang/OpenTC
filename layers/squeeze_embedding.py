# -*- coding： utf-8 -*-

import torch
import torch.nn as nn


class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch.
    """
    def __init__(self, batch_first=True):
        self.batch_first = batch_first
        super(SqueezeEmbedding, self).__init__()

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack -> unsort
        :param x: Sequence embedding vectors.
        :param x_len: numpy/tensor list.
        :return:
        """
        # sort
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]

        # pack
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # unpack: out
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        out = out[0]

        # unsort
        out = out[x_unsort_idx]
        return out
