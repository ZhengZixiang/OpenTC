# -*- coding: utf-8 -*-
"""
Data utils to process data.
"""

import os
import pickle
from collections import Counter

import numpy as np
from torch.utils.data import Dataset


def build_tokenizer(fnames, max_seq_len, ngram, min_count, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        corpus = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for line in lines:
                label, text = line.strip().split('\t')
                corpus += text
        tokenizer = Tokenizer(max_seq_len, ngram, min_count)
        tokenizer.fit_on_text(corpus)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_token_vec(path, token2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    token_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if token2idx is None or tokens[0] in token2idx.keys():
            token_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return token_vec


def build_embedding_matrix(token2idx, embed_dim, embed_file, dat_fname):
    if os.path.exists(dat_fname):
        print('Loading embedding matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('Loading token vectors ...')
        embedding_matrix = np.zeros((len(token2idx) + 2, embed_dim))  # idx 0 and len(token2idx) + 1 are all-zeros
        token_vec = _load_token_vec(embed_file, token2idx=token2idx)
        print('Building embedding matrix:', dat_fname)
        for token, i in token2idx.items():
            vec = token_vec.get(token)
            if vec is not None:
                # Tokens not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, max_seq_len, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(max_seq_len) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-max_seq_len:]
    else:
        trunc = sequence[:max_seq_len]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, ngram, min_count, lower=True):
        super(Tokenizer, self).__init__()
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.token2idx = {}
        self.idx2token = {}
        if ngram > 1:
            self.ngram = ngram
            self.ngram2idx = {}
            self.idx2ngram = {}
            self.min_count = min_count
            for n in range(2, ngram + 1):
                self.ngram2idx[n] = {}
                self.idx2ngram[n] = {}

    def fit_on_text(self, text):
        counters = {}
        for n in range(2, self.ngram + 1):
            counters[n] = Counter()
        if self.lower:
            text = text.lower()
        tokens = text.split()
        for i, token in enumerate(tokens):
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = token
            if self.ngram > 1:
                for n in range(2, self.ngram + 1):
                    grams = ' '.join(tokens[i:i + n])
                    if grams not in counters[n]:
                        counters[n][grams] = 1
                    else:
                        counters[n][grams] += 1
        if self.ngram > 1:
            for n in range(2, self.ngram + 1):
                for key, value in sorted(counters[n].items(), key=lambda x: x[1], reverse=True):
                    if value >= self.min_count:
                        self.ngram2idx[n][key] = len(self.ngram2idx[n])
                        self.idx2ngram[n][len(self.idx2ngram[n])] = key
                    else:
                        break

    def text_to_sequence(self, text, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        tokens = text.split()
        unknown_idx = len(self.token2idx) + 1
        sequence = [self.token2idx[tok] if tok in self.token2idx else unknown_idx for tok in tokens]
        ngram_seqs = []
        if self.ngram > 1:
            for n in range(2, self.ngram + 1):
                unknown_idx = len(self.ngram2idx[n]) + 1
                ngram_seqs.append([self.ngram2idx[n][' '.join(tokens[i:i + n])]
                                   if ' '.join(tokens[i:i + n]) in self.ngram2idx[n] else unknown_idx
                                   for i, tok in enumerate(tokens)])
        if len(sequence) == 0:
            sequence = [0]
            if self.ngram > 1:
                for n in range(2, self.ngram + 1):
                    ngram_seqs.append([0])

        x = pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        if self.ngram > 1:
            for i, seq in enumerate(ngram_seqs):
                ngram_seqs[i] = pad_and_truncate(seq, self.max_seq_len, padding=padding, truncating=truncating)
        return x, ngram_seqs


class TCDataset(Dataset):
    def __init__(self, fname, label_mapping, ngram, tokenizer):
        super(TCDataset, self).__init__()
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for line in lines:
            label, text = line.strip().split('\t')
            text_indices, ngram_indices_li = tokenizer.text_to_sequence(text)
            data = {
                'text_indices': text_indices,
                'label': label_mapping[label]
            }
            if ngram_indices_li:
                for n in range(2, ngram + 1):
                    data[str(n) + 'gram_indices'] = ngram_indices_li[n - 2]
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
