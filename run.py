# -*- coding: utf-8 -*-
"""
Runner.
"""

import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, TCDataset
from models import AttnBiLSTM, FastText, TextCNN, TextRNN, TextRCNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Inferrer(object):

    def __init__(self, opt):
        super(Inferrer, self).__init__()
        self.opt = opt
        self.tokenizer = build_tokenizer(fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                                         max_seq_len=opt.max_seq_len,
                                         ngram=opt.ngram,
                                         min_count=opt.min_count,
                                         dat_fname='./bin/{0}_tokenizer.dat'.format(opt.dataset))
        opt.vocab_size = len(self.tokenizer.token2idx) + 2
        opt.ngram_vocab_sizes = [len(self.tokenizer.ngram2idx[n]) + 2 for n in range(2, opt.ngram + 1)]
        embedding_matrix = build_embedding_matrix(
            token2idx=self.tokenizer.token2idx,
            embed_dim=opt.embed_dim,
            embed_file=opt.embed_file,
            dat_fname='./bin/{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(opt, embedding_matrix)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # Switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def infer(self, raw_text):
        text_indices, ngram_indices_li = self.tokenizer.text_to_sequence(raw_text)
        t_inputs = {
            'text_indices': text_indices,
        }
        if ngram_indices_li:
            for n in range(2, self.opt.ngram + 1):
                t_inputs[str(n) + 'gram_indices'] = ngram_indices_li[n - 2]
        t_inputs = [torch.tensor([t_inputs[col]], dtype=torch.int64).to(self.opt.device) for col in self.opt.input_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


class Instructor(object):

    def __init__(self, opt):
        super(Instructor, self).__init__()
        self.opt = opt

        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            ngram=opt.ngram,
            min_count=opt.min_count,
            dat_fname='./bin/{0}_tokenizer.dat'.format(opt.dataset))
        opt.vocab_size = len(tokenizer.token2idx) + 2
        opt.ngram_vocab_sizes = [len(tokenizer.ngram2idx[n]) + 2 for n in range(2, opt.ngram + 1)]
        if opt.embed_file is not None and os.path.exists(opt.embed_file):
            embedding_matrix = build_embedding_matrix(
                token2idx=tokenizer.token2idx,
                embed_dim=opt.embed_dim,
                embed_file=opt.embed_file,
                dat_fname='./bin/{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        else:
            embedding_matrix = None
        self.model = opt.model_class(opt, embedding_matrix).to(opt.device)

        self.train_set = TCDataset(opt.dataset_file['train'], opt.label_mapping, opt.ngram, tokenizer)
        self.test_set = TCDataset(opt.dataset_file['test'], opt.label_mapping, opt.ngram, tokenizer)
        assert 0 <= opt.val_set_ratio < 1
        if opt.val_set_ratio > 0:
            val_set_len = int(len(self.train_set) * opt.val_set_ratio)
            self.train_set, self.val_set = random_split(self.train_set,
                                                        (len(self.train_set) - val_set_len, val_set_len))
        else:
            self.val_set = self.test_set

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_non_trainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_non_trainable_params += n_params
        logger.info('n_trainable_prams: {0}, n_non_trainable_params: {1}'
                    .format(n_trainable_params, n_non_trainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc, max_val_f1 = 0, 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            # Switch model to training mode.
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # Clear gradient accumulators.
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                outputs = self.model(inputs)
                targets = sample_batched['label'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # Switch model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                t_targets = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                      labels=list(self.opt.label_mapping.values()), average='macro')
        return acc, f1

    def run(self):
        # Loss and optimizer.
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.val_set, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    # Setting about initialization and dataset.
    parser.add_argument('--mode', default='train', type=str,
                        help='mode could be `train` or `infer`.')
    parser.add_argument('--pretrained_model_name', default='bert-base-uncased', type=str,
                        help='Name of pretrained language model to initialize bert-type model.')
    parser.add_argument('--model_name', default='attnbilstm', type=str,
                        help='Name of model to build like FastText, TextCNN, TextRCNN and so on.')
    parser.add_argument('--dataset', default='sst-2', type=str,
                        help='Name of dataset that you want to load.')
    parser.add_argument('--val_set_ratio', default=0, type=float,
                        help='Splitting specified ratio of train set as validation set.')
    # Setting about inferrer.
    parser.add_argument('--state_dict_path', default='./state_dict/textcnn_sst-2_val_acc0.8612', type=str,
                        help='Set your trained model path.')
    parser.add_argument('--raw_text', default='hide new secretions from the parental units', type=str,
                        help='Input your raw data.')
    # Setting about optimization.
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    # Setting about vocabulary and embedding.
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--embed_file', default='./embed/glove.42B.300d.txt', type=str)  # None to initialize randomly
    parser.add_argument('--ngram', default=3, type=int)
    parser.add_argument('--min_count', default=2, type=int)  # only use min_count to control ngram vocab
    # Setting about trainable parameter.
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--kernel_sizes', default='3,4,5', type=str)  # `3,4,5` are used in TextCNN paper.
    parser.add_argument('--num_kernels', default=100, type=int)  # 100 is used in TextCNN paper.
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    # Other setting.
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=12333, type=int)

    opt = parser.parse_args()

    set_seed(opt)
    model_classes = {
        'fasttext': FastText,
        'textcnn': TextCNN,
        'textrnn': TextRNN,
        'textrcnn': TextRCNN,
        'attnbilstm': AttnBiLSTM,
    }
    dataset_files = {
        'sst-2': {
            'train': './dataset/sst-2/train.tsv',
            'test': './dataset/sst-2/dev.tsv',
        },
    }
    label_mappings = {
        'sst-2': {
            '0': 0,
            '1': 1
        },
    }
    initializers = {
        'xavier_uniform_': nn.init.xavier_uniform_,
        'xavier_normal_': nn.init.xavier_normal,
        'orthogonal_': nn.init.orthogonal_
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'adamw': torch.optim.AdamW,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    input_colses = {
        'fasttext': ['text_indices'] + [str(n) + 'gram_indices' for n in range(2, opt.ngram + 1)],
        'textcnn': ['text_indices'],
        'textrnn': ['text_indices'],
        'textrcnn': ['text_indices'],
        'attnbilstm': ['text_indices'],
    }
    opt.input_cols = input_colses[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.label_mapping = label_mappings[opt.dataset]
    opt.reverse_label_mapping = {val: key for key, val in opt.label_mapping.items()}
    opt.num_labels = len(opt.label_mapping)
    opt.kernel_sizes = list(map(int, opt.kernel_sizes.split(',')))

    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('./bin'):  # Create dir for save pickle file.
        os.makedirs('./bin')

    if opt.mode == 'train':
        log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, time.strftime('%y%m%d-%H%M', time.localtime()))
        logger.addHandler(logging.FileHandler(log_file))
        ins = Instructor(opt)
        ins.run()
    elif opt.mode == 'infer':
        inf = Inferrer(opt)
        t_probs = inf.infer(opt.raw_text)
        print(opt.reverse_label_mapping[int(t_probs.argmax(axis=-1))])
    else:
        raise ValueError('Unexpected mode to run model!')


if __name__ == '__main__':
    main()
