# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
import copy
from torch.nn.utils import weight_norm


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def CrossEntropy(predict, target, device='cpu'):
    predict = torch.softmax(predict, dim=1)
    one_hot = torch.zeros_like(predict)
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    log_P = torch.log(predict)
    loss = -one_hot * log_P
    loss = loss.sum(dim=1)
    return loss.mean()


def JS_div(predict, target):
    kl0 = KL_div(predict, (predict + target) / 2)
    kl1 = KL_div(target, (predict + target) / 2)
    js = (kl0 + kl1) / 2
    return js.mean()


def KL_div(predict, target):
    kl = torch.sum(target * torch.log(target / predict), dim=1)
    return kl


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Chomp1d(nn.Module):
    ''' 这里chomp_size = padding = kernel_size - 1
        去除卷积后的padding个元素，达到因果卷积的效果
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        ''' Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
            input:  (N, C_in,  L_in),  N: batch_size, C_in: input channel,   L_in: input sequence length
            output: (N, C_out, L_out), N: batch_size, C_out: output channel, L_out: output sequence length
                L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        '''
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: bs * emb_size * T
        out = self.net(x)   # out: bs * emb_size * T
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MutiHeadAttn(nn.Module):
    def __init__(self, num_words, anchor_num, N, topic_mat, dropout, hid_size, lr, l2, lr_dc_cate, device='cpu', seed = 200):
        super(MutiHeadAttn, self).__init__()
        self.num_words = num_words
        self.anchor_num = anchor_num
        self.topic_mat = torch.from_numpy(topic_mat).float().to(device)
        self.N = N
        self.dropout = dropout
        self.hid_size = hid_size
        self.device = device
        self.lr = lr
        self.l2 = l2
        self.lr_dc_cate = lr_dc_cate
        
        self.seed = seed
        init_seed(self.seed)

        self.dim_reduction = nn.Sequential(
            nn.Linear(self.anchor_num, self.hid_size),
            nn.ReLU(),
        )

        self.w1s = clones(nn.Linear(self.hid_size, self.hid_size, bias=True), N)
        self.w2s = clones(nn.Linear(self.hid_size, self.hid_size, bias=True), N)
        self.vs = clones(nn.Linear(self.hid_size, 1, bias=False), N)
        self.linear0 = nn.Sequential(
            nn.Linear(2 * N * self.hid_size, self.hid_size),
            nn.ReLU()
        )

        self.linears = nn.Sequential(
            nn.Linear(self.hid_size, self.hid_size, bias=True),
            nn.BatchNorm1d(self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size, bias=True),
            nn.BatchNorm1d(self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size, bias=True),
            nn.BatchNorm1d(self.hid_size),
            nn.ReLU(),
        )

        self.out_net = nn.Linear(self.hid_size, self.num_words - 1, bias=False)

        self.init_weight()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.l2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_dc_cate, gamma=0.1)

    def init_weight(self):
        init_seed(self.seed)
        stdv = 1 / math.sqrt(self.hid_size)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, inp_seq):
        init_seed(self.seed)
        inp_fea = self.topic_mat[inp_seq]   # bs * T * anchor_num
        inp_fea = self.dim_reduction(inp_fea)   # bs * T * hs

        mask = torch.where(inp_seq > 0, torch.tensor([1], device=self.device), torch.tensor([0], device=self.device))
        c = []
        for i in range(self.N):
            # ht = inp_fea[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
            ht = inp_fea[:, -1, :]
            q1 = self.w1s[i](ht).view(ht.shape[0], 1, ht.shape[1])
            q2 = self.w2s[i](inp_fea)
            alpha = self.vs[i](torch.sigmoid(q1 + q2))
            a = torch.sum(alpha * inp_fea * mask.view(mask.shape[0], -1, 1).float(), 1)
            c.append(torch.cat([a, ht], dim=1))
        c_final = torch.cat(c, dim=1)

        c = self.linear0(c_final)

        c_final = self.linears(c)
        c_final = c_final + c
        # c_final = torch.cat((c_final, inp_fea[:, -1, :]), dim=1)

        score = self.out_net(c_final)

        return score


class GRUTCN(nn.Module):
    def __init__(self, num_words, item_emb_size, hidden_size, num_channels,
                 kernel_size=3, dropout=0.3, emb_dropout=0.1, lr = 0.001, lr_dc = [10, 15], l2 = 1e-5, device='cpu', seed = 2000):
        
        super(GRUTCN, self).__init__()
        
        self.seed = seed
        init_seed(self.seed)
        
        self.num_words = num_words
        self.item_emb_size = item_emb_size
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr
        self.lr_dc = lr_dc
        self.l2 = l2
        

        self.item_embeddings = nn.Embedding(num_words, item_emb_size, padding_idx=0)
        self.gru = nn.GRU(input_size=item_emb_size, hidden_size=hidden_size,
                          num_layers=1, dropout=0, bidirectional=False)
        self.tcn = TemporalConvNet(item_emb_size, num_channels, kernel_size, dropout=dropout)
        self.w1_item = nn.Linear(item_emb_size + hidden_size, item_emb_size + hidden_size, bias=True)
        self.w2_item = nn.Linear(item_emb_size + hidden_size, item_emb_size + hidden_size, bias=True)
        self.v_item = nn.Linear(item_emb_size + hidden_size, 1, bias=False)

        self.W_trans = nn.Linear(2 * (item_emb_size + hidden_size), item_emb_size, bias=False)

        self.init_weight()
        self.drop = nn.Dropout(emb_dropout)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.l2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_dc, gamma=0.1)

    def init_weight(self):
        init_seed(self.seed)
        
        stdv = 1.0 / math.sqrt(self.item_emb_size)
        for weight in self.parameters():
            if weight.shape[0] == self.num_words and weight.shape[1] == self.item_emb_size:
                weight.data[1:].normal_(0, stdv)
                continue
            weight.data.normal_(0, stdv)

    def init_h0(self, batch_size):
        init_seed(self.seed)
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_seq_behind):
        init_seed(self.seed)
        seq_emb = self.drop(self.item_embeddings(inp_seq_behind))  # bs * T * es
        h0 = self.init_h0(seq_emb.size(0))
        H, ht = self.gru(seq_emb.permute(1, 0, 2), h0)  # H: T * bs * hs
        U = self.tcn(seq_emb.transpose(2, 1))  # U: bs * hs * T
        H = H.permute(1, 0, 2)  # bs * T * hs
        U = U.permute(0, 2, 1)  # bs * T * hs
        y = torch.cat((H, U), dim=2)  # bs * T * (hs + es)
        mask_behind = torch.where(inp_seq_behind > 0, torch.tensor([1.], device=self.device),
                                  torch.tensor([0.], device=self.device))
        yt = y[:, -1, :]  # bs * (hs + es)
        q1_item = self.w1_item(yt).view(yt.shape[0], 1, yt.shape[1])  # bs * 1 * (hs + es)
        q2_item = self.w2_item(y)  # bs * T * (hs + es)
        alpha = self.v_item(torch.sigmoid(q1_item + q2_item))  # bs * T * 1
        s = torch.sum(alpha * y * mask_behind.view(mask_behind.shape[0], -1, 1).float(), 1)
        seq_repr0 = torch.cat([s, yt], dim=1)  # bs * 2(hs + es)
        seq_repr = self.W_trans(seq_repr0)
        item_embeddings = self.item_embeddings.weight[1:]
        item_score = torch.matmul(seq_repr, item_embeddings.transpose(1, 0))

        return item_score


class Fusion(nn.Module):
    def __init__(self, num_words, device='cpu', seed = 2000):
        super(Fusion, self).__init__()
        
        self.seed = seed
        init_seed(self.seed)
        
        
        self.device = device
        self.num_words = num_words

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, item_score, cate_score):
        init_seed(self.seed)
        item_score0 = torch.softmax(item_score, dim=1)
        cate_score0 = torch.softmax(cate_score, dim=1)

        beta = ((cate_score0 * item_score0).sum(dim=1) / (
                    torch.norm(cate_score0, p=2, dim=1) * torch.norm(item_score0, p=2, dim=1))).unsqueeze(1)
        score = item_score + beta * cate_score

        # mean_score = (item_score0 + cate_score0) / 2
        # alpha = ((item_score0 * mean_score).sum(dim=1) / (
        #             torch.norm(item_score0, p=2, dim=1) * torch.norm(mean_score, p=2, dim=1))).unsqueeze(1)
        # beta = ((cate_score0 * mean_score).sum(dim=1) / (
        #             torch.norm(cate_score0, p=2, dim=1) * torch.norm(mean_score, p=2, dim=1))).unsqueeze(1)
        #
        # score = alpha * item_score + beta * cate_score

        return score

