
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F
import copy
from torch.nn.utils import weight_norm


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


class Short_Term(nn.Module):

    def __init__(self, num_words, item_emb_size, hidden_size, num_channels,
                 kernel_size=3, dropout=0.3, emb_dropout=0.1, args=None, device='cpu'):
        super(Short_Term, self).__init__()
        self.num_words = num_words
        self.item_emb_size = item_emb_size
        self.hidden_size = hidden_size
        self.device = device

        self.item_embeddings = nn.Embedding(num_words, item_emb_size, padding_idx=0)

        self.tcn = TemporalConvNet(item_emb_size, num_channels, kernel_size, dropout=dropout)
        self.w1_item = nn.Linear(item_emb_size, item_emb_size, bias=True)
        self.w2_item = nn.Linear(item_emb_size, item_emb_size, bias=True)
        self.v_item = nn.Linear(item_emb_size, 1, bias=False)

        self.W_trans = nn.Linear(2 * item_emb_size, item_emb_size, bias=False)

        self.init_weight()
        self.drop = nn.Dropout(emb_dropout)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.l2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_dc, gamma=0.1)

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.item_emb_size)
        for weight in self.parameters():
            if weight.shape[0] == self.num_words and weight.shape[1] == self.item_emb_size:
                weight.data[1:].normal_(0, stdv)
                continue
            weight.data.normal_(0, stdv)

    def forward(self, inp_seq):
        seq_emb = self.drop(self.item_embeddings(inp_seq))  # bs * T * es
        U = self.tcn(seq_emb.transpose(2, 1))  # U: bs * hs * T
        U = U.permute(0, 2, 1)  # bs * T * hs
        mask = torch.where(inp_seq > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        ut = U[:, -1, :]

        q1_item = self.w1_item(ut).view(ut.shape[0], 1, ut.shape[1])  # bs * 1 * (hs + es)
        q2_item = self.w2_item(U)  # bs * T * (hs + es)
        alpha = self.v_item(torch.sigmoid(q1_item + q2_item))  # bs * T * 1
        s = torch.sum(alpha * U * mask.view(mask.shape[0], -1, 1).float(), 1)

        s = torch.cat((s, ut), dim=1)
        s = self.W_trans(s)

        item_embedding = self.item_embeddings.weight[1:]
        score = torch.matmul(s, item_embedding.transpose(1, 0))

        return score

