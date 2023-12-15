# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import math
import copy

class SGINM(nn.Module):

    def __init__(self, lr, embedding_size, hidden_size, dropout, num_word, l2):
        super(SGINM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = 1 # by defualt its value 1
        self.embedding_size = embedding_size
        self.lr = lr
        self.l2 = l2 # l2 penality
        self.dropout = dropout
        self.lr_dc = [8, 10]
        
        
        # Defined embedding layer
        self.embedding = nn.Embedding(num_word, self.embedding_size)
        
        # Defined GRU layer
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                          dropout=self.dropout, bidirectional=False)
        
        # Defined attention layer
        self.attn0 = Attn(hidden_size=self.hidden_size)
        
        self.Linear_0 = nn.Linear(self.embedding_size, 2 * self.hidden_size + 128, bias=False)
        # self.Linear_0 = nn.Linear(self.embedding_size, 2 * self.hidden_size)
        # self.Linear_0 = nn.Linear(self.embedding_size, 128)
        self.fc0 = nn.Linear(self.embedding_size, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()

        self.resetParameters()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.l2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_dc, gamma=0.1)

    def forward(self, input_seq, input_lengths, mask0, mask1, maskinf, hidden_0=None):
        # Convert item indexes to embeddings
        input_emb = self.embedding(input_seq)  # T * bs * emb_size
        
        # RNN module for input seq and attn
        packed = nn.utils.rnn.pack_padded_sequence(input_emb, input_lengths)
        outputs, hidden = self.gru(packed, hidden_0)  # hidden: 1 * bs * hs
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs)  # outputs: T * bs * hs
        alpha = self.attn0(hidden, outputs)
        alpha = alpha + maskinf
        alpha = torch.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(1)  # bs * 1 * T

        outputs = outputs.permute(1, 0, 2)  # bs * T * hs
        local = torch.bmm(alpha, outputs)
        local = local.squeeze(1)
        hidden = hidden.squeeze(0)
        x = torch.cat((hidden, local), dim=1)

        # ######################################################
        # MLP for seq
        # bs * T * emb_size MLP 需要这样维度的输入
        input_emb = input_emb.permute(1, 0, 2)
        mask0 = mask0.unsqueeze(2)  # bs * T * 1
        order_y = torch.mul(input_emb, mask0)  # bs * T * emb_size
        order_y = torch.sum(order_y, dim=1)  # bs * emb_size
        mask1 = mask1.unsqueeze(1)  # bs * 1
        order_y = order_y / mask1  # bs * emb_size

        order_y = self.fc0(order_y)
        rest = self.fc1(order_y)
        order_y = order_y + rest
        order_y = self.relu(order_y)
        rest = self.fc2(order_y)
        order_y = order_y + rest
        order_y = self.relu(order_y)
        # ########################################################

        # concate
        y = torch.cat((x, order_y), dim=1)
        # y = order_y
        # y = x

        items_embeddings = self.embedding.weight[1:].clone()  # item_num * emb_size
        items_embeddings = self.Linear_0(items_embeddings)  # item_num * (2 * hidden_size)

        # calculate score
        score = torch.matmul(y, items_embeddings.transpose(1, 0))  # batch_size * item_num
        return score

    def resetParameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weigth in self.parameters():
            weigth.data.normal_(0, stdv)


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        
        self.w2 = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        

    def forward(self, state1, state2):
        x1 = torch.matmul(state1, self.w1)
        x2 = torch.matmul(state2, self.w2)
        x = x1 + x2
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.v.transpose(1, 0))
        x = x.squeeze(2)
        x = x.transpose(1, 0)
        # x = torch.softmax(x, dim=1)
        # x = x.unsqueeze(1)

        return x




