#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import math
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv,  SAGEConv, SignedConv, TAGConv, ARMAConv,TransformerConv
#from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, SignedConv, TAGConv, ARMAConv,TransformerConv///SuperGAT, GATV2, GAT, 


class GAT(torch.nn.Module):
    def __init__(self, node_dim, n_dim, heads): # 节点维度和边的维度
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(node_dim, n_dim, heads) # GAT
        self.gat6 = GATv2Conv( n_dim* heads, 128, heads)
        self.line = nn.Linear(128*heads, 64)


    def forward(self, data): # forward0
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat6(x, edge_index)
        x = self.line(x)
        return x

# 基于session graph　图的定义
class SessionGraph(Module):
    
    def __init__(self, lr, embedding, l2, dropout, lr_dc, lr_dc_step, nonhybrid, num_node, kernel_type='exp-1', num_heads=2):
        super(SessionGraph, self).__init__()
        self.lr = lr
        
        self.l2 = l2
        self.dropout = dropout
        self.lr_dc = lr_dc
        self.nonhybrid = nonhybrid
        self.num_node = num_node
        self.heads = 7
        self.hidden_size = embedding
        self.lr_dc_step = lr_dc_step
        self.embedding = nn.Embedding(self.num_node, self.hidden_size) # 将节点嵌入成向量（数据集节点数，hiddensize）
        self.position_embed = PositionalEncoding(self.dropout,)
        # self.gnn = GNN(self.hidden_size, step=opt.step)


        self.gat = GAT(node_dim=self.hidden_size+1, n_dim=256, heads=self.heads)  # n_dim本身自设256, self.hidden_size+1是嵌入的特征加一个维度的时间特征
        # self.linear = nn.Linear(self.hidden_size + 1, self.hidden_size, bias=True)

        # 加 mlp 将1维的时间 变换为10维
        self.linear = nn.Linear(self.hidden_size+70, self.hidden_size, bias=True)  # +10
        self.linear_t = nn.Linear(1, 70, bias=True)  # 10表示 由前面的1维映射到10维

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_four = nn.Linear(1, 1, bias=False) # 为了对比实验中mean函数
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_dc_step, gamma=self.lr_dc)
        self.reset_parameters()
        # self.decoder = MultiheadAttention(self.hidden_size, num_heads, dropout=dropout)

        self.kernel_num = self.num_node

        parts = kernel_type.split('-')
        kernel_types = []

        self.params = []
        for i in range(len(parts)):
            pi = parts[i]
            if pi in {'exp', 'exp*', 'log', 'lin', 'exp^', 'exp*^', 'log^', 'lin^', 'ind', 'const', 'thres'}:
                if pi.endswith('^'):
                    var = (nn.Parameter(torch.rand(1, requires_grad=True) * 5 + 10),
                           nn.Parameter(torch.rand(1, requires_grad=True)))
                    kernel_types.append(pi[:-1])
                else:
                    var = (nn.Parameter(torch.rand(1, requires_grad=True) * 0.01),
                           nn.Parameter(torch.rand(1, requires_grad=True)))
                    kernel_types.append(pi)

                self.register_parameter(pi + str(len(self.params)) + '_0', var[0])
                self.register_parameter(pi + str(len(self.params)) + '_1', var[1])

                self.params.append(var)

            elif pi.isdigit():
                val = int(pi)
                if val > 1:
                    pi = parts[i - 1]
                    for j in range(val - 1):
                        if pi.endswith('^'):
                            var = (nn.Parameter(torch.rand(1, requires_grad=True) * 5 + 10),
                                   nn.Parameter(torch.rand(1, requires_grad=True)))
                            kernel_types.append(pi[:-1])
                        else:
                            var = (nn.Parameter(torch.rand(1, requires_grad=True) * 0.01),
                                   nn.Parameter(torch.rand(1, requires_grad=True)))
                            kernel_types.append(pi)

                        self.register_parameter(pi + str(len(self.params)) + '_0', var[0])
                        self.register_parameter(pi + str(len(self.params)) + '_1', var[1])

                        self.params.append(var)


            else:
                print('no matching kernel ' + pi)

        self.kernel_num = len(kernel_types)
        print(kernel_types, self.params)

        def decay_constructor(t):
            kernels = []
            for i in range(self.kernel_num):
                pi = kernel_types[i]
                if pi == 'log':
                    kernels.append(torch.mul(self.params[i][0], torch.log1p(t)) + self.params[i][1])
                elif pi == 'exp':
                    kernels.append(1000 * torch.exp(torch.mul(self.params[i][0], torch.neg(t))) + self.params[i][1])
                elif pi == 'exp*':
                    kernels.append(torch.mul(self.params[i][0], torch.exp(torch.neg(t))) + self.params[i][1])
                elif pi == 'lin':
                    kernels.append(self.params[i][0] * t + self.params[i][1])
                elif pi == 'ind':
                    kernels.append(t)
                elif pi == 'const':
                    kernels.append(torch.ones(t.size()))
                elif pi == 'thres':
                    kernels.append(torch.reciprocal(1 + torch.exp(-self.params[i][0] * t + self.params[i][1])))

            return torch.stack(kernels, dim=2)

        self.decay = decay_constructor
#
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size # 局部
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size  # 全局
        # print(q1.shape, '\n',q2.shape)
        # q2 = torch.mean(q2, )
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # print(alpha.shape)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
    def minmaxscaler(self, time_data, max, min):

        return (time_data - min) / (max - min)

    def forward(self, inputs, A_edge, input_times, batch_size):  # forward1
        # time_embedding = input_times
        self.batch_size = batch_size
        time_embedding = self.decay(input_times)
        # time_embedding = input_times.reshape(input_times.shape[0], input_times.shape[1], 1)
        time_embedding = time_embedding * math.sqrt(self.batch_size) + self.position_embed(time_embedding, 1000)  # [100, 69, 1]
        time_embedding = time_embedding.view(len(input_times), len(input_times[0]), 1)
        # 4.10 加MLP
        # time_embedding = self.linear_t(time_embedding)  #这里传入原始的一维度的time特征，没有映射到多维度

        hidden = self.embedding(inputs)  # 此处的inputs是items传来的

        hidden = hidden.view(len(inputs), len(inputs[0]), -1)
        #  4.10 加MLP 后 再cat
        hidden = torch.cat([hidden, time_embedding], 2)
        graph_data = Deal_data(hidden, A_edge, self.batch_size)  # 调用本文件中的Deal_data()
        hidden = self.gat(graph_data)
        hidden = hidden.view(len(inputs), len(hidden) // len(inputs), -1)
        # hidden = self.linear(hidden)

        # #  4.10 加MLP 后 再cat
        # hidden = torch.cat([hidden, time_embedding], 2)
        # hidden = self.linear(hidden)
        #  4.08 拼接
        # hidden = torch.cat([hidden, time_embedding], 2)
        # hidden = self.linear(hidden)

        # 4.09 加位置函数后相乘(效果不好降低8%)
        # hidden = torch.mul(hidden, time_embedding)
        # hidden = self.linear(hidden)

        # hidden = hidden * time_embedding
        # hidden = torch.cat([hidden, time_embedding], 2)
        # print(hidden.shape)

        return hidden


# def trans_to_cuda(variable):
#     if torch.cuda.is_available():
#         return variable.cuda()
#     else:
#         return variable


# def trans_to_cpu(variable):
#     if torch.cuda.is_available():
#         return variable.cpu()
#     else:
#         return variable

def Deal_data(items, A_edge, batch_size): #
    # print(len(items), len(A_edge))
    data_list = []
    # print(len(items),items)
    for i in range(len(items)):
        edge_index = torch.tensor(A_edge[i], dtype=torch.long)
        x = torch.tensor(items[i], dtype = torch.float)
        # print(edge_index, x)
        data = Data(x = x, edge_index = edge_index.t().contiguous())
        data_list.append(data)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    # print(len(loader))
    for i, batch in enumerate(loader):
        # print(i)
        data1 = batch
    # print(data1[0])
    return data1

def deal_time(lens, input_times):
    # l = []
    # for h in input_times:
    #     l.append(len(h))
    # max_len = np.max(np.array(l))

    us_times= [input_times[i] + [0]*(lens-len(input_times[i])) for i in range(len(input_times))]
    us_times = np.asarray(us_times)
    return us_times
    #
def forward(model, i, data, batch_size): # i为大小为100的数组  forward3
# 调用utils中data类中的get_slice()函数
    alias_inputs, A, items, mask, targets, A_edge, input_times, max_time, min_time = data.get_slice(i) # 添加A_edge存边
    # graph_data = Deal_data(items, A_edge)11.02 无用到
    length_ = len(items[0])
    input_times = deal_time(length_, input_times)
    # 时间归一化,time = (input_times-min_time) / (max_time - min_time),这里不用这个的原因是因为时间不够长的后面补了0， 然后相当于min_time=0
    #time= input_times / max_time  #
    #input_times = time.reshape(time.shape[0], time.shape[1], 1)
    # 4.08  加时间处理
   
    input_times= input_times / max_time  #

    alias_inputs = torch.Tensor(alias_inputs).long() # 转为tensor # alias_inputs.shape[100, 145]
    items =torch.Tensor(items).long()    # items.shape[100,17]
    A = torch.Tensor(A).float() # A.shape[100,17,17]
    mask = torch.Tensor(mask).long()# [100, 145]
    input_times = torch.Tensor(input_times).float()
    # print(items.shape, A.shape, alias_inputs.shape, mask.shape) # 输出是torch.Size([100, 17]) torch.Size([100, 17, 17]) torch.Size([100, 145]) torch.Size([100, 145])

    # 把item和制作的邻接矩阵丢入模型（这里图表示是但的邻接矩阵，他是处理本身图的关系，也把行和列的关系进行了统一，自己理解的)

    hidden = model(items, A_edge, input_times, batch_size)  # 调用class SessionGraph 中的forward（）
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # print(seq_hidden, mask)
    return targets, model.compute_scores(seq_hidden, mask)


# def train_test(model, train_data, test_data): # model传递的是SessionGraph中所有参数
#     model.scheduler.step()

#     print('start training: ', datetime.datetime.now())# 输出当前时间
#     model.train()
#     total_loss = 0.0
#     #　调用util中的generate_batch()函数
#     slices = train_data.generate_batch(model.batch_size) # batch_size=100，slices为一个大小为100的数组
#     for i, j in zip(slices, np.arange(len(slices))): # np.arange()一个参数时，参数值为终点，起点取默认值0，步长取默认值1
#         model.optimizer.zero_grad()
#         # 调用本文件中的forward()函数
#         targets, scores = forward(model, i, train_data) # 掉用forward3
#         targets = trans_to_cuda(torch.Tensor(targets).long())
#         loss = model.loss_function(scores, targets - 1)
#         loss.backward()
#         model.optimizer.step()
#         total_loss += loss
#         if j % int(len(slices) / 5 + 1) == 0:
#             print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
#     print('\tLoss:\t%.3f' % total_loss)

#     print('start predicting: ', datetime.datetime.now())
#     model.eval()
#     hit, mrr = [], []
#     slices = test_data.generate_batch(model.batch_size)
#     for i in slices:
#         targets, scores = forward(model, i, test_data)
#         sub_scores = scores.topk(20)[1]
#         sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         for score, target, mask in zip(sub_scores, targets, test_data.mask):
#             hit.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr.append(0)
#             else:
#                 mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
#     hit = np.mean(hit) * 100
#     mrr = np.mean(mrr) * 100
#     return hit, mrr

class PositionalEncoding(nn.Module):
    def __init__(self, dropout,):
        super(PositionalEncoding, self).__init__()

    def forward(self, X, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = X.size()[1]
        # X为wordEmbedding的输入,PositionalEncoding与batch没有关系
        # max_seq_len越大,sin()或者cos()的周期越小,同样维度
        # 的X,针对不同的max_seq_len就可以得到不同的positionalEncoding
        assert X.size()[1] <= max_seq_len
        # X的维度为: [batch_size, seq_len, embed_size]
        # 其中: seq_len = l, embed_size = d
        l, d = X.size()[1], X.size()[-1]
        # P_{i,2j}   = sin(i/10000^{2j/d})
        # P_{i,2j+1} = cos(i/10000^{2j/d})
        # for i=0,1,...,l-1 and j=0,1,2,...,[(d-2)/2]
        max_seq_len = int((max_seq_len//l)*l)
        P = np.zeros([1, l, d])
        # T = i/10000^{2j/d}
        T = [i*1.0/10000**(2*j*1.0/d) for i in range(0, max_seq_len, max_seq_len//l) for j in range((d+1)//2)]
        T = np.array(T).reshape([l, (d+1)//2])
        if d % 2 != 0:
            P[0, :, 1::2] = np.cos(T[:, :-1])
        else:
            P[0, :, 1::2] = np.cos(T)
        P[0, :, 0::2] = np.sin(T)
        return torch.tensor(P, dtype=torch.float)
