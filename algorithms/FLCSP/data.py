# -*- coding: utf-8 -*-
__author__ = 'zzz'

import networkx as nx
import os
import pickle
import numpy as np
import torch
# Default Padding item
PAD_token = 0

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Dictionary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {PAD_token: 'PAD'}
        self.num_words = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.num_words += 1

    def __len__(self):
        return self.num_words



def sort_pairs(data_pairs):
    # 根据tensor长度进行排序
    return sorted(data_pairs, key=lambda x: len(x[0]), reverse=True)


def batchify(data, batch_size, seed = 2000):
    init_seed(seed)
    
    nbatch = len(data) // batch_size + 1
    data_batch = []
    for i in range(nbatch):
        if i * batch_size + batch_size >= len(data):
            data_batch.append(data[i * batch_size:])
        else:
            data_batch.append(data[i * batch_size:(i + 1) * batch_size])
    return data_batch


def zero_padding(data):
    max_len = 0
    for seq, _ in data:
        if len(seq) > max_len:
            max_len = len(seq)
    data = sort_pairs(data)
    data_list = []
    len_list = []
    lab_list = []
    for seq, lab in data:
        tmp_data = [0] * max_len
        for i in range(len(seq)):
            tmp_data[i + max_len - len(seq)] = seq[i]
        data_list.append(tmp_data)
        len_list.append(len(seq))
        lab_list.append(lab)
    data_tensor = torch.LongTensor(data_list)
    lab_tensor = torch.LongTensor(lab_list)
    len_tensor = torch.LongTensor(len_list)

    return data_tensor, lab_tensor, len_tensor

def zero_padding_on_front(data):
    max_len = 0
    for seq, _ in data:
        if len(seq) > max_len:
            max_len = len(seq)
    data = sort_pairs(data)
    data_list = []
    len_list = []
    lab_list = []
    for seq, lab in data:
        tmp_data = [0] * max_len
        for i in range(len(seq)):
            tmp_data[i] = seq[i]
        data_list.append(tmp_data)
        len_list.append(len(seq))
        lab_list.append(lab)
    data_tensor = torch.LongTensor(data_list)
    lab_tensor = torch.LongTensor(lab_list)
    len_tensor = torch.LongTensor(len_list)

    # get A
    A, items, alias = get_A(data_list)
    A = torch.Tensor(A).float()
    items = torch.Tensor(items).long()
    alias = torch.Tensor(alias).long()
    return data_tensor, lab_tensor, len_tensor, A, items, alias



def zero_padding_front(data):
    max_len = 0
    for seq, _ in data:
        if len(seq) > max_len:
            max_len = len(seq)
    data = sort_pairs(data)
    data_list = []
    lab_list = []
    for seq, lab in data:
        tmp_data = [0] * max_len
        for i in range(len(seq)):
            tmp_data[i] = seq[i]
        data_list.append(tmp_data)
        lab_list.append(lab)
    data_tensor = torch.LongTensor(data_list)
    lab_tensor = torch.LongTensor(lab_list)

    return data_tensor, lab_tensor

def zero_padding_behind(data):
    max_len = 0
    for seq, _ in data:
        if len(seq) > max_len:
            max_len = len(seq)
    data = sort_pairs(data)
    data_list = []
    lab_list = []
    for seq, lab in data:
        tmp_data = [0] * max_len
        for i in range(len(seq)):
            tmp_data[i + max_len - len(seq)] = seq[i]
        data_list.append(tmp_data)
        lab_list.append(lab)
    data_tensor = torch.LongTensor(data_list)
    lab_tensor = torch.LongTensor(lab_list)

    return data_tensor, lab_tensor

# ###################################################################



def get_A(data):
    A, items, n_node, alias = [], [], [], []
    for seq in data:
        n_node.append(len(np.unique(seq)))
    max_n_node = np.max(n_node)
    for seq in data:
        node = np.unique(seq)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(seq) - 1):
            if seq[i + 1] == 0:
                break
            u = np.where(node == seq[i])[0][0]
            v = np.where(node == seq[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias.append([np.where(node == j)[0][0] for j in seq])

    return A, items, alias


# graph code
def build_graph(num_word, data):
    
    Adj = np.zeros((num_word, num_word))
    graph = nx.DiGraph()
    for seq, lab in data:
        seq = seq + [lab]
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i+1]) is None:
                weight = 1
                Adj[seq[i], seq[i+1]] = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i+1])['weight'] + 1
                Adj[seq[i], seq[i+1]] += 1
            graph.add_edge(seq[i], seq[i+1], weight=weight)

    # 邻接矩阵归一化得到转移概率矩阵(按行归一化)
    Trans_adj = np.zeros_like(Adj)
    Adj_sum = Adj.sum(axis=1)
    for i in range(num_word):
        if Adj_sum[i] > 0:
            Trans_adj[i, :] = Adj[i, :] / Adj_sum[i]

    # Adj_sum = Adj.sum(axis=0)
    # for i in range(num_word):
    #     if Adj_sum[i] > 0:
    #         Trans_adj[:, i] = Adj[:, i] / Adj_sum[i]

    return Trans_adj, graph

def anchor_select(graph, anchor_num):
    pagerank = nx.pagerank(graph)
    pagerank_sort = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    pagerank_sort = pagerank_sort[:anchor_num]
    anchors = [x[0] for x in pagerank_sort]

    return anchors


def random_walk(Trans_adj, anchors, alpha):
    # 得到每个物品针对锚点物品的收敛概率，视为针对不同话题的概率分布
    print('start random walk...')
    anchor_num = len(anchors)
    num_word = Trans_adj.shape[0]
    # 节点分布矩阵
    prob_node = np.zeros((num_word, anchor_num))
    # 重启动矩阵
    restart = np.zeros((num_word, anchor_num))
    for i in range(anchor_num):
        restart[anchors[i]][i] = 1
        prob_node[anchors[i]][i] = 1

    count = 0
    while True:
        count += 1
        prob_t = alpha * np.dot(Trans_adj, prob_node) + (1 - alpha) * restart
        residual = np.sum(np.abs(prob_node - prob_t))
        prob_node = prob_t
        if abs(residual) < 1e-8:
            prob = prob_node.copy()
            print('random walk convergence, iteration: %d' % count)
            break

    # prob作为个收敛矩阵(是按列为概率分布的)，现在按行归一化为概率分布
    for i in range(prob.shape[0]):
        if prob[i, :].sum() != 0:
            prob[i, :] = prob[i, :] / prob[i, :].sum()
        else:
            if i == 0:
                continue
            prob[i, :] = 1.0 / prob[i, :].shape[0]
    return prob







