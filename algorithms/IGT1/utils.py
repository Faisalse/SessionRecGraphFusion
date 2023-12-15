#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np

def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph

# 传入参数是inputs,[0]   返回值是us_pois, us_msks, len_max＝１６
# 数据掩码？
# all_usr_pois(应该是所有用户点击的所有商品编号)：[[[1, 2], [1], [4], [6]...] [[282], [281, 308], [281]...]
# item_tail [0] [0]
# 这一步主要是规范化数据
def data_masks(all_usr_pois, item_tail):
    # print(all_usr_pois, item_tail)
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # 这是将所有的用户点击商品数扩大成16，　不足的话使用0填充
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # 构造16长度的列表，这一步设定是否是用户点击的商品，是，则对应列表的位置设为１，否则设为0
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

# 将训练集的部分拆分为验证集
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]# list形式
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# 这里的data是将用户点击序列，序列[0:-1]作为用户行为，序列[-1]作为预测的标签，即训练目标
class Data1():
    def __init__(self, data, shuffle=False, graph=None):
        # print("..................")
        # # print(data)
        # data.to_csv('data.tsv', sep='\t', index=False)
        # data数据格式：训练集：([[1, 2], [1], [4]...], [5,2,5...])
        inputs = data[0] # 取文件中第一列
        input_times = data[2]
        times = np.array(input_times)
        
        self.max_time = times.max()[0]
        self.min_time = times.min()[0]
        # print(max_time, min_time)
        # print(input_times, '!!!!!!!!!!')
        inputs, mask, len_max = data_masks(inputs, [0]) #　调用本文件的data_masks()函数
        # inputs_times, time_mask, len_time_max = data_masks(input_times, [0])

        self.inputs = np.asarray(inputs)
        self.inputs_times = np.asarray(input_times)

        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])# 标签
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    #生成batch函数
    def generate_batch(self, batch_size):# batch_size=100
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.inputs_times = self.inputs_times[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
    # 把数组中n_batch * batch_size个数平均分成n_batch份
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))] # [-1]表示数组最后一个元素
        return slices

    def get_slice(self, i): # i为大小为100的数组
        # inputs:(100, 16),mask:(100, 16), targets:(100,)是长度为100的列表
        inputs, mask, targets, input_times = self.inputs[i], self.mask[i], self.targets[i], self.inputs_times[i]
        # input_times = input_times[:,0:len(inputs[1])]
        items, n_node, A, alias_inputs = [], [], [], []
        # 这里去重是为了构造邻接矩阵用的
        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) # 各个u_input的长度集合
        max_n_node = np.max(n_node)        # 每个batch_size的max_n_node是不同的
        A_edge = []  # 存边

        for u_input in inputs:
            node = np.unique(u_input)#去重--第一次循环结果[0,85]
            items.append(node.tolist() + (max_n_node - len(node)) * [0])# 后面追加0的个数为((max_n_node - len(node))个
            u_A = np.zeros((max_n_node, max_n_node))# 生成max_n_node　*　max_n_node的矩阵
            # 这一步是构建邻接矩阵, 将用户点击的商品构成了一个无方向图??
            edge = []
            for i in np.arange(len(u_input) - 1): # 遍历各个元素 # i循环当前序列长度145
                if u_input[i + 1] == 0:
                    break
                # print(node, u_input, u_input[i])
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                # print('---', u, v)
                u_A[u][v] = 1
                edge.append([u,v])

            A.append(u_A) # 变形的邻接矩阵，怎么用
           # print( A.append(u_A))
            A_edge.append(edge) #
    # alias_inputs 到底是个啥？目的实干啥？1020
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # np.where(node == i)[0][0] 渠道的是符合要求的第0个下标
       # print("0000",len(A))#200
       # print("1111", len(A_edge))#100
       # print("0000", A)
       # print("11110", A_edge)
        return alias_inputs, A, items, mask, targets, A_edge, input_times, self.max_time, self.min_time
