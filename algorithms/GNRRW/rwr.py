import pickle
import numpy as np
import networkx as nx
import argparse
import time
import datetime
import os


def build_graph(Adj_matrix):
    
    num_items = Adj_matrix.shape[0]
    Trans_adj = np.zeros_like(Adj_matrix, dtype=np.float)
    Adj_sum = Adj_matrix.sum(axis=1)
    for i in range(num_items):
        if Adj_sum[i] > 0:
            Trans_adj[i, :] = Adj_matrix[i, :] / Adj_sum[i]

    graph = nx.Graph()
    i_idx, j_idx = np.nonzero(Adj_matrix)
    for i in range(len(i_idx)):
        graph.add_edge(i_idx[i], j_idx[i], weight=Adj_matrix[i_idx[i], j_idx[i]])

    return Trans_adj, Adj_sum, graph


def anchor_select(graph, anchor_num):
    pagerank = nx.pagerank(graph)
    pagerank_sort = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    pagerank_sort = pagerank_sort[:anchor_num]
    anchors = [x[0] for x in pagerank_sort]

    return anchors


def random_walk(Trans_adj, anchors, alpha):
    anchor_num = len(anchors)
    num_items = Trans_adj.shape[0]

    # 节点分布矩阵
    prob_node = np.zeros((num_items, anchor_num))
    # 重启动矩阵
    restart = np.zeros((num_items, anchor_num))
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
            # print('random walk convergence, iteration: %d' % count)
            break

    
    for i in range(prob.shape[0]):
        if prob[i, :].sum() != 0:
            prob[i, :] = prob[i, :] / prob[i, :].sum()
        else:
            if i == 0:
                continue
            prob[i, :] = 1.0 / prob[i, :].shape[0]

    return prob

