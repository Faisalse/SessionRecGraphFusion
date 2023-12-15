# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:32:08 2022

@author: shefai
"""
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import time
DATA_PATH = r'./algorithms/GNRRW/'
import networkx as nx
import pickle

from algorithms.GNRRW.model_cls import *
from algorithms.GNRRW.data import *
import warnings
warnings.filterwarnings("ignore")
import os



USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

class GNRRW:
    def __init__(self, epoch = 1, lr = 0.001, batch_size = 300, embedding_size = 100, l2 = 0.0001, seed = 2000):
        
        self.seed = seed
        init_seed(self.seed)
        
        
        
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_size = embedding_size
        self.sessionid = -1
        self.session_length = 12
        self.hop = 1
        self.dropout_local = 0
        self.dropout_global = 0.5
        self.activate = "relu"
        self.anchor_num = 40
        self.alpha = 0.5
        self.lr_dc = 0.1
        self.lr_dc_epoch = [3, 6, 9, 12]
        self.routing_iter = 4
        self.n_factors = 40
        self.l2 = l2
        
       
    def fit(self, train, test):
        init_seed(self.seed)
        
        # start from here
        session_key = "SessionId"
        item_key = "ItemId"
        index_session = train.columns.get_loc( session_key)
        index_item = train.columns.get_loc( item_key )
        
        session_item_train = {}
        # Convert the session data into sequence
        for row in train.itertuples(index=False):
            
            if row[index_session] in session_item_train:
                session_item_train[row[index_session]] += [(row[index_item])] 
            else: 
                session_item_train[row[index_session]] = [(row[index_item])]
        
        word2index ={}
        index2word = {}
        item_no = 1
        
        for key, values in session_item_train.items():
            length = len(session_item_train[key])
            for i in range(length):
                if session_item_train[key][i] in word2index:
                    session_item_train[key][i] = word2index[session_item_train[key][i]]
                    
                else:
                    word2index[session_item_train[key][i]] = item_no
                    index2word[item_no] = session_item_train[key][i]
                    session_item_train[key][i] = item_no
                    item_no +=1
                               
        features = []
        targets = []
        for value in session_item_train.values():
            for i in range(1, len(value)):
                targets.append(value[-i])
                features.append(value[:-i])
                
                
        session_item_test = {}
        # Convert the session data into sequence
        for row in test.itertuples(index=False):
            if row[index_session] in session_item_test:
                session_item_test[row[index_session]] += [(row[index_item])] 
            else: 
                session_item_test[row[index_session]] = [(row[index_item])]
        
        for key, values in session_item_test.items():
            length = len(session_item_test[key])
            for i in range(length):
                if session_item_test[key][i] in word2index:
                    session_item_test[key][i] = word2index[session_item_test[key][i]]
                else:
                    word2index[session_item_test[key][i]] = item_no
                    index2word[item_no] = session_item_test[key][i]
                    session_item_test[key][i] = item_no
                    item_no +=1
                    
        features1 = []
        targets1 = []
        
        
        for value in session_item_test.values():
            for i in range(1, len(value)):
                targets1.append(value[-i])
                features1.append(value[:-i])
        
        all_train_sequence = []
        
        for value in session_item_train.values():
                all_train_sequence.append(value)  
       
        
        
        
        maxList = max(all_train_sequence, key = lambda i: len(i))
        maxLength = len(maxList) 
        
        
        self.session_length = maxLength
        self.number_of_unique_items = item_no
        
        
        num_node = item_no
        relation = []
        
        
        # dictionaries -----> total number of nodes...
        adj1 = [dict() for _ in range(self.number_of_unique_items)]
        # create list ------> total number of nodes.
        
        adj = [[] for _ in range(self.number_of_unique_items)]
        
        # make relations to make graph......
        for i in range(len(all_train_sequence)):
            data = all_train_sequence[i]
            for k in range(1, 4):
                
                for j in range(len(data)-k):
                    
                    relation.append([data[j], data[j+k]])
                    relation.append([data[j+k], data[j]])
                    
        # adjancy list with weights to represent graph....        
        for tup in relation:
            if tup[1] in adj1[tup[0]].keys():
                adj1[tup[0]][tup[1]] += 1
            else:
                adj1[tup[0]][tup[1]] = 1
                
                
        weight = [[] for _ in range(self.number_of_unique_items)]
        
        for t in range(self.number_of_unique_items):
            
            x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
            # sorted adjancy list and weighted list.........
            adj[t] = [v[0] for v in x]
            weight[t] = [v[1] for v in x]
            
         
        # Create adjancy numpy matrix for fast processing..........    
        adj_numpy = np.zeros((self.number_of_unique_items, self.number_of_unique_items), dtype=np.int)
        
        for i in range(1, self.number_of_unique_items):
            for idx, val in adj1[i].items():
                adj_numpy[i][idx] = val
                
        # mechanism to learn random graph structure.....  
        
        # Trans_adj ---> transition metric,
        # Adj_sum ---> sum of columns of Adjancy_metrix
        # graph ----> build a graph.....
        Trans_adj, Adj_sum, graph = self.build_graph(adj_numpy)
        
        # anchors means refer points.... to perform random walk or perform graph traversing tasks.....
        anchors = self.anchor_select(graph, anchor_num=self.anchor_num)
        # random walk.....
        prob_conver = self.random_walk(Trans_adj, anchors, alpha=self.alpha)
        
        self.word2index = word2index
        self.index2word = index2word
        train_data = (features, targets)  
        train_data = Data(train_data, self.number_of_unique_items, self.seed)
        
        test_data = (features1, targets1)  
        test_data = Data(test_data, self.number_of_unique_items, self.seed)
        
    
        adj_items, weight_items = self.handle_adj(adj, weight, num_node, self.session_length)
    
        model = NeighRoutingGnnCls2Scores(self.hidden_size, self.routing_iter, self.n_factors, self.hop, self.session_length,
                                          num_node, adj_items, weight_items, prob_conver, device, self.seed)
    
        model = model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_dc_epoch, gamma=self.lr_dc)
    
       
        train_slices = train_data.generate_batch(self.batch_size)
        
        test_slices = test_data.generate_batch(1)
        Mrr20List = []
        counter = 0 
        for epoch in range(self.epoch):
            model.train()
            total_loss = []
            total_rec_loss = []
            total_rec_loss1 = []
            total_rec_loss2 = []
            for index in tqdm(train_slices):
                
                optimizer.zero_grad()
                scores, scores1, scores2, targets = self.forward(model, index, train_data)
                
                rec_loss1 = model.loss_function1(scores1, targets - 1)
                rec_loss2 = model.loss_function2(scores2, targets - 1)
                rec_loss = model.loss_function(scores, targets - 1)
                loss = rec_loss1 + rec_loss2 + rec_loss
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                total_rec_loss1.append(rec_loss1.item())
                total_rec_loss2.append(rec_loss2.item())
                total_rec_loss.append(rec_loss.item())
                
            with torch.no_grad(): 
                valid_Mrr = 0 
                for index in test_slices:
                    score, scores1, scores2, out_var = self.forward(model, index, test_data)
                    
                    sub_scores_k20_index = score.topk(20)[1]
                    sub_scores_k20_score = score.topk(20)[0]
                    
                    sub_scores_k20_index = np.array(sub_scores_k20_index.cpu())
                    sub_scores_k20_score = sub_scores_k20_score.detach().cpu()
                    sub_scores_k20_score = sub_scores_k20_score.detach().numpy()
                    
                    preds = pd.Series(data = list(sub_scores_k20_score[0]), index = list(sub_scores_k20_index[0]))
                    out_var = out_var.cpu()
                    out_var = out_var.numpy()
                    
                    out_var = out_var[0] - 1
                    if out_var in preds.index:
                        rank = preds.index.get_loc( out_var ) + 1
                        valid_Mrr += ( 1.0/ rank )
                
                valid_Mrr = valid_Mrr/len(test_data)
                if epoch < 2:
                    Mrr20List.append(valid_Mrr)
                else:
                    if(valid_Mrr > Mrr20List[-1]):
                        Mrr20List.append(valid_Mrr)
                    else:
                        counter +=1
                            
                print("test_Mrr20  ", valid_Mrr) 
                
            
                if  counter > 4:
                  print("We stop at Epoch:", epoch +1)
                  # Store the best values
                  max_value = max(Mrr20List)
                  max_index = Mrr20List.index(max_value)
                  name = "Epoch:"+str(max_index+1)+"-Lr:"+str(self.lr)+"-BatchSize:"+str(self.batch_size)+"-L2:"+str(self.l2)
                  Dict = {"Parameters" : [name],
                          "MRR20": [max_value]}
                  Dict = pd.DataFrame.from_dict(Dict)
                  if(os.path.isfile(DATA_PATH+"results.csv")):
                      result = pd.read_csv(DATA_PATH+"results.csv")
                      frames = [result, Dict]
                      result = pd.concat(frames)
                      result.to_csv(DATA_PATH+"results.csv", index = False)
                  else:
                      Dict.to_csv(DATA_PATH+"results.csv", index = False)
                  break    
        self.model = model
        
    def forward(self, model, index, data):
        init_seed(self.seed)
        
        
        inp_sess, targets, mask_1, mask_inf, lengths = data.get_slice_sess_mask(index)
        inp_sess = torch.LongTensor(inp_sess).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        targets = torch.LongTensor(targets).to(device)
        mask_1 = torch.FloatTensor(mask_1).to(device)
        mask_inf = torch.FloatTensor(mask_inf).to(device)
        scores, scores1, scores2 = model(inp_sess, mask_1, mask_inf, lengths)
        return scores, scores1, scores2, targets
    
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        init_seed(self.seed)
        
        
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
         
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        # temp_list
        temp_list = []
        temp_list = ([self.testList], [prev_iid])
        test_data = Data(temp_list, self.number_of_unique_items, self.seed)
        
        self.model.eval()
        train_slices = test_data.generate_batch(1)
        for index in train_slices:
            tes_scores, tes_scores1, tes_scores2, tes_targets = self.forward(self.model, index, test_data)
        sub_scores_k100_index = tes_scores.topk(100)[1]
        sub_scores_k100_score = tes_scores.topk(100)[0]
       
        sub_scores_k100_index = np.array(sub_scores_k100_index.cpu())
        sub_scores_k100_score = sub_scores_k100_score.cpu()
        sub_scores_k100_score = sub_scores_k100_score.detach().numpy()
        tempList = []
        sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index[0]]
        for key in sub_scores_k100_index:
            tempList.append(self.index2word[key])
        preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
        return preds 
       # preds = pd.Series(data = list(sub_scores_k100_score[0]), index = list(sub_scores_k100_index[0]))
       # return preds
           
   
    def clear(self):
        pass
    
    def build_graph(self, Adj_matrix):
        init_seed(self.seed)
        
        num_items = Adj_matrix.shape[0]
        # Return an array of zeros with the same shape and type as a given array.
        
        Trans_adj = np.zeros_like(Adj_matrix, dtype=np.float)
        # take sum of all rows....
        Adj_sum = Adj_matrix.sum(axis=1)
        
        for i in range(num_items):
            if Adj_sum[i] > 0:
                # What is meaning of trans adjancy....
                Trans_adj[i, :] = Adj_matrix[i, :] / Adj_sum[i]

        graph = nx.Graph()
        i_idx, j_idx = np.nonzero(Adj_matrix)
        for i in range(len(i_idx)):
            graph.add_edge(i_idx[i], j_idx[i], weight=Adj_matrix[i_idx[i], j_idx[i]])

        return Trans_adj, Adj_sum, graph


    def anchor_select(self, graph, anchor_num):
        init_seed(self.seed)
        
        pagerank = nx.pagerank(graph)
        pagerank_sort = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        pagerank_sort = pagerank_sort[:anchor_num]
        anchors = [x[0] for x in pagerank_sort]

        return anchors


    def random_walk(self,Trans_adj, anchors, alpha):
        init_seed(self.seed)
        
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
    
    
    def handle_adj(self, adj_items, weight_items, n_items, sample_num):
        init_seed(self.seed)
        adj_entity = np.zeros((n_items, sample_num), dtype=np.int)
        wei_entity = np.zeros((n_items, sample_num))
        for entity in range(1, n_items):
            neighbor = list(adj_items[entity])
            neighbor_weight = list(weight_items[entity])
            n_neighbor = len(neighbor)
            if n_neighbor == 0:
                continue
            if n_neighbor >= sample_num:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
            tmp, tmp_wei = [], []
            for i in sampled_indices:
                tmp.append(neighbor[i])
                tmp_wei.append(neighbor_weight[i])

            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
            wei_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
        return adj_entity, wei_entity
       



















        