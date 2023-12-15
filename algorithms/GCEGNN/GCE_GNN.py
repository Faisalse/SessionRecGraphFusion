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
import pickle

from algorithms.GCE_GNN.utils import *
from algorithms.GCE_GNN.model import *
class GCE_GNN:
    def __init__(self, epoch = 1, lr = 0.001, batch_size = 300, embedding_size = 100, l2 = 0.00001):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_size = embedding_size
        self.sessionid = -1
        self.session_length = 12
        self.number_hop = 2
        self.dropout_local = 0
        self.dropout_global = 0.5
        self.activate = "relu"
        self.l2 = l2
       
    def fit(self, train, test):
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
        
        all_train_sequence = []
        for value in session_item_train.values():
                all_train_sequence.append(value)  
                
                
        # for value in session_item_test.values():
        #         all_train_sequence.append(value) 
                
                
        maxList = max(all_train_sequence, key = lambda i: len(i))
        maxLength = len(maxList)  
        self.session_length = maxLength
    
            
            
    
        number_of_unique_items = item_no
        num_node = item_no
        
        relation = []
       
        adj1 = [dict() for _ in range(number_of_unique_items)]
        adj = [[] for _ in range(number_of_unique_items)]
        
        for i in range(len(all_train_sequence)):
            data = all_train_sequence[i]
            for k in range(1, 4):
                for j in range(len(data)-k):
                    relation.append([data[j], data[j+k]])
                    relation.append([data[j+k], data[j]])
        
        for tup in relation:
            if tup[1] in adj1[tup[0]].keys():
                adj1[tup[0]][tup[1]] += 1
            else:
                adj1[tup[0]][tup[1]] = 1
        
        weight = [[] for _ in range(number_of_unique_items)]
        
        for t in range(number_of_unique_items):
            x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
            adj[t] = [v[0] for v in x]
            weight[t] = [v[1] for v in x]
        
        # for i in range(number_of_unique_items):
        #     adj[i] = adj[i][:self.session_length]
        #     weight[i] = weight[i][:self.session_length]
        
        
        self.word2index = word2index
        self.index2word = index2word      
        
            
        train_data = (features, targets)  
        train_data = Data(train_data)
    
        adj, weight = handle_adj(adj, num_node, self.session_length, weight)
        model = trans_to_cuda(CombineGraph(self.lr, self.batch_size, self.hidden_size, self.session_length, self.number_hop, self.dropout_local, self.dropout_global, self.activate, num_node, adj, weight, self.l2))
        
        print('start training: ', datetime.datetime.now())
        for epoch in range(self.epoch):
                 
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            model.train()
            total_loss = 0.0
            train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                                       shuffle=True, pin_memory=True)
            for data in tqdm(train_loader):
                model.optimizer.zero_grad()
                targets, scores = self.forward(model, data)
                targets = trans_to_cuda(targets).long()
                loss = model.loss_function(scores, targets - 1)
                loss.backward()
                model.optimizer.step()
                total_loss += loss
            print('\tLoss:\t%.3f' % total_loss)
            # model.scheduler.step()
    
            print('\tLoss:\t%.3f' % total_loss)
            
        self.model = model
        
    def forward(self, model, data):
        alias_inputs, adj, items, mask, targets, inputs = data
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        adj = trans_to_cuda(adj).float()
        mask = trans_to_cuda(mask).long()
        inputs = trans_to_cuda(inputs).long()
    
        hidden = model(items, adj, mask, inputs)
        get = lambda index: hidden[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return targets, model.compute_scores(seq_hidden, mask)
    
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
         
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        # temp_list
        temp_list = []
        temp_list = ([self.testList], [prev_iid])
        test_data = Data(temp_list)
        
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=1,
                                              shuffle=False, pin_memory=True)
        
        
        for data in test_loader:
            targets, scores = self.forward(self.model, data)
        sub_scores_k100_index = scores.topk(100)[1]
        sub_scores_k100_score = scores.topk(100)[0]
       
        sub_scores_k100_index = trans_to_cpu(sub_scores_k100_index).detach().numpy()
        sub_scores_k100_score = trans_to_cpu(sub_scores_k100_score).detach().numpy()
       
       
       
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
       



















        