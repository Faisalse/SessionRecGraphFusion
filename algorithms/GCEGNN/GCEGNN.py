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
import os
DATA_PATH = r'./algorithms/GCEGNN/'
from algorithms.GCEGNN.utils import *
from algorithms.GCEGNN.model import *
class GCEGNN:
    def __init__(self, epoch = 1, lr = 0.001, batch_size = 300, embedding_size = 100, dropout = 0.3, l2 = 0.00001):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.embedding = embedding_size
        self.sessionid = -1
        self.session_length = 12
        self.number_hop = 2
        self.dropout_local = 0
        self.dropout_global = dropout
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
                
                
        features1 = []
        targets1 = []
        for value in session_item_test.values():
            for i in range(1, len(value)):
                targets1.append(value[-i])
                features1.append(value[:-i])
                
                
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
        
        test_data = (features1, targets1)  
        test_data = Data(test_data)
    
        adj, weight = handle_adj(adj, num_node, self.session_length, weight)
        model = trans_to_cuda(CombineGraph(self.lr, self.batch_size, self.embedding, self.session_length, self.number_hop, self.dropout_local, self.dropout_global, self.activate, num_node, adj, weight, self.l2))
        
        Mrr20List = []
        counter = 0 
        for epoch in range(self.epoch):
                 
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            model.train()
            total_loss = 0.0
            train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                                       shuffle=True, pin_memory=True)
            
            validation_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=1,
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
            
            with torch.no_grad(): 
                valid_Mrr = 0 
                for data in validation_loader:
                    out_var, score = self.forward(model, data)
                    
                    
                    sub_scores_k20_index = score.topk(20)[1]
                    sub_scores_k20_score = score.topk(20)[0]
                    
                    sub_scores_k20_index = trans_to_cpu(sub_scores_k20_index).detach().numpy()
                    sub_scores_k20_score = trans_to_cpu(sub_scores_k20_score).detach().numpy()
                    
                    preds = pd.Series(data = list(sub_scores_k20_score[0]), index = list(sub_scores_k20_index[0]))
                    out_var = trans_to_cpu(out_var)
                    out_var = out_var[0] - 1
                    
                    
                    if out_var in preds.index:
                        rank = preds.index.get_loc( out_var ) + 1
                        valid_Mrr += ( 1.0/ rank )
                
                valid_Mrr = valid_Mrr/len(features1)        
                
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
                  name = "Epoch:"+str(max_index+1)+"-Lr:"+str(self.lr)+"-BatchSize:"+str(self.batch_size)+"-Embedding:"+str(self.embedding)+"-L2:"+str(self.l2)+"-dropout:"+str(self.dropout_global)
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
       



















        