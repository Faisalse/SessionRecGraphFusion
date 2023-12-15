# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:32:08 2022

@author: shefai
"""
import datetime
import numpy as np
import torch
from torch import nn
import numpy as np
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from algorithms.TAGNN.utils import *
from algorithms.TAGNN.model import *
import pandas as pd
DATA_PATH = r'./algorithms/TAGNN/'
import os


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


init_seed(2000)
class TAGNN:
    def __init__(self, epoch = 1 ,  lr = 0.001, batch_size = 300, embedding_size = 100, l2 = 0.0004, seed = 2000):
        self.seed = seed
        init_seed(self.seed)
        
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_size = embedding_size
        self.sessionid = -1
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
        index2wiord = {}
        item_no = 1
        for key, values in session_item_train.items():
            length = len(session_item_train[key])
            for i in range(length):
                if session_item_train[key][i] in word2index:
                    session_item_train[key][i] = word2index[session_item_train[key][i]]
                else:
                    word2index[session_item_train[key][i]] = item_no
                    index2wiord[item_no] = session_item_train[key][i]
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
                    index2wiord[item_no] = session_item_test[key][i]
                    session_item_test[key][i] = item_no
                    item_no +=1
        
        features1 = []
        targets1 = []
        for value in session_item_test.values():
            for i in range(1, len(value)):
                targets1.append(value[-i])
                features1.append(value[:-i])
        
                
        item_no = item_no +1
        self.word2index = word2index
        self.index2wiord = index2wiord      
        
        print('start training: ', datetime.datetime.now())         
        
        
        train_data = (features, targets)  
        train_data = Data(train_data, shuffle=True, seed = self.seed)
        
        test_data = (features1, targets1)  
        test_data = Data(test_data, shuffle=True, seed = self.seed)
        
        lenn  = len(features1)
        model = trans_to_cuda(SessionGraph(self.lr, self.batch_size, self.hidden_size, item_no, self.l2, self.seed))
        model.scheduler.step()
        model.train()
        total_loss = 0.0
        
        slices = train_data.generate_batch(model.batch_size)
        slices1 = test_data.generate_batch(1)
        
        Mrr20List = []
        counter = 0 
        for epoch in range(self.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            for i in tqdm(slices):
                model.optimizer.zero_grad()
                targets, scores = self.forward(model, i, train_data)
                targets = trans_to_cuda(torch.Tensor(targets).long())
                loss = model.loss_function(scores, targets - 1)
                loss.backward()
                model.optimizer.step()
                total_loss += loss.item()
                # if j % int(len(slices) / 5 + 1) == 0:
                #     print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    
            print('\tLoss:\t%.3f' % total_loss)
            
            with torch.no_grad(): 
                valid_Mrr = 0 
                for index in slices1:
                    out_var, scores = self.forward(model, index, test_data)
                    sub_scores_k20_index = scores.topk(20)[1]
                    sub_scores_k20_score = scores.topk(20)[0]
                    
                    sub_scores_k20_index = trans_to_cpu(sub_scores_k20_index).detach().numpy()
                    sub_scores_k20_score = trans_to_cpu(sub_scores_k20_score).detach().numpy()
                    
                    preds = pd.Series(data = list(sub_scores_k20_score[0]), index = list(sub_scores_k20_index[0]))
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
                
            
                if  counter > 3 or epoch == (self.epoch-1):
                  print("We stop at Epoch:", epoch +1)
                  # Store the best values
                  max_value = max(Mrr20List)
                  max_index = Mrr20List.index(max_value)
                  name = "Epoch:"+str(max_index+1)+"-Lr:"+str(self.lr)+"-BatchSize:"+str(self.batch_size)+"-EmbeddingSize:"+str(self.hidden_size)+"-L2:"+str(self.l2)
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
            
        self.model = model
    def forward(self, model, i, data):
        init_seed(self.seed)
        
        alias_inputs, A, items, mask, targets = data.get_slice(i)
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        hidden = model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return targets, model.compute_scores(seq_hidden, mask)
        
    
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
       test_data = Data(temp_list, shuffle=False, seed = self.seed)
       self.model.eval()
      
       slices = test_data.generate_batch(1)
       targets, scores = self.forward(self.model, slices[0], test_data)
       
       sub_scores_k100_index = scores.topk(100)[1]
       sub_scores_k100_score = scores.topk(100)[0]
       
       sub_scores_k100_index = trans_to_cpu(sub_scores_k100_index).detach().numpy()
       sub_scores_k100_score = trans_to_cpu(sub_scores_k100_score).detach().numpy()
       
       
       tempList = []
       sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index[0]]
       for key in sub_scores_k100_index:
           tempList.append(self.index2wiord[key])
       preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
       return preds 
       
           
   
    def clear(self):
        pass
       



















        