import time
import pickle
from algorithms.CM_HGCN.model import *
from algorithms.CM_HGCN.utils import *
import pandas as pd
import torch
import torch.nn as nn
import datetime
import numpy as np
from tqdm import tqdm
import os
DATA_PATH = r'./algorithms/CM_HGCN/'



def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class CM_HGCN:
    def __init__(self, epoch = 1,  lr = 0.1, batch_size = 100, embedding_size = 50, l2 = 0.00005, seed = 2000):
        self.seed = seed
        init_seed(self.seed)
        
        
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.sessionid = -1
        self.l2 = l2
        self.embedding = embedding_size
        
    def fit(self, train, test):
        init_seed(self.seed)
        
        
        session_key = "SessionId"
        item_key = "ItemId"
        CatId = "CatId"
        index_session = train.columns.get_loc( session_key)
        index_item = train.columns.get_loc( item_key )
        index_cat = train.columns.get_loc( CatId )
        
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
                targets += [value[-i]]
                features += [value[:-i]]
                
                
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
                targets1 += [value[-i]]
                features1 += [value[:-i]]
        
        
        # old item 2 category and category 2 item    
        item2Cate = {}
        for row in train.itertuples(index=False):
            item2Cate[row[index_item]] = row[index_cat]
        
        for row in test.itertuples(index=False):
            item2Cate[row[index_item]] = row[index_cat]
            
        # Assigning new indexing category

        self.num_node =  item_no
        item_no = item_no
        n_category = 1
        Cate2index = {}
        
        for key,value in item2Cate.items():
            if value not in Cate2index:
                Cate2index[value] = item_no
                item_no +=1
                n_category += 1
        
        
        item2CateNew = {}
        for key, value in word2index.items():
            oldcate = item2Cate[key]
            newcategory = Cate2index[oldcate]
            item2CateNew[value] = newcategory
   
        self.word2index = word2index
        self.index2wiord = index2wiord 
        self.category = item2CateNew
        self.n_category = n_category
        
        train_data = (features, targets)
        train_data = Data(train_data, self.category, self.seed)
        
        test_data = (features1, targets1)
        test_data = Data(test_data, self.category, self.seed)
        
        
        model = trans_to_cuda(CombineGraph(self.num_node + self.n_category-1, self.n_category, self.category, self.lr, self.batch_size, self.l2, self.embedding, self.seed))
        Mrr20List = []
        counter = 0
        # start learning...... 
        for epoch in range(self.epoch):
            print('start training: ', datetime.datetime.now())
            model.train()
            total_loss = 0.0
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                       shuffle=True, pin_memory=True)
            validation_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
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
                    out_var = out_var.numpy()
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
                 # counter > 4 or            
                print("test_Mrr20  ", valid_Mrr) 
                if  counter > 4:
                  print("We stop at Epoch:", epoch +1)
                  # Store the best values
                  max_value = max(Mrr20List)
                  max_index = Mrr20List.index(max_value)
                  name = "Epoch:"+str(max_index+1)+"-Lr:"+str(self.lr)+"-BatchSize:"+str(self.batch_size)+"-embedding:"+str(self.embedding)+"-L2:"+str(self.l2)
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
        test_data = Data(temp_list, self.category)
        self.model.eval()
        
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
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
            tempList.append(self.index2wiord[key])
        preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
        return preds
        
    
    
    
    def forward(self, model, data):
        init_seed(self.seed)
        
        (alias_inputs, adj, items, mask, targets, inputs, alias_inputs_ID, adj_ID, items_ID, 
        alias_items, alias_category, total_adj, total_items)= data
        
        
        alias_items = trans_to_cuda( alias_items).long()
        alias_category = trans_to_cuda(alias_category).long()
        total_adj = trans_to_cuda(total_adj).float()
        total_items = trans_to_cuda(total_items).long()
        
        alias_inputs_ID = trans_to_cuda(alias_inputs_ID).long()
        items_ID = trans_to_cuda(items_ID).long()
        adj_ID = trans_to_cuda(adj_ID).float()
        
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        adj = trans_to_cuda(adj).float()
        mask = trans_to_cuda(mask).long()
        inputs = trans_to_cuda(inputs).long()
    
        hidden1, hidden2, hidden_mix = model(items, adj, mask, inputs,  items_ID, adj_ID , total_items, total_adj)
        
        get1 = lambda i: hidden1[i][alias_inputs[i]]     
        seq_hidden1 = torch.stack([get1(i) for i in torch.arange(len(alias_inputs)).long()])
        get2 = lambda i: hidden2[i][alias_inputs_ID[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
        seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs_ID)).long()])
        
        get1_mix = lambda i: hidden_mix[i][alias_items[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
        seq_hidden1_mix = torch.stack([get1_mix(i) for i in torch.arange(len(alias_items)).long()])
        get2_mix = lambda i: hidden_mix[i][alias_category[i]]      #alias_inputs表示的是每个session中按序点击的商品在此session的“item”列表中所对应的相对位置
        seq_hidden2_mix = torch.stack([get2_mix(i) for i in torch.arange(len(alias_category)).long()])
        # Error is here.....
        return targets, model.compute_scores(seq_hidden1,seq_hidden2,seq_hidden1_mix, seq_hidden2_mix, mask)
    
    def clear(self):
        pass



    
        