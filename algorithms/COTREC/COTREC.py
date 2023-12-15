import time
import pickle
from algorithms.COTREC.model import *
from algorithms.COTREC.util import *

import pandas as pd
import torch
import torch.nn as nn
import datetime
import numpy as np
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings('ignore')


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 



class COTRECModel:
    def __init__(self, epoch = 1,  lr = 0.1, batch_size = 100, embedding_size = 50, l2 = 0.00005, seed = 2000):
        
        
  
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.embedding = embedding_size
        self.l2 = l2
        
        self.layer = 2
        self.beta = 0.005
        self.lam = 0.005
        self.eps = 0.2
        self.sessionid = -1
        
        self.seed = seed
        init_seed(self.seed)
        
        
    
        
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
                    item_no += 1
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
        self.num_node =  item_no
        self.word2index = word2index
        self.index2wiord = index2wiord 
        train_data = (features, targets)
        
        self.features = features
        
        train_data = Data(train_data, self.features, shuffle = True, n_node = self.num_node, seed = self.seed)
        
        model = COTREC(train_data.adjacency, self.num_node, self.lr, self.layer, self.l2, self.beta, self.lam, self.eps, self.embedding, device, self.seed)
        model = model.to(device)
        
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        
        # start learning...... 
        for epoch in range(self.epoch):
            total_loss = 0.0
            model.train()
            slices = train_data.generate_batch(self.batch_size)
            for i in tqdm(slices):
                
                
                tar, session_len, session_item, reversed_sess_item, mask, diff_mask = train_data.get_slice(i)
                diff_mask = torch.Tensor(diff_mask).long()
                diff_mask = diff_mask.to(device)
                
                
                
                A_hat, D_hat = train_data.get_overlap(session_item)
                A_hat = torch.Tensor(A_hat)
                D_hat = torch.Tensor(D_hat)
                
                A_hat = A_hat.to(device)
                D_hat = D_hat.to(device)
                
                session_item = torch.Tensor(session_item).long()
                session_item = session_item.to(device)
                
                session_len = torch.Tensor(session_len).long()
                session_len = session_len.to(device)
                
                tar = torch.Tensor(tar).long()
                tar = tar.to(device)
                
                mask = torch.Tensor(mask).long()
                mask = mask.to(device)
                
                reversed_sess_item = torch.Tensor(reversed_sess_item).long()
                reversed_sess_item = reversed_sess_item.to(device)
                
                model.optimizer.zero_grad()
                con_loss, loss_item, scores_item, loss_diff = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, tar, diff_mask)
                loss = loss_item + con_loss + loss_diff
                loss.backward()
                model.optimizer.step()
                total_loss += loss.item()
        self.model = model

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
        test_data = Data(temp_list, self.features, shuffle=True, n_node=self.num_node, seed = self.seed)
        self.model.eval()
        slices = test_data.generate_batch(1)
        for i in slices:
            #tar, score, con_loss, loss_item, loss_diff = forward(self.model, i, test_data, train=False)
            tar, session_len, session_item, reversed_sess_item, mask, diff_mask = test_data.get_slice(i)
            diff_mask = torch.Tensor(diff_mask).long()
            diff_mask = diff_mask.to(device)
            
            A_hat, D_hat = test_data.get_overlap(session_item)
            A_hat = torch.Tensor(A_hat)
            D_hat = torch.Tensor(D_hat)
            A_hat = A_hat.to(device)
            D_hat = D_hat.to(device)
            
            session_item = torch.Tensor(session_item).long()
            session_len = torch.Tensor(session_len).long()
            session_item = session_item.to(device)
            session_len = session_len.to(device)
            
            
            tar = torch.Tensor(tar).long()
            mask = torch.Tensor(mask).long()
            reversed_sess_item = torch.Tensor(reversed_sess_item).long()
            
            tar = tar.to(device)
            mask = mask.to(device)
            reversed_sess_item = reversed_sess_item.to(device)
            
            
            con_loss, loss_item, score, loss_diff = self.model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask,tar, diff_mask)
            
            
            
            
        
        sub_scores_k100_index = score.topk(100)[1]
        sub_scores_k100_score = score.topk(100)[0]
        
        sub_scores_k100_index = np.array(sub_scores_k100_index.cpu())
        sub_scores_k100_score = sub_scores_k100_score.detach().cpu()
        sub_scores_k100_score = sub_scores_k100_score.detach().numpy()
        
        
        tempList = []
        sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index[0]]
        for key in sub_scores_k100_index:
            tempList.append(self.index2wiord[key])
        preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
        return preds
        

    def clear(self):
        pass


    
        