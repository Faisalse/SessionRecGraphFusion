#import time
import argparse
import numpy as np
import os
import random
from algorithms.FLCSP.data import *
from algorithms.FLCSP.model import *

import pickle
from tqdm import tqdm
import torch
import pandas as pd


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    



class FLCSP_cate:
    def __init__(self, epoch =  1, lr= 0.001, batch_size= 100, embedding_size = 50, hidden_size= 100, dropout= 0.1, l2 = 0.00001, seed = 2000):
        self.seed = seed
        init_seed(self.seed)
        
        
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.short_or_long = 0
        self.sessionid = -1
        self .levels = 1
        self.nhid = 50
        self.ksize = 3
        self.emb_dropout = 0.2
        self.anchor_num = 1000
        self.N = 1
        self.hid_size = 200
        self.l2 = l2
        self.lr_dc_cate = [10, 15]
        self.lr_dc = [10, 15]
        self.clip = 1.0
       
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
        
        
        number_of_unique_items = len(word2index) +1
        num_node = item_no = len(word2index) +1
        self.word2index = word2index
        self.index2word = index2word      
        # features = []
        # targets = []
        train_data = []
        for value in session_item_train.values():
            for i in range(1, len(value)):
                # targets.append(value[-i])
                # features.append(value[:-i])
                train_data.append([value[:-i], value[-i]])
        
        Trans_adj, graph = build_graph(number_of_unique_items, train_data)
        
        
        anchors = anchor_select(graph, self.anchor_num)
        prob_conver = random_walk(Trans_adj, anchors, alpha=0.5)
        
        # Load the batches 
        num_chans = [self.nhid] * (self .levels - 1) + [self.embedding_size]
        
        
        model_item = GRUTCN(number_of_unique_items, self.embedding_size, self.hidden_size, num_chans, self.ksize,
                       self.dropout, self.emb_dropout, self.lr, self.lr_dc, self.l2, device=device, seed = self.seed)
        
        
        model_cate = MutiHeadAttn(number_of_unique_items, self.anchor_num, self.N, prob_conver, self.dropout, self.hid_size, self.lr, self.l2, self.lr_dc_cate, device=device, seed = self.seed)
        
        
        model = Fusion(number_of_unique_items, device=device, seed = self.seed)
    
        model_item = model_item.to(device)
        model_cate = model_cate.to(device)
        model = model.to(device)
        
        train_data = batchify(train_data, self.batch_size, self.seed)
        for epoch in range(self.epoch):
            model_item.train()
            model_cate.train()
            model.train()
            
            # model_item.scheduler.step()
            # model_cate.scheduler.step()
            # !model.scheduler.step()
            
            
            total_loss = []
            total_loss_item = []
            total_loss_cate = []
                
            for i in tqdm(range(len(train_data))):
                inp_behind_tensor, lab_tensor = zero_padding_behind(train_data[i])

                # inp_front_tensor = inp_front_tensor.to(device)
                inp_behind_tensor = inp_behind_tensor.to(device)
                lab_tensor = lab_tensor.to(device)
        
                model_item.optimizer.zero_grad()
                model_cate.optimizer.zero_grad()
                # model.optimizer.zero_grad()
        
                item_score = model_item(inp_behind_tensor)
                cate_score = model_cate(inp_behind_tensor)
                tra_score = model(item_score, cate_score)
        
                loss_item = model_item.loss_function(item_score, lab_tensor - 1)
                loss_cate = model_cate.loss_function(cate_score, lab_tensor - 1)
                loss = model.loss_function(tra_score, lab_tensor - 1)
        
                loss_item.backward()
                if self.clip > 0:
                    self.gradClamp(model_item.parameters(), self.clip)
                model_item.optimizer.step()
        
                loss_cate.backward()
                if self.clip > 0:
                    self.gradClamp(model_cate.parameters(), self.clip)
                model_cate.optimizer.step()
        
        
                # loss.backward()
                # if args.clip > 0:
                #     gradClamp(model.parameters(), args.clip)
                # model.optimizer.step()
        
                total_loss.append(loss.item())
                total_loss_item.append(loss_item.item())
                total_loss_cate.append(loss_cate.item())
            
            print("Loss after Epoch: ", epoch)
            print('Loss:\t%0.4f' % (np.mean(total_loss)))
            print('LossItem:\t%0.4f' % np.mean(total_loss_item))
            print('LossCate:\t%0.4f' % np.mean(total_loss_cate))
            
        self.model_item = model_item
        self.model_cate = model_cate
        self.model = model
                        
    def gradClamp(self, parameters, clip=0.5):
        init_seed(self.seed)
        for p in parameters:
            p.grad.data.clamp_(min=-clip, max=clip)            
        
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp=0):
        init_seed(self.seed)
        # Set model for evaluations....
        self.model_item.eval()
        self.model_cate.eval()
        self.model.eval()
        
        
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
            
            
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        # temp_list
        temp_list = []
        temp_list.append([self.testList, prev_iid])
        
        test_data = batchify(temp_list, 1, self.seed)
        
        # for i in range(len(test_data)):
            # inp_front_tensor_tes, lab_tensor_tes = zero_padding_front(tes_data[i])
        inp_behind_tensor_tes, lab_tensor_tes = zero_padding_behind(test_data[0])
            # inp_front_tensor_tes = inp_front_tensor_tes.to(device)
        inp_behind_tensor_tes = inp_behind_tensor_tes.to(device)
        lab_tensor_tes = lab_tensor_tes.to(device)
    
        tes_score_item = self.model_item(inp_behind_tensor_tes)
        tes_score_cate = self.model_cate(inp_behind_tensor_tes)
        test_score = self.model(tes_score_item, tes_score_cate)
        
        sub_scores_k100_index = tes_score_cate.topk(100)[1]
        sub_scores_k100_score = tes_score_cate.topk(100)[0]
        
        
        sub_scores_k100_index = np.array(sub_scores_k100_index.cpu())
        sub_scores_k100_score = sub_scores_k100_score.cpu()
        sub_scores_k100_score = sub_scores_k100_score.detach().numpy()
        
        tempList = []
        sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index[0]]
        for key in sub_scores_k100_index:
            tempList.append(self.index2word[key])
        preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
        return preds
        
        
        
    def clear(self):
        pass
        
        
def trans_to_cuda_SG(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    
def trans_to_SG(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable    
    
#%%        
# obj1 =   FLCSP(0.001,300, 1, 50, 100, 0.1) 

# obj1.fit("a", "b")
