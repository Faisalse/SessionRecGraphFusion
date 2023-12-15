import time
import pickle
from algorithms.MGS.model import *
from algorithms.MGS.utils import *
import pandas as pd
import torch
import torch.nn as nn
import datetime
import numpy as np
from tqdm import tqdm
import os

class MGS:
    def __init__(self, epoch = 1,  lr = 0.1, batch_size = 100, embedding_size = 50, l2 = 0.00005, dropout = 0.3):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.sessionid = -1
        self.l2 = l2
        self.embedding = embedding_size
        self.dropout = dropout
        self.attribute_kinds = 1
        self.phi = 2.5
        self.sample_num = 40
        self.decay_count = 0
        self.decay_num = 3
        
    def fit(self, train, test):
        session_key = "SessionId"
        item_key = "ItemId"
        CatId = "CatId"
        index_session = train.columns.get_loc( session_key)
        index_item = train.columns.get_loc( item_key )
        index_cat = train.columns.get_loc( CatId )
        combine = [train, test]
        combine = pd.concat(combine)
        
        cate_to_item = combine.groupby(CatId)[item_key].apply(list).to_dict()
        item_to_cate = combine.groupby(item_key)[CatId].apply(list).to_dict()
        
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
        
        self.product_attributes = dict()
        for m in index2wiord:
            # get original item....
            item = index2wiord[m]
            # get category
            cat = item_to_cate[item]
            itemList = cate_to_item[cat[0]]
            new_indexing_item = []
            for i in itemList:
                new_indexing_item.append(word2index[i])
            
            tem_dic = dict()
            tem_dic["category"] = cat[0]
            tem_dic["same_cate"] = new_indexing_item
            self.product_attributes[m] = tem_dic
                    
        # Assigning new indexing category
        self.num_node =  item_no
        print("number of nodes...  ",self.num_node)
        item_no = item_no
        
        self.word2index = word2index
        self.index2wiord = index2wiord 

        
        train_data = (features, targets)
        features1 = features1[:5]
        targets1 = targets1[:5]
        self.test_data = (features1, targets1)
        
        
        train_data = Data(train_data, self.product_attributes, self.attribute_kinds, self.sample_num)
        #test_data = Data(test_data, self.product_attributes, self.attribute_kinds, self.sample_num)
        
        model = trans_to_cuda(CombineGraph(self.num_node,  self.lr, self.l2, self.embedding, self.dropout))
        # start learning...... 
        for epoch in range(self.epoch):
            print('start training: ', datetime.datetime.now())
            model.train()
            total_loss = 0.0
            
            train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=self.batch_size,
                                               shuffle=True, pin_memory=False)
            
            for i, data in enumerate(tqdm(train_loader)):
                targets, scores, loss = self.forward(model, data)
                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad()
                total_loss += loss
               
            if self.decay_count < self.decay_num:
                model.scheduler.step()
                self.decay_count += 1
            
            print('\tLoss:\t%.3f' % total_loss)
        print("Faisal")
        self.model = model

    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
            
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        
        if len(self.testList) == 1:
            self.testList.append(prev_iid)
        
        # temp_list
        temp_list = []
        temp_list = ([self.testList, self.testList, self.testList, self.testList, self.testList], [prev_iid, prev_iid, prev_iid, prev_iid, prev_iid])
        
        test_data = Data(temp_list, self.product_attributes, self.attribute_kinds, self.sample_num)
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=5,
                                           shuffle=True, pin_memory=False)
        
        for i, data in enumerate(test_loader):
            targets, scores, loss = self.forward(self.model, data)
        scores = scores[0]
        sub_scores_k100_index = scores.topk(100)[1]
        sub_scores_k100_score = scores.topk(100)[0]
        sub_scores_k100_index = trans_to_cpu(sub_scores_k100_index).detach().numpy()
        sub_scores_k100_score = trans_to_cpu(sub_scores_k100_score).detach().numpy()
        
        
        tempList = []
        sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index]
        for key in sub_scores_k100_index:
             tempList.append(self.index2wiord[key])
        preds = pd.Series(data = list(sub_scores_k100_score), index = tempList)
        return preds
         
    
    def clear(self):
        pass

    def forward(self,model, data):
        adj, items, targets, last_item_mask, as_items, as_items_SSL, simi_mask = data
        items = trans_to_cuda(items).long()
        adj = trans_to_cuda(adj).float()
        last_item_mask = trans_to_cuda(last_item_mask)
        for k in range(self.attribute_kinds):
            as_items[k] = trans_to_cuda(as_items[k]).long()
            as_items_SSL[k] = trans_to_cuda(as_items_SSL[k]).long()
        targets_cal = trans_to_cuda(targets).long()
        simi_mask = trans_to_cuda(simi_mask).long()
    
        simi_loss, scores = model(items, adj, last_item_mask, as_items, as_items_SSL, simi_mask)
        loss = model.loss_function(scores, targets_cal - 1)
        loss = loss + self.phi * simi_loss
        return targets, scores, loss



# def train_test(model, opt, train_data, test_data):
#     print('start training: ', datetime.datetime.now())
#     model.train()
#     total_loss = 0.0
#     train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=opt.batch_size,
#                                                shuffle=True, pin_memory=False)
#     for i, data in enumerate(tqdm(train_loader)):
#         targets, scores, loss = forward(model, data, opt)
#         loss.backward()
#         model.optimizer.step()
#         model.optimizer.zero_grad()
#         total_loss += loss
#     print('\tLoss:\t%.3f' % total_loss)
#     if opt.decay_count < opt.decay_num:
#         model.scheduler.step()
#         opt.decay_count += 1

#     print('start predicting: ', datetime.datetime.now())
#     model.eval()
#     test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=int(opt.batch_size / 8),
#                                               shuffle=False, pin_memory=False)
#     result_20 = []
#     hit_20, mrr_20 = [], []
#     result_10 = []
#     hit_10, mrr_10 = [], []
#     for data in test_loader:
#         targets, scores, loss = forward(model, data, opt)
#         targets = targets.numpy()
#         sub_scores_20 = scores.topk(20)[1]
#         sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
#         for score, target in zip(sub_scores_20, targets):
#             hit_20.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr_20.append(0)
#             else:
#                 mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

#         sub_scores_10 = scores.topk(10)[1]
#         sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
#         for score, target in zip(sub_scores_10, targets):
#             hit_10.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr_10.append(0)
#             else:
#                 mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

#     result_20.append(np.mean(hit_20) * 100)
#     result_20.append(np.mean(mrr_20) * 100)

#     result_10.append(np.mean(hit_10) * 100)
#     result_10.append(np.mean(mrr_10) * 100)

#     return result_10, result_20

    
        