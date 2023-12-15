import pandas as pd
import torch


from tqdm import tqdm
import os
import time
from algorithms.IGT1.utils import *

from algorithms.IGT1.model1 import *
import numpy as np
import pickle
class IGT:
    def __init__(self, epoch = 1, lr = 0.1, batch_size = 100, embedding = 64, l2 = 0.00005, dropout = 0.2):
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.embedding = embedding
        self.l2 = l2
        self.dropout = dropout
        
        self.lr_dc = 0.1
        self.lr_dc_step = 5
        self.nonhybrid = "store_true"
        
        self.sessionid = -1
        
    def fit(self, train, test):
        
        session_key = "SessionId"
        item_key = "ItemId"
        Time = "Time"
        index_session = train.columns.get_loc( session_key)
        index_item = train.columns.get_loc( item_key )
        index_time = train.columns.get_loc( Time )
        
        session_item_train = {}
        session_item_train_time = {}
        # Convert the session data into sequence
        for row in train.itertuples(index=False):
            
            if row[index_session] in session_item_train:
                session_item_train[row[index_session]]       += [(row[index_item])] 
                session_item_train_time[row[index_session]]  += [(int(row[index_time]))] 
            else: 
                session_item_train[row[index_session]]      = [(row[index_item])]
                session_item_train_time[row[index_session]] = [(int(row[index_time]))] 
        
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
            if (len(value)) < 2:
                pass
            else:
                targets += [value[-1]]
                features += [value[:-1]]
       
        time0 = []    
        for value in session_item_train_time.values():
            if (len(value)) < 2:
                pass
            else:
                time0 += [value[:-1]]        
                
        session_item_test = {}
        session_item_test_time = {}
        # Convert the session data into sequence
        for row in test.itertuples(index=False):
            if row[index_session] in session_item_test:
                session_item_test[row[index_session]] += [(row[index_item])] 
                session_item_test_time[row[index_session]] += [(int(row[index_time]))] 
            else: 
                session_item_test[row[index_session]] = [(row[index_item])]
                session_item_test_time[row[index_session]] = [(int(row[index_time]))] 
                
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
        
       
        # Assigning new indexing category
        self.num_node =  item_no
        self.word2index = word2index
        self.index2wiord = index2wiord 
        
        tr_deal_items, tr_deal_times = self.process_times(features, time0)
        #te_deal_items, te_deal_times = self.process_times(features1, time1)
        
        train_data = (features, targets, tr_deal_times, tr_deal_items)
        # test_data = (features1, targets1, te_deal_times, te_deal_items)
        train_data = Data1(train_data, shuffle=False)
        # test_data = Data1(test_data, shuffle=False)
        model = SessionGraph(self.lr, self.embedding, self.l2, self.dropout, self.lr_dc, self.lr_dc_step, self.nonhybrid,  self.num_node)
        Mrr20List = []
        counter = 0
        # start learning...... 
        for epoch in range(self.epoch):
            model.train()
            total_loss = 0.0
            #　调用util中的generate_batch()函数
            print("Batch size  ", self.batch_size)
            slices = train_data.generate_batch(self.batch_size) # batch_size=100，slices为一个大小为100的数组
            for i, j in zip(slices, tqdm(np.arange(len(slices)))): # np.arange()一个参数时，参数值为终点，起点取默认值0，步长取默认值1
                model.optimizer.zero_grad()
                # 调用本文件中的forward()函数
                targets, scores = forward(model, i, train_data, self.batch_size) # 掉用forward3
                targets = torch.Tensor(targets).long()
                loss = model.loss_function(scores, targets - 1)
                loss.backward()
                model.optimizer.step()
                total_loss += loss
                
                
            print('\tLoss:\t%.3f' % total_loss)
            print('start predicting: ', datetime.datetime.now())
                 
        self.model = model
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        if(sid !=self.sessionid):
            self.testList = []
            self.testtime = []
            self.sessionid = sid
            
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        self.testtime.append(timestamp)
        

        temp1 = self.testList
        temptime = self.testtime
        if len(self.testList) < 2:
            first = prev_iid + 1
            second = first + 1
            temp1.append(first)
            temp1.append(second)
            
            first_time = timestamp + 50000 
            second_time = first_time + 50000
            temptime.append(first_time)
            temptime.append(second_time)
        
        # temp_list
        temp_list = [temp1]
        temp_target = [prev_iid]
        temp_time = [temptime]
    
        te_deal_items, te_deal_times = self.process_times(temp_list, temp_time)
        test_data = (temp_list, temp_target, te_deal_times, te_deal_items)
        test_data = Data1(test_data, shuffle=False)
        
        self.model.eval()
        slices = test_data.generate_batch(1) # batch_size=100，slices为一个大小为100的数组
        targets, scores = forward(self.model, [0], test_data, 1)
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
    

    def process_times(self, p_seqs, p_times):
        deal_times = []
        deal_items = []
        for i in range(len(p_seqs)):
            # print(p_seqs[i], p_times[i])
            if len(p_seqs[i]) == len(p_times[i]):
                tmp_list = []
                tmp_dict = dict()
                tis = p_times[i]
                if len(p_seqs[i]) == 1:
                    deal_times.append(p_times[i])
                    deal_items.append(p_seqs[i])

                else:
                    for j, it in enumerate(p_seqs[i]):
                        if it not in tmp_list:
                            tmp_list.append(it)

                        if it in tmp_dict:
                            tmp_dict[it] += [tis[j]]
                        else:
                            tmp_dict[it] = [tis[j]]
                    # print(tmp_dict)
                    one_time = []

                    for tp in tmp_list:
                        if len(tmp_dict[tp]) == 1:
                            one_time += tmp_dict[tp]
                        else:
                            mean_time = [int(np.mean(np.array(tmp_dict[tp])))]
                            tmp_dict[tp] = mean_time
                            one_time += mean_time
                    # if len(one_time) == len(tmp_list):
                    deal_times.append(one_time)
                    deal_items.append(tmp_list)

            else:
                print('________________________________', 'error')
                print(p_seqs[i])
                print(p_times[i])
        for i in range(len(deal_items)):
            # print('p_seqs', p_seqs[i])
            # print('deal_items',deal_items[i])
            # print('deal_times',deal_times[i])
            t_input = list(np.unique(np.array(p_seqs[i])))
            if len(t_input) == len(deal_items[i]) and len(t_input) == len(deal_times[i]):
                continue
            else:
                print('出错')
                print(p_seqs[i])
                print(deal_items[i])
                print(deal_times[i])
        return deal_items, deal_times
    
    def clear(self):
        pass



    
        