#import time
import argparse
import numpy as np
import os
import random
from algorithms.SGINM.data import *
from algorithms.SGINM.model import *
import os
DATA_PATH = r'./algorithms/SGINM/'
import pickle
from tqdm import tqdm
import torch
import pandas as pd

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


init_seed(2000)
class SGINM_Call:
    def __init__(self, epoch = 1, lr = 0.001, batch_size = 300,  embedding_size = 50, dropout = 0.1, l2=0.0001):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.l2 = l2
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.dropout = dropout
        self.short_or_long = 0
        self.sessionid = -1
    def all_datafunction(self, short_or_long, train_data, test_data):
        if short_or_long == 0:
            all_data = train_data + test_data
        else:
            tra_short, tra_long = self.split_short_long(train_data, thred=5)
            tes_short, tes_long = self.split_short_long(test_data, thred=5)
            if self.short_or_long == 1:
                all_data = tra_short + tes_short
                train_data = tra_short
                test_data = tes_short
            else:
                all_data = tra_long + tes_long
                train_data = tra_long
                test_data = tes_long
        return all_data, train_data, test_data
    
    
    def split_short_long(self, data_pairs, thred=5):
        short_pairs = []
        long_pairs = []
        for seq, lab in data_pairs:
            if len(seq) <= thred:
                short_pairs.append((seq, lab))
            else:
                long_pairs.append((seq, lab))
        print('Short session: %d, %0.2f\tLong session: %d, %0.2f' %
              (len(short_pairs), 100.0 * len(short_pairs) / len(data_pairs), len(long_pairs), 100.0 * len(long_pairs) / len(data_pairs)))
        return short_pairs, long_pairs
    
    
    def fit(self, train, test):
        
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
         
        train_data = []
        count = 0
        total_items = []
        for value in session_item_train.values():
            count += len(value)
            total_items += value
            for i in range(1, len(value)):
                tar = value[-i]
                features1 = value[:-i]
                train_data.append((features1, tar))
                
                
                
              
        
        # print("No. of items in trainig data   ", len(set(total_items)))
        # print("Size of trainig data   ", train.shape)
        # print("Number of training sessions:  ", len(session_item_train))   
        # print("Average length of training sessions: ", count/len(session_item_train))
         
                
        index_session = test.columns.get_loc( session_key)
        index_item = test.columns.get_loc( item_key ) 
        session_item_test = {}
        # Convert the session data into sequence
        for row in test.itertuples(index=False):
            if row[index_session] in session_item_test:
                session_item_test[row[index_session]] += [(row[index_item])] 
            else: 
                session_item_test[row[index_session]] = [(row[index_item])]
         
        test_data = []
        count = 0
        total_items = []
        for value in session_item_test.values():
            count +=len(value)
            total_items += value
            for i in range(1, len(value)):
                tar = value[-i]
                features2 = value[:-i]
                test_data.append((features2, tar))
                
        # print("No. of items in testing data   ", len(set(total_items)))        
        # print("Size of testing data   ", test.shape)        
        # print("Number of testing sessions:  ", len(session_item_test))   
        # print("Average length of testing sessions: ", count/len(session_item_test))        
        
        
                
        self.all_data, self.train_data,  self.test_data =  self.all_datafunction(self.short_or_long, train_data, test_data)
        self.all_data = Data(self.all_data, status='all')  
        
        self.train_data = Data(self.train_data, status='train')
        self.train_data.word2index, self.train_data.index2word, self.train_data.num_words = self.all_data.word2index, self.all_data.index2word, self.all_data.num_words
        
        self.test_data = Data(self.test_data, status='test')
        self.test_data.word2index, self.test_data.index2word, self.test_data.num_words = self.all_data.word2index, self.all_data.index2word, self.all_data.num_words
        
        
        
        self.word2index = self.all_data.word2index

        model = SGINM(self.lr, self.embedding_size, self.hidden_size, self.dropout, self.train_data.num_words, self.l2)
        model = model.to(device)
        Mrr20List = []
        counter = 0 
        for epoch in range(self.epoch):
            print('Epoch: ', epoch)
            model.train()
            model.scheduler.step()
            total_loss = []
            val_dationLoss = []
            
            # function to generate the batches.....
            slices = self.train_data.generate_batch_slices(self.batch_size)
            for i in tqdm(range(len(slices))):
                inp_var, lengths, mask0, mask1, maskinf, out_var = self.train_data.batch2TrainData(slices[i])
                inp_var = inp_var.to(device)
                lengths = lengths
                mask0 = mask0.to(device)
                mask1 = mask1.to(device)
                maskinf = maskinf.to(device)
                out_var = out_var.to(device)
                
    
                model.optimizer.zero_grad()
                score = model(inp_var, lengths, mask0, mask1, maskinf)
                loss = model.loss_function(score, out_var - 1)
                loss.backward()
                model.optimizer.step()
                total_loss.append(loss.item())
                
                
            trainLoss = np.mean(total_loss)    
            with torch.no_grad(): 
                # Calculation of MRR on training data
                # slices = self.train_data.generate_batch_slices(1)
                
                # train_Mrr = 0 
                # for i in range(len(slices)):
                #     inp_var, lengths, mask0, mask1, maskinf, out_var = self.train_data.batch2TrainData(slices[i])
                #     score = model(inp_var, lengths, mask0, mask1, maskinf)
                #     sub_scores_k20_index = score.topk(20)[1]
                #     sub_scores_k20_score = score.topk(20)[0]
                #     sub_scores_k20_index = sub_scores_k20_index.numpy()
                #     sub_scores_k20_score = sub_scores_k20_score.detach().numpy()
                #     preds = pd.Series(data = list(sub_scores_k20_score[0]), index = list(sub_scores_k20_index[0]))
                #     out_var = np.array(out_var)
                #     out_var = out_var[0] -1
                #     if out_var in preds.index:
                #         rank = preds.index.get_loc( out_var ) + 1
                #         train_Mrr += ( 1.0/rank )
                        
                slices = self.test_data.generate_batch_slices(1)
                valid_Mrr = 0 
                for i in range(len(slices)):
                    inp_var, lengths, mask0, mask1, maskinf, out_var = self.test_data.batch2TrainData(slices[i])
                    
                    inp_var = inp_var.to(device)
                    lengths = lengths
                    mask0 = mask0.to(device)
                    mask1 = mask1.to(device)
                    maskinf = maskinf.to(device)
                    
                    score = model(inp_var, lengths, mask0, mask1, maskinf)
                    
                    
                    
                    sub_scores_k20_index = score.topk(20)[1]
                    sub_scores_k20_score = score.topk(20)[0]
                    
                    sub_scores_k20_index = np.array(sub_scores_k20_index.cpu())
                    sub_scores_k20_score = sub_scores_k20_score.detach().cpu()
                    sub_scores_k20_score = sub_scores_k20_score.detach().numpy()
                    
                    
                    preds = pd.Series(data = list(sub_scores_k20_score[0]), index = list(sub_scores_k20_index[0]))
                    out_var = np.array(out_var)
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
            if counter > 4:
              print("We stop at Epoch:", epoch +1)
              # Store the best values
              max_value = max(Mrr20List)
              max_index = Mrr20List.index(max_value)
              name = "Epoch:"+str(max_index+1)+"-Lr:"+str(self.lr)+"-BatchSize:"+str(self.batch_size)+"-EmbeddingSize:"+str(self.embedding_size)+"-Dropout:"+str(self.dropout)+"-L2:"+str(self.l2)
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
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp=0):
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
        # incomming elements of a session....
        
        self.testList.append(prev_iid)
        # temp_list
        temp_list = []
        temp_list.append((self.testList, prev_iid))
        
        self.test_data = Data(temp_list, status='test')
        self.test_data.word2index, self.test_data.index2word, self.test_data.num_words = self.all_data.word2index, self.all_data.index2word, self.all_data.num_words
        
        
        self.word2index = self.all_data.word2index
        self.index2word = self.all_data.index2word
        self.model.eval()
        slices = self.test_data.generate_batch_slices(1)
        te_inp_var, te_lengths, te_mask0, te_mask1, te_maskinf, te_out_var = self.test_data.batch2TrainData(slices[0])
        te_inp_var = te_inp_var.to(device)
        te_mask0 = te_mask0.to(device)
        te_mask1 = te_mask1.to(device)
        te_maskinf = te_maskinf.to(device)
        
        score = self.model(te_inp_var, te_lengths, te_mask0, te_mask1, te_maskinf)
        
        
        
        sub_scores_k100_index = score.topk(100)[1]
        sub_scores_k100_score = score.topk(100)[0]
        
        sub_scores_k100_index = np.array(sub_scores_k100_index.cpu())
        sub_scores_k100_score = sub_scores_k100_score.detach().cpu()
        
        sub_scores_k100_score = sub_scores_k100_score.numpy()
        tempList = []
        sub_scores_k100_index = [x + 1 for x in sub_scores_k100_index[0]]
        for key in sub_scores_k100_index:
            tempList.append(self.index2word[key])
        preds = pd.Series(data = list(sub_scores_k100_score[0]), index = tempList)
        return preds
        
        # preds = pd.Series(data = list(sub_scores_k100_score[0]), index = list(sub_scores_k100_index[0]))
        # return preds
    def init_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def clear(self):
        pass



class EarlyStopping():
    def __init__(self, tolerance=1):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False

    def __call__(self, currentMRR, Pre_MRR):
        if (currentMRR < Pre_MRR[-1]):
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                
                
                
                
                
                
                
                