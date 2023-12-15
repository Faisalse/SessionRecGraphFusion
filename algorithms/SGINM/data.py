# -*- coding: utf-8 -*-

import torch
import itertools
import numpy as np

# Default word tokens
PAD_token = 0  # Used for padding short sentences

class Data(object):

    def __init__(self, data_pairs, status='train'):
        
        self.data_pairs = data_pairs
        self.data_length = len(self.data_pairs)
        self.num_words = 1
        self.word2index = {}
        self.index2word = {PAD_token: 'PAD'}
        if status == 'all':
            #this function is called for embedding of all items.... in worlds to make dictionary.... 
            self.travel_around_data()

    def travel_around_data(self):
        for seq, lab in self.data_pairs:
            seq = seq + [lab]
            for s in seq:
                if s not in self.word2index:
                    self.word2index[s] = self.num_words
                    self.index2word[self.num_words] = s
                    self.num_words += 1
                    

    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence]

    # This function is used to generate the batche size::: Number of training samples used in one batch....
    def generate_batch_slices(self, batch_size):
        n_batch = int(self.data_length / batch_size)
        if self.data_length % batch_size != 0:
            n_batch += 1
            
        slices = [0] * n_batch
        
        for i in range(n_batch):
            
            if i != n_batch - 1:
                slices[i] = self.data_pairs[i*batch_size:(i+1)*batch_size]
                
            else:
                
                slices[i] = self.data_pairs[i*batch_size:]
            
        return slices

    def inputVar(self, l):
        indexes_batch = [self.indexesFromSentence(sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        max_len = len(indexes_batch[0])
        mask0 = np.zeros((len(indexes_batch), max_len), dtype=np.float32)
        mask1 = []
        mask_inf = np.full((len(indexes_batch), max_len), float('-inf'), dtype=np.float32)
        
        for i in range(len(indexes_batch)):
            mask0[i, :len(indexes_batch[i])] = 1.0
            mask1.append(len(indexes_batch[i]))
            mask_inf[i, :len(indexes_batch[i])] = 0.0

        indexes_pad = self.zeroPadding(indexes_batch)
        
        indexes_var = torch.LongTensor(indexes_pad)
        
        mask0_var = torch.FloatTensor(mask0)
        mask1_var = torch.FloatTensor(mask1)
        maskinf_var = torch.FloatTensor(mask_inf)

        return indexes_var, lengths, mask0_var, mask1_var, maskinf_var

    def outputVar(self, l):
        indexes_l = [self.word2index[word] for word in l]
        labelVar = torch.LongTensor(indexes_l)
        return labelVar

    def batch2TrainData(self, pair_batch):
        
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        input_batch, output_batch = [], []
        for pair in pair_batch:
            # seperate the features and labels..... 
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp_var, lengths, mask0, mask1, mask_inf = self.inputVar(input_batch)
        
        out_var = self.outputVar(output_batch)
        return inp_var, lengths, mask0, mask1, mask_inf, out_var









