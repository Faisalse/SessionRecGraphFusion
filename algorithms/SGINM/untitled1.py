# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:46:33 2022

@author: shefai
"""
import csv
import pandas as pd
import operator
 
train = pd.read_csv(r"C:\Users\shefai\Desktop\PhDProjects\SBR\session-rec-master\data\30music\raw/30music-200ks.txt")
#%%
train.sort_values(by=['Time'])
item_dic = {}
# Convert the session data into sequence
for index, row in train.iterrows():
    if row["SessionId"] in item_dic:
        item_dic[row["SessionId"]] += [(row["ItemId"])] 
    else: 
        item_dic[row["SessionId"]] = [(row["ItemId"])]

train_data = []     
for value in item_dic.values():
    for i in range(1, len(value)):
        tar = value[i]
        features = value[:-i]
        train_data.append((features, tar))
#%%      

trai = train.iloc[:100000, :]

trai.to_csv("normal_train_full.txt", sep = "\t", index = False)
#%%      

import torch
  
# define tensor
tens = torch.Tensor([5.344, 8.343, -2.398, -0.995, 5, 30.421])
print("Original tensor: ", tens)
  
# find top 2 elements
values = torch.topk(tens, 2)[0]
  
# print top 2 elements
print("Top 2 element values:", values)
  
indexes = torch.topk(tens, 2)[1]
    
# print index of top 2 elements
print("Top 2 element indices:", indexes)

#%%   

a = {2:"Faisal"}