import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
import time
import operator
# data config (all methods)

DATA_PATH = r'./data/rec22/raw/'
DATA_PATH_PROCESSED = r'./data/rec22/prepare/'


DATA_FILE = "rec2022_clicks"

# COLS=[0,1,2]
COLS = [0, 2, 3, 4]
# TYPE = 3
TYPE = 2

# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2016-05-07'

# slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 45
DAYS_SHIFT = 18
DAYS_TRAIN = 25
DAYS_TEST = 1

# retraining default config
DAYS_RETRAIN = 1
# preprocessing from original gru4rec -  uses just the last day as test
def preprocess_org(path=DATA_PATH, data_file =DATA_FILE , path_proc=DATA_PATH_PROCESSED, tes_days = DAYS_TEST,  min_item_support=MIN_ITEM_SUPPORT,
                   min_session_length=MIN_SESSION_LENGTH):
    data= load_data(path + data_file)
    data = filter_data(data, min_session_length,  min_item_support)
    train, test = split_data_only( data, tes_days)
    saveTrainData(train, path_proc)
    saveTestData(test, path_proc)
    
    
def load_data(file): 
    #load csv
    data = pd.read_csv( file+'.txt')
    data.rename(columns = {'session_id':'SessionId', 'item_id':'ItemId'}, inplace = True)
   
    
    #data['Time'] = data.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').timestamp())
    data["Time"] = data["date"].apply(lambda x: datetime.strptime(x.split(".")[0], '%Y-%m-%d %H:%M:%S').timestamp())  
    del data["date"]
    return data

def filter_data(data, min_session_length, min_items_support ):
    # train data
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_items_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data

def split_data_only( data, tes_days):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    # Last day data is used  for testing of models.
    session_train = session_max_times[session_max_times < tmax-(86400 *tes_days)].index
    session_test = session_max_times[session_max_times >= tmax-(86400 *tes_days)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
    
    return train, test    

def saveTrainData(data, path):
    filename = "rec22_train_full"
    data.to_csv(path+"fulltrain/" + filename+'.txt', sep='\t', index=False)
    
    records1By2 = int(len(data) / 2)
    records1By4 = int(len(data) / 4)
    records1By8 = int(len(data) / 8)
    records1By12 = int(len(data) / 12)
    records1By16 = int(len(data) / 16)
    records1By20 = int(len(data) / 20)
    records1By32 = int(len(data) / 32)
    records1By64 = int(len(data) / 64)
    records1By128 = int(len(data) / 128)
    records1By256 = int(len(data) / 256)
    records1By512 = int(len(data) / 512)
    records1By1024 = int(len(data) / 1024)
    
    
    train1By2 = data.iloc[:records1By2, :]
    train1By4 = data.iloc[:records1By4, :]

    train1By8 = data.iloc[:records1By8, :]
    rec15_train_tr, rec15_train_valid = split_data_only(train1By8, 1)
    
    train1By12 = data.iloc[:records1By12, :]
    train1By16 = data.iloc[:records1By16, :]
    train1By20 = data.iloc[:records1By20, :]
    train1By32 = data.iloc[:records1By32, :]
    train1By64 = data.iloc[:records1By64, :]
    train1By128 = data.iloc[:records1By128, :]
    train1By256 = data.iloc[:records1By256, :]
    train1By512 = data.iloc[:records1By512, :]
    train1By1024 = data.iloc[:records1By1024, :]
    
    # Save data.....
    train1By2.to_csv(path+"train1By2/" + filename+'.txt', sep='\t', index=False)
    train1By4.to_csv(path+"train1By4/" + filename+'.txt', sep='\t', index=False)
    train1By8.to_csv(path+"train1By8/" + filename+'.txt', sep='\t', index=False)
    rec15_train_tr.to_csv(path+"train1By8/" + "rec15_train_tr"+'.txt', sep='\t', index=False)
    rec15_train_valid.to_csv(path+"train1By8/" + "rec15_train_valid"+'.txt', sep='\t', index=False)
    
    train1By12.to_csv(path+"train1By12/" + filename+'.txt', sep='\t', index=False)
    train1By16.to_csv(path+"train1By16/" + filename+'.txt', sep='\t', index=False)
    train1By20.to_csv(path+"train1By20/" + filename+'.txt', sep='\t', index=False)
    train1By32.to_csv(path+"train1By32/" + filename+'.txt', sep='\t', index=False)
    train1By64.to_csv(path+"train1By64/" + filename+'.txt', sep='\t', index=False)
    train1By128.to_csv(path+"train1By128/" + filename+'.txt', sep='\t', index=False)
    train1By256.to_csv(path+"train1By256/" + filename+'.txt', sep='\t', index=False)
    train1By512.to_csv(path+"train1By512/" + filename+'.txt', sep='\t', index=False)
    train1By1024.to_csv(path+"train1By1024/" + filename+'.txt', sep='\t', index=False)
    
    
    
def saveTestData(data, path):
    filename = "rec22_test"
    data.to_csv(path + filename+'.txt', sep='\t', index=False)
    
    
if __name__ == '__main__':
    
    preprocess_org();
