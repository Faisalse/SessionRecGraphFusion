# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

#data config (all methods)
DATA_PATH = r'./data/eCommerce/raw/'
DATA_PATH_PROCESSED = r'./data/eCommerce/prepare/Nov/'


DATA_FILE_original = '2020-Jan_original'

DATA_FILE = '2019-Nov_raw'

SESSION_LENGTH = 30 * 60 #30 minutes

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#slicing default config
DAYS_TRAIN = 9
DAYS_TEST = 3




def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, 
                      days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    data = load_data_only( path+file )
    data = filter_data( data, min_item_support, min_session_length)
    train, test = dataSpilting(data, days_test)
    saveTrainData(train, path_proc)
    saveTestData(test, path_proc)

def preprocess_original_data( path=DATA_PATH, file=DATA_FILE_original, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, 
                      days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    data = load_data( path+file )
    saveTrainData_or(data, DATA_PATH)

def load_data_only(file):
    data = pd.read_csv( file+'.txt', encoding='utf-8-sig', sep='\s*,\s*', engine='python')
    data.columns = data.columns.str.strip()
    return data
  
     
def load_data( file ) : 
    #load csv
    data = pd.read_csv( file+'.csv')
    ab = data.head()
    #data = data.iloc[:10000, :]
    del data["user_session"]
    del data["category_code"]
    data.dropna(how='all', inplace = True)
    print("shape   ", data.shape)
    data["Time"] = data["event_time"].apply(lambda x: datetime.strptime(x.split(" UTC")[0], '%Y-%m-%d %H:%M:%S').timestamp())    
    del data["event_time"]
    data["Time"] = (data.Time /1).astype( int )
    data.rename(columns = {'event_type':'EventType', 'product_id':'ItemId', 'category_id':'CatId', 'brand':'Brand', 'user_id':'UserId', 'price':'Price'}, inplace = True)
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)
    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids
    data.sort_values(['SessionId','Time'], ascending=True, inplace=True)
    return data


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH): 
    print("Faisal")
    session_lengths = data.groupby('SessionId').size()
    
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths <30 ].index)]
    #filter item support
    print("Faisal")
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    # data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    # data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    # print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
    #       format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    return data



def dataSpilting(data, TestDays):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - (86400 * TestDays)].index
    session_test = session_max_times[session_max_times >= tmax - (86400 * TestDays)].index
    
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    
    # training data
    test = data[np.in1d(data.SessionId, session_test)]
    # All test items must be appear in traing items..... 
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    return train, test

def saveTrainData_or(data, path):
    fileName = "2020-Jan_raw"
    data.to_csv(path + fileName +'.txt', sep='\t', index=False)


def saveTrainData(data, path):
    fileName = "eCommerce_train_full"
    data.to_csv(path+"fulltrain/" + fileName+'.txt', sep='\t', index=False)
    records1By2 = int(len(data) / 2)
    records1By4 = int(len(data) / 4)
    records1By8 = int(len(data) / 8)
    records1By12 = int(len(data) / 12)
    records1By16 = int(len(data) / 16)
    records1By20 = int(len(data) / 20)
    records1By64 = int(len(data) / 64)
    records1By128 = int(len(data) / 128)
    records1By256 = int(len(data) / 256)
    records1By512 = int(len(data) / 512)
    records1By1024 = int(len(data) / 1024)
    
    train1By2 = data.iloc[:records1By2, :]
    train1By4 = data.iloc[:records1By4, :]
    train1By8 = data.iloc[:records1By8, :]
    train1By12 = data.iloc[:records1By12, :]
    train1By16 = data.iloc[:records1By16, :]
    train1By20 = data.iloc[:records1By20, :]
    train1By64 = data.iloc[:records1By64, :]
    train1By128 = data.iloc[:records1By128, :]
    train1By256 = data.iloc[:records1By256, :]
    train1By512 = data.iloc[:records1By512, :]
    train1By1024 = data.iloc[:records1By1024, :]
    
    # Save data.....
    train1By2.to_csv(path+"train1By2/" + fileName+'.txt', sep='\t', index=False)
    train1By4.to_csv(path+"train1By4/" + fileName+'.txt', sep='\t', index=False)
    train1By8.to_csv(path+"train1By8/" + fileName+'.txt', sep='\t', index=False)
    train1By12.to_csv(path+"train1By12/" + fileName+'.txt', sep='\t', index=False)
    train1By16.to_csv(path+"train1By16/" + fileName+'.txt', sep='\t', index=False)
    train1By20.to_csv(path+"train1By20/" + fileName+'.txt', sep='\t', index=False)
    train1By64.to_csv(path+"train1By64/" + fileName+'.txt', sep='\t', index=False)
    train1By128.to_csv(path+"train1By128/" + fileName+'.txt', sep='\t', index=False)
    train1By256.to_csv(path+"train1By256/" + fileName+'.txt', sep='\t', index=False)
    train1By512.to_csv(path+"train1By512/" + fileName+'.txt', sep='\t', index=False)
    train1By1024.to_csv(path+"train1By1024/" + fileName+'.txt', sep='\t', index=False)
    
    
    
def saveTestData(data, path):
    fileName = "eCommerce_test"
    data.to_csv(path + fileName+'.txt', sep='\t', index=False)
    



if __name__ == '__main__':
    preprocess_slices()