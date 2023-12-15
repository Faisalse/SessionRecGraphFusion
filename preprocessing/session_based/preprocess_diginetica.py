import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
import time
import pickle
import operator
# data config (all methods)
DATA_PATH = r'./data/diginetica/raw/'
DATA_PATH_PROCESSED = r'./data/diginetica/prepare/'
PRODUCT_FILE = "product_categories"
PRODUCT_PRICE = "products"
# DATA_FILE = 'yoochoose-clicks-10M'
# DATA_FILE = 'train-clicks'
# MAP_FILE = 'train-queries'
# MAP_FILE2 = 'train-item-views'
DATA_FILE = 'train-item-views'

# COLS=[0,1,2]
COLS = [0, 2, 3, 4]
# TYPE = 3
TYPE = 2
# filtering config (all methods)
MIN_SESSION_LENGTH = 10
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2016-05-07'

# days test default config
DAYS_TEST = 7

# slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 45
DAYS_SHIFT = 18
DAYS_TRAIN = 25
DAYS_TEST = 7

# retraining default config
DAYS_RETRAIN = 1
# preprocessing from original gru4rec -  uses just the last day as test
def preprocess_org(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                   min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    split_data_org(data, path_proc + file)

def preprocess_HCGCNN(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                   min_session_length=MIN_SESSION_LENGTH, product_file=  PRODUCT_FILE, item_price_token = PRODUCT_PRICE,  test_day = DAYS_TEST):
    
    dataset = load_data_HCGCNN(path + file+".csv")
    # In case of HGCNN min_item_support = 5, min_session_length = 2
    item_cate_dic = load_data_make_dictionary(path+product_file)
    
    item_price_dic, item_name_token_dic =  load_items_price_name_tokens(path+item_price_token)
    
    dataset = filter_items_HCGCNN(dataset, min_item_support, min_session_length, item_cate_dic, item_price_dic, item_name_token_dic)
    
    # Test days for HGCNN....
    train, test, train_tr, valid  = data_spilting_HCGCNN(dataset, test_day)
    #saveTraiValidationData(train_tr, valid, path_proc)
    saveTrainData(train, test, train_tr, valid,   path_proc)
    #saveTestData(test, path_proc)
   

# preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                            min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                            min_date=MIN_DATE, days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_min_date(data, min_date)
    data = filter_data(data, min_item_support, min_session_length)
    split_data(data, path_proc + file, days_test)


# preprocessing adapted from original gru4rec
def preprocess_days_test(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                         min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)

    split_data(data, path_proc + file, days_test)


# preprocessing to create data slices with a window
def preprocess_slices(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                      min_session_length=MIN_SESSION_LENGTH,
                      num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN,
                      days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    slice_data(data, path_proc + file, num_slices, days_offset, days_shift, days_train, days_test)


# just load and show info
def preprocess_info(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)


def preprocess_save(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    data.to_csv(path_proc + file + '_preprocessed.txt', sep='\t', index=False)


# just load and show info
def preprocess_buys(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED):
    data = load_data(path + file)
    data.to_csv(path_proc + file + '.txt', sep='\t', index=False)

def load_data_HCGCNN(file):
    timestampp = list()
    dataset = pd.read_csv(file, sep =";")
    
    with open(file, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        for data in reader:  
            Time = time.mktime(time.strptime(data['eventdate'], '%Y-%m-%d'))
            timestampp.append(Time)   
        dataset["Time"] = timestampp
    # change the columns name...
    dataset.rename(columns = {"sessionId":"SessionId", 'itemId':'ItemId'}, inplace = True)   
    del dataset["userId"]
    return dataset
    
def filter_items_HCGCNN(dataset, item_supports, session_lenth, item_cate_dic, price, token):
    # remove sessions with one lenth
    session_lengths = dataset.groupby('SessionId').size()
    dataset = dataset[np.in1d(dataset.SessionId, session_lengths[ session_lengths>= session_lenth ].index)]   

    # filterour out items ids that appear less than five times...
    items_lengths = dataset.groupby('ItemId').size()
    dataset = dataset[np.in1d(dataset.ItemId, items_lengths[ items_lengths>= item_supports ].index)]
    item_property_dic = item_cate_dic
    
    itemslist = list(dataset["ItemId"])
    catelist = []
    for item in itemslist:
        catelist.append(item_property_dic[item])
        
    pricelist = []
    for item in itemslist:
        pricelist.append(price[item])
    
    
    tokenlist = []
    for item in itemslist:
        tokenlist.append(token[item])
        
    
    dataset["CatId"] = catelist
    dataset["Price"] = pricelist
    dataset["Name"] = tokenlist
    
    return dataset


def load_data_make_dictionary(file):
    dataset = pd.read_csv(file+".csv", sep =";")
    print("data shape:   ", dataset.shape)
    
    item_property_dic = {}
    for i in range(len(dataset)):
        if dataset.iloc[i,0] not in item_property_dic:
            item_property_dic[dataset.iloc[i,0]] = dataset.iloc[i,1]
            
    return item_property_dic

def load_items_price_name_tokens(file):
    dataset = pd.read_csv(file+".csv", sep =";")
    print("data shape:   ", dataset.shape)
    
    item_price_dic = {}
    item_name_token = {}
    for i in range(len(dataset)):
        if dataset.iloc[i,0] not in item_price_dic:
            item_price_dic[dataset.iloc[i,0]] = dataset.iloc[i,1]
            item_name_token[dataset.iloc[i,0]] = dataset.iloc[i,2]
    return item_price_dic, item_name_token


def data_spilting_HCGCNN(data, test_days):  
    data['Time'] = data.Time.astype(float)
    del data["timeframe"]
    del data["eventdate"]
    
    
    data = short(data)
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - ( 86400 * test_days)].index
    session_test = session_max_times[session_max_times >= tmax - (86400 * test_days)].index
    
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)] 
    
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId','Time'], inplace = True)
    
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    
    
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    
    # data_sorting by time.....
    return train, test, train_tr, valid

# long sessions
def long(data):
    df1 = data.groupby('SessionId').filter(lambda x: len(x) > 9)    
    g = df1.groupby("SessionId").size()
    return df1
    
# medium sessions   
def medium(data):
    df1 = data.groupby('SessionId').filter(lambda x: len(x) < 10 and len(x) >=4)
    g = df1.groupby("SessionId").size()
    return df1

# short sessions
def short(data):
    df1 = data.groupby('SessionId').filter(lambda x: len(x) < 4 and len(x) > 1)
    g = df1.groupby("SessionId").size()
    return df1

        
def saveTrainData(train, test, train_tr, valid, path):
    trainName = "diginetica_train_full"
    testName = "diginetica_test"
    train_tr_Name = "diginetica_train_tr"
    valid_name = "diginetica_train_valid"
    
    train.to_csv(path+"fulltrain/" + trainName+'.txt', sep='\t', index=False)
    test.to_csv(path+"fulltrain/" + testName+'.txt', sep='\t', index=False)
    train_tr.to_csv(path+"fulltrain/" + train_tr_Name+'.txt', sep='\t', index=False)
    valid.to_csv(path+"fulltrain/" + valid_name+'.txt', sep='\t', index=False)
    
    # records1By2 = int(len(data) / 2)
    # records1By4 = int(len(data) / 4)
    # records1By8 = int(len(data) / 8)
    # records1By12 = int(len(data) / 12)
    # records1By16 = int(len(data) / 16)
    # records1By20 = int(len(data) / 20)
    # records1By32 = int(len(data) / 32)
    # records1By64 = int(len(data) / 64)
    # records1By128 = int(len(data) / 128)
    # records1By256 = int(len(data) / 256)
    # records1By512 = int(len(data) / 512)
    # records1By1024 = int(len(data) / 1024)
    
    # train1By2 = data.iloc[:records1By2, :]
    # train1By4 = data.iloc[:records1By4, :]
    # train1By8 = data.iloc[:records1By8, :]
    # train1By12 = data.iloc[:records1By12, :]
    # train1By16 = data.iloc[:records1By16, :]
    # train1By20 = data.iloc[:records1By20, :]
    # train1By32 = data.iloc[:records1By32, :]
    # train1By64 = data.iloc[:records1By64, :]
    # train1By128 = data.iloc[:records1By128, :]
    # train1By256 = data.iloc[:records1By256, :]
    # train1By512 = data.iloc[:records1By512, :]
    # train1By1024 = data.iloc[:records1By1024, :]
    
    # Save data.....
    # train1By2.to_csv(path+"train1By2/" + filename+'.txt', sep='\t', index=False)
    # train1By4.to_csv(path+"train1By4/" + filename+'.txt', sep='\t', index=False)
    # train1By8.to_csv(path+"train1By8/" + filename+'.txt', sep='\t', index=False)
    # train1By12.to_csv(path+"train1By12/" + filename+'.txt', sep='\t', index=False)
    # train1By16.to_csv(path+"train1By16/" + filename+'.txt', sep='\t', index=False)
    # train1By20.to_csv(path+"train1By20/" + filename+'.txt', sep='\t', index=False)
    # train1By32.to_csv(path+"train1By32/" + filename+'.txt', sep='\t', index=False)
    # train1By64.to_csv(path+"train1By64/" + filename+'.txt', sep='\t', index=False)
    # train1By128.to_csv(path+"train1By128/" + filename+'.txt', sep='\t', index=False)
    # train1By256.to_csv(path+"train1By256/" + filename+'.txt', sep='\t', index=False)
    # train1By512.to_csv(path+"train1By512/" + filename+'.txt', sep='\t', index=False)
    # train1By1024.to_csv(path+"train1By1024/" + filename+'.txt', sep='\t', index=False)
    
def saveTestData(data, path):
    filename = "diginetica_test"
    data.to_csv(path + filename+'.txt', sep='\t', index=False)
    
    
def saveTraiValidationData(train_tr, valid, path):  
    trainfile = "diginetica_train_tr"
    validfile = "diginetica_train_valid"
    train_tr.to_csv(path + trainfile+'.txt', sep='\t', index=False)
    valid.to_csv(path+ validfile+'.txt', sep='\t', index=False)

def load_data(file):
    if TYPE is 1:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int64, 1: np.int64, 2: np.int64})
        mapping = pd.read_csv(DATA_PATH + MAP_FILE + '.csv', sep=';', usecols=[0, 1, 4, 5],
                              dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        mapping2 = pd.read_csv(DATA_PATH + MAP_FILE2 + '.csv', sep=';', usecols=[0, 2, 3, 4],
                               dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        # specify header names
        data.columns = ['QueryId', 'Time', 'ItemId']
        mapping.columns = ['QueryId', 'SessionId', 'Time2', 'Date']
        # mapping2.columns = ['SessionId', 'ItemId','Time3','Date2']

        data = data.merge(mapping, on='QueryId', how='inner')
        # data = data.merge( mapping2, on=['SessionId','ItemId'], how='outer' )
        del mapping
        # del mapping2
        data.to_csv(file + '.1.csv', index=False)

        # convert time string to timestamp and remove the original column
        #         start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        #         data['Time'] = (data['Time'] / 1000) + start.timestamp()
        #         data['TimeO'] = data.Time.apply( lambda x: datetime.fromtimestamp( x, timezone.utc ) )
        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['TimeNew'] = (data['Time2'] / 1000) + data['Datestamp'] + (data['Time'] / 1000000)
        data['TimeO'] = data.TimeNew.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))
        data.to_csv(file + '.2.csv', index=False)
        print("Columns name: ", data.columns)
        data['Time'] = data['TimeNew']
        data.sort_values(['SessionId', 'TimeNew'], inplace=True)
        print("Columns name: ", data.columns)


    elif TYPE is 2:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int32, 1: np.int64, 2: np.int64, 3: str})
        # specify header names

        # data.columns = ['SessionId', 'Time', 'ItemId','Date']
        data.columns = ['SessionId', 'ItemId', 'Time', 'Date']
        #print("Before Data type 2", data)
        data = data[['SessionId', 'Time', 'ItemId', 'Date']]
        #print("Data type 2", data)
        data['Time'] = data.Time.fillna(0).astype(np.int64)
        print("Columns name: ", data.columns)
        # convert time string to timestamp and remove the original column
        # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['Time'] = (data['Time'] / 1000)
        data['Time'] = data['Time'] + data['Datestamp']
        data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))
        print("Columns name: ", data.columns)
    elif TYPE is 4:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, header=0, dtype={0: np.int32, 1: np.int64, 2: np.int32, 3: str})
        # specify header names
        # data.columns = ['sessionId', 'TimeStr', 'itemId']
        data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.+
        data['SessionId'] = data['sessionId']
        data['ItemId'] = data['itemId']
        del data['itemId'], data['sessionId'], data['eventdate']
        
        data['TimeAdd'] = 1
        data['TimeAdd'] = data.groupby('SessionId').TimeAdd.cumsum()
        data['Time'] += data['TimeAdd']
        print(data)
        del data['TimeAdd']

    elif TYPE is 3:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int64, 1: np.int64, 2: np.int64})
        mapping = pd.read_csv(DATA_PATH + MAP_FILE + '.csv', sep=';', usecols=[0, 1, 4, 5],
                              dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        # specify header names
        data.columns = ['QueryId', 'Time', 'ItemId']
        mapping.columns = ['QueryId', 'SessionId', 'Time2', 'Date']

        data = data.merge(mapping, on='QueryId', how='inner')
        # data = data.merge( mapping2, on=['SessionId','ItemId'], how='outer' )
        del mapping

        # convert time string to timestamp and remove the original column
        # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        # data['Time'] = (data['Time'] / 1000) + start.timestamp()
        # data['TimeO'] = data.Time.apply( lambda x: datetime.fromtimestamp( x, timezone.utc ) )

        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['Time'] = (data['Time'] / 1000000) + data['Datestamp']
        data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))

        print(data)
        data['SessionId'] = data['QueryId']
        data.sort_values(['QueryId', 'Time'], inplace=True)

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data = data.groupby('SessionId').apply(lambda x: x.sort_values('Time'))     # data = data.sort_values(['SessionId'],['Time'])
    data.index = data.index.get_level_values(1)
    return data;


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH):
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    session_lengths = session_lengths[ session_lengths >= min_session_length ]
    data = data[np.in1d(data.SessionId, session_lengths.index)]

    # filter item support
    data['ItemSupport'] = data.groupby('ItemId')['ItemId'].transform('count')
    data = data[data.ItemSupport >= min_item_support]

    # filter session length again, after filtering items
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    
    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set default \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def filter_min_date(data, min_date='2014-04-01'):
    
    print('filter_min_date')
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    # filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[session_max_times > min_datetime.timestamp()].index

    data = data[np.in1d(data.SessionId, session_keep)]

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set min date \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def split_data_org(data, output_file):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times >= tmax - 86400].index
    
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    
    
    
    
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)


def split_data(data, output_file, days_test = DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    valid_from = data_end - timedelta(days=days_test)
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)

def slice_data(data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ):
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))

    start = datetime.fromtimestamp(data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(slice_id) + '.txt', sep='\t', index=False)

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(),
                 end.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)


# def retrain_data(data, output_file_path, output_file_name, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
def retrain_data(data, output_file, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
    retrain_num = int(days_test/days_retrain)
    for retrain_n in range(0, retrain_num):
        # output_f = output_file_path + 'set_' + str(retrain_n) + '/' + output_file_name
        # split_data_retrain(data, output_file, days_train, days_retrain, retrain_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)
        train = split_data_retrain_train(data, output_file, days_train, days_retrain, retrain_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)
        test_set_num = retrain_num - retrain_n
        for test_n in range(0,test_set_num):
            split_data_retrain_test(data, train, output_file, days_train, days_retrain, retrain_n, test_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)


def split_data_retrain_train(data, output_file, days_train, days_test, retrain_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    train_from = data_start
    new_days = retrain_num * days_test
    train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    # todo: test_from
    # test_to = train_to + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index
    # session_test = session_max_times[(session_max_times > train_to.timestamp()) & (session_max_times <= test_to.timestamp())].index

    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    # test = data[np.in1d(data.SessionId, session_test)]
    # test = test[np.in1d(test.ItemId, train.ItemId)]
    # tslength = test.groupby('SessionId').size()
    # test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), train_from.date().isoformat(),
                 train_to.date().isoformat()))
    train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
    #                                                                    test.ItemId.nunique()))
    # test.to_csv(output_file + '_test.' + str(retrain_num) + '.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    valid_from = data_end - timedelta(days=days_test)
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.' + str(retrain_num) + '.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.' + str(retrain_num) + '.txt', sep='\t', index=False)

    return train


def split_data_retrain_test(data, train, output_file, days_train, days_test, retrain_num, test_set_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    # train_from = data_start
    # new_days = retrain_num * days_test
    # new_days = test_set_num * days_test
    new_days = (retrain_num + test_set_num) * days_test
    # train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_from = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_to = test_from + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    # session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index
    # session_test = session_max_times[(session_max_times > train_to.timestamp()) & (session_max_times <= test_to.timestamp())].index
    session_test = session_max_times[(session_max_times > test_from.timestamp()) & (session_max_times <= test_to.timestamp())].index

    # train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    # print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
    #                                                                          train.ItemId.nunique()))
    # train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test_from.date().isoformat(),
                 test_to.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(retrain_num) + '_' + str(test_set_num) + '.txt', sep='\t', index=False)

    # data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    # valid_from = data_end - timedelta(days=days_test)
    # session_max_times = train.groupby('SessionId').Time.max()
    # session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    # session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    # train_tr = train[np.in1d(train.SessionId, session_train)]
    # valid = train[np.in1d(train.SessionId, session_valid)]
    # valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    # tslength = valid.groupby('SessionId').size()
    # valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    # print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
    #                                                                     train_tr.ItemId.nunique()))
    # train_tr.to_csv(output_file + '_train_tr.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
    #                                                                          valid.ItemId.nunique()))
    # valid.to_csv(output_file + '_train_valid.' + str(retrain_num) + '.txt', sep='\t', index=False)

# -------------------------------------
# MAIN TEST
# --------------------------------------

if __name__ == '__main__':
    #preprocess_info()
    #preprocess_org_min_date(min_date=MIN_DATE, days_test=DAYS_TEST)
    #preprocess_days_test(days_test=DAYS_TEST)
    #preprocess_slices()
    # ab = load_data_make_dictionary()
    # print(len(ab))
    preprocess_HCGCNN()