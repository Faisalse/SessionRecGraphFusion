from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
DATA_PATH_PROCESSED = r'./data/rsc15/prepared/'


# This Data_Loader file is copied online
class Data_Loader:
    
    def __init__(self, train, test):

        '''positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]'''
        
        self.train = train.sort_values(by=['Time'], ascending=True)
        self.test = test.sort_values(by=['Time'], ascending=True)
        self.session_key = "SessionId"
        self.session = -1
        self.item_key = "ItemId"
        self.time_key = "Time"
        
    def getSequence(self):
        itemId = {}
        print("ggggggggggg")
        for index, row in self.train.iterrows():
            if(row["SessionId"] != self.session):
                self.session = row["SessionId"]
                
                if  row["SessionId"] not in itemId:
                    itemId[row["SessionId"]] = [row["ItemId"]]
                else:
                    itemId[row["SessionId"]]   += [row["ItemId"]]
            else:
                if  row["SessionId"] not in  itemId:
                    itemId[row["SessionId"]] = [row["ItemId"]]
                else:
                    itemId[row["SessionId"]]   += [row["ItemId"]]
                    
                
       
        
        sequence = list()
        for item in itemId:
            if len(itemId[item]) > 2:
                sequence.append(itemId[item])
        
                
        predictList = list()
        for i in range(len(sequence)):
            seq = sequence[i]
            pr = seq[-1]
            sequence[i] = seq[:-1]
            predictList.append(pr)
            
        for i in range(len(sequence)):
            convertIntoString = ""
            seq1 = sequence[i]
            for j in range(len(seq1)):
                convertIntoString = convertIntoString+" " +str(int(seq1[j]))
                
            sequence[i] = convertIntoString
        
        
        train_texts = sequence
        self.tk = Tokenizer(num_words=None, oov_token='UNK')
        self.tk.fit_on_texts(train_texts)

        vocab_size = len(self.tk.word_index)
        
        session_Size = 10
        
        train_sequences = self.tk.texts_to_sequences(train_texts)
        train_data = pad_sequences(train_sequences, maxlen = session_Size, padding='pre')
        return train_data ,  predictList, vocab_size, session_Size
                               
#%%

# import pandas as pd        
# train = pd.read_csv(DATA_PATH_PROCESSED+"yoochoose-clicks-100k_train_full.txt", sep = "\t")  
# train.dropna(how='all', inplace = True)  

# #%%
# test = pd.read_csv(DATA_PATH_PROCESSED+"yoochoose-clicks-100k_train_full.txt", sep = "\t")   
# obj1  = Data_Loader(train, test)
# ab, b, c, cb = obj1.getSequence()


