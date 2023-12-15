import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.regularizers import l2
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
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
        self.max_session_length = 8
        self.CharacterOfEachFeature = 15
        self.NumberOfClicksConsidered = 7
        self.tokenization = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        
        
        alphabet="0123456789"
        char_dict = {}
        for i, char in enumerate(alphabet):
            char_dict[char] = i 
        # Use char_dict to replace the tk.word_index
        self.tokenization.word_index = char_dict 
        # Add 'UNK' to the vocabulary 
        self.tokenization.word_index[self.tokenization.oov_token] = max(char_dict.values()) + 1
        
        self.characterVocabSize =     len(self.tokenization.word_index)
   
    def getSequence(self):
        itemId = {}
        
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
                    
                
       
       # remove the repeated items from sessions......
        copydic = itemId.copy()
        for session in copydic:
           b = itemId[session]
           b = sorted(set(b), key=b.index)
           if(len(b) > 1):
               itemId[session]  = b  
           else:
               del itemId[session]
       
        # make slices
        itemId_slicing = list()
        self.predictionList = list()
        for key in itemId:
             
            a = itemId[key]
            
            for i in range(len(itemId[key])):
                if (i < 2):
                    pass
                else:
                    slic = slice(i)
                    itemId_slicing.append(a[slic])

                    
        for i in range(len(itemId_slicing)):
            
            if(len(itemId_slicing[i]) >= self.max_session_length):
                
                 self.predictionList.append(itemId_slicing[i][self.max_session_length-1])
                 itemId_slicing[i] = itemId_slicing[i][:self.max_session_length -1]
            
            else:
                itemId_slicing[i][:0]    = [0] * (self.max_session_length - len(itemId_slicing[i]))
                self.predictionList.append(itemId_slicing[i][self.max_session_length-1])
                itemId_slicing[i] = itemId_slicing[i][:self.max_session_length -1]
        
        
        session_lengths = self.test.groupby('SessionId').size()
        self.test = self.test[np.in1d(self.test.SessionId, session_lengths[ session_lengths  <= 7 ].index)]
        
        
        self.test = self.test[np.in1d(self.test.ItemId, self.predictionList)]    
        
        return pd.DataFrame(itemId_slicing), self.predictionList
   
    # Convert the slicing sequence into encoding....
    def Sessionsequence(self, dataFrameWithSequence):
        def encodingList(NumberOfNestedList, elementsInEachNestedList):
            encodingListt = list()
            for i in range(NumberOfNestedList):
                encodingListt.append([0 for i in range(elementsInEachNestedList)])
            return encodingListt
        
        NumberofTraingSequence = dataFrameWithSequence.shape[0]
        CompleteSessionData = np.zeros((NumberofTraingSequence, self.NumberOfClicksConsidered, 
                                        self.characterVocabSize, self.CharacterOfEachFeature))
        
        
        for i in range(dataFrameWithSequence.shape[0]):
            for j in range(dataFrameWithSequence.shape[1]):
                # No of columns in dataFrame.....
                feature = str(dataFrameWithSequence.iloc[i, j]) # Convert into string type becuase tokenizer work on string type
                feature = [s.lower() for s in feature] # convert to lower case 
                sequences = self.tokenization.texts_to_sequences(feature) # Convert text into sequence using character level encoding... 
                # Feature Encoded Matrix  rows = characterVocabSize-- in this case 52 and columns = CharacterOfEachFeature in this case considered 52
                
                featuresEncodingMatrix = encodingList(self.characterVocabSize, self.CharacterOfEachFeature)
                for k in range(len(sequences)):
                    if(k > self.CharacterOfEachFeature):
                        break
                    else:
                        row = sequences[k][0] # Sequence shape --> [[3], [5], [6], [6], [5]]
                        featuresEncodingMatrix[row][k] = 1
                CompleteSessionData[i, j , : , :] =  np.array(featuresEncodingMatrix)
        return CompleteSessionData        


    def  returnFinalOutData(self): 
        itemId_slicing,  predictionList = Data_Loader.getSequence(self)
        itemId_slicing = Data_Loader.Sessionsequence(self, itemId_slicing)
        return itemId_slicing
#%%
# import pandas as pd        
# train = pd.read_csv(DATA_PATH_PROCESSED+"yoochoose-clicks-100k_train_full.txt", sep = "\t")  
# train.dropna(how='all', inplace = True)  

# #%%
# test = pd.read_csv(DATA_PATH_PROCESSED+"yoochoose-clicks-100k_train_full.txt", sep = "\t")   
# obj1  = Data_Loader(train, test)

# ab = obj1.returnFinalOutData()

