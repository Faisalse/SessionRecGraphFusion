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
DATA_PATH_PROCESSED = r'./data/eCommerce/prepared/'

# This Data_Loader file is copied online
class Data_Loader:
    
    def __init__(self, train, test):

        '''positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]'''
        
        self.train = train.sort_values(by=['UserId', 'Time'], ascending=True)
        self.test = test.sort_values(by=['UserId', 'Time'], ascending=True)
        self.session_key = "SessionId"
        self.session = -1
        self.item_key = "ItemId"
        self.time_key = "Time"
        self.max_session_length = 8
        self.characterVocabSize = 52
        self.CharacterOfEachFeature = 50
        self.NumberOfClicksConsidered = 7
        self.tokenization = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        
        
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:/\|_@#$%"
        char_dict = {}
        for i, char in enumerate(alphabet):
            char_dict[char] = i 
        # Use char_dict to replace the tk.word_index
        self.tokenization.word_index = char_dict 
        # Add 'UNK' to the vocabulary 
        self.tokenization.word_index[self.tokenization.oov_token] = max(char_dict.values()) + 1
             
    def getSequence(self):
        
        prod_cat = {}
        itemId = {}
        brand = {}
        
        for index, row in self.train.iterrows():
            
            if(row["SessionId"] != self.session):
                
                self.session = row["SessionId"]
                
                if  row["SessionId"] not in itemId:
                    
                    prod_cat[row["SessionId"]] = [row["CatId"]]
                    itemId[row["SessionId"]] = [row["ItemId"]]
                    brand[row["SessionId"]] = [row["Brand"]]
                
                else:
                
                    prod_cat[row["SessionId"]] += [row["CatId"]]
                    itemId[row["SessionId"]]   += [row["ItemId"]]
                    brand[row["SessionId"]]    += [row["Brand"]]
                
            else:
                if  row["SessionId"] not in  itemId:
                    
                    prod_cat[row["SessionId"]] = [row["CatId"]]
                    itemId[row["SessionId"]] = [row["ItemId"]]
                    brand[row["SessionId"]] = [row["Brand"]]
                
                else:
                
                    prod_cat[row["SessionId"]] += [row["CatId"]]
                    itemId[row["SessionId"]]   += [row["ItemId"]]
                    brand[row["SessionId"]]    += [row["Brand"]]
                
       
       # remove the repeated items from sessions......
        copydic = prod_cat.copy()
        for session in copydic:
        
           a = prod_cat[session]
           b = itemId[session]
           c = brand[session]
           
           a = sorted(set(a), key=a.index)
           b = sorted(set(b), key=b.index)
           c = sorted(set(c), key=c.index)
           
           if(len(a) > 1):
               
               prod_cat[session] = a
               itemId[session]  = b
               brand[session] = c
           else:
               del prod_cat[session]
               del itemId[session]
               del brand[session]
               
       
        # make slices
        itemId_slicing = list()
        prod_cat_slicing = list()
        brand_slicing = list()
        self.predictionList = list()
        for key in itemId:
             
            a = itemId[key]
            b = prod_cat[key]
            c = brand[key]
            for i in range(len(itemId[key])):
                if (i < 2):
                    pass
                else:
                    slic = slice(i)
                    itemId_slicing.append(a[slic])
                    prod_cat_slicing.append(b[slic])
                    brand_slicing.append(c[slic])
                    
        
        
        
        
        for i in range(len(itemId_slicing)):
            
            if(len(itemId_slicing[i]) >= self.max_session_length):
                
                 self.predictionList.append(itemId_slicing[i][self.max_session_length-1])
                 itemId_slicing[i] = itemId_slicing[i][:self.max_session_length -1]
                 prod_cat_slicing[i] = prod_cat_slicing[i][:self.max_session_length -1]
                 brand_slicing[i] = brand_slicing[i][:self.max_session_length -1]
                 
            else:
                
                
                itemId_slicing[i][:0]    = [0] * (self.max_session_length - len(itemId_slicing[i]))
                prod_cat_slicing[i][:0]  = [0] * (self.max_session_length - len(prod_cat_slicing[i]))
                brand_slicing[i][:0]     = [0] * (self.max_session_length - len(brand_slicing[i]))
                
                self.predictionList.append(itemId_slicing[i][self.max_session_length-1])
                
                itemId_slicing[i] = itemId_slicing[i][:self.max_session_length -1]
                prod_cat_slicing[i] = prod_cat_slicing[i][:self.max_session_length -1]
                brand_slicing[i] = brand_slicing[i][:self.max_session_length -1]
        
            
        # remove those clicks that are not appear in training data
        
        session_lengths = self.test.groupby('SessionId').size()
        self.test = self.test[np.in1d(self.test.SessionId, session_lengths[ session_lengths  <= 7 ].index)]
        self.test = self.test[np.in1d(self.test.ItemId, self.predictionList)]    
        
        return pd.DataFrame(itemId_slicing), pd.DataFrame(prod_cat_slicing), pd.DataFrame(brand_slicing), self.predictionList
   
    
   
    
   
    # Convert the slicing sequence into encoding....
    def Sessionsequence(self, dataFrameWithSequence):
        
        def encodingList(NumberOfNestedList, elementsInEachNestedList):
            encodingListt = list()
            for i in range(NumberOfNestedList):
                encodingListt.append([0 for i in range(elementsInEachNestedList)])
            return encodingListt
        
        NumberofTraingSequence = dataFrameWithSequence.shape[0]
        CompleteSessionData = np.zeros((NumberofTraingSequence, self.NumberOfClicksConsidered,  self.characterVocabSize, self.CharacterOfEachFeature))
        
        for i in range(dataFrameWithSequence.shape[0]):
            # count = 0
            
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
            
        print("Sessionsequence")    
        return CompleteSessionData        

    # Combine all sequence
    def CombineAllSequence(self, iDSequence, categorySequence, brandSequence):
        numberofFeatureTaken = 3
        CompleteSequencewithAllFeatures = np.zeros((iDSequence.shape[0],  7,  self.characterVocabSize * numberofFeatureTaken, 50))
        print(CompleteSequencewithAllFeatures.shape)
        for i in range(CompleteSequencewithAllFeatures.shape[0]):
            
            for j in range(CompleteSequencewithAllFeatures.shape[1]):
                CompleteSequencewithAllFeatures[i, j , 0:52, :] = iDSequence[i, j ,:, :]
                CompleteSequencewithAllFeatures[i,  j , 52:104, :] = categorySequence[i, j ,:, :]
                CompleteSequencewithAllFeatures[i, j , 104: 156, :] = brandSequence[i, j,:,  :]  
        print("CombineAllSequence")        
        return CompleteSequencewithAllFeatures   

    def  returnFinalOutData(self): 
        
        itemId_slicing, prod_cat_slicing, brand_slicing,  predictionList = Data_Loader.getSequence(self)
        itemId_slicing = Data_Loader.Sessionsequence(self, itemId_slicing)
        prod_cat_slicing = Data_Loader.Sessionsequence(self, prod_cat_slicing)
        brand_slicing = Data_Loader.Sessionsequence(self, brand_slicing)
        CombineData = Data_Loader.CombineAllSequence(self, itemId_slicing, prod_cat_slicing, brand_slicing)
        
        return CombineData



#%%
# import pandas as pd        
# train = pd.read_csv(DATA_PATH_PROCESSED+"2019-Dec_train_full.txt", sep = "\t")  
# train.dropna(how='all', inplace = True)  
#%%
# test = pd.read_csv(DATA_PATH_PROCESSED+"2019-Dec_test_full.txt", sep = "\t")   
# obj1  = Data_Loader(train, test)

# a= obj1.returnFinalOutData()

# print(obj1.max_session_length)


