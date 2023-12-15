import keras as keras
import algorithms.model3D.data_loader_cosmec_images as data_loader_cosmec
import time
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.layers import Flatten
from keras.optimizers import adam
from keras.layers import Conv3D
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing

# #%% Additional parts
# DATA_PATH_PROCESSED = r'./data/eCommerce/prepared/'
# train = pd.read_csv(DATA_PATH_PROCESSED+"2019-Dec_train_full.txt", sep = "\t")  
# # train.dropna(how='all', inplace = True)  
# test = pd.read_csv(DATA_PATH_PROCESSED+"2019-Dec_test_full.txt", sep = "\t")   

#%%

class Model3D:
    '''
    Code based on work by Yuan et al., A Simple but Hard-to-Beat Baseline for Session-based Recommendations, CoRR abs/1808.05163, 2018.

    # Strongly suggest running codes on GPU with more than 10G memory!!!
    # if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
    '''

    def __init__(self, learning_rate = 0.01, batch_size = 32, epoch = 1):

        '''
        :param top_k: Sample from top k predictions
        :param beta1: hyperpara-Adam
        :param eval_iter: Sample generator output evry x steps
        :param save_para_every: save model parameters every
        :param is_negsample:False #False denotes no negative sampling

        '''
        
        self.top_k = 5
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.session_key = 'SessionId'
        self.item_key = 'ItemId'
        self.time_key = 'Time'
        self.epoch = epoch
        self.session = -1
        
        
    def fit(self, train, test):
        self.train = train
        self.test = test
        print("one_hot_encode")
        self.dataKoaderObject = data_loader_cosmec.Data_Loader(self.train, self.test)
        
        trainingEncodingData = self.dataKoaderObject.returnFinalOutData()
        
        
        trainingEncodingData = trainingEncodingData.reshape(trainingEncodingData.shape[0], 
                                                            trainingEncodingData.shape[1], trainingEncodingData.shape[2], trainingEncodingData.shape[3], 1)
        
        self.targetItems = self.dataKoaderObject.predictionList
        
        
        # model input and output settings..... 
        uniqueItems = len(set(self.targetItems))
        width = trainingEncodingData.shape[1]
        height = trainingEncodingData.shape[2]
        No_of_Frames = trainingEncodingData.shape[3]
        
        
        # self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
        self.inputShape = Input(shape=(width, height, 1))
        self.NumberOfUniqueItems = uniqueItems
        
        # Label encoding,,,
        le = preprocessing.LabelEncoder()
        le.fit(self.targetItems)
        self.LabelsEncodedForm = le.transform(self.targetItems)
        
        one_hot_encoded = to_categorical(self.LabelsEncodedForm)
        
        print("one_hot_encoded   ", one_hot_encoded.shape)
        
        inputShape = Input(shape=(width, height, No_of_Frames, 1))
        conv1 = Conv3D(16, kernel_size=(5, 55, 3), strides = (2,2, 2), padding = "same", activation='relu', data_format="channels_last")(inputShape)
        conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv1)
        
        conv3 = Conv3D(16, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv2)
           
        # skipConnection = keras.layers.Add()([conv1, conv3])
             
        # conv4 = Conv3D(32, kernel_size=(3, 3, 3), strides = (2,2, 1), padding = "same", activation='relu')(skipConnection)
        # conv5 = Conv3D(32, kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv4)
                   
        # conv6 = Conv3D(64, kernel_size=(3, 3, 3), strides = (2,2,1), padding = "same", activation='relu')(conv5)
        # conv7 = Conv3D(64, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv6)
              
        # conv8 = Conv3D(128, kernel_size=(3, 3, 3), strides = (2,2,1), padding = "same", activation='relu')(conv7)
        # conv9 = Conv3D(128, kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv8)
                
                   
        # conv10 = Conv3D(256, kernel_size=(3, 3, 3), strides = (2,2, 1), padding = "same", activation='relu')(conv9)
        # conv11 = Conv3D(256, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv10)
         
        
        # conv12 = Conv3D(512 , kernel_size=(3, 3,3), strides = (2,2,1), padding = "same", activation='relu')(conv11)
        # conv13 = Conv3D(512 , kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv12)
        # #pooling = AveragePooling3D((3, 3,3), strides = (2,2,2))(conv13)
        print("one_hot_encoded   1")      
        flat = Flatten()(conv3)
        drop = Dropout(0.5)(flat)
        output = Dense(self.NumberOfUniqueItems, kernel_regularizer=l2(0.02), activation='softmax')(drop)
        model = Model(inputs=inputShape, outputs=output)
       
        #optimizer = adam(learning_rate=self.learning_rate, decay=self.learning_rate/50)
        optimizer = adam(learning_rate=self.learning_rate)
        print("one_hot_encoded   2") 
        model.compile( optimizer= optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
       
        print("one_hot_encoded   3")  
        model.fit(trainingEncodingData, one_hot_encoded, epochs=self.epoch, batch_size = self.batch_size , verbose=1, shuffle=False)
        self.model = model
    
    def predict_next(self, session_id, prev_iid, prev_cat,  prev_brand):
        if(session_id != self.session):
            session_sequence_id = list()
            session_sequence_cat = list()
            session_sequence_brand = list()
            
            
        session_sequence_id.append(prev_iid)
        session_sequence_cat.append(prev_cat)
        session_sequence_brand.append(prev_brand)
        
        length = self.dataKoaderObject.max_session_length -1
        
        if(len(session_sequence_id) < (length - 1)):
            session_sequence_id[:0] = [0] * (length - len(session_sequence_id))
            session_sequence_cat[:0] = [0] * (length - len(session_sequence_id))
            session_sequence_brand[:0] = [0] * (length - len(session_sequence_id))

        
        itemId_slicing = self.dataKoaderObject.Sessionsequence(self, pd.DataFrame(session_sequence_id))
        prod_cat_slicing = self.dataKoaderObject.Sessionsequence(self, pd.DataFrame(session_sequence_cat))
        brand_slicing = self.dataKoaderObject.Sessionsequence(self, pd.DataFrame(session_sequence_brand)) 
        
        CombineData = self.dataKoaderObject.CombineAllSequence(itemId_slicing, prod_cat_slicing, brand_slicing)
        
        CombineData = CombineData.reshape(CombineData.shape[0], 
                                                            CombineData.shape[1], CombineData.shape[2], CombineData.shape[3], 1)
        
        prediction = self.model.predict(CombineData)
        # new addition.....
        prediction_df = pd.DataFrame()

        print("prediction__list:       ", prediction)
        prediction_df["items_id"] = self.targetItems
        prediction_df["items_id_encoded"] = self.LabelsEncodedForm
        prediction_df.sort_values(by=['items_id_encoded'], inplace = True)

        prediction_df["score"] = prediction[0]

        prediction_df.sort_values(by=["score"], ascending = False, inplace = True)
        
        series = pd.Series(data=prediction_df["score"].tolist(), index=prediction_df["items_id"].tolist())
        
        return series
#%%
# def ggg():
    
#     inputShape = Input(shape=(52, 350, 3, 1))
#     conv1 = Conv3D(16, kernel_size=(5, 55, 3), strides = (2,2, 3), padding = "same", activation='relu', data_format="channels_last")(inputShape)
#     conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv1)
    
#     conv3 = Conv3D(16, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv2)
       
#     # skipConnection = keras.layers.Add()([conv1, conv3])
         
#     # conv4 = Conv3D(32, kernel_size=(3, 3, 3), strides = (2,2, 1), padding = "same", activation='relu')(skipConnection)
#     # conv5 = Conv3D(32, kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv4)
               
#     # conv6 = Conv3D(64, kernel_size=(3, 3, 3), strides = (2,2,1), padding = "same", activation='relu')(conv5)
#     # conv7 = Conv3D(64, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv6)
          
#     # conv8 = Conv3D(128, kernel_size=(3, 3, 3), strides = (2,2,1), padding = "same", activation='relu')(conv7)
#     # conv9 = Conv3D(128, kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv8)
            
               
#     # conv10 = Conv3D(256, kernel_size=(3, 3, 3), strides = (2,2, 1), padding = "same", activation='relu')(conv9)
#     # conv11 = Conv3D(256, kernel_size=(3, 3, 3), strides = (1,1, 1), padding = "same", activation='relu')(conv10)
     
    
#     # conv12 = Conv3D(512 , kernel_size=(3, 3,3), strides = (2,2,1), padding = "same", activation='relu')(conv11)
#     # conv13 = Conv3D(512 , kernel_size=(3, 3, 3), strides = (1,1,1), padding = "same", activation='relu')(conv12)
#     # #pooling = AveragePooling3D((3, 3,3), strides = (2,2,2))(conv13)
          
#     flat = Flatten()(conv3)
#     drop = Dropout(0.5)(flat)
#     output = Dense(4342, kernel_regularizer=l2(0.02), activation='softmax')(drop)
#     model = Model(inputs=inputShape, outputs=output)
   
#     #optimizer = adam(learning_rate=self.learning_rate, decay=self.learning_rate/50)
#     optimizer = adam(learning_rate=0.05)
    
#     model.compile( optimizer= optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
#     return model
    

# ab = ggg()

