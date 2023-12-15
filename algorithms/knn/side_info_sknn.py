from _operator import itemgetter
from math import sqrt
import random
import time
import numpy as np
import pandas as pd


class ContextKNN:
    '''
    ContextKNN(k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, extend=False, normalize=True,
                 session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time', catId = "CatId", bran = "Brand"):
       
        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.catId = catId
        self.brand = bran
        
        
        self.extend = extend
        self.normalize = normalize
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        # session_item stors occuring of items in a session
        self.session_item_map = dict() 
        self.session_time = dict()
        self.session_cat_map = dict() 
        self.session_brand_map = dict() 
        
        # item_session_map stors occuring of a items in total number of sessions.... 
        self.item_session_map = dict()
        self.cat_session_map = dict()
        self.brand_session_map = dict()
        
        
        self.sim_time = 0
        
    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        # get_loc returns the location columns-- 0,1,2
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        index_cat = train.columns.get_loc( self.catId )
        index_brand = train.columns.get_loc( self.brand )
        
        
        session = -1
        session_items = set()
        session_cat = set()
        session_brand = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    self.session_cat_map.update({session : session_cat})
                    self.session_brand_map.update({session : session_brand})
                    
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                session = row[index_session]
                session_items = set()
                session_cat = set()
                session_brand = set()
                
            time = row[index_time]
            session_items.add(row[index_item])
            session_cat.add(row[index_cat])
            session_brand.add(row[index_brand])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
            
            
            map_iss =  self.cat_session_map.get(row[index_cat] )
            if map_iss is None:
                map_iss = set()
                self.cat_session_map.update({row[index_cat] : map_iss})
            map_iss.add(row[index_session])
            
            
            map_isb = self.brand_session_map.get(row[index_brand] )
            if map_isb is None:
                map_isb = set()
                self.brand_session_map.update({row[index_brand] : map_is})
            map_isb.add(row[index_session])
        
           
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_cat_map.update({session : session_cat})
        self.session_brand_map.update({session : session_brand})
        self.session_time.update({session : time})
        
    def predict_next(self, session_id, input_item_id, input_item_cat, input_item_brand,  predict_for_item_ids, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
#         gc.collect()
#         process = psutil.Process(os.getpid())
#         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        # if(type(session_id) is np.ndarray):
        #     session_id = session_id[0]
        # if(type(input_item_id) is np.ndarray):
        #     input_item_id = input_item_id[0]

        # print("session_id::::::   ", session_id)
        # print("input_item_id::::::   ", input_item_id)
        # print("predict_for_item_ids::::::   ", predict_for_item_ids)
        # print("predict_for_item_ids length::::::   ", len(predict_for_item_ids))
        # print("session_id::::::   ", session_id)
        
        
        if( self.session != session_id ): #new session
            
            if(self.extend):
                item_set = set( self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
                
            self.session = session_id
            self.session_items = list()
            self.session_items_cats = list()
            self.session_items_brands = list()
            
            
            self.relevant_sessions = set()
            
        
        if mode_type == 'view':
            self.session_items.append( input_item_id )
            self.session_items_cats.append( input_item_cat )
            self.session_items_brands.append( input_item_brand )
        
        if skip:
            return
        # neighbors: sorted list most similar session of current session iD   input_item_id, input_item_cat, input_item_brand              
        neighbors = self.find_neighbors(set(self.session_items), set(self.session_items_cats), set(self.session_items_brands),  input_item_id, input_item_cat, input_item_brand)
        scores = self.score_items( neighbors )
          
        
          
        
        # add some reminders
        if self.remind:
             
            reminderScore = 5
            takeLastN = 3
            
            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                #reminderScore = reminderScore + (cnt/100)
                 
                oldScore = scores.get( elem )
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                #print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt/100)
                 
                scores.update({elem : newScore})
        
        #push popular ones
        if self.pop_boost > 0:
            pop = self.item_pop( neighbors )
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})
         
        
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        
        items = predict_for_item_ids[mask]
        
        
        values = [scores[x] for x in items]
        
        
        predictions[mask] = values
                
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        if self.normalize:
            series = series / series.max()
        
        #print return the series where each item has item iD has specefic score..... 
        return series 

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session( session )
            for item in items:
                
                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})
                    
                if( result.get(item) > max_pop ):
                    max_pop =  result.get(item)
         
        for key in result:
            result.update({key: ( result[key] / max_pop )})
                   
        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
       
        # if (len(second) > 2):
        #     print("second", second)
        #     print("second", len(second))
            
        # elif (len(first) > 2):
        #     print("first", first)
        #     print("first", len(first))
        # print("First :     ", type(first))
        # print("second:     ", type(second))
        
        
        
        li = len(first&second)
        
        la = len(first)
        lb = len(second)
        # number of given items in a session are more similar to a neighbour session, similarity value will be high..... 
        result = li / sqrt(la) * sqrt(lb)

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        return random.random()
    

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    def cat_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_cat_map.get(session);
    
    def brand_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_brand_map.get(session)
    
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
	
        return self.item_session_map.get( item_id )
    
    def sessions_for_cat(self, cat):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
	
        return self.cat_session_map.get( cat )
    
    def sessions_for_brand(self, brand):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
	
        return self.brand_session_map.get( brand )
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, input_item_id, input_item_cat, input_item_brand):
        
        
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        # it takes union between sets... 
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_cat( input_item_cat )
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_brand( input_item_brand )
        
        
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            
            return self.relevant_sessions

        else: #sample some sessions
             
            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
                        

    def calc_similarity(self, session_items, session_items_cats, session_items_brands, sessions ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        
        
        
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        #print("All sessions:     ", sessions)
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session( session )
            session_items_cat_test = self.cat_for_session( session )
            session_items_brand_test = self.brand_for_session( session )
            
            # print("---------------------------")
            
            # print("session_items_test  :", session_items_test)
            # print("session_items_test  :", len(session_items_test))
            
            
            similarity1 = getattr(self , self.similarity)(session_items_test, session_items)
            similarity2 = getattr(self , self.similarity)(session_items_cat_test, session_items_cats)
            similarity3 = getattr(self , self.similarity)(session_items_brand_test, session_items_brands)
            
            similarity = similarity1 * similarity2 * similarity3
           
            
            if similarity > 0:
                neighbors.append((session, similarity))
        
        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, session_items_cats, session_items_brands,  input_item_id, input_item_cat, input_item_brand):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        
        possible_neighbors = self.possible_neighbor_sessions(input_item_id, input_item_cat, input_item_brand)
        
        
        possible_neighbors = self.calc_similarity( session_items, session_items_cats, session_items_brands,  possible_neighbors)
        
        
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        
       
        possible_neighbors = possible_neighbors[:self.k]
        # Range of cosine similarity is -1 to +1.... But our sorted list have values more than 1.... 
        #print("possible_neighborspossible_neighbors     ", possible_neighbors)
        return possible_neighbors
    
            
    def score_items(self, neighbors):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            
            for item in items:
                old_score = scores.get( item )
                new_score = session[1]
                
                if old_score is None:
                    scores.update({item : new_score})
                else: 
                    new_score = old_score + new_score
                    scores.update({item : new_score})
        
        # return the score each item gain that occur in neighbouring sessions... 
        #print("scores          ", scores)
        return scores
    
    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.cat_session_map = dict()
        self.brand_session_map = dict()
        self.session_time = dict()

    def support_users(self):
        '''
          whether it is a session-based or session-aware algorithm
          (if returns True, method "predict_with_training_data" must be defined as well)

          Parameters
          --------

          Returns
          --------
          True : if it is session-aware
          False : if it is session-based
        '''
        return False

