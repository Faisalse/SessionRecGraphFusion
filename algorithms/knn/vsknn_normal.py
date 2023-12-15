from _operator import itemgetter
from math import sqrt
import random
import time

from pympler import asizeof
import numpy as np
import pandas as pd
from math import log10
from datetime import datetime as dt
from datetime import timedelta as td
import math

class VMContextKNN:
    '''
    VMContextKNN( k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time')
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
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    weighting_time : boolean
        Experimental function to give less weight to items from older sessions (default: False)
    dwelling_time : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    last_n_days : int
        Use only data from the last N days. (default: None)
    last_n_clicks : int
        Use only the last N clicks of the current session when recommending. (default: None)
    extend : bool
        Add evaluated sessions to the maps.
    normalize : bool
        Normalize the scores in the end.
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''
    def __init__( self, k, sample_size=1000, sampling='recent', similarity='vec', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, remind=True, push_reminders=False, add_reminders=False, extend=False, weighting_score='div', weighting_time=False, normalize=True, idf_weighting=False, idf_weighting_session=False, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time', cat_key='CatId', add_cate_info = False, picked_items = 0, give_importance=10):
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting
        self.dwelling_time = dwelling_time
        self.weighting_score = weighting_score
        self.weighting_time = weighting_time
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.cat_key = cat_key
        self.extend = extend  # to add evaluated sessions to the maps
        self.remind = remind
        self.push_reminders = push_reminders  # give more score to the items that belongs to the current session
        self.add_reminders = add_reminders  # force the last 3 items of the current session to be in the top 20
        self.idf_weighting = idf_weighting
        self.idf_weighting_session = idf_weighting_session
        self.normalize = normalize
        self.last_n_days = last_n_days
        self.last_n_clicks = last_n_clicks
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.item_cat_map = dict()
        self.cate_items_map = dict()
        self.min_time = -1
        self.sim_time = 0
        # Faisal contributions.....
        self.add_cate_info = add_cate_info
        self.picked_items = picked_items
        self.give_importance = give_importance
        
    def fit(self, data, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        # this will not be run.....
        if self.last_n_days != None:
            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]
        
        else: 
            train = data
        
        #self.num_items = train[self.item_key].max()
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        index_cat = train.columns.get_loc( self.cat_key )
        session = -1
        session_items = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            # Create a two dictionaries for sessions_to_item and item_to_sessions....
            # We also say this.... session to item map -r item to sessions map......
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                    if time < self.min_time:
                        self.min_time = time
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])
            # cache sessions involving an item
            
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
            
            if row[index_item] not in self.item_cat_map:
                self.item_cat_map[row[index_item]] = row[index_cat]
        
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        
        self.cate_items_map = train.groupby("CatId")["ItemId"].apply(list).to_dict()
        
        if self.idf_weighting or self.idf_weighting_session: 
            self.idf = pd.DataFrame()
            self.idf['idf'] = train.groupby( self.item_key ).size()
            # counts total total number of documents in the corpus. in this case,  total number of sessions in the training data.....
            self.idf['idf'] = np.log( train[self.session_key].nunique() / self.idf['idf'] )
            # idf contains the inverse document frequecy of each items, where popukar items have less value while unpopular items have more value or more importance. 
            self.idf = self.idf['idf'].to_dict()
        
        
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0, skip=False, mode_type='view'):
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
        
        if( self.session != session_id ): #new session
        # in this case; extent has False
            if( self.extend ):
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
            self.last_ts = -1 
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()
        
        if mode_type == 'view':
            self.session_items.append( input_item_id )
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append( timestamp - self.last_ts )
                self.last_ts = timestamp
        
        if skip:
            return
         
        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        # neighbors: find the neighbors of current session items.....
        neighbors = self.find_neighbors( items, input_item_id, session_id, self.dwelling_times, timestamp )
        # we goes for different story to calculate the scores for items
        scores = self.score_items(neighbors, items, timestamp)
        
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.push_reminders:

            session_series = pd.Series( self.session_items )
            session_count = session_series.groupby( session_series ).count() + 1
            series[ session_count.index ] *= session_count

        if self.add_reminders:
            session_series = pd.Series( index=self.session_items, data=series[ self.session_items  ])
            session_series = session_series[ session_series > 0 ]

            if len(session_series) > 0:
                session_series = session_series.iloc[:3]
                series.sort_values( ascending=False, inplace=True )
                session_series = session_series[session_series < series.iloc[19-3] ]
                series[ session_series.index ] = series.iloc[19-3] + 1e-4
        
        series[np.isnan(series)] = 0
        series.sort_values( ascending=False, inplace=True )
        if self.add_cate_info:
            series = self.shuffle_items_order_based_category(set(items), series, self.picked_items, self.give_importance)
        
        
        
        # for it in reversed(items):
        #     last_item = it
        #     break
        
        # count = 0
        # for ser in series.index:
        #     count +=1
        #     last_item_cate = self.check_category(last_item)
        #     list_cat = self.check_category(ser)
            
        #     if last_item_cate == list_cat:
        #         pass
        #     else:
        #         print("Fais")
        #         series.drop(index = ser, inplace=True) 
            
        #     if count == 100:
        #         break
            
        # series.sort_values( ascending=False, inplace=True )
        
        if self.normalize:
            series = series / series.max()
        
        return series 
    def shuffle_items_order_based_category(self, current_session_items, sorted_ranking_list, picked_items, give_importance):
        if picked_items:
            sorted_ranking_list = sorted_ranking_list[:picked_items]
        else:
            sorted_ranking_list = sorted_ranking_list[:]
        if len(current_session_items) > 0:
            for item in current_session_items:
                category = self.check_category(item)
                # find the items with this same category.....
                items_with_same_category = list(set(self.find_item_of_category(category)))
                for items_same in items_with_same_category:
                    if items_same in sorted_ranking_list.index:
                        sorted_ranking_list[items_same] = sorted_ranking_list[items_same] + give_importance
        return sorted_ranking_list
    
    
    def check_category(self, item):
        return self.item_cat_map[item]
    
    def find_item_of_category(self, category):
        if category in self.cate_items_map:
            return self.cate_items_map[category]
        else:
            return []
        
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
    
    def vec(self, current, neighbor, pos_map):
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
        intersection = current & neighbor
        
        if pos_map is not None:
            vp_sum = 0
            current_sum = len(pos_map)
            for i in intersection:
                vp_sum += pos_map[i]
                
        else:
            vp_sum = len( intersection )
            current_sum = len( current )
        
        result = vp_sum / current_sum

        return result
    
    def cosine(self, current, neighbor, pos_map):
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
                
        lneighbor = len(neighbor)
        intersection = current & neighbor
        
        if pos_map is not None:
            
            vp_sum = 0
            current_sum = 0
            for i in current:
                current_sum += pos_map[i] * pos_map[i]
                if i in intersection:
                    vp_sum += pos_map[i]
        else:
            vp_sum = len( intersection )
            current_sum = len( current )
                
        result = vp_sum / (sqrt(current_sum) * sqrt(lneighbor))
        
        return result
    
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
    
    def vec_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_vec_map.get(session);
    
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
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()
        
        
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
        
        
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
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
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                         
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
                        

    def calc_similarity(self, session_items, sessions, dwelling_times, timestamp ):
        
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids: it contains set of sessions ids in which items of current session occur. 
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        
        pos_map = {}
        length = len( session_items )
        
        count = 1
        for item in session_items:
            if self.weighting is not None: # to give the weights of items in current sessions.... this function give more weight to last occuring items in currents 
                pos_map[item] = getattr(self, self.weighting)( count, length )
                count += 1
            else:
                pos_map[item] = 1
            
        if self.dwelling_time:
            dt = dwelling_times.copy()
            dt.append(0)
            dt = pd.Series(dt, index=session_items)  
            dt = dt / dt.max()
            #dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
            dt[session_items[-1]] = 1
        
            #print(dt)
            for i in range(len(dt)):
                pos_map[session_items[i]] *= dt.iloc[i]
            #print(pos_map)    
        
        if self.idf_weighting_session:
            max = -1
            for item in session_items:
                pos_map[item] = self.idf[item] if item in self.idf else 0
#                 if pos_map[item] > max:
#                     max = pos_map[item]
#             for item in session_items:
#                 pos_map[item] = pos_map[item] / max
                
            
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            n_items = self.items_for_session( session )
            sts = self.session_time[session]

            #dot product
            #similarity = self.vec(items, n_items, pos_map)
            similarity = getattr(self, self.similarity)( items, n_items, pos_map )
            
            if similarity > 0:
                
                if self.weighting_time:
                    diff = timestamp - sts
                    days = round( diff/ 60 / 60 / 24 )
                    decay = pow( 7/8, days )
                    similarity *= decay
                
                #print("days:",days," => ",decay)
                
                neighbors.append((session, similarity))
                
                
        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id, dwelling_times, timestamp ):
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
        # find the relevant sessions where items of current sessions
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors, dwelling_times, timestamp )
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        # possile_neighbour is a list of sessions [(session_id, score), (session_id, score)]
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors, current_session_items, timestamp):
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
        iset = set( current_session_items )
        # iterate over the sessions
        for session in neighbors:
            # get the items of neighbour session
            items = self.items_for_session( session[0] )
            step = 1
            for item in reversed( current_session_items ):
                if item in items: # if last interacted item presents in current session, then decay the 
                    decay = getattr(self, self.weighting_score+'_score')( step )
                    
                    break
                step += 1
            
            
            # for item in reversed( current_session_items ):
            #     last_item_category = self.check_category(item)
            #     sim = 1
            #     dis_sim = 1
            #     for i in items:
            #         category1 = self.check_category(i)
            #         if (last_item_category == category1):
            #             sim +=1
            #         else:
            #             dis_sim +=1
            #     decay2 = sim/ len(items)
            #     print("Decy   ", decay2)        
            #     break
                
                        
            for item in items:
                
                if not self.remind and item in iset:
                    continue
                
                old_score = scores.get( item )
                new_score = session[1]
                new_score = new_score if not self.idf_weighting else new_score + ( new_score * self.idf[item] * self.idf_weighting )
                new_score = (new_score * decay) #* decay2
                
                if not old_score is None:
                    new_score = old_score + new_score
                    
                scores.update({item : new_score})
                    
        return scores
    
    
    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same_score(self, i):
        return 1
    
    def div_score(self, i):
        return 1/i
    
    def log_score(self, i):
        return 1/(log10(i+1.7))
    
    
    def quadratic_score(self, i):
        return 1/(i*i)
    
    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0
    
    def same(self, i, length):
        return 1
    
    def div(self, i, length):
        return i/length
    
    def log(self, i, length):
        return 1/(log10((length-i)+1.7))
    
    def quadratic(self, i, length):
        return (i/length)**2
    
    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict() 
        self.item_session_map = dict()
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