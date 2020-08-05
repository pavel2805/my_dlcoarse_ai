import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 2:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
            #print('dist 0 0',dists[0,0])
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)               
            
            
    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                #pass
                dists[i_test][i_train]=np.sum(np.abs(self.train_X[i_train]-X[i_test]))
        return dists
    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            #pass
            aa=(np.abs(self.train_X-X[i_test]))
            #print('aa.shape',aa.shape) #expect 121.3072
            bb=np.sum(aa,axis=1)
            #print('bb.shape, after np.sum',bb.shape)
            dists[i_test]=bb
        return dists
        

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        #print('self.train_X.shape',self.train_X.shape)
        #print('ttest_X.shape',X.shape)
        aaa_train=self.train_X[np.newaxis,...]
        bbb_test=X[:,np.newaxis,:]
        #print('aaa_train_X.shape',aaa_train.shape)
        #print('bbb_test_X.shape',bbb_test.shape)
        ccc=np.abs(aaa_train-bbb_test)
        #print(ccc.shape)
        dists=np.sum(ccc,axis=2)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            #aa=dists[i].argmin()
            sorted_index=dists[i].argsort() #array of sorted indexes
            #print('sorted index',sorted_index)
            
            #test print
            if i == 0:
                #print('sorted inex.shape',sorted_index.shape)
                for index in sorted_index:
                    #print('dists[0]', dists[i][index])
                    pass
                    
            k_nearest = np.zeros(2,)
            #print('k__nearet',k_nearest)
            
            #print('train_y',train_y[:10])
            for kin in (range(self.k)):
                if self.train_y[sorted_index[kin]]==False:
                    k_nearest[0]+=1
                else:
                    k_nearest[1]+=1
            #print('k__nearet',k_nearest)
            if k_nearest[0]>k_nearest[1]:
                pred[i]=False
            else:
                pred[i]=True
            #pred[i]=self.train_y[sorted_index[0]]
            #print(pred[i])
            pass
        
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        #print('train_y',self.train_y[:10])
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            
            sorted_index=dists[i].argsort() #array of sorted indexes
            #print('sorted index',sorted_index)
            
            #test print
            if i < 5:
                #print('sorted inex.shape',sorted_index.shape)
                for index in sorted_index:
                    #print('dists[0]', dists[i][index])
                    pass
                    
           
            k_nearest = np.zeros(10,)
            
            for kin in (range(self.k)):
                k_nearest[self.train_y[sorted_index[kin]]]+=1
                
            k_nearest_ind=k_nearest.argsort()
            
            pred[i]=k_nearest_ind[-1]
                
            #print(pred[i])
            pass
        
       
        return pred
