import numpy as np


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y, **kwargs):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y, **kwargs):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LambdaMeans(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.miu = None

    def fit(self, X, _, **kwargs):
        """  Fit the lambda means model  """
        assert 'lambda0' in kwargs, 'Need a value for lambda'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        lambda0 = kwargs['lambda0']
        iterations = kwargs['iterations']
        # TODO: Write code to fit the model.  NOTE: labels should not be used here.

        try:
            X = X.toarray()
        except:
            pass

        num_of_samples = X.shape[0]
        num_of_features = X.shape[1]

        #Initializing lambda0 value
        if lambda0 == 0:
            mean = np.mean(X,0) 
            sum = 0
            for i in range(num_of_samples):
                temp = np.square(X[i,]-mean).sum()
                sum += np.sqrt(temp)
            lambda0 = sum/num_of_samples

        self.miu = np.mean(X,0)

        #Numbers of clusters + 1, so we can set iteration numbers to be k
        num_kplus1 = 1

        for i in range(iterations):
            #initialize training label
            y_hat = np.zeros(num_of_samples)


            for j in range(num_of_samples):
                current_cluster_index = 0
                min_distance = 10000000


                for k in range(num_kplus1):
                    temp_distance = np.square(X[j]-u[k]).sum()
                    temp_distance = np.sqrt(temp_distance)

                    if temp_distance < minValue:
                        current_cluster_index = k
                        min_distance = temp_distance

                if min_distance < lambda0:
                    y_hat[j] = current_cluster_index

                else:
                    self.miu = np.vstack((miu,X[i,:]))
                    y_hat[j] = num_kplus1
                    num_kplus1 += 1
            for i in range(num_k):
                self.miu[i] = np.mean(X[y_hat ==i,:],0)
    





    def predict(self, X):

        try:
            X = X.toarray()
        except:
            pass

        num_of_samples = X.shape[0]
        num_of_features = X.shape[1]


        #Initial prediction labels        
        prediction = np.zeros(num_of_samples)

        if num_of_features < self.miu.shape[1]:
            features = num_of_features
        else:
            features = self.miu.shape[1]

        X_used = X[:,:features]
        miu_used = self.miu[:,:features]

        

        for i in range(num_of_samples):
            min_distance = 100000000
            
            #Iterate through each existing group
            for j in range(miu_used.shape[0]):
                temp_distance = np.square(X.used[i]-miu_used[j]).sum()
                temp_distance = np.sqrt(temp_distance)
                
                if temp_distance < min_distance:
                    prediction[i] = j
                    min_distance = temp_distance

        return prediction


class StochasticKMeans(Model):

    def __init__(self):
        super().__init__()
        self.miu = None

    def fit(self, X, _, **kwargs):
        assert 'num_clusters' in kwargs, 'Need the number of clusters (K)'
        assert 'iterations' in kwargs, 'Need the number of EM iterations'
        num_clusters = kwargs['num_clusters']
        iterations = kwargs['iterations']

        try:
            X = X.toarray()
        except:
            pass

        num_of_samples = X.shape[0]
        num_of_features = X.shape[1]
        X_minimum = X.min(0)
        X_maximum = X.max(0)

        #Initialize centers
        self.miu = np.zeros(num_clusters,num_of_features)
        self.miu[0] = X_minimum
        if num_clusters > 1:
            pieces = num_clusters - 1
            for i in range(1,pieces):
                self.miu[i] = (i/pieces)*X_minimum + (1-(i/pieces))*X_maximum
            self.miu[-1] = X_maximum

        for iter in range(iterations):
            #Initialize beta
            c = 2
            beta = c * (iter+1)
            #Initialize pnk
            p_nk = np.zeros(num_of_samples,num_clusters)

            for n in range(num_of_samples):

                #Get distance of samples to each cluster
                distance = []
                for k in range(num_clusters):
                    temp = np.square(X[n]-self.miu[k]).sum()
                    distance.append(np.sqrt(temp))
                avg_distance = np.mean(distance)


                #Calc sum of distances for p_nk
                total_distance = 0
                for k in range(num_clusters):
                    total_distance += np.exp((-1*beta*distance[k])/avg_distance)
                
                #Updating p_nk
                for k in range(num_clusters):
                    p_nk[n,k] = (np.exp((-1*beta*distance[k])/avg_distance)) / total_distance

            for k in range(num_clusters):
                temp = np.zeros(num_of_features)
                for n in range(num_of_samples):
                    temp +=p_nk[n,k] * X[i]
                self.miu[k] = temp/np.sum(p[:,k])
        
        

        
        

    def predict(self, X):
        try:
            X = X.toarray()
        except:
            pass

        num_of_samples = X.shape[0]
        num_of_features = X.shape[1]


        #Initial prediction labels        
        prediction = np.zeros(num_of_samples)

        if num_of_features < self.miu.shape[1]:
            features = num_of_features
        else:
            features = self.miu.shape[1]

        X_used = X[:,:features]
        miu_used = self.miu[:,:features]

        for i in range(num_of_samples):
            min_distance = 1000000000
            for j in range(self.miu.shape[0]):
                temp = (np.square(X_used[i]-miu.used[j])).sum()
                temp = np.sqrt(temp)
            if temp < min_distance:
                min_distance = temp
                prediction[i] = j

        return prediction

    
