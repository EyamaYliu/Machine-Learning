import numpy as np
from scipy.special import expit


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
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

    def fit(self, X, y):
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


class LogisticRegression(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        pass

    def fit(self, X, y):

        numFeatureSelected = 10
	
        # TODO: Write code to fit the model.
        try:
            X = X.toarray()
        except:
            pass #Transform X from whatever to ndarray, if X 

        #if X.shape[1] > numFeatureSelected:
        X = featureSelect(X,y,numFeatureSelected)
        

        self.num_input_features = X.shape[1]
        self.reference_example = X[0, :]
        self.reference_label = y[0]
        learning_rate = 0.01 #arbitrarily setting
        iterations = 20
        self.w = np.array(self.num_input_features * [0])

        for num in range(iterations):
            self.w = gradDecent(y,X,learning_rate,self.w)




    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
                raise Exception('fit must be called before predict.')

	#Standard input testing
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        
        try:
            X = X.toarray()
        except:
            pass
        

        #Initializing prediction
        y_hat = []
        
        
        cnt = 0
        y_hato = 0
        for cnt in range(num_examples):
                Xi = X[cnt,:]
                wx = np.dot(self.w,Xi)
                probYis1 = sigmoid(wx)

                if probYis1 >= 0.5:
                        y_hato = 1
                else:
                        y_hato = 0

                y_hat.append(y_hato)
        
        y_hat = np.asarray(y_hat)
        return y_hat

#Calculate conditional entropy and return corresponding X array
def entropyCalc(x,y):
        
        entropy = 0
        y_labels = np.unique(y)
        for i in y_labels:
                yid = np.where(y == i)
                #Frequency of specific y appearing in y
                pyi = np.shape(np.where(y == i))[1] / np.shape(y)[0]  #
                xColumns = x[yid]
                x_labels = np.unique(np.array(xColumns))
                for j in x_labels:

                        pxj = np.shape(np.where(x == j))[1]/np.shape(y)[0] #p(x_j)

                        xid = np.where(xColumns == j)
                        pxjyi = (np.shape(xid)[1]/np.shape(yid)[1])*pyi #p(x_j,y_i)
                        entropy = entropy - pxjyi * np.log(pxjyi/pxj)
        return entropy

def featureSelect(X,y,n):
        entropy = []
        order = []
        features = []
 
        discardNum = X.shape[1]-n
        

        for i in range(X.shape[1]):
                x = X[:,i]
                xmean = np.mean(x)
                x = np.where(x>=xmean,1,0) #set all data into binary outcome for entropy calc

                entropy.append(entropyCalc(x,y))
        order = list(range(0,len(entropy)))
        point = zip(entropy,order)
        sp = sorted(point)


        new_ord = [point[1] for point in sp] #change order with sorted entropy 


        for j in range(discardNum):
                del new_ord[-1] #discard the greatest value of conditional entropys

        discardColomnIndexes = list(set(order)-set(new_ord))


        for i in discardColomnIndexes:
                X[:,i] = 0

        return X


def sigmoid(z):
        return expit(z)

def gradDecent(y,X,n,weight):

        i = 0
        dw = np.array(X.shape[1] * [0])
        for feature_lines in X:
                dot = weight.dot(feature_lines)
                dw = dw + y[i]*sigmoid(-dot)*feature_lines + (1-y[i])*sigmoid(dot)*(-feature_lines)
                i += 1
        weight = weight + n*dw
        return weight



# TODO: Add other Models as necessary.
