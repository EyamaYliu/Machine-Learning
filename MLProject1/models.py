import numpy as np


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


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        # Design first and second half
        self.num_input_features = X.shape[1]


    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        first_half = None
        second_half = None

        num_examples, num_input_features = X.shape

        
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
	
	
        X=X.copy()
        y_hat = np.empty([num_examples], dtype=np.int)
        


        cnt = 0
        while cnt < num_examples:

                if self.num_input_features % 2 == 0:		
                        self.first_half = X[cnt,0:int(self.num_input_features/2)]
                        self.second_half = X[cnt,int(self.num_input_features/2):self.num_input_features]
                else:
                        self.first_half = X[cnt,0:int((self.num_input_features-1)/2)]
                        self.second_half = X[cnt,int((self.num_input_features+1)/2):self.num_input_features]

                firsthalfSum = self.first_half.sum()
                secondhalfSum = self.second_half.sum()

                

                if firsthalfSum > secondhalfSum:
                        y_hat[cnt] = 1
                else:
                        y_hat[cnt] = 0
                cnt += 1

        return y_hat


class Perceptron(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.w = []

    def fit(self, X, y,n,I):
        # TODO: Write code to fit the model.
        #Extract input info
        X = X.copy()
        num_examples, self.num_input_features = X.shape
        self.reference_label = y.copy()
        for line in self.reference_label:
                if line == 0:
                        line = -1;

        #Initializing traing information
        Iteration = I
        self.w = self.num_input_features *[0]
        y_hat = np.empty([num_examples], dtype=np.int)
        learning_rate = n
        
        k = 0
        cnt = 0
        
        while k < Iteration:
                while cnt < num_examples:
                        featurecnt = 0
                        dotproduct = 0
                        while featurecnt < self.num_input_features:
                                dotproduct += self.w[featurecnt]*X[cnt,featurecnt]
                                featurecnt += 1


                        if dotproduct >= 0:
                                y_hat[cnt] = 1
                        else:
                                y_hat[cnt] = -1
                        if self.reference_label[cnt] == 0:
                                self.reference_label[cnt] = -1


                        #Updating weight if prediction is wrong
                        if y_hat[cnt] != self.reference_label[cnt]:
                                featurecnt = 0
                                for weight in self.w:
                                        weight = weight + learning_rate*self.reference_label[cnt]*X[cnt,featurecnt]
                                        featurecnt += 1
                        cnt +=1
                        dotproduct = 0
                        featurecnt = 0
                cnt = 0
                k += 1

        

    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
                raise Exception('fit must be called before predict.')

#Standard inpute testing
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        #Initializing prediction
        y_hat = np.empty([num_examples], dtype=np.int)
        
        cnt = 0
        while cnt < num_examples:
                featurecnt = 0
                dotproduct = 0
                while featurecnt < self.num_input_features:
                        dotproduct += self.w[featurecnt]*X[cnt,featurecnt]
                        featurecnt += 1

                if np.sign(dotproduct) >= 0:
                        y_hat[cnt] = 1
                else:
                        y_hat[cnt] = 0
                dotproduct = 0
                cnt += 1
        
        return y_hat


# TODO: Add other Models as necessary.
