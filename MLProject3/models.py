import numpy as np
from scipy.special import expit
import math



class Adaboost:

    def __init__(self):
        super().__init__()
        self.j_sets = []
        self.cutoff_sets = []
        self.alpha_sets = []
        self.y_dim = []

    def fit(self, X, y,iteration):

        try:
            X = np.asarray(X)
        except:
            pass #Transform X from whatever to ndarray 

        #num_input_features,n
        nof = X.shape[1]
        
        #num of samples,m
        nos = X.shape[0]
        
        D = [1.0/nof]*nof
	
        for i in range(iteration):
            error,yhat,j,c,dim = find_hjc(X,y,nof,D)
            self.y_dim.append(dim)

            self.j_sets.append(j)
            self.cutoff_sets.append(c)

            if error >=0.000001:
                alpha = find_alpha(error)
                #Updating weight D
                temp = np.exp(np.multiply(-alpha*y,yhat))
                D = np.multiply(D,temp)
                D /= D.sum()
                self.alpha_sets.append(alpha)
            else:
                break


    def predict(self, X):
        try:
            X = np.asarray(X)
        except:
            pass #Transform X from whatever to ndarray

        yhat = y_predict =  [0]*X.shape[0]
        yhat = np.asarray(yhat)

        for i in range(len(self.j_sets)):
            yjc = [0]*X.shape[0]
            yjc = np.asarray(yjc)
            Xc = X[:,self.j_sets[i]]
            yjc[Xc >= c] = self.y_dim[i]
            yjc[Xc < c] = -1*self.y_dim[i]
            yhat += yjc
        y[yhat>=0] = 1
        y[yhat<0] = 0
        return y






def find_yhat(j,c,X,y):
    
    yhat = np.ones(X.shape[0])
    
    for i in range(X.shape[0]):
        if X[i,j] <= c:
            yhat[i] = -1
    
    if np.sum(y_hat == y) >= np.sum(y_hat != y):
        return yhat,1
    else:
        return -yhat,-1

def find_hjc(X,y,nof,D):
    

    error = float('inf')
    

    for j in range(nof):
        sortj = np.sort(X[:,j])

        #For each row j of X input, store all the cutoffs
        cutoffs = []
        for i in range(sortj.shape[0]-1):
            cutoffs.append((sortj[i]+sortj[i+1])/2)
        cutoffs=np.unique(np.asarray(cutoffs))
                

        for c in cutoffs:
            yhat,yhatdim = find_yhat(j,c,X,y)

            #Get the error from cutoffs
            errors = np.zeros(nos)

            for i in range(errors.shape[0]):
                if yhat[i] != y[i]:
                    errors[i] = 1
            
            weightederror= D.dot(error)
            #Update            
            if weightederror < error:
                error = weightederror
                bestyhat = yhat
                bestj = j
                bestc = c
    
    return error,bestyhat,bestj,bestc,yhatdim



def find_alpha(error):

    alpha = math.log((1-error)/error) * 0.5
    return 



# TODO: Add other Models as necessary.
