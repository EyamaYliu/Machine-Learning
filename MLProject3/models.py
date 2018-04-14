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
            X = X.toarray()
        except:
            pass #Transform X from whatever to ndarray 

       

        #num of samples,m
        nos = X.shape[0]
        #num_input_features,n
        nof = X.shape[1]
        
        
        D = [1.0/nos]*nos
	
        for i in range(iteration):
            error,yhat,j,c,dim = find_hjc(X,y,D)
            print(j,c)
            
        
            self.y_dim.append(dim)
            self.j_sets.append(j)
            self.cutoff_sets.append(c)

            if error >=0.000005:
                alpha = find_alpha(error)
                print(alpha)
                #Updating weight D
                temp = np.exp(np.multiply(y*-alpha,yhat))
                D = np.multiply(D,temp)
                D /= D.sum()
                self.alpha_sets.append(alpha)
            else:
                break


    def predict(self, X):
        try:
            X = X.toarray()
        except:
            pass #Transform X from whatever to ndarray

        yhat = y_predict =  np.zeros(X.shape[0])

        for i in range(len(self.j_sets)):

            c = self.cutoff_sets[i]
            #print(c)
            ydim = self.y_dim[i]
            j = self.j_sets[i]
            alpha = self.alpha_sets[i]

            #Initiating yjc
            yjc = [0]*X.shape[0]
            yjc = np.asarray(yjc)
            #Extracting j'th column of X
            Xc = X[:,j]

            #Make prediction
            if ydim == 1:
                yjc[Xc >= c] = 1
                yjc[Xc < c] = -1
            if ydim == -1:
                yjc[Xc >= c] = -1
                yjc[Xc < c] = 1

            yhat += yjc*alpha

        
        y_predict[yhat>=0] = 1
        y_predict[yhat<0] = 0
        return y_predict






def find_yhat(j,c,X,y):
    
    
    yhat = np.ones(X.shape[0])
    
    for i in range(X.shape[0]):
        if X[i,j] <= c:
            yhat[i] = -1

    
    if np.sum(yhat == y) >= np.sum(yhat != y):
        return yhat,1
    else:
        return -yhat,-1

def find_hjc(X,y,D):      
    
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] = -1    

    error = float('inf')
    nos = X.shape[0]
    nof = X.shape[1]

    bestc = 0

    for j in range(nof):

        sortj = np.unique(np.sort(X[:,j]))

        #For each row j of X input, store all the cutoffs
        cutoffs = []
        
        for i in range(sortj.shape[0]-1):
            cutoffs.append((sortj[i]+sortj[i+1])/2)
        #cutoffs=np.unique(np.asarray(cutoffs))
        cutoffs = np.asarray(cutoffs)

        for c in cutoffs:
            yhat,yhatdim = find_yhat(j,c,X,y)
            #Get the error from cutoffs
            errors = np.zeros(nos)

            for i in range(errors.shape[0]):
                if yhat[i] != y[i]:
                    errors[i] = 1
            
            weightederror= np.dot(D,errors)
         
            #Update  
            if weightederror < error:

                error = weightederror
                bestyhat = yhat
                bestj = j
                bestc = c
                bestdim = yhatdim
            #print(bestj)
            #print(bestc)
        #print(bestj)
        #print(bestc)
    
    return error,bestyhat,bestj,bestc,bestdim



def find_alpha(error):

    alpha = math.log((1-error)/error) * 0.5
    return alpha



# TODO: Add other Models as necessary.
