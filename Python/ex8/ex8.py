import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

data = sio.loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

def estimateGaussian(X):
    mu = X.mean(0)
    Sig2 = X.var(0)
    return mu,Sig2

mu,Sig2 = estimateGaussian(X)

def multivariateGaussian(X,mu,Sig2):
    k = len(mu)
    if Sig2.ndim < 2 or Sig2.shape[1] == 1:
        Sig2 = np.diagflat(Sig2)
    X = X - mu
    p = (((2.0*np.pi)**(-k/2.0)) * (np.linalg.det(Sig2)**(-0.5)))*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(Sig2))*X,1))
    return p

p = multivariateGaussian(X,mu,Sig2)
pval = multivariateGaussian(Xval,mu,Sig2)
print pval[:10]

#note : np.sum(X.dot(np.linalg.pinv(Sig2))*X,1) -> after doing X*(Sig2^-1), we want to vector multiply each row of the result with each row in X. This is done here with element wise multiplication and sum. 

def selectThreshold(yval,pval):
    bestEpsilon = 0
    bestF1 = 0
    for epsilon in np.linspace(np.min(pval),np.max(pval),1000):
        predictions = pval < epsilon
        pos = np.argwhere(predictions)
        neg = np.argwhere(predictions==0)
        TP = np.sum(yval[pos]) #True positive
        FP = len(pos) - TP # or np.sum(yval[pos]==0) # False positive
        FN = np.sum(yval[neg]) #False negative
        TN = len(neg) - FN # or np.sum(yval[neg]==0) # True negative
        precision = TP/float(TP+FP)
        recall = TP/float(TP+FN)
        F1 = 2.0*precision*recall/float(precision+recall)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

epsilon, F1 = selectThreshold(yval,pval)
print epsilon,F1