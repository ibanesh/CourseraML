import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sio
from sklearn import svm

data = sio.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].ravel()

svc = svm.SVC(C=0.1,kernel='linear')
svc.fit(X,y)
print svc.score(X,y)

data = sio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest']
print svc.score(Xtest,ytest)

sorted_idx = np.argsort(-svc.coef_).ravel()
weight = svc.coef_.ravel()
data = pd.read_csv('vocab.txt',header=None,delim_whitespace=True)
for i in xrange(20):
    print data[1][sorted_idx[i]], weight[sorted_idx[i]]
