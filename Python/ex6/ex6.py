import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sio

data = sio.loadmat('ex6data1.mat')
X = data['X']
y = np.squeeze(data['y'])

pos_idx = np.where(y==1)
neg_idx = np.where(y==0)
plt.scatter(X[pos_idx,0],X[pos_idx,1],marker='o')
plt.scatter(X[neg_idx,0],X[neg_idx,1],marker='x')
plt.show()

from sklearn import svm
svc = svm.LinearSVC(C=1,loss= 'hinge')

print svc

svc.fit(X,y)
print svc.coef_
print svc.intercept_

print svc.predict(X)

print svc.score(X,y)

svc2 = svm.LinearSVC(C=100,loss= 'hinge')

svc2.fit(X,y)
print svc2
print svc2.coef_
print svc2.intercept_

print svc2.predict(X)

print svc2.score(X,y)

svc3 = svm.SVC(kernel='linear',C=1)

svc3.fit(X,y)

print svc3
print svc3.coef_
print svc3.intercept_

print svc3.predict(X)

print svc3.score(X,y)

svc4 = svm.SVC(kernel='linear',C=100)

svc4.fit(X,y)

print svc4
print svc4.coef_
print svc4.intercept_

print svc4.predict(X)

print svc4.score(X,y)

#form reading around LinearSVC penalize the intercept which is not desired, so use svc with linear kernel

x_plt = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
y1_plt =  -(svc.coef_[0][0]*x_plt + svc.intercept_[0])/svc.coef_[0][1]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1], s=50, c=svc.decision_function(X), cmap='seismic')
ax.plot(x_plt,y1_plt,'-b')
plt.show()

y2_plt =  -(svc2.coef_[0][0]*x_plt + svc2.intercept_[0])/svc2.coef_[0][1]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1], s=50, c=svc2.decision_function(X), cmap='seismic')
ax.plot(x_plt,y2_plt,'-r')
plt.show()

y3_plt =  -(svc3.coef_[0][0]*x_plt + svc3.intercept_[0])/svc3.coef_[0][1]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1], s=50, c=svc3.decision_function(X), cmap='seismic')
ax.plot(x_plt,y3_plt,'-g')
plt.show()

y4_plt =  -(svc4.coef_[0][0]*x_plt + svc4.intercept_[0])/svc4.coef_[0][1]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1], s=50, c=svc4.decision_function(X), cmap='seismic')
ax.plot(x_plt,y4_plt,'-k')
plt.show()

def gaussianKernel(x1,x2,sigma):
    #temp = x1-x2
    #return np.exp(-(temp.T.dot(temp))/(2.0*sigma**2))
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))
    
x1 = np.array([1,2,1])
x2 = np.array([0,4,-1])
sigma = 2

print "Testing gaussian kernel function for x1 = [1; 2; 1], x2 = [0; 4; -1], sigma =2 : "
print gaussianKernel(x1,x2,sigma)

data = sio.loadmat('ex6data2.mat')
X = data['X']
y = np.squeeze(data['y'])
pos_idx = np.where(y==1)
neg_idx = np.where(y==0)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[pos_idx,0],X[pos_idx,1],marker='o')
ax.scatter(X[neg_idx,0],X[neg_idx,1],marker='x')
plt.show()

svc = svm.SVC(C=1,gamma=50,probability=True)
#gamma is the value of 1/(2*sigma^2) in the gaussiankernel.
# set probability=False (which is default) if not using predict_prob method. 
# prbability=True will slow down the fit method
svc.fit(X,y)
svc.score(X,y)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1], s=30, c=svc.predict_proba(X)[:,0], cmap='seismic')
plt.show()

predictions = svc.predict(X)
fig, ax = plt.subplots(figsize=(12,8))
x1plt = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
x2plt = np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
X1,X2 = np.meshgrid(x1plt,x2plt)
predictions = np.zeros(X1.shape)
for i in xrange(X1.shape[1]):
    predictions[:,i] = svc.predict(np.column_stack([X1[:,i],X2[:,i]]))

ax.scatter(X[pos_idx,0],X[pos_idx,1],s=30,marker='o')
ax.scatter(X[neg_idx,0],X[neg_idx,1],s=30,marker='x')
ax.contour(X1,X2,predictions,levels=[0.5])
plt.show()

data = sio.loadmat('ex6data3.mat')
X = data['X']
y = np.squeeze(data['y'])
Xval = data['Xval']
yval = data['yval']

def dataset3Params(X,y,Xval,yval):
    C_vals = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    gamma_vals = 1/(2.0*(np.array([0.01,0.03,0.1,0.3,1,3,10,30])**2 ))#gamma = 1/(2*sig^2)
    min_error = 1
    ret = [1,0.3]
    for c in C_vals:
        for g in gamma_vals:
            svc = svm.SVC(C=c,gamma=g)
            svc.fit(X,y)
            val_error = svc.score(Xval,yval)
            if  val_error < min_error:
                min_error = val_error
                ret = [c,g]
    return ret
            
C,sigma = dataset3Params(X, y, Xval, yval)
