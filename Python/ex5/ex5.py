import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sio

data = sio.loadmat('ex5data1.mat')
Xtest = data['Xtest']
Xval = data['Xval']
X = data['X']
y = data['y']
ytest = data['ytest']
yval = data['yval']

m = X.shape[0]

def linearRegCostFunction(theta,X,y,lam):
    m = X.shape[0]
    h = X.dot(theta) - np.squeeze(y)
    J = (1.0/(2*m))*(h.T.dot(h)) + ((lam/(2.0*m))*theta[1:].dot(theta[1:]))
    grad = (1.0/m)*(X.T.dot(h) + lam*np.concatenate([[0],theta[1:]]))
    return J,grad

theta = np.ones(2)
J,grad = linearRegCostFunction(theta,np.concatenate([np.ones([m,1]),X],1),y,1)
print "Cost at theta = [1;1] :"
print J
print "Gradient:"
print grad

def trainLinearReg(X,y,lam):
    initial_theta = np.ones(X.shape[1])
    theta = opt.minimize(fun=linearRegCostFunction,x0 = initial_theta,args=(X,y,lam),method='TNC', jac=True, options={'maxiter': 200} )
    return theta

lam = 0
X1 = np.concatenate([np.ones([X.shape[0],1]),X],1)
res = trainLinearReg(X1,y,lam)
theta = res.x
plt.plot(X,y,'rx')

X_plt = np.ones([2,2])
X_plt[0,1] = np.min(X)
X_plt[1,1] = np.max(X)
h_plt = X_plt.dot(theta)

plt.plot(X_plt[:,1],h_plt)
plt.show()

def learningCurve(X,y,Xval,yval,lam):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in xrange(m):
        res = trainLinearReg(X[:i+1],y[:i+1],lam)
        theta_train = res.x
        error_train[i], buf =  linearRegCostFunction(theta_train,X[:i+1],y[:i+1],0)
        error_val[i], buf = linearRegCostFunction(theta_train,Xval,yval,0)
    return error_train,error_val

lam = 0
error_train,error_val = learningCurve(np.concatenate([np.ones([m,1]),X],1),y,np.concatenate([np.ones([Xval.shape[0],1]),Xval],1),yval,lam)
print error_train
print error_val
plt.plot(np.arange(1,m+1),error_train,np.arange(1,m+1),error_val)
plt.show()

def polyFeatures(X,p):
    res = X
    for i in xrange(2,p+1):
        res = np.column_stack([res,X**i]) #column stack seems to work even if the input vector is in (x,)  size form
    return res


def featureNormalize(X):
    mu = X.mean(0)
    X = X - mu
    std = X.std(0,ddof=1)
    X = X/std
    return X,mu,std
    
p=8
Xpoly = polyFeatures(X,p)
Xpoly,mu,std = featureNormalize(Xpoly)
Xpoly = np.concatenate([np.ones([Xpoly.shape[0],1]),Xpoly],1)

Xpoly_test = polyFeatures(Xtest,p)
Xpoly_test = Xpoly_test - mu
Xpoly_test = Xpoly_test/std
Xpoly_test = np.concatenate([np.ones([Xpoly_test.shape[0],1]),Xpoly_test],1)

Xpoly_val = polyFeatures(Xval,p)
Xpoly_val = Xpoly_val - mu
Xpoly_val = Xpoly_val/std
Xpoly_val = np.concatenate([np.ones([Xpoly_val.shape[0],1]),Xpoly_val],1)

print "Normalized training example 1: "
print Xpoly[0,:]


def plotFit(min_x,max_x,mu,sigma,theta,p):
    x = np.arange(min_x-15,max_x+25,0.05)
    X_poly = polyFeatures(x,p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma
    X_poly = np.column_stack([np.ones([x.shape[0],1]),X_poly])
    plt.plot(x, X_poly.dot(theta), '--', lw=2)
    
lam = 0
res = trainLinearReg(Xpoly,y,lam)
theta = res.x
plt.plot(X,y,'rx')
plotFit(np.min(X),np.max(X),mu,std,theta,p)
plt.show()

def validationCurve(X,y,Xval,yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)
    for i in xrange(lambda_vec.size):
        lam = lambda_vec[i]
        res = trainLinearReg(X,y,lam)
        theta_train = res.x
        error_train[i],buf = linearRegCostFunction(theta_train,X,y,0)
        error_val[i],buf = linearRegCostFunction(theta_train,Xval,yval,0)
    return lambda_vec,error_train,error_val

lam_vec,error_train,error_val = validationCurve(Xpoly,y,Xpoly_val,yval)
print error_train
print error_val

plt.plot(lam_vec,error_train,lam_vec,error_val)
plt.show()


lam = 3
res = trainLinearReg(Xpoly,y,lam)
theta_train = res.x
error_test, buf = linearRegCostFunction(theta_train,Xpoly_test,ytest,0)
print "Test error:"
print error_test