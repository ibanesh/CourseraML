import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sio



def sigmoid(z):
    return 1.0/(1+np.exp(-z))
    
    
def costFunctionReg(theta,X,y,lam):
    h = np.squeeze(sigmoid(X.dot(theta)))
    m = X.shape[0]
    return (1.0/m)*(-y.dot(np.log(h))-(1-y).dot(np.log(1-h))) + ((lam/(2.0*m))*(sum(theta[1:]**2)))


def gradientReg(theta,X,y,lam):
    m = X.shape[0]
    h = np.squeeze(sigmoid(X.dot(theta)))
    return (1.0/m)*(X.T.dot(h-y)) + (lam/float(m))*np.concatenate(([0],theta[1:]))


#m = X.shape[0]

theta_t = np.array([-2,-1,1,2])
X_t = np.concatenate((np.ones([5,1]),np.reshape(np.arange(1,16),[3,5]).T/10.0),1)
y_t = np.array([1,0,1,0,1]) >= 0.5
lambda_t = 3
J = costFunctionReg(theta_t,X_t,y_t,lambda_t)
grad = gradientReg(theta_t,X_t,y_t,lambda_t)
print "Cost : "
print J
print "Gradient : "
print grad

def oneVsAll(X,y,num_labels,lam):
    m = X.shape[0]
    n = X.shape[1]
    
    all_theta = np.zeros([num_labels,n+1])
    X = np.concatenate((np.ones([m,1]),X),1)
    initial_theta = np.zeros(n+1)
    for c in xrange(num_labels): #xrange(num_labels):
        all_theta[c,:] = opt.fmin_bfgs(f=costFunctionReg,x0=initial_theta,
                                       fprime=gradientReg,args=(X,y == c+1,lam))
    return all_theta

mat_content = sio.loadmat('ex3data1.mat')
X = mat_content['X']
y = np.squeeze(mat_content['y'])
input_layer_size = 400
num_labels = 10
lam = 0.1
all_theta = oneVsAll(X,y,num_labels,lam)

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    X = np.concatenate((np.ones([m,1]),X),1)
    return np.argmax(sigmoid(X.dot(all_theta.T)),1)+1

pred = predictOneVsAll(all_theta, X)
print "Accuracy: "
print np.mean(pred == y)*100

#-----------------------------------------------------------
#Part 2

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
#X and y already loaded from part 1
mat_content = sio.loadmat('ex3weights.mat')
Theta1 = mat_content['Theta1']
Theta2 = mat_content['Theta2']

def predict(Theta1,Theta2,X):
    m = X.shape[0]
    X = np.concatenate((np.ones([m,1]),X),1)
    a2 = sigmoid(Theta1.dot(X.T))
    a2 = np.concatenate((np.ones([1,a2.shape[1]]),a2),0)
    h = sigmoid(Theta2.dot(a2))
    return np.squeeze(h.argmax(0)+1)

pred = predict(Theta1,Theta2,X)
print "Accuracy: "
print np.mean(pred == y)*100