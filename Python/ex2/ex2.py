import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

data = pd.read_csv('ex2data1.txt',header=None,names=['Exam 1','Exam 2','Admitted'])
pos = np.where(data['Admitted'] == 1)
neg = np.where(data['Admitted'] == 0)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
X = X.values
X = np.concatenate((np.ones((len(X),1)),X),axis = 1)
y = y.values
m = len(X)
#data = np.loadtxt('ex2data1.txt', delimiter=',')
theta = np.zeros(3)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(theta,X,y):
    m = X.shape[0]
    h = np.squeeze(sigmoid(X.dot(theta)))
    return (1.0/m)*(-y.dot(np.log(h))-(1-y).dot(np.log(1-h)))

def gradient(theta,X,y):
    m = X.shape[0]
    h = np.squeeze(sigmoid(X.dot(theta)))
    return (1.0/m)*(X.T.dot(h-y))

opt_theta = opt.fmin_bfgs(f=costFunction, x0=theta, fprime=gradient, args=(X,y))

# opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X,y))

plt.scatter(X[pos,1],X[pos,2],marker='+',color='b')
plt.scatter(X[neg,1],X[neg,2],marker='o',color='r')
points = np.array([min(X[:,1])-2,max(X[:,1])+2])
plt.plot(points,(-1.0/opt_theta[1])*(opt_theta[0]+ opt_theta[2]*points))
plt.xlabel("Exam 1 score")
plt.ylabel('Exam 2 score')
plt.show()

def predict(theta,X):
    return (sigmoid(X.dot(theta)) >= 0.5)

p = predict(opt_theta,X)
print "Train Accuracy: "
print np.mean(p==y)*100


data = pd.read_csv('ex2data2.txt',header=None,names=['Exam 1','Exam 2','Admitted'])
pos = np.where(data['Admitted'] == 1)
neg = np.where(data['Admitted'] == 0)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
X = X.values
y = y.values
m = len(X)

def mapFeature(X1,X2,degree):
    X1 = X1.reshape((X1.size,1))
    X2 = X2.reshape((X2.size,1))
    res = np.ones((len(X1),1))
    for i in xrange(1,degree+1):
        for j in xrange(0,i+1):
            res = np.concatenate((res,(X1**(i-j))*(X2**j)),axis=1)
    return res
    

X = mapFeature(X[:,0],X[:,1],6)
theta = np.zeros(X.shape[1])

lam = 1.0

def costFunctionReg(theta,X,y,lam):
    h = np.squeeze(sigmoid(X.dot(theta)))
    return (1.0/m)*(-y.dot(np.log(h))-(1-y).dot(np.log(1-h))) + ((lam/(2*m))*(sum(theta[1:]**2)))

def gradientReg(theta,X,y,lam):
    h = np.squeeze(sigmoid(X.dot(theta)))
    return (1.0/m)*(X.T.dot(h-y)) + (lam/m)*np.concatenate(([0],theta[1:]))

print "Cost Function at initial theta (all 0's) :"
print costFunctionReg(theta,X,y,lam)
print "Gradients:"
print gradientReg(theta,X,y,lam)
#lam =10.0
#print costFunctionReg(np.ones(X.shape[1]),X,y,lam)
#print gradientReg(np.ones(X.shape[1]),X,y,lam)

opt_theta = opt.fmin_bfgs(f=costFunctionReg, x0=theta, fprime=gradientReg, args=(X,y,lam))
print opt_theta


u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (mapFeature(np.array(u[i]), np.array(v[j]),6).dot(opt_theta))
        
z = z.T

plt.scatter(X[pos,1],X[pos,2],marker='+',color='b')
plt.scatter(X[neg,1],X[neg,2],marker='o',color='r')
plt.xlabel("Exam 1 score")
plt.ylabel('Exam 2 score')

plt.contour(u,v,z,levels=[0])
plt.show()




p = predict(opt_theta,X)
print "Train Accuracy: "
print np.mean(p==y)*100