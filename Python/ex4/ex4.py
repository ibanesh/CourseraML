import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
import scipy.io as sio

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

mat_content = sio.loadmat('ex4data1.mat')
X = mat_content['X']
y = mat_content['y']

mat_content = sio.loadmat('ex4weights.mat')
Theta1 = mat_content['Theta1']
Theta2 = mat_content['Theta2']

nn_params = np.concatenate((np.ravel(Theta1),np.ravel(Theta2)),0)

def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lam):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],[hidden_layer_size,input_layer_size+1])
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],[num_labels,hidden_layer_size+1])
    
    m = X.shape[0]
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    X = np.concatenate((np.ones([m,1]),X),1)
    
    #Feed Forward
    z2 = Theta1.dot(X.T)  # 25x5000
    a2 = np.concatenate((np.ones([1,m]),sigmoid(z2)),0) # 26x5000
    z3 = Theta2.dot(a2)
    h = sigmoid(z3.T)
    
    temp = np.zeros([m,num_labels])
    for i in xrange(m):
        temp[i][y[i]-1] = 1
    y = temp
    
    h_flat = h.ravel()
    y_flat = y.ravel()
    J = (1.0/m)*(-y_flat.T.dot(np.log(h_flat)) - (1-y_flat).dot(np.log(1-h_flat)) )
    
    # Regularization of Cost
    reg = (lam/(2.0*m))*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))
    J += reg
    
    #Back propagation
    del3 = h-y #5000x10
    del2 = (del3.dot(Theta2))*((a2*(1-a2)).T) #5000x26
    del2 = del2[:,1:]
    
    Theta2_grad = (1.0/m)*(a2.dot(del3)).T
    Theta1_grad = (1.0/m)*(del2.T.dot(X))
    
    #adding regularization
    Theta1_grad[:,1:] += (1.0/m)*(lam*Theta1[:,1:])
    Theta2_grad[:,1:] += (1.0/m)*(lam*Theta2[:,1:])
    
    grad = np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])
    
    return J, grad
    
def nnGradient(nn_params, input_layer_size, hidden_layer_size,
               num_labels, X, y, lam):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],[hidden_layer_size,input_layer_size+1])
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],[num_labels,hidden_layer_size+1])
    
    m = X.shape[0]
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    X = np.concatenate((np.ones([m,1]),X),1)
    
    #Feed Forward
    z2 = Theta1.dot(X.T)  # 25x5000
    a2 = np.concatenate((np.ones([1,m]),sigmoid(z2)),0) # 26x5000
    z3 = Theta2.dot(a2)
    h = sigmoid(z3.T)
    
    #Back propagation
    del3 = h-y #5000x10
    del2 = (del3.dot(Theta2))*((a2*(1-a2)).T) #5000x26
    del2 = del2[:,1:]
    
    Theta2_grad = (1.0/m)*(a2.dot(del3)).T
    Theta1_grad = (1.0/m)*(del2.T.dot(X))
    
    #adding regularization
    Theta1_grad[:,1:] += (1.0/m)*(lam*Theta1[:,1:])
    Theta2_grad[:,1:] += (1.0/m)*(lam*Theta2[:,1:])
    
    grad = np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])

    return grad

def randInitWeights(L_in, L_out):
    epsilonInit = 0.12
    return np.random.rand(L_out, 1 + L_in) *((2 * epsilonInit) - epsilonInit)
    
initial_Theta1 = randInitWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitWeights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate([Theta1.ravel(),Theta2.ravel()])

lam =1

nn_params_opt = opt.minimize(fun=nnCostFunction,x0=initial_nn_params,
              args=(input_layer_size, hidden_layer_size,num_labels,X,y,lam),method='TNC', jac=True, options={'maxiter': 50})
    
Theta1 = np.reshape(nn_params_opt.x[0:hidden_layer_size*(input_layer_size+1)],[hidden_layer_size,input_layer_size+1])
Theta2 = np.reshape(nn_params_opt.x[hidden_layer_size*(input_layer_size+1):],[num_labels,hidden_layer_size+1])

def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    
    p = np.zeros([m,1])
    
    h1 = sigmoid(Theta1.dot(np.concatenate((np.ones([m,1]),X),1).T))
    h2 = sigmoid(Theta2.dot(np.concatenate((np.ones([1,m]),h1),0)))
    
    return np.squeeze(h2.argmax(0)+1)

pred = predict(Theta1,Theta2,X)
print "Accuracy: "
print np.mean(pred == y.squeeze())*100
    