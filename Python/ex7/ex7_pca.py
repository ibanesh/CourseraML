import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

data = sio.loadmat('ex7data1.mat')
X = data['X']

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0],X[:,1])
plt.show()

def featureNormalize(X):
    mu = X.mean()
    sig = X.std()
    return (X-X.mean())/float(X.std()),mu,sig

def pca(X):
    m = X.shape[0]
    cov = (1.0/m)*X.T.dot(X)
    U,S,V = np.linalg.svd(cov)
    return U,S,V

X_norm,mu,sig = featureNormalize(X)
U,S,V = pca(X_norm)
print U
print S

def projectData(X,U,K):
    return X.dot(U[:,:K])

K = 1
Z = projectData(X_norm,U,K)
print "Dimension reduced data:"
print Z

def recoverData(Z,U,K):
    return Z.dot(U[:,:K].T)

X_rec = recoverData(Z,U,K)
print X_rec

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X_rec[:, 0], X_rec[:, 1])
x_plt = np.linspace(-2.5,2.0,100)
plt.plot(x_plt,(x_plt*U[1,0])/float(U[0,0]),'r') # plotting line of form y = mx, where m  is the slope, here U[:,0] the vector is from 0,0 to give the point U[0,0],U[0,1]
plt.show()

face_data = sio.loadmat('ex7faces.mat')
X = face_data['X']
print X.shape

fig,ax = plt.subplots(10,10,figsize=(12,8),sharex=True,sharey=True)
for i in xrange(10):
    for j in xrange(10):
        face = X[10*i + j,:].reshape(32,32).T
        ax[i,j].imshow(face,cmap='gray')
        ax[i,j].axis('off')
plt.show()

X_norm,mu,sig = featureNormalize(X)
U,S,V = pca(X_norm)
fig,ax = plt.subplots(6,6,figsize=(12,8),sharex=True,sharey=True)
for i in xrange(6):
    for j in xrange(6):
        eigen_face = U[:,6*i + j].reshape(32,32).T
        ax[i,j].imshow(eigen_face,cmap='gray')
        ax[i,j].axis('off')
plt.show()

K = 100
Z = projectData(X,U,K)
print Z.shape

X_rec = recoverData(Z,U,K)
fig,ax = plt.subplots(10,10,figsize=(12,8),sharex=True,sharey=True)
for i in xrange(10):
    for j in xrange(10):
        face = X_rec[10*i + j,:].reshape(32,32).T
        ax[i,j].imshow(face,cmap='gray')
        ax[i,j].axis('off')
plt.show()
