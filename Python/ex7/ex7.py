import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc


data = sio.loadmat('ex7data2.mat')
X = data['X']
K =3

def findClosestCentroids(X,centroids):
    K = centroids.shape[0]
    dist = np.zeros((X.shape[0],K))
    for i in xrange(K):
        diff = X - centroids[i,:]
        dist[:,i] = np.sum(diff**2,1)
    return dist.argmin(1) #np.argmin(dist,1)
        
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = findClosestCentroids(X,initial_centroids)
print idx[0:3]

def computeCentroids(X,idx,K):
    centroids = np.zeros((K,X.shape[1]))
    for i in xrange(K):
        centroids[i,:]=np.mean(X[np.argwhere(idx==i).ravel(),:],0)
    return centroids

centroids = computeCentroids(X,idx,K)
print centroids

def runkMeans(X,initital_centroids,max_iters,plot=False):
    m = X.shape[0]
    K = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    #skipped code for plotting the progress
    #to plot progress we can use something like
    # from IPython import display
    # for i in xrange(10):
    #     plt.scatter(X[:,0],X[:,1])
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
    #     raw_input("Press Enter to continue...")
    for i in xrange(max_iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,K)

    return idx, centroids

max_iters = 10
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx,centroids = runkMeans(X,initial_centroids,max_iters)

fig,ax = plt.subplots(figsize=(12,8))
K = centroids.shape[0]
colors = iter(cm.rainbow(np.linspace(0, 1, K)))
for i in xrange(K):
    cluster = X[np.argwhere(idx==i).ravel(),:]
    ax.scatter(cluster[:,0],cluster[:,1],color=next(colors))
plt.show()
    
def kMeansInitCentroids(X,K):
    m = X.shape[0]
    rnd_idx = np.random.permutation(m)
    return X[rnd_idx[:K],:]

print kMeansInitCentroids(X,3)

#from scipy import misc
A = misc.imread('bird_small.png') # uses the Python Imaging Library (PIL) module, install with "pip install Pillow"
img_size = A.shape
plt.imshow(A) #also requires PIL to be 
plt.show()

A = A/255.0
X = A.reshape(img_size[0]*img_size[1],img_size[2])

K =16
max_iters = 10
initial_centroids = kMeansInitCentroids(X,K)
idx, centroids = runkMeans(X,initial_centroids,max_iters)

idx = findClosestCentroids(X,centroids)
X_recovered = centroids[idx,:]
X_recovered = X_recovered.reshape(img_size)
plt.imshow(X_recovered)
plt.show()