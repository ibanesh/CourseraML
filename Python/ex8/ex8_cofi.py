import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = sio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
print "Average rating for movie 1:"
print Y[0,np.where(R[0,:])].mean()

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(Y,cmap='gray_r',aspect='auto') 
#plt.imshow(Y,aspect='auto')
fig.tight_layout()
plt.show()

data = sio.loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']

def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,lam):
    X = np.reshape(params[:num_movies*num_features],(num_movies,num_features))
    Theta = np.reshape(params[num_movies*num_features:],(num_users,num_features))
    
    J = (1/2.0)*np.sum(((X.dot(Theta.T) - Y)*R)**2) + (lam/2.0)*np.sum(X**2) + (lam/2.0)*np.sum(Theta**2)
    X_grad = ((X.dot(Theta.T) - Y)*R).dot(Theta) + lam*X
    Theta_grad = ((X.dot(Theta.T) - Y)*R).T.dot(X) + lam*Theta
    grad = np.concatenate((X_grad.ravel(),Theta_grad.ravel()))
    return J,grad

num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

J,grad = cofiCostFunc(np.concatenate((X.ravel(),Theta.ravel())),Y,R,num_users,num_movies,num_features,0)
print "Without Regularization:"
print "Cost " + str(J)
print "Gradients " +str(grad)

J,grad = cofiCostFunc(np.concatenate((X.ravel(),Theta.ravel())),Y,R,num_users,num_movies,num_features,1.5)
print "With Regularization: (lam = 1.5)"
print "Cost " + str(J)
print "Gradients " +str(grad)

with open('movie_ids.txt') as f:
    movieList = f.read().splitlines() #f.readlines() show '\n' at the end of the movie names
    
movieList = [' '.join(movie.split()[1:]) for movie in movieList]

myRatings = np.zeros(len(movieList))
myRatings[0] = 4
myRatings[97] = 2
myRatings[6] = 3
myRatings[11] = 5
myRatings[53] = 4
myRatings[63] = 5
myRatings[65] = 3
myRatings[68] = 5
myRatings[182] = 4
myRatings[225] = 5
myRatings[354] = 5

for i in xrange(len(myRatings)):
    if myRatings[i] > 0:
        print "Rated "+str(myRatings[i])+" for "+ str(movieList[i])


data = sio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
Y = np.column_stack((Y,myRatings))
R = np.column_stack((R, myRatings != 0))
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
lam = 10

X = np.random.randn(num_movies,num_features)
Theta = np.random.randn(num_users,num_features)
params = np.concatenate((X.ravel(),Theta.ravel()))

def normalizeRatings(Y,R):
    num_ratings = np.sum(R,1)
    Ymean = np.sum(Y*R,1)/num_ratings
    Ynorm = Y - Ymean.reshape((Y.shape[0],1)) 
    #actually subracts the mean from 0 entries in Y, which correspond to unrated, i.e R(i,j)=0. In my  cost calculation code I multiply the error matrix with R, so whatever value Ynorm has at the unrated entries will not matter.
    Ynorm[np.where(R==0)] = 0  # To be on the safer side we can set to unrated entries back to zero
    return Ynorm,Ymean

Ynorm,Ymean = normalizeRatings(Y,R)
print Y[0,:10]
print Ynorm[0,:10]


fmin = minimize(fun=cofiCostFunc,x0=params,args=(Ynorm,R,num_users,num_movies,num_features,lam),method='CG',jac=True,options={'maxiter': 100})
print fmin

X = np.reshape(fmin.x[:num_movies*num_features],(num_movies,num_features))
Theta = np.reshape(fmin.x[num_movies*num_features:],(num_users,num_features))

p = X.dot(Theta.T)
my_predictions = p[:,-1] + Ymean

print "Top recommendations for you: "
sorted_idx = np.argsort(-my_predictions)
for i in xrange(10):
    print "Predicting rating "+str(my_predictions[sorted_idx[i]])+" for movie "+ str(movieList[sorted_idx[i]])