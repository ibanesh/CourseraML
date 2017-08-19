import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# identity matrix
eye = np.identity(5)


path = os.getcwd()+'/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#peek at the head of the data
data.head()
#get a description of the data
data.describe()
#plot the data



data.plot(kind='scatter', x='Population', y='Profit')
plt.show()

# can also load data without pandas
# data = np.loadtxt('ex1data1.txt', delimiter=',')

#add 1's to the dataframe 
data.insert(0, 'Ones', 1)
X = data.iloc[:,0:2]
X = X.values
#alternatively, instead of adding ones to dataframe we can add ones to the numpy array X
# use np.concatenate, np.hstack


y = data.iloc[:,2]
y = y.values # creates np array of shape (2,)

#y = data.iloc[:,2:3]
#y = y.values # creates np array of shape (2,1)

theta = np.array([0,0])
#theta = np.array([0,0]).reshape(2,1)

# for matrix/vector mulitplication use np.dot(v1,v2) or v1.dot(v2)


def computeCost(X,y,theta):
    z = np.squeeze(X.dot(theta)) - y # squeeze incase of using theta shape of (2,1)
    return z.dot(z)/(2*len(y))


print computeCost(X,y,theta)

print computeCost(X,y,np.array([-1,2]))


def gradientDescent(X,y,theta,alpha,iterations):
    J_history = np.zeros(iterations)
    for i in xrange(iterations):
        theta = theta - (alpha/len(y))*np.dot(np.squeeze(X.dot(theta)) - y,X)
        J_history[i] = computeCost(X,y,theta)
    return theta, J_history

alpha = 0.01
iterations = 1500
theta, J_history = gradientDescent(X,y,theta,alpha,iterations)
print theta

#plot linear fit
plt.scatter(X[:,1],y)
plt.plot(X[:,1],X.dot(theta))
plt.show() #In non-interactive mode, display all figures and block until the figures have been closed; 


print 'For population = 35,000, we predict a profit of: '
predict1 = theta.dot([1,3.5])
print predict1*10000
print 'For population = 70,000, we predict a profit of: '
predict2 = theta.dot([1,7])
print predict2*10000


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


theta0 = np.arange(-10,10,0.2)
theta1 = np.arange(-1,4,0.05)
T0, T1 = np.meshgrid(theta0,theta1)
J_vals = np.array([computeCost(X,y,[i,j]) for i,j in zip(np.ravel(T0),np.ravel(T1))])
J_vals = J_vals.reshape(T0.shape)

ax.plot_surface(T0,T1,J_vals)
plt.show()

levels = np.arange(0,800,10)
plt.contour(T0,T1,J_vals,levels = levels)
plt.show()