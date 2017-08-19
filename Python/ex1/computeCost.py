def computeCost(X,y,theta):
    z = (X.dot(theta)) - y
    return z.dot(z)/(2*len(y))