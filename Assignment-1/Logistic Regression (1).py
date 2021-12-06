#!/usr/bin/env python
# coding: utf-8

# In[285]:


import numpy as np


# In[286]:


# read Data from excel file .
data = pd.read_csv('heart.csv')

#Input / features
X = np.array(data[['trestbps','chol','thalach','oldpeak']])

#Output
Y = np.array(data['target'])
rows, cols = X.shape

# add new term to input .
X = np.concatenate([np.ones((rows, 1)), X], axis=1)


# In[287]:


def sigmoid(theta_Dot_X):
    return 1 / (1 + np.exp(-theta_Dot_X))


# In[288]:


def costFunction(theta, X, y):
    m = y.size
    return -(1/m)* np.sum( y*np.log(sigmoid(np.dot(X,theta))) + (1-y)*np.log(1-sigmoid(np.dot(X,theta))) ) 


# In[289]:


def gradiantDescent(alpha , theta , X , y , iterations):

    m = X.shape[0] # num of rows = 303 in this case.
    CostFunction = []
    theta = theta.copy()
    for i in range(iterations):
        theta = theta - (alpha / m) * (np.dot(X.T, ( sigmoid(np.dot(X,theta)) - y ))) 
        CostFunction.append( costFunction(theta, X, y) )
        
    return theta , CostFunction


# In[290]:


def predict(theta, X):
    m = X.shape[0]
    for i in range(m):
        if(sigmoid(np.dot(X[i], theta)) >=0.5 ):
                 p[i] = 1
        else:       
                 p[i] = 0       
    return p


# In[291]:


# Test (1)
intial_theta = np.zeros(X.shape[1])
theta , CostFunction = gradiantDescent(0.000081, intial_theta , X , Y , 10000)

print('Optimized Theta : ' , theta)
p = predict(theta, X)
print('Train Accuracy: {:.5f} %'.format(np.mean(p == Y) * 100))
print("Cost: ",CostFunction)


# In[292]:


# Test (2)
intial_theta = np.zeros(X.shape[1])
theta , CostFunction = gradiantDescent(0.000081, intial_theta , X , Y , 20000)

print('Optimized Theta : ' , theta)
p = predict(theta, X)
print('Train Accuracy: {:.5f} %'.format(np.mean(p == Y) * 100))
print("Cost: ",CostFunction)


# In[293]:


# Test (3)
# Best result until now.
intial_theta = np.zeros(X.shape[1])
theta , CostFunction = gradiantDescent(0.000081, intial_theta , X , Y , 30000)

print('Optimized Theta : ' , theta)
p = predict(theta, X)
print('Train Accuracy: {:.5f} %'.format(np.mean(p == Y) * 100))
print("Cost: ",CostFunction)


# In[294]:


# Test (4)
intial_theta = np.zeros(X.shape[1])
theta , CostFunction = gradiantDescent(0.000081, intial_theta , X , Y , 50000)

print('Optimized Theta : ' , theta)
p = predict(theta, X)
print('Train Accuracy: {:.5f} %'.format(np.mean(p == Y) * 100))
print("Cost: ",CostFunction)


# In[295]:


# Test (5)
intial_theta = np.zeros(X.shape[1])
theta , CostFunction = gradiantDescent(0.00005, intial_theta , X , Y , 100000)

print('Optimized Theta : ' , theta)
p = predict(theta, X)
print('Train Accuracy: {:.5f} %'.format(np.mean(p == Y) * 100))
print("Cost: ",CostFunction)


# In[ ]:




