#!/usr/bin/env python
# coding: utf-8

# In[13]:


#import Libraries with their alias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # **Simple Example**

# In[21]:


#Creating dataset
X = np.array([45, 60, 20, 66, 32, 46, 18, 65, 15, 70]).reshape(-1,1)
Y = np.array([55, 75, 31, 76, 44, 59, 33, 70, 26, 83]).reshape(-1,1)


# In[22]:


#Using the train_test_split() function from sklearn
x_train, x_test, y_train, y_train = train_test_split(X, Y, test_size=0.1)


# In[23]:


#Created a LinearRegression Model
model = LinearRegression()
model.fit(X, Y)


# In[24]:


#calculating the Accuracy of our model
model.score(X, Y)


# In[25]:


#Plotting the datapoints and visualizing it through a scatter plot
plt.scatter(X, Y)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.show()


# In[42]:


#Predicting new values
x_arr = np.array([int(input("Enter a value : "))]).reshape(-1,1)
x_new = np.array([x_arr]).reshape(-1,1)
y_predict = model.predict(x_new)
print(y_predict)


# In[ ]:




