#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)


# In[2]:


import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html


# In[5]:


from sklearn import datasets
digits = datasets.load_digits()


# In[6]:


dir(digits)


# In[7]:


print (type(digits.images))
print (type(digits.target))


# In[46]:


print(digits.images.shape)
plt.gray()
plt.matshow(digits.images[2])


# In[37]:


print (digits.images[1236])


# In[36]:


import matplotlib.pyplot as plt
plt.imshow(digits.images[1236],cmap='binary')
plt.show()


# In[11]:


print (digits.target.shape)
print (digits.target)


# In[51]:


def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()


# In[43]:


plot_multi(2)


# In[17]:


y = digits.target
x = digits.images.reshape((len(digits.images), -1))
x.shape


# In[18]:


x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]


# In[19]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)


# In[20]:


mlp.fit(x_train,y_train)


# In[21]:


predictions = mlp.predict(x_test)
predictions[:797]


# In[22]:


y_test[:797]


# In[23]:


predictions[:797]-y_test[0:797]


# In[24]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[ ]:




