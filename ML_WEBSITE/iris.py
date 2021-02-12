#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


df=pd.read_csv(r'C:\Users\Dell\Downloads\Iris.csv')


# In[3]:


df


# In[21]:


x=df.iloc[:,1:5].values


# In[22]:
print(x)



# In[23]:


y=df.iloc[:,5:].values
print(y)

# In[24]:


y


# In[25]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[36]:


from sklearn.svm import SVC



# In[37]:


sv=SVC(kernel='linear').fit(x_train,y_train)


# In[ ]:


pickle.dump(sv,open('iri.pkl','wb'))

