#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# # Load the Dataset: 

# In[21]:


df = pd.read_csv("C:\\Users\\ANJU VERMA\\Downloads\\IRIS.csv", encoding='utf-8')


# In[46]:


df.head(10)


# In[53]:


df['species'].value_counts()


# In[54]:


df.tail()


# In[25]:


df.info()


# # No missing Data:

# In[56]:


df.isnull().sum()


# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df.head()


# In[28]:


species_name = le.classes_
print(species_name)


# In[30]:


X = df.drop(columns=['species'])
y = df['species']
X.head(3)


# In[31]:


print(y[:5])


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=69)


# In[33]:


x_train.shape


# In[38]:


y_train.shape


# In[34]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(x_train[:1])
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:1])


# In[35]:


y_train = keras.utils.to_categorical(y_train, num_classes=3)
print(y_train[:5])


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[40]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


log_clf=LogisticRegression()
log_clf.fit(X_train, y_train)


# # Predictions on the Test Data:

# In[42]:


y_pred=log_clf.predict(X_test)


# # Evaluation of the model Performance on the Logistic Regression:

# In[43]:


print(classification_report(y_test, y_pred))


# In[ ]:




