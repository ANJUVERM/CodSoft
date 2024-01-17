#!/usr/bin/env python
# coding: utf-8

# # Task 1 

# # Titanic Survival Prediction

# # Import required dependencies

# In[46]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


# In[47]:


t_data = pd.read_csv(r"C:\Users\ANJU VERMA\Downloads\Titanic-Dataset.csv")


# In[48]:


t_data


# # learning about the data set

# In[49]:


t_data.head()


# In[50]:


t_data.tail()


# In[51]:


t_data.shape


# In[52]:


t_data.info()


# In[53]:


t_data.isnull().sum()


# # we need to handle missing values   

# In[12]:


# o handle missing values in the cabin column we will drop it as they are numerous


# In[54]:


t_data.drop(columns='Cabin', axis=1, inplace=True)


# In[14]:


# to handle missing values in the age and fare column we will replace them with the mean age and fare


# In[55]:


Age = t_data['Age'].mean()


# In[56]:


t_data['Age'].fillna(Age, inplace = True)


# In[57]:


Fare = t_data['Fare'].mean()


# In[58]:


t_data['Fare'].fillna(Fare, inplace = True)


# In[59]:


t_data.info()


# In[20]:


# Our data is now consistent


# In[60]:


t_data


# # Data Visualization  

# In[61]:


sns.set()


# In[62]:


sns.countplot(x='Sex', data=t_data)    


# In[63]:


t_data['Survived'].value_counts()


# In[25]:


sns.countplot(x='Survived', data=t_data)    


# In[26]:


sns.countplot (x='Sex', hue = 'Survived', data = t_data)


# # this is clearly visible that those who survived were most of the females
# 

# In[27]:


t_data[['Survived', 'Sex']]


# In[28]:


t_data[['Survived', 'Pclass' ]]


# In[29]:


sns.countplot(x ='Pclass',hue= 'Survived', data=t_data)


# # converting the categorical variables into numerical data

# In[30]:


t_data.replace({'Sex':{'male':0, 'female':1},'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[31]:


t_data


# In[33]:


# Now drop the columns which are irrelevant for the survival prediction, such as PassengerId, Name and Ticket


# In[34]:


t_data.drop(columns={'PassengerId','Name','Ticket'},axis=1, inplace=True)


# In[35]:


t_data


# # separating features and target

# In[36]:


X = t_data.drop(columns='Survived', axis=1)
Y = t_data['Survived']


# In[37]:


print(X)
print(Y)


# # splitting the data into training and testing

# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[39]:


X_train.isnull().sum()


# # Model Training

# In[40]:


# we're using logistic regression model that uses binary classification for the prediction


# In[68]:


X_train.fillna (X_train ['Embarked'].mean(),inplace = True)


# In[69]:


X_train.isnull().sum()


# In[70]:


model = LogisticRegression()   


# In[71]:


model.fit(X_train,Y_train)


# In[88]:


X_test.isnull().sum()


# In[89]:


X_test.fillna (X_test ['Embarked'].mean(),inplace = True)


# In[90]:


X_test.isnull().sum()


# In[92]:


y_pred1 = model.predict(X_test)


# In[93]:


X_test_prediction = model.predict(X_test)


# In[94]:


print(X_test_prediction)


# In[95]:


testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[96]:


print('Accuracy score of test data is : ',testing_data_accuracy)


# In[97]:


test_data_precision = precision_score(Y_test, X_test_prediction)


# In[98]:


print('test data precion is :', test_data_precision)


# In[103]:


from sklearn import metrics


# In[104]:


score = model.score(X_test,Y_test)
print(score)


# In[106]:


cm = metrics.confusion_matrix(Y_test, X_test_prediction)
print(cm)


# In[107]:


sns.heatmap(cm, annot = True, fmt = "d", square = True, cmap= "inferno")
plt.ylabel('Actual label')
plt.xlabel('predicted label')
title = ('Accuracy Score :',score)
plt.title(title, size = 10)


# # Thank you

# In[ ]:




