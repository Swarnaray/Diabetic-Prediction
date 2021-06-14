#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Dataset 

# In[2]:


dataset = pd.read_csv("diabetes.csv")
dataset


# In[3]:


dataset.head(6)


# In[4]:


dataset.tail(6)


# ## Getting the detailed information about the Database

# In[5]:


dataset.shape


# In[6]:


diabetes_true_count = len(dataset.loc[dataset['Outcome'] == 1])
diabetes_false_count = len(dataset.loc[dataset['Outcome'] == 0])
(diabetes_true_count,diabetes_false_count)


# In[7]:


dataset.info()


# In[8]:


dataset.corr()  ## Show the pairwise correlation of all columns in the dataframe


# In[9]:


# Get correlations of each features in dataset and plot into a heat map
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ## Data Visualization

# In[10]:


plt.figure(figsize=(15, 20))
j=1
for i in range(1,9):
        plt.subplot(5,2,i)
        plt.scatter(dataset.iloc[:,0],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[0])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[11]:


plt.figure(figsize=(15, 20))
j=1
for i in range(2,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,1],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[1])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[12]:


plt.figure(figsize=(15, 20))
j=1
for i in range(3,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,2],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[2])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[13]:


plt.figure(figsize=(15, 20))
j=1
for i in range(4,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,3],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[3])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[14]:


plt.figure(figsize=(15, 20))
j=1
for i in range(5,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,4],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[4])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[15]:


plt.figure(figsize=(15, 20))
j=1
for i in range(6,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,5],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[5])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[16]:


plt.figure(figsize=(15, 20))
j=1
for i in range(7,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,6],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[6])
        plt.ylabel(dataset.columns[i])
        j+=1


# In[17]:


plt.figure(figsize=(15, 20))
j=1
for i in range(8,9):
        plt.subplot(5,2,j)
        plt.scatter(dataset.iloc[:,7],dataset.iloc[:,i])
        plt.xlabel(dataset.columns[7])
        plt.ylabel(dataset.columns[i])
        j+=1


# ## Checking for Clean Data

# In[18]:


dataset.isnull().values.any()


# In[19]:


dataset.isnull().sum()


# In[20]:


sns.heatmap(dataset.isnull())


# ## Creating Dependent and Independent variable

# In[21]:


X=dataset.drop('Outcome', axis=1)
X


# In[22]:


y=dataset['Outcome']
y


# ## Dividing into Train and Test Dataset

# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.1, random_state=0)


# In[24]:


y_test


# ## Model Training

# In[25]:


logmodl=LogisticRegression()


# In[26]:


logmodl.fit(X_train, y_train)


# In[27]:


X_test


# In[28]:


y_test


# ## Model Evaluation

# In[29]:


y_pred=logmodl.predict(X_test)


# In[30]:


y_pred


# In[31]:


df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# In[32]:


accuracy_score(y_test, y_pred)*100


# In[33]:


cm=confusion_matrix(y_test, y_pred)
((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100


# In[34]:


((48+19)/(48+3+7+19))*100


# In[45]:


classification_report(y_test, y_pred)


# ## Model Testing

# In[46]:


predict={0:'Not Diabetic',1:'Diabetic'}


# In[47]:


predict


# In[68]:


z=logmodl.predict([[2,126,60,30,1,47,31.2,24]])


# In[69]:


z


# In[64]:


predict[z[0]]


# In[65]:


z=logmodl.predict([[4,76,62,0,0,34.0,0.391,25]])


# In[66]:


z


# In[67]:


predict[z[0]]

