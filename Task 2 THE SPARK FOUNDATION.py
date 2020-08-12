#!/usr/bin/env python
# coding: utf-8

# # Task2 - To Explore Supervised Machine Learning
# **In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# This is a simple linear regression task as it involves just two variables**

# ## Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data

# In[4]:


# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head()


# In[5]:


## Exploratory Data Analysis (EDA)


# In[6]:



print(data.describe(),'\n')
print(data.isnull().sum())


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


data.dtypes


# In[10]:


data.info()


# In[11]:


data.isnull().head()


# ## Scatter Plot

# In[12]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## Correlation Plot
# 

# In[30]:


corr = data.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);


# In[33]:


X=data[['Hours']]
Y=data['Scores']


# In[35]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)


# ## Training and Predicting

# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print("Training complete.")#training the algorithm


# In[38]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# ### Regration Plot for the test data
# 
# 

# In[40]:


# Now compare the actual output values for X_test with the predicted values
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# ### if a student study for 9.25 hrs in a day

# In[41]:


#What will be predicted score if a student study for 9.25 hrs in a day?
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# #### You can also test with your own data
# 

# 

# In[ ]:





# In[ ]:




