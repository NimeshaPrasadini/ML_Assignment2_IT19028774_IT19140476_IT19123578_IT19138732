#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Heart Disease Prediction using Logistic Regression
#The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD)


# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading Heart Data from framingham.csv
chd_df=pd.read_csv("framingham.csv")
#drop a column
chd_df.drop(['education'],axis=1,inplace=True)
chd_df.head()


# In[3]:


#Rename 'male' column name
chd_df.rename(columns={'male':'sex_male'},inplace=True)


# In[4]:


#Find missing values
chd_df.isnull().sum()


# In[5]:


#Counting total no of rows with missing values
count=0
for i in chd_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values =', count)
print('Percentage of rows with missing values in the dataset =',round((count/len(chd_df.index))*100),'%')
print('Therefore, the missing values are eliminated.')


# In[6]:


#dropping the missing values
chd_df.dropna(axis=0,inplace=True)


# In[7]:


#Exploratory Analysis by drawing histograms for CHD features
def draw_chd_histograms(dataframe, features, rows, cols):
    fig_chd=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax_chd=fig_chd.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax_chd,facecolor='maroon')
        ax_chd.set_title(feature+" Visualization",color='navy')
        
    fig_chd.tight_layout()  
    plt.show()
#Call the histogram function
draw_chd_histograms(chd_df,chd_df.columns,6,3)


# In[8]:


#TenYearCHD feature values counting
chd_df.TenYearCHD.value_counts()


# In[9]:


#Plot a graph for the TenYearCHD feature value data 
sn.countplot(x='TenYearCHD',data=chd_df)


# In[10]:


print('Therefore, there are',(chd_df.TenYearCHD == 1).sum(),'patients with risk of heart disease and',(chd_df.TenYearCHD == 0).sum(),'patents with no heart disease.')


# In[11]:


# Plot graphs for all feature data in the dataframe
sn.pairplot(data=chd_df)


# In[12]:


#Description of the all feature data in the dataframe
#count - no of non-empty values
#mean - average (mean) value
#std - standard deviation
#min - minimum value
#25% - 25% percentile
#50% - 50% percentile
#75% - 75% percentile
#max - maximum value
chd_df.describe()


# In[ ]:




