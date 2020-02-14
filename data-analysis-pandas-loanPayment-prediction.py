#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initial step:
# import fundamental pkgs
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import datetime as dt

# import visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML
get_ipython().run_line_magic('matplotlib', 'inline')

# import machine learning pkgs
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc


# In[ ]:


# Next step:
# load data and name the dataFrame as df
df = pd.read_csv('data.csv',low_memory=False)


# In[ ]:


# Have a look at data structure
df.head()


# In[ ]:


# Check the number of rows and cols
df.shape


# In[ ]:


# Check the column info in list form
list(df.columns)


# In[ ]:


# Check the target variable
df['loan_status'].value_counts()


# ##### From the values under loan_status col we can see there are some people who did not meet the credit policy, these data should be removed.

# In[ ]:


# Remove the improper data set
df = df.loc[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]


# In[ ]:


# Check the new one
df['loan_status'].value_counts()


# In[ ]:


# After rearranged the target variable,
# check the missing values, and return the missing rate.
# Define a function:
def missingVals(df):
    '''
    This function return a table of number of missing values, 
    and it's percentage.
    '''
    # Proportion of missing values
    propMissingVals = df.isnull.sum() / len(df) * 100
    
    # Create a table
    tableMissingVals = pd.concat([df.isnull().sum(), propMissingVals], axis=1)
    
    # Rename the cols
    renTableMissVals = tableMissingVals.rename(
        columns = {0: "Missing Value Total", 1: "Percentage (%) of Missing Value"})
    
    # Sort the table by percentage of missing values
    # in descending order (ie. highest ot lowest).
    renTableMissVals = renTableMissVals[
        renTableMissVals.iloc[:,1] != 0].sort_values(
        "Percentage (%) of Missing Value",
        ascending = False).round(2)
    
    # Print conclustion messages for readers.
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(renTableMissVals.shape[0]) +
              " columns that have missing values.")
    
    # Return the data frame in table with missing values not removed.
    return renTableMissVals


# In[ ]:




