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


# ### First stage

# In[2]:


# Next step:
# load data and name the dataFrame as df
df = pd.read_csv('data.csv',low_memory=False)


# In[3]:


# Have a look at data structure
df.head()


# In[4]:


# Check the number of rows and cols
df.shape


# In[5]:


# Check the column info in list form
list(df.columns)


# In[6]:


# Check the target variable
df['loan_status'].value_counts()


# ##### From the values under loan_status col we can see there are some people who did not meet the credit policy, these data should be removed.

# In[7]:


# Remove the improper data set
df = df.loc[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]


# In[8]:


# Check the new one
df['loan_status'].value_counts()


# In[9]:


# After rearranged the target variable,
# check the missing values, and return the missing rate.
# Define a function:
def missingVals(df):
    '''
    This function return a table of number of missing values, 
    and it's percentage.
    '''
    # Total missing values
    missingVals = df.isnull().sum()
        
    # Proportion of missing values
    propMissingVals = missingVals / len(df) * 100
    
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


# In[10]:


missing = missingVals(df)
print(missing)


# In[11]:


# Drop the columns with high percentage (80%)
missingCols = list(missing.index[
    missing["Percentage (%) of Missing Value"] > 80])

dfSub = df.drop(columns = missingCols)


# In[12]:


# Check
missingVals(dfSub)


# In[13]:


# After check missings, check duplicated values and drop.
dfSub[dfSub.duplicated()]
dfSub.drop_duplicates(inplace= True)


# In[14]:


# Check once, how many rows and cols remaining.
dfSub.shape


# ### Stage Two: Exploratory Data Analysis

# In[15]:


# Check the data is balanced or imbalanced
# Create a new col named charged_off, where fully paid is 0 and charged off is 1.
dfSub["charged_off"] = dfSub["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)


# In[16]:


# charge off rate (in %)
chargeOffRate = dfSub["charged_off"].sum() / dfSub.shape[0] * 100
chargeOffRate.round(2)


# The charge off rate is relatively high (14%), we consider the data set as imbalanced.

# In[17]:


# Drop the cols that potentially cause data leakage.
leakCols = ["total_pymnt_inv","total_rec_prncp","funded_amnt","funded_amnt_inv","total_pymnt",
            "total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee",
            "last_pymnt_amnt","chargeoff_within_12_mths","debt_settlement_flag"]


# In[18]:


dfSub.drop(columns = leakCols, inplace=True)


# In[19]:


dfSub.shape


# In[20]:


54 - 43 == len(leakCols) - 1


# ### Numerical features

# In[21]:


# Get the statistical parameters again.
dfSub.describe()


# In[23]:


dfSub["delinq_amnt"].value_counts()


# In[24]:


# Drop zero values cols since they are unnecessary,
# These cols are those with zero means and SDs.
dfSub.drop(columns=[
    'collections_12_mths_ex_med','tax_liens','out_prncp','out_prncp_inv','delinq_amnt','acc_now_delinq']
         , inplace=True)


# In[27]:


dfSub.shape


# ### Categorical features

# In[34]:


print(dfSub.info())


# In[40]:


# To manipulate categorical columns,
# we need to create a subset data frame which
# only contains categorical variables.
dfCat = dfSub.select_dtypes(include=['object'])
dfCat.shape


# In[79]:


# Drop the target variable (y)
catCols = dfCat.drop(columns=['loan_status']).columns.tolist()
catCols


# In[66]:


# Strip leading and trailing space of each categorical column
for i in catCols:
    dfSub[i] = dfSub[i].str.strip()


# In[83]:


pd.set_option("display.max_columns", None)
# Display the categorical columns
dfSub[catCols].head(10)


# In[82]:


# Here catCols is a 'list'
type(catCols)


# In[ ]:




