#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


# In[2]:


def get_list_of_json_files():
    list_of_files = os.listdir('/Users/muditagarg/Downloads/Drug War Data Analysis/2011-2021')
    return list_of_files


# In[3]:


#Printing the list of JSON files present in your folder.
get_list_of_json_files()


# In[4]:



#Reaading the JSON files one by one and write them to different CSV files on the basis of 'tweets' and 'data' attributes.
def write_csv():
    list_of_files = get_list_of_json_files()
    for file in list_of_files:
        with open(f'/Users/muditagarg/Downloads/Drug War Data Analysis/2011-2021/{file}') as f:
            data = json.load(f)  
            #'data' attribute is nested in 'tweet' attribute in our JSON structure.
            df1 = json_normalize(data,['tweets',['data']])            
        outputfile=file.rsplit( ".", 1 )[ 0 ] 

        df1.to_csv(f'/Users/muditagarg/Downloads/Drug War Data Analysis/output1/{outputfile}.csv', index=False)


# In[5]:


if __name__=="__main__":
    write_csv()


# In[6]:


def get_list_of_csv_files():
    list_of_csv_files = os.listdir('/Users/muditagarg/Downloads/Drug War Data Analysis/output1')
    return list_of_csv_files


# In[7]:


get_list_of_csv_files()


# In[8]:



# Reading CSV files one by one and concatenate it in single data frame. 

csv_data = pd.DataFrame()
list_of_csv_files = get_list_of_csv_files()
for file in list_of_csv_files:
    with open(f'/Users/muditagarg/Downloads/Drug War Data Analysis/output1/{file}', encoding='utf-8') as f:
        single_csv_data = pd.read_csv(f, header=None)
        csv_data=csv_data.append(single_csv_data)

print(csv_data)


# In[10]:


df=csv_data


# In[11]:


df.rename(columns=df.iloc[0]).drop(df.index[0])


# In[12]:


df.to_csv(r'/Users/muditagarg/Downloads/Drug War Data Analysis/Ouput.csv', index = False)


# EDA

# In[13]:


import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt
# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math


# In[14]:


df= pd.read_csv("/Users/muditagarg/Downloads/Drug War Data Analysis/Ouput.csv")


# In[15]:


print("Shape of data =",df.shape)


# In[19]:


df.head(5)


# In[20]:


df=df.rename(columns=df.iloc[0]).drop(df.index[0]) # to remove the first row


# In[25]:


df.head(5)


# In[ ]:


#Checking missing values


# In[22]:


df.isnull()


# In[ ]:


#checking the data types of every variable


# In[26]:


df.dtypes


# In[ ]:


#Duplicate rows in the dataset


# In[82]:


duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[83]:


df.count() # counting number of rows in the data


# In[27]:


df = df.drop_duplicates() #dropping duplicates


# In[28]:


df.count()


# In[42]:


import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


#df['lang'] = df['lang'].apply(str) #changing datatype


# In[75]:


df.dtypes

