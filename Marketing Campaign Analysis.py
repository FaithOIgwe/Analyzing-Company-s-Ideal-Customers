#!/usr/bin/env python
# coding: utf-8

# ## Introduction 
# 
#    Companies carryout marketing through several means for different reasons: to create more awareness  about their products and services, increase sales, get new customers, and also to be top of mind for their current customers and intending customers.
# 
# A marketing campaign is successful when the objective of the marketing is accomplished. To achieve this, a company must first understand their customers, so they can market apporpriately speaking the right language, and this all boils down to analyzing their historical data to understand their customers.
# 
# In this Project,I am a Data Analyst Consulting for a small business company to help them understand their ideal customers using their historical data from previous marketing campaigns.
# 

# ### Problem Statement
# 
# In this project I take the position of a Marketign Data Analyst Consulting for a small business who sells retail products such as wines,Fruits, Fish products, sweet products, and Gold products. 
# 
# I have been provided with historical data of their past campaign, to analyze and help them better understand who their ideal customers are.
# 
#  

# In[2]:


#Improting the needed the needed libraries
import pandas as pd
import numpy as np
import os
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} missingno')

import missingno as msno


# In[3]:


# Load the csv file
marketing = pd.read_csv(r"C:\Users\AKHIGBE\Downloads\Company's Ideal Customers Marketing Strategy\marketing_campaign Project.csv")


# ### Data Preprocessing
# 
# 

# In[4]:


#Viewing the first rows of data 

marketing.head()


# In[5]:


# Checking the columns
marketing.columns


# In[6]:


#Viewing the datatypes are properly represented in each columns and also checking  for missing data

marketing.info()


# In[7]:


marketing


# In[8]:


#There are some missing values in the income column, we will need to input "Nan" or "0" in the cell

# we will use the current date and time to determine the customers age by minusing the current year by their year of birth to better understand their Age


marketing['Age']=datetime.now().year - marketing['Year_Birth']
marketing.head()


# In[9]:


# For better understanding we will convert the Dt_customer to Tenure using our current date and time to ensure we better understand how long the customer has been with us


marketing['now']=datetime.now()
marketing['Tenure'] = marketing['now'] - (marketing['Dt_Customer']) * pd.offsets.Day()


# In[10]:


marketing['quotient'] = marketing['Dt_Customer'] - marketing['now']
marketing['quotient'] = marketing['quotient'].dt.days


# In[12]:


import pandas as pd

# Create two DatetimeArray
dates1 = pd.date_range(start='2022-01-01', end='2022-01-05')
dates2 = pd.date_range(start='2022-01-02', end='2022-01-06')

# Subtract 
result = dates1 - dates2

print(result)


# In[13]:


marketing.describe(include='all')


# In[14]:


pd.unique(marketing['Education'])


# In[15]:


pd.unique(marketing['Marital_Status'])


# In[ ]:





# In[16]:


pd.unique(marketing['AcceptedCmp1'])


# In[17]:


pd.unique(marketing['AcceptedCmp2'])


# In[18]:


pd.unique(marketing['AcceptedCmp3'])


# In[19]:


pd.unique(marketing['AcceptedCmp5'])


# In[20]:


# We will be cleaning the following columns to better set the tone for our analysis 

# 'Education': group 'Graduation' and '2n Cycle' in the same category as 'Master'

# 'Marital_Status': replace 'Alone' with 'Single'; replace 'Married' and 'Together' with 'Coupled';

# replace 'YOLO' and 'Absurd' with 'Other'

# ' Income ': get rid of the spaces in the col name, and deal with missing values


# In[21]:


# 'Graduation' and '2n Cycle' both mean 'Master'

marketing['Education_cleaned']=marketing['Education']


# In[22]:


#Cleaning the 'Education Column' replacing Graduation and 2n Cycelw with  Master

marketing['Education_Cleaned']=marketing['Education']
marketing['Education_Cleaned']=marketing['Education_Cleaned'].replace(['Graduation', '2n Cycle'], 'Master')

assert pd.unique(marketing['Education_Cleaned']).all() in ['Basic', 'Master', 'PhD']


# In[23]:



# cleaning the 'Marital_Status' col by adding a new column

marketing['Marital_Status_Cleaned']=marketing['Marital_Status']
pd.unique(marketing['Marital_Status_Cleaned'])


# In[24]:


# Cleaning the 'Marital status' column  to only include Single, Married, Divorced and others as the only options in this column
marketing['Marital_Status_Cleaned']=marketing['Marital_Status_Cleaned'].replace(['Alone'], 'Single')
marketing['Marital_Status_Cleaned']=marketing['Marital_Status_Cleaned'].replace([ 'Together', 'Coupled'], 'Married')
marketing['Marital_Status_Cleaned']=marketing['Marital_Status_Cleaned'].replace(['YOLO', 'Absurd'], 'Other')

assert pd.unique(marketing['Marital_Status_Cleaned']).all() in ['Divorced', 'Single', 'Widow', 'Other'] 
               


# In[25]:


#Checking if the Cleaned columns was dully effected
pd.unique(marketing['Marital_Status_Cleaned'])


# In[26]:


marketing.head()


# In[30]:


# cleaning the ' Income ' 
# first, start a new, identical column without spaces in the col name

marketing['Income']=marketing['Income']

assert marketing['Income'].equals(marketing['Income'])


# In[31]:


# Deal with missing values 
get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(marketing)


# In[32]:


# we will assume that they are missing at random and fill the values with the mean

missing_value=marketing['Income'].mean()
marketing['Income'].fillna(value=missing_value, inplace=True)


# In[33]:


marketing['Income'].describe()


# In[60]:


# our dataset has been fully cleaned, let us view them
marketing_cleaned = marketing.drop(columns=['Marital_Status', 'Education', 'Year_Birth', 'Dt_Customer'])
marketing_cleaned.columns


# In[61]:


#Our data is all clean and ready for Analysis


#  # Univariate Exploration - Categorical Variables

# In[62]:


Education_order = ['Basic', 'Master', 'PhD']
sns.countplot(x='Education_Cleaned', data=marketing_cleaned, order=Education_order)


# In[63]:


sns.countplot(x='Marital_Status_Cleaned', data=marketing_cleaned)


# In[64]:


marketing_cleaned


# In[65]:


fig, axes = plt.subplots(1, 5, sharey=True)

sns.countplot(data=marketing_cleaned, x='AcceptedCmp1', ax=axes[0])
sns.countplot(data=marketing_cleaned, x='AcceptedCmp2', ax=axes[1])
sns.countplot(data=marketing_cleaned, x='AcceptedCmp3', ax=axes[2])
sns.countplot(data=marketing_cleaned, x='AcceptedCmp4', ax=axes[3])
sns.countplot(data=marketing_cleaned, x='AcceptedCmp5', ax=axes[4])

fig.subplots_adjust(wspace=1.5)


# In[66]:


sns.countplot(data=marketing_cleaned, x='Response')


# In[67]:


sns.countplot(data=marketing_cleaned, x='Complain')


# # Univariate Exploration - Numerical Variables

# In[68]:


sns.histplot(data=marketing_cleaned, x='Age', bins=30)


# In[69]:


sns.boxplot(data=marketing_cleaned, x='Income')


# In[70]:


sns.countplot(data=marketing_cleaned, x='Kidhome')


# In[71]:


sns.countplot(data=marketing_cleaned, x='Teenhome')


# In[72]:


sns.histplot(data=marketing_cleaned, x='Recency', bins=20)


# In[73]:


# we will be visualizing the amounts column using a Box and Whisker Plot

marketing_mnt=marketing_cleaned[['MntWines', 'MntFruits', 'MntFishProducts', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
marketing_mnt_melted=pd.melt(marketing_mnt)

sns.boxplot(x='variable', y='value', data=marketing_mnt_melted)
plt.xticks(rotation=45)


# In[74]:


# Visualizing the 'number of purchases' columns in box plots
marketing.head()


# In[75]:


marketing_num=marketing_cleaned[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases']]


# In[76]:


# visualize all the 'number of purchases' columns together with box plots

marketing_num=marketing_cleaned[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']]
marketing_num_melted=pd.melt(marketing_num)

sns.boxplot(x='variable', y='value', data=marketing_num_melted)
plt.xticks(rotation=45)


# In[77]:


sns.boxplot(data=marketing_cleaned, y='NumWebVisitsMonth')


# ## Findings From Univariate Exploration:
# 
# 1) campaign 2 was the least perfoRming campaign
# 
# 2) customers spent the most on wine, followed by meat
# 
# 3) store seems to be customers' favourite channel of purchasing, followed by web
# 
# 4) more than 10% of customers accepted offer from the last campaign
# 
# 5) customers seem to be overall satisfied since there's minimal complaints

# ## Bivariate Analysis

# In[78]:


# first, correlation matrix and a heatmap to visualize it so I know which variables to look into next

corr_mat= marketing.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_mat, annot=True)


# In[79]:


# I want to single out the columns with the strongest correlations

c=corr_mat.abs()
s=c.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
so=so[so!=1] #excluse the 1s
so


# In[80]:


# select moderate to strong correlations (>=0.5, anything above 0.7 is considered strong, above 0.5 is moderate)

so_corr=so[so>=0.5].drop_duplicates()
so_corr


# In[82]:


# plot NumCatalogPurchases against: MntMeatProducts, MntWines, Income, MntFishProducts

def jitter(values, j):
    return values+np.random.normal(j, 0.5, values.shape)

sns.set(font_scale = 1.5)
fig, ax=plt.subplots(1,4,figsize=(30,10), sharey=True)

sns.regplot(y=jitter(marketing_cleaned.NumCatalogPurchases, 0.5), x=marketing_cleaned.MntMeatProducts, scatter_kws={'alpha':0.25, 's':5}, ax=ax[0])
sns.regplot(y=jitter(marketing_cleaned.NumCatalogPurchases, 0.5), x=marketing_cleaned.MntWines, scatter_kws={'alpha':0.25, 's':5}, ax=ax[1])
sns.regplot(y=jitter(marketing_cleaned.NumCatalogPurchases, 0.5), x=marketing_cleaned.Income, scatter_kws={'alpha':0.25, 's':5}, ax=ax[2])
sns.regplot(y=jitter(marketing_cleaned.NumCatalogPurchases, 0.5), x=marketing_cleaned.MntFishProducts, scatter_kws={'alpha':0.25, 's':5}, ax=ax[3])


# In[83]:


# plot NumStorePurchases against: MntWines, Income

fig, ax=plt.subplots(1,2, figsize=(30,10), sharey=True)

sns.regplot(y=jitter(marketing_cleaned.NumStorePurchases, 0.5), x=marketing_cleaned.MntWines, scatter_kws={'alpha':0.25, 's':5}, ax=ax[0])
sns.regplot(y=jitter(marketing_cleaned.NumStorePurchases, 0.5), x=marketing_cleaned.Income, scatter_kws={'alpha':0.25, 's':5}, ax=ax[1])


# In[85]:


# plot NumWebPurchases against: MntWines

sns.regplot(y=jitter(marketing_cleaned.NumWebPurchases, 0.5), x=marketing_cleaned.MntWines, scatter_kws={'alpha':0.25, 's':5})


# In[87]:


# plot NumWebVisitsMonth against NumWebPurchases

sns.regplot(y=jitter(marketing_cleaned.NumWebVisitsMonth, 0.5), x=jitter(marketing_cleaned.NumWebPurchases, 1), scatter_kws={'alpha':0.25, 's':5})


# In[88]:


# There seems to be a weak and negative correlation between number of web visits and number of web purchases
# More attention shoudl be drawn to our website, as it may not be woorking well.


# In[89]:


# plot MntWines against Income

sns.regplot(y=jitter(marketing_cleaned.MntWines, 0.5), x=jitter(marketing_cleaned.Income, 1), scatter_kws={'alpha':0.25, 's':5})


# In[90]:


# plot MntMeatProducts against: Income and NumWebVisitsMonth

fig, ax=plt.subplots(1,2, figsize=(30,10), sharey=True)

sns.regplot(y=jitter(marketing_cleaned.MntMeatProducts, 0.5), x=marketing_cleaned.Income, scatter_kws={'alpha':0.25, 's':5}, ax=ax[0])
sns.regplot(y=jitter(marketing_cleaned.MntMeatProducts, 0.5), x=jitter(marketing_cleaned.NumWebVisitsMonth, 0.5), scatter_kws={'alpha':0.25, 's':5}, ax=ax[1])


# In[91]:


# explore the relationship between Education and amounts spent on different kinds of products, sales channels and their favourite campaigns

marketing_mnt_edu_pivot=pd.pivot_table(data=marketing_cleaned, values=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], columns='Education_Cleaned', aggfunc=np.sum)
marketing_num_edu_pivot=pd.pivot_table(data=marketing_cleaned, values=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases'], columns='Education_Cleaned', aggfunc=np.sum)
marketing_cmp_edu_pivot=pd.pivot_table(data=marketing_cleaned, values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], columns='Education_Cleaned', aggfunc=np.sum)

fig, ax = plt.subplots(1,3,figsize=(27,9))

marketing_mnt_edu_pivot.plot(kind='bar', stacked=True, ax=ax[0])
marketing_num_edu_pivot.plot(kind='bar', stacked=True, ax=ax[1])
marketing_cmp_edu_pivot.plot(kind='bar', stacked=True, ax=ax[2])


# In[92]:


# explore the relationship between Marital Status and amounts spent on different kinds of products, sales channels and their favourite campaigns

marketing_mnt_ms_pivot=pd.pivot_table(data=marketing_cleaned, values=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], columns='Marital_Status_Cleaned', aggfunc=np.sum)
marketing_num_ms_pivot=pd.pivot_table(data=marketing_cleaned, values=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases'], columns='Marital_Status_Cleaned', aggfunc=np.sum)
marketing_cmp_ms_pivot=pd.pivot_table(data=marketing_cleaned, values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], columns='Marital_Status_Cleaned', aggfunc=np.sum)

fig, ax = plt.subplots(1,3,figsize=(30,18))

marketing_mnt_ms_pivot.plot(kind='bar', stacked=True, ax=ax[0])
marketing_num_ms_pivot.plot(kind='bar', stacked=True, ax=ax[1])
marketing_cmp_ms_pivot.plot(kind='bar', stacked=True, ax=ax[2])


# In[93]:


# does number of kids home have anything to do with the kinds of products people buy?

marketing_mnt_kh_pivot=pd.pivot_table(data=marketing_cleaned, values=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], columns='Kidhome', aggfunc=np.sum)
marketing_mnt_th_pivot=pd.pivot_table(data=marketing_cleaned, values=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], columns='Teenhome', aggfunc=np.sum)

fig, ax = plt.subplots(1,2,figsize=(8,8),sharey=True)

marketing_mnt_kh_pivot.plot(kind='bar', stacked=True, ax=ax[0])
marketing_mnt_th_pivot.plot(kind='bar', stacked=True, ax=ax[1])


# In[94]:


# explore the relationships between # of kids/teens home and sales channels

marketing_num_kh_pivot=pd.pivot_table(data=marketing_cleaned, values=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases'], columns='Kidhome', aggfunc=np.sum)
marketing_num_th_pivot=pd.pivot_table(data=marketing_cleaned, values=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases'], columns='Teenhome', aggfunc=np.sum)

fig, ax = plt.subplots(1,2,figsize=(8,8), sharey=True)

marketing_num_kh_pivot.plot(kind='bar', stacked=True, ax=ax[0])
marketing_num_th_pivot.plot(kind='bar', stacked=True, ax=ax[1])


# In[ ]:


# explore the relationships between number of kids/teens in the home and campaign successes

marketing_cmp_kh_pivot=pd.pivot_table(data=marketing_cleaned, values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], columns='Kidhome', aggfunc=np.sum)
marketing_cmp_th_pivot=pd.pivot_table(data=marketing_cleaned, values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], columns='Teenhome', aggfunc=np.sum)

fig, ax = plt.subplots(1,2,figsize=(30,15), sharey=True)

marketing.plot(kind='bar', stacked=True, ax=ax[0])
marketing_cmp_th_pivot.plot(kind='bar', stacked=True, ax=ax[1])


# In[ ]:


# explore the relationship between Country and amounts spent on different kinds of products, sales channels and their favourite campaigns

marketing_mnt_cty_pivot=pd.pivot_table(data=marketing_cleaned, values=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], columns='Country', aggfunc=np.sum)
marketing_num_cty_pivot=pd.pivot_table(data=marketing_cleaned, values=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases','NumStorePurchases'], columns='Country', aggfunc=np.sum)
marketing_cmp_cty_pivot=pd.pivot_table(data=marketing_cleaned, values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], columns='Country', aggfunc=np.sum)

fig, ax = plt.subplots(1,3,figsize=(32,15))

marketing_mnt_cty_pivot.plot(kind='bar', stacked=True, ax=ax[0])
marketing_num_cty_pivot.plot(kind='bar', stacked=True, ax=ax[1])
marketing_cmp_cty_pivot.plot(kind='bar', stacked=True, ax=ax[2])


# In[ ]:


# explore the relationships between customers' preferred sales channels and amounts spent on wine

fig, ax = plt.subplots(1, 4, figsize=(20,5), sharey=True)

sns.lineplot(data=marketing_cleaned, x='NumDealsPurchases', y='MntWines', ci=None, ax=ax[0])
sns.lineplot(data=marketing_cleaned, x='NumWebPurchases', y='MntWines', ci=None, ax=ax[1])
sns.lineplot(data=marketing_cleaned, x='NumCatalogPurchases', y='MntWines', ci=None, ax=ax[2])
sns.lineplot(data=marketing_cleaned, x='NumStorePurchases', y='MntWines', ci=None, ax=ax[3])


# In[ ]:


# explore the relationships between customers' preferred sales channels and amounts spent on meat

fig, ax = plt.subplots(1, 4, figsize=(20,5), sharey=True)

sns.lineplot(data=marketing_cleaned, x='NumDealsPurchases', y='MntMeatProducts', ci=None, ax=ax[0])
sns.lineplot(data=marketing_cleaned, x='NumWebPurchases', y='MntMeatProducts', ci=None, ax=ax[1])
sns.lineplot(data=marketing_cleaned, x='NumCatalogPurchases', y='MntMeatProducts', ci=None, ax=ax[2])
sns.lineplot(data=marketing_cleaned, x='NumStorePurchases', y='MntMeatProducts', ci=None, ax=ax[3])


# In[ ]:


# Recency by accepted campaigns - is there one campaign that works well for inactive customers?

cmp1=marketing_cleaned[marketing_cleaned['AcceptedCmp1']==1]
cmp2=marketing_cleaned[marketing_cleaned['AcceptedCmp2']==1]
cmp3=marketing_cleaned[marketing_cleaned['AcceptedCmp3']==1]
cmp4=marketing_cleaned[marketing_cleaned['AcceptedCmp4']==1]
cmp5=marketing_cleaned[marketing_cleaned['AcceptedCmp5']==1]

fig, ax = plt.subplots(5,1,figsize=(3,15),sharex=True)


sns.ecdfplot(data=cmp1, x='Recency', ax=ax[0])
sns.ecdfplot(data=cmp2, x='Recency', ax=ax[1])
sns.ecdfplot(data=cmp3, x='Recency', ax=ax[2])
sns.ecdfplot(data=cmp4, x='Recency', ax=ax[3])
sns.ecdfplot(data=cmp5, x='Recency', ax=ax[4])
ax[0].set(title='Campaign 1')
ax[1].set(title='Campaign 2')
ax[2].set(title='Campaign 3')
ax[3].set(title='Campaign 4')
ax[4].set(title='Campaign 5')


# # Multivariate Exploration

# In[ ]:


# Income, Education and Marital Status

marketing_edu_ms=df_cleaned[['Education_Cleaned', 'Marital_Status_Cleaned', 'Income']]
marketing_edu_ms_melted=pd.melt(marketing_edu_ms, value_name='Income', value_vars='Income', id_vars=['Education_Cleaned', 'Marital_Status_Cleaned'])

marketing_edu_ms_melted.columns

plt.figure(figsize=(8,12))
sns.boxplot(data=marketing_edu_ms_melted, x='Marital_Status_Cleaned', y='Income', hue='Education_Cleaned')


# In[ ]:


# Income, Education and Number of kids/teens

marketing_edu_kh=df_cleaned[['Education_Cleaned', 'Kidhome', 'Income']]
marketing_edu_kh_melted=pd.melt(marketing_edu_kh, value_name='Income', value_vars='Income', id_vars=['Education_Cleaned', 'Kidhome'])
marketing_edu_th=df_cleaned[['Education_Cleaned', 'Teenhome', 'Income']]
marketing_edu_th_melted=pd.melt(marketing_edu_th, value_name='Income', value_vars='Income', id_vars=['Education_Cleaned', 'Teenhome'])


fig, ax = plt.subplots(1,2,figsize=(16,12),sharey=True)
sns.boxplot(data=marketing_edu_kh_melted, x='Kidhome', y='Income', hue='Education_Cleaned', ax=ax[0])
sns.boxplot(data=marketing_edu_th_melted, x='Teenhome', y='Income', hue='Education_Cleaned', ax=ax[1])


# # Conclusion 
# 
# #### Our Average Customer Profile
# 
#  Is a Master's degree holder, Married, in their 40s-70s, earns 35-68k per year and  has 0 -1 child
#  
# #### Our most popular product types 
# Most of our customers spend the most on wine, followed by meat
# 
# Most of these customers who spend on meat, do so usually, using the catalogue.
# 
# These customers tend not to have kids.
# 
# #### What is our most popular sales channel
# Our Customers prefer to make their purchases instore, followed by online purchase 
# 
# #### What is the performance of our web sales channel?
# Even if web is relatively popular as a sales channel, it is alarming that higher website visits don't result in higher web purchases, infact, there seems to be a slightly negative correlation between website visits and website purchases.
# 
# This needs to be looked into by the business, there could be something with our website that makes people click on the website, but make no purchases
# 
# #### Campaign Performance
# Our best performing campaign is campaign 4, while campaign 2 recieved very poor results
# 
# Campaign 3 seems to be very attractive to customers with basic education, and it was also common with customers who had 1-2 kids or teens at home.
# 
# Lastly, customers with teens(at least one) at home really appreciated campaign 4
# 
# #### More Findings
# A large portion of our customers are educated with at least a Master's Degree
# 
# The Higher education Column seems to be highly correlated with basic education.
# 
# Which makes me wonder if we as a business is not affordable for lower income customers
# 
# Customers with no kids seem to have higher income, which means that they might fit our targeted profile better. AS a result, it might make sense for us to figure out how to tap into their disposable income (that they have no kids to spend on!) with our campaigns
# 
# Finally one thought on recency, I didn't find any correlation between any campaign and recency, But I think potential future campaigns that get inactive customers making purchases could be interesting

# In[ ]:




