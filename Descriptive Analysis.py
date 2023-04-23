#!/usr/bin/env python
# coding: utf-8

# # Tesla:Stock Price Analysis with Python

# Time Series data is a series of data points indexed in time order. Time series data is everywhere, so manipulating them is important for any data analyst or data scientist.
# 
# In this notebook, we will discover and explore data from the stock market, particularly Tesla. We will learn how to use finance data to get information, and visualize different aspects of it using Seaborn and Matplotlib. we will look at a few ways of analyzing the risk of a stock, based on its previous performance history.
# 

# In[5]:


# importing important libraries
get_ipython().system('pip install missingno')
get_ipython().system('pip install plotly')
get_ipython().system('pip install mplfinance')


# In[6]:


import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns 
from datetime import datetime
import missingno as msno
import plotly.graph_objects as go
import mplfinance as mpf
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore', category=UserWarning)


# In[7]:


A = pd.read_csv("G:Data_Science_Vrashali/Descriptive Analysis with Python/Tesla_stock_Price.csv")


# In[8]:


A.head()


# In[9]:


for i in range(0,len(A.columns),1):
            X = A.iloc[:,0:i]


# In[10]:


X


# In[11]:


A.info()


# In[12]:


print(A.shape)
print('-'*30)
print(A.columns)
print('-'*30)
print(A.head())
print('-'*30)
print(A.tail())
print('-'*30)


# In[13]:


{i:str(A[i].dtype) for i in A.columns if A[i].dtype== object}


# In[14]:


A.info()


# # Data Cleaning

# In[15]:


def low_fun(n):
    try:
        return float(n)
    except:
        return np.nan
    
def vol_fun(num):
    try:
        return float(num)
    except:
        return np.nan


# In[16]:


#for date column
A['Date'] = pd.to_datetime(A['Date'],errors='coerce')
#for Low column
A['Low'] = A['Low'].apply(low_fun) 
# for Volume column 
A['Volume'] = A['Volume'].str.split('M').str.get(0).apply(vol_fun) 
# for change% column 
A['Chg%'] = A['Chg%'].str.split('%').str.get(0).astype('float32')


# In[17]:


#Checking number of null values in each column
A.isnull().sum()


# In[18]:


A['Price'].fillna(A['Price'].mean() ,inplace=True)
A['Open'].fillna(A['Open'].mean() ,inplace=True)
A['High'].fillna(A['High'].mean() ,inplace=True)
A['Low'].fillna(A['Low'].mean() ,inplace=True)
A['Volume'].fillna(A['Volume'].mean() ,inplace=True)
A['Chg%'].fillna(A['Chg%'].mean() ,inplace=True) 

A.dropna(subset={'Date'},inplace=True)
round(100*(A.isnull().sum()/len(A.index)), 2) 


# In[19]:


#Calculating the data loss
100-round(100*len(A.index)/3258,2) 


# # data Exploration

# What was the change in price of the stock over time?
# 
# Perform different time sampling
# 
# What was the daily return of the stock on average?
# 
# What was the moving average of the various stocks?
# 
# What was the correlation between different component of stocks'?

# In[20]:


A.set_index('Date', inplace=True)
A.sort_index(inplace=True) 


# In[21]:


A.head()


# In[ ]:


A.describe()


# In[24]:


A.plot(subplots=True,figsize=(16,25),title='Stocks Analysis over Years',linestyle='--',linewidth=2)
plt.show() 


# In[25]:


A = A.rename(columns={'Price':'Close'})
A1 = A[['Close','Open','High','Low','Volume']] 


# Candlestick chart are also known as a Japanese chart. These are widely used for technical analysis in trading as they visualize the price size within a period. They have four points Open, High, Low, Close (OHLC). Candlestick charts can be created in python using a matplotlib module called mplfinance. 
# 
# Installation:
# !pip install mplfinance
# mplfinance.candlestick_ohlc()
# This function is used to plot Candlestick charts.
# 
# Syntax: mplfinance.candlestick_ohlc(ax, quotes, width=0.2, colorup=’k’, colordown=’r’, alpha=1.0)
# Parameters: 
# 
# ax: An Axes instance to plot to.
# quotes: sequence of (time, open, high, low, close, …) sequences.
# width: Fraction of a day for the rectangle width.
# colorup: The color of the rectangle where close >= open.
# colordown: The color of the rectangle where close < open.
# alpha: (float) The rectangle alpha level.
# Returns: returns (lines, patches) where lines are a list of lines added and patches is a list of the rectangle patches added.

# In[26]:


# create a custom style for the chart

custom_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10})
# plot the candlestick chart
mpf.plot(A, type='candle', style=custom_style, volume=True, ylabel='Price',figsize=(16,12),title=('Stocks Analysis Over Time'))
mpf.show() 


# In[27]:


# Which is the date of the lowest price of the stock? 
y = A[A['Close'] == A['Close'].min()].index[0]   
print(y.day_name(),y.day,'th',y.month_name(),y.year) 


# In[28]:


#show the graph ,rose subsequently
# apply xlimit and y limit
plt.style.use('classic')
A['Close'].plot(xlim=['2020/01/01','2022/09/29'],ls='--',lw=4, figsize=(15,4),color='#212f3d') 
plt.legend() 
plt.show() 


# In[29]:


# tesla stocks open for 2021 
share_open = A.loc['2021/01/01':'2021/12/31']['Open']  
index = share_open.index 
share_open.head(5) 


# In[30]:


share_open.plot(ls='--', figsize=(15,4),color='#61c9c9') 
plt.legend() 
plt.show() 


# Perform different time sampling
# In time series, data consistency is of prime importance, resampling ensures that the data is distributed with a consistent frequency. Resampling can also provide a different perception of looking at the data, in other words, it can add additional insights about the data based on the resampling frequency.

# In[31]:


# minimum share price in evey year end
A.resample(rule='A').min() 


# In[32]:


# maximum share price in evey year
grid = A.resample(rule='A').max() 
grid


# In[33]:


# Showing the volume of 2021 is very high
plt.style.use('fivethirtyeight') 
plt.figure(figsize=(20,10))
plt.imshow(grid)
plt.colorbar()


# In[35]:


# maximum share price in evey quarter start
A.resample(rule='QS').max()['Close'].plot(figsize=(12,4)) 


# In[36]:


# maximum share open in evey Business year end
A.resample(rule='BA').max()['Open'].plot(figsize=(12,4)) 


# In[37]:


# mean share volume in every month
A.resample(rule='M').mean()['Open']


# In[38]:


# plot
A.resample(rule='M').mean()['Open'].plot(kind='line',figsize=(12,4)) 


# What was the moving average of the various stocks?
# The moving average (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks, or any time period the trader chooses.

# In[39]:


A['Open: 10 days rolling']= A['Open'].rolling(10).mean() 
A['Open: 30 days rolling']= A['Open'].rolling(30).mean() 
A['Open: 50 days rolling']= A['Open'].rolling(50).mean() 


# In[41]:


plt.style.use('ggplot') 

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(18,12), )
ax[0,0].plot(A.index,A['Open'])  
ax[0,0].set_title('Open price without MA')
ax[0,0].set_ylabel('Opening Price') 


ax[0,1].plot(A.index,A['Open: 10 days rolling'])  
ax[0,1].set_title('MA with 10 days window')

ax[1,0].plot(A.index,A['Open: 30 days rolling'])
ax[1,0].set_title('MA with 30 days window')
ax[1,0].set_ylabel('Opening Price') 
ax[1,0].set_xlabel('Years') 



ax[1,1].plot(A.index,A['Open: 50 days rolling'])  
ax[1,1].set_title('MA with 50 days window')
ax[1,1].set_xlabel('Years') 


# What was the daily return of the stock on average
# Now that we've done some baseline analysis, let's go ahead and dive a little deeper. We're now going to analyze the risk of the stock. In order to do so we'll need to take a closer look at the daily changes of the stock, and not just its absolute value. Let's go ahead and use pandas to retrieve teh daily returns for the stock.

# In[42]:


A['Chg%'].plot(kind='line',ylabel='Change(%)',figsize=(15,6),legend=True,color='Turquoise')  


# In[44]:


# with histogram 
A['Chg%'].plot(kind='hist',xlabel='Daily Returns',bins=[-20,-15,-10,-5,0,5,10,15],figsize=(12,5))
plt.yscale('log') 


# #What was the correlation between stocks closing and opening prices?
# Correlation is a statistic that measures the degree to which two variables move in relation to each other which has a value that must fall between -1.0 and +1.0. Correlation measures association, but doesn’t show if x causes y or vice versa — or if the association is caused by a third factor.

# In[45]:


# between opening and closing price 
A[['Close','Open']].plot(kind='scatter',x='Close',y='Open') 


# In[46]:


A.plot(kind='scatter',x='High',y='Low')


# # End
