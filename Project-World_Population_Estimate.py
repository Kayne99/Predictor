


import statsmodels.api as sm


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[5]:


url = 'https://en.wikipedia.org/wiki/Estimates_of_historical_world_population'
dataset = pd.read_html(url, header = 0)


# In[6]:


dataset = dataset[2]


# In[7]:


dataset.head()


# In[8]:


dataset.to_excel('World_Population_Estimate.xlsx')




# In[9]:


dataset_for_analysis = dataset[['Year', 'United States Census Bureau (2017)[29]']]


# In[10]:


dataset_for_analysis.head()


# In[11]:


dataset_for_analysis = dataset_for_analysis.dropna()


# In[12]:


dataset_for_analysis.describe()


# ## Data Exploration

# In[13]:


y = dataset_for_analysis['United States Census Bureau (2017)[29]'] # Defining the independent and dependent variables
x1 = dataset_for_analysis['Year']


# In[16]:


fig, ax = plt.subplots(figsize = (14,7))
plt.scatter(x1,y)
plt.title('World Population', fontsize=45)
plt.xlabel('Year', fontsize=40)
plt.ylabel('Census', fontsize=40)
plt.show()


# ## Regression Model

# In[28]:


y = dataset_for_analysis['United States Census Bureau (2017)[29]'] # Defining the independent and dependent variables
x1 = dataset_for_analysis['Year']


# ### Using Ordinary Least Squares

# In[29]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# ## Explanation of table
# ## With our R- squared value of 0.997, this shows our model explains the variability of the data to a large extent

# In[15]:


fig, ax = plt.subplots(figsize = (14,8))
plt.scatter(x1,y)
yhat = 75290000*x1 - 144500000000 # census = 75290000 * x1 + (-144500000000)
fig = plt.plot(x1, yhat, lw=4, c='yellow', label = 'regression line')
plt.title('World Population Growth', fontsize=45)
plt.xlabel('Year', fontsize=40)
plt.ylabel('Census', fontsize=40)
plt.show()


# 
# # Interpretation of Regression Analysis
# 
# ## From the analysis, we calculated the values for our constants in the regression equation:
# ### yhat = -144500000000 + 75290000 * x1
# ### Where;
# ### x1 = Year
# ### yhat = Census
# ## This equation can be use to predict the World's population
# 

# In[ ]:




