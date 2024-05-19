#!/usr/bin/env python
# coding: utf-8

# # Predicting Median House Values

#  ## Data set Attributes
# 
#  The data set contains information about houses in Boston, Massachusetts. The data set was collected by the U.S. Census Service and first published by Harrison and Rubenfeld in 1978.
# 
#  It contains the following variables:
# * **crim:** per capita crime rate by town
# * **zn:** proportion of residential land zoned for lots over 25,000 sq. ft
# * **indus:** proportion of non-retail business acres per town
# * **chas:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# * **nox:** nitric oxide concentration (parts per 10 million)
# * **rm:** average number of rooms per dwelling
# * **age:** proportion of owner-occupied units built prior to 1940
# * **dis:** weighted distances to five boston employment centers
# * **rad:** index of accessibility to radial highways
# * **tax:** full-value property tax rate per \$10,000
# * **ptratio:** pupil-teacher ratio by town
# * **b:** 1000(bk — 0.63)², where bk is the proportion of [people of African American descent] by town
# * **lstat:** percentage of lower status of the population
# * **medv:** median value of owner-occupied homes in $1000s
# 
# 
# *Harrison, David, and Daniel L. Rubinfeld, Hedonic Housing Prices and the Demand for Clean Air, Journal of Environmental Economics and Management, Volume 5, (1978), 81-102. Original data.*
# 

#  ## Objective
# 
#  The goal of this task is to analyse the relationship between these variables and build a multiple linear regression model to predict the median value based on the 'lm' and 'lstat` variables.
# 

# In[20]:


# Import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[21]:


# Read in the data set
df = pd.read_csv('boston.csv')


# Clean and pre-process the data if neccessary

# In[22]:


# View data info and description 
print(df.info())
df.describe()

# Check for missing values 
print('\nSum of null values:')
print(df.isnull().sum())


# Explore the data with visualisations such as histograms and correlation matrices

# In[23]:


# Create a historams for 'rm', 'lstat' and 'medv' 
plt.figure(figsize=(16, 6))

# Histogram for 'rm'
plt.subplot(1, 3, 1)
sns.histplot(df['rm'], bins=20, kde=True, color='green')
plt.title('Histogram of rm')
plt.xlabel('rm (Average Number of Rooms)')
plt.ylabel('Frequency')

# Histogram for 'lstat'
plt.subplot(1, 3, 2)
sns.histplot(df['lstat'], bins=20, kde=True, color='orange')
plt.title('Histogram of lstat')
plt.xlabel('lstat (Percentage of Lower Status)')
plt.ylabel('Frequency')

# Histogram for 'medv'
plt.subplot(1, 3, 3)
sns.histplot(df['medv'], bins=20, kde=True, color='blue')
plt.title('Histogram of medv')
plt.xlabel('medv(Median Value)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate correlation matrix 
corr_matrix = df.corr()

# Create a heatmap for correlation matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix  Heatmap')
plt.show()


# In[24]:


# Split the independent variables from the dependent variable
X = df[['rm', 'lstat']]
y = df['medv']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[25]:


# Explore relationships between the independent and dependent variables
plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
sns.scatterplot(x='rm', y='medv', data=df)
plt.title('Relationship between rm and medv')

plt.subplot(1, 2, 2)
sns.scatterplot(x='lstat', y='medv', data=df)
plt.title('Relationship between lstas and medv')

plt.show()


# In[26]:


# Create a training and test set with a 75:25 split ratio
X_train, X_test , y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)


# In[27]:


# Build a multiple linear regression model using 'rm' and 'lstat'
model = LinearRegression()
model.fit(X_train, y_train)


# In[28]:


# Print the model intercept and coefficients
print(f'Intercept: {model.intercept_}')
print('Coeffients:')
for feature, coefficient in zip(['rm', 'lstat'], model.coef_):
    print(f'{feature}: {coefficient}')


# In[29]:


# Generate predictions for the test set
y_pred =model.predict(X_test)


# In[30]:


# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


# Generate a plot

# In[31]:


# Calculate residuals (differences between actual and predicted values)
residuals = y_test - y_pred

# Generate an error plot 
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals)
plt.axhline(y=0, color='r', linestyle='--') # A horizontal line at y=0 for reference 
plt.xlabel('Data Points')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Error Plot: Residuals in Linear Regression Model')
plt.show()


# In[32]:


# Print the coefficients
print('Coefficients:')
for feature, coefficient in zip(['rm', 'lstat'], model.coef_):
    print(f'{feature}: {coefficient}')


# **Interpret coefficients in the context of the prediction:**
# 
# - The coefficient for 'rm' is approximately 3.70. This means that, on average, for each additional room ('rm'), the predicted median value of owner-occupied homes ('medv') is expected to increase by about 3.70 units, assuming all other variables are held constant.
# 
# - The coefficient for 'lstat' is approximately -4.63. This means that, on average, for each one-unit increase in the percentage of the population with lower status ('lstat'), the predicted median value of owner-occupied homes ('medv') is expected to decrease by about 4.63 units, assuming all other variables are held constant.
# 
# **Summarise findings**
# 
# - The 'rm' coefficient suggests a positive relationship between the average number of rooms and the median home value, while the 'lstat' coefficient suggests a negative relationship between the percentage of lower-status population and the median home value.
# 
# - With regards to "the model's predictive performance", an RMSE of 5.44 indicates that, on average, the model's predictions deviates by about 5.44 units from the actual values. Interpreting the RMSE inline with the targeted variable, the median value is in $1000s, so an RMSE of 5.44 suggest the model's predictions are, on average, within that range of dollars from the true values. 
