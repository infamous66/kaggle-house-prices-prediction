#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


train_data = pd.read_csv("train.csv")


# In[3]:


train_data.shape


# In[4]:


train_data.head()


# In[5]:


train_data.isnull().sum().sum()


# In[6]:


train_data.info()


# In[7]:


train_data.drop(columns=["Id"], inplace=True)

excessive_na_cols = ["Alley", "PoolQC", "Fence", 'MasVnrType', "MiscFeature", 'FireplaceQu']

train_data.drop(columns=excessive_na_cols, inplace=True)


# In[8]:


for column in train_data.select_dtypes(include='object').columns:
    most_frequent_value = train_data[column].value_counts(normalize=True).max()
    
    if most_frequent_value >= 0.90:
        print(f"Column: {column}")
        print(f"Most frequent category makes up {most_frequent_value * 100:.2f}% of the values.\n")


# In[9]:


unneccessary_cols = [
    "Street", "Utilities", "LandSlope", "Condition2", "RoofMatl", "BsmtCond", 
    "Heating", "Electrical", "Functional", "GarageQual", "GarageCond", "PavedDrive"
]

train_data.drop(columns=unneccessary_cols, inplace=True)


# In[10]:


from sklearn.impute import SimpleImputer

num_cols = train_data.select_dtypes(include=['number']).columns
mean_imputer = SimpleImputer(strategy='mean')

train_data[num_cols] = mean_imputer.fit_transform(train_data[num_cols])


# In[11]:


# categorical columns
categorical_cols = train_data.select_dtypes(include=['object']).columns

mode_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols] = mode_imputer.fit_transform(train_data[categorical_cols])


# In[12]:


train_data.isnull().sum().sum()


# In[13]:


relevant_columns = [
    'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'GarageArea', 'YearBuilt', 'YearRemodAdd', 'Neighborhood',
    'Condition1', 'LotArea', 'HeatingQC', 'TotRmsAbvGrd',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'SaleType', 
    'SaleCondition', 'LotFrontage','MasVnrArea', 'MoSold', 
    'YrSold', 'SalePrice',
]
train_data = train_data[relevant_columns]


# In[14]:


train_data.columns


# In[15]:


numerical_columns = [
    'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
  'GarageArea', 'LotArea', 'LotFrontage', 'MasVnrArea'
]


def handle_outliers(train_data, column, method='remove'):
    Q1 = train_data[column].quantile(0.25)
    Q3 = train_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if method == 'remove':
        train_data = train_data[(train_data[column] >= lower_bound) & (train_data[column] <= upper_bound)]
    elif method == 'cap':
        train_data[column] = train_data[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)

    return train_data

for column in numerical_columns:
    train_data = handle_outliers(train_data, column, method='remove') # cap


# In[16]:


train_data.describe()


# In[17]:


# corr with salprice
numerical_df = train_data.select_dtypes(include=['float64', 'int64'])

corr_matrix = numerical_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.show()

corr_with_saleprice = corr_matrix['SalePrice'].sort_values(ascending=False)
print(corr_with_saleprice)


# In[18]:


weak_features = ["BedroomAbvGr", "MoSold", "YrSold"]
train_data = train_data.drop(columns=weak_features)


# In[19]:


train_data["SalePrice"]


# In[20]:


from scipy.stats import skew
from numpy import log1p

skewed_features = train_data[numerical_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_features})
print(skewness)

skewed_features = skewness[skewness['Skew'] > 0.75].index
train_data[skewed_features] = train_data[skewed_features].apply(lambda x: log1p(x))


# In[21]:


train_data = pd.get_dummies(train_data, columns=[
    'Neighborhood', 'Condition1', 'SaleType', 'SaleCondition', 'HeatingQC'
], drop_first=True)


# In[22]:


train_data.columns


# In[23]:


train_data.shape[1]


# In[24]:


from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                      'GarageArea', 'YearBuilt', 'YearRemodAdd',
                      'LotArea', 'LotFrontage', 'MasVnrArea',
                      'TotRmsAbvGrd', 'FullBath', 'HalfBath']

train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# In[26]:


X_train.shape


# In[27]:


X_val.shape


# In[28]:


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_val)
y_pred


# In[29]:


mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')
# Calculate RMSE
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
print(f'Root Mean Squared Error: {rmse}')


# In[30]:


y_val


# In[31]:


y_pred[0]


# In[32]:


# 13% in average misses the price
22456.94088435092 / 168774.807420 * 100


# In[ ]:





# In[36]:


from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=7)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_val)

mse_ridge = mean_squared_error(y_val, y_pred_ridge)
print(f'Mean Squared Error (Ridge): {mse_ridge}')

rmse_ridge = np.sqrt(np.mean((y_val - y_pred_ridge) ** 2))
print(f'Square Root Mean Squared Error: {rmse_ridge}')


# In[34]:


from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_val)

mse_lasso = mean_squared_error(y_val, y_pred_lasso)
print(f'Mean squared error (lasso): {mse_lasso}')

rmse_lasso = np.sqrt(np.mean((y_val - y_pred_lasso) ** 2))
print(f'Square Root Mean Squared Error: {rmse_lasso}')


# In[35]:


# Doing the same preproccessing steps on test dataset and creating a submission file

relevant_columns_test = ['OverallQual',
 'GrLivArea',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GarageArea',
 'YearBuilt',
 'YearRemodAdd',
 'Neighborhood',
 'Condition1',
 'LotArea',
 'HeatingQC',
 'TotRmsAbvGrd',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'SaleType',
 'SaleCondition',
 'LotFrontage',
 'MasVnrArea',
 'MoSold',
 'YrSold',]

numerical_columns = [
    'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
  'GarageArea', 'LotArea', 'LotFrontage', 'MasVnrArea'
]

test_data = pd.read_csv("test.csv")
test_data = test_data.set_index("Id")
test_data.drop(columns=excessive_na_cols, inplace=True)
test_data.drop(columns=unneccessary_cols, inplace=True)
num_cols = test_data.select_dtypes(include=['number']).columns
test_data[num_cols] = mean_imputer.fit_transform(test_data[num_cols])
categorical_cols = test_data.select_dtypes(include=['object']).columns
test_data[categorical_cols] = mode_imputer.fit_transform(test_data[categorical_cols])
test_data = test_data[relevant_columns_test]
for column in numerical_columns:
    test_data = handle_outliers(test_data, column, method='cap') # cap
test_data = test_data.drop(columns=weak_features)
test_data = pd.get_dummies(test_data, columns=[
    'Neighborhood', 'Condition1', 'SaleType', 'SaleCondition', 'HeatingQC'
], drop_first=True)
test_data = test_data[X_train.columns]
test_data[numerical_features] = scaler.fit_transform(test_data[numerical_features])
y_test_pred = ridge_model.predict(test_data)
# Prepare submission DataFrame
submission = pd.DataFrame({
    'Id': test_data.index,       
    'SalePrice': y_test_pred     
})
submission.to_csv("results.csv", index=False)

