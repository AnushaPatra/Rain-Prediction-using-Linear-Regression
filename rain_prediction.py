#!/usr/bin/env python
# coding: utf-8

# In[1]:


#RA2111027010022-Anusha Patra

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# In[2]:


#Q1 connecting data into pandas
RUL_database = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\weather.csv")
print(RUL_database.head())


# In[3]:


# showing random 10 rows from the dataset
RUL_database.sample(10)


# In[4]:


# No. of rows and columns
RUL_database.shape


# In[5]:


#Q2 No. of cells having NULL values
RUL_database.isnull().sum()


# In[6]:


#Q3 drop the columns and rows with all null values
RUL_database = RUL_database.dropna(axis=1, how='all')
print(RUL_database)


# In[7]:


#Q4 Assuming RUL_database is a pandas DataFrame containing the mentioned columns
columns_to_encode = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]
encoded_data = pd.DataFrame()

for column in columns_to_encode:
    label_encoder = LabelEncoder()
    encoded_col = label_encoder.fit_transform(RUL_database[column])
    encoded_data[column] = encoded_col


# In[8]:


#Q5 Print the encoded values
print("Encoded values for WindGustDir:")
print(encoded_data["WindGustDir"])

print("\nEncoded values for WindDir9am:")
print(encoded_data["WindDir9am"])

print("\nEncoded values for WindDir3pm:")
print(encoded_data["WindDir3pm"])


# In[9]:


#Q6 apply minmax scaler
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(encoded_data)

# Convert the scaled_data array back to a DataFrame
scaled_data_df = pd.DataFrame(scaled_data, columns = encoded_data.columns)

# Print the scaled data
print("Scaled data:")
print(scaled_data_df)


# In[10]:


# Perform one-hot encoding on categorical columns
encoded_data = pd.get_dummies(scaled_data_df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])

# Split the data into features (X) and the target variable (y)
X = encoded_data.drop(columns=["RainTomorrow"])
y = encoded_data["RainTomorrow"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


#Q7 Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)


# In[14]:


#Q8 r2 score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)


# In[15]:


#Q9 Apply 6-fold cross-validation and get R2 scores for each fold
cv_scores = cross_val_score(model, X, y, cv=6, scoring='r2')


# In[16]:


#Q10 Print the R2 scores for each fold
print("R2 Scores for each fold:", cv_scores)


# In[ ]:




