#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
get_ipython().system('pip install xgboost')
import xgboost as xgb
from xgboost import XGBClassifier


# In[2]:


#read the sales data
sales = pd.read_csv("sales.csv")
sales.head()


# In[3]:


df1 = pd.DataFrame(sales, columns=["DATE", "STORE LOCATION", "STORE TYPE", "SALES", "CLOSEST WEATHER CENTER LOCATION"])
print(df1)


# In[4]:


#deleting unecessary columns
del sales['Unnamed: 5']


# In[5]:


#changing the data types for store location and store types to string to better use them for my visualizations
sales['STORE LOCATION'] = sales['STORE LOCATION'].astype(str)
sales['STORE TYPE'] = sales['STORE TYPE'].astype(str)


# In[6]:


#Changing Date from object to date time
sales['DATE'] = pd.to_datetime(sales["DATE"])


# In[7]:


#print out the date to ensure changes were made
print(sales["DATE"])


# In[8]:


#remove null values from the data set
sales.dropna()


# In[9]:


#changes the sales column to integer
sales = sales.astype({'SALES': 'int64'})
print(sales.dtypes)


# In[10]:


#read the weather data
weather = pd.read_csv("weather.csv")
weather.head()


# In[11]:


#Changing Date from object to date time
weather['DATE'] = pd.to_datetime(weather["DATE"])


# In[12]:


#Drop Null values
weather.dropna()


# In[13]:


#merge sales and weather data using outer join on date 
salesweather = pd.merge(sales, weather, how = "outer", on = "DATE")
salesweather.head()


# In[14]:


#printout the data types for the columns in weather
weather.dtypes


# In[15]:


#print out the types of data in the columns for salesweather data
print(salesweather.dtypes)


# In[16]:


import plotly.graph_objects as go


# In[17]:


#create new dataframe for sales weather to only filter for 2016 data
filtered_df = salesweather.query("DATE >= '2016-01-01'                        & DATE <= '2016-12-31'")


# In[18]:


##create new dataframe for sales weather to only filter for 2017 data
filtered_df1 = salesweather.query("DATE >= '2017-01-01'                        & DATE <= '2017-12-31'")


# In[19]:


#create new dataframe for sales weather to only filter for store types
store_type1 = salesweather[(salesweather["STORE TYPE"] == "1")]
store_type2 = salesweather[(salesweather["STORE TYPE"] == "2")]
store_type3 = salesweather[(salesweather["STORE TYPE"] == "3")]


# In[20]:


#create new dataframe for salesweather to filter for 2016 and 2017 data based on store type
filtered_df16_1 = store_type1.query("DATE >= '2016-01-01'                        & DATE <= '2016-12-31'")
filtered_df16_2 = store_type2.query("DATE >= '2016-01-01'                        & DATE <= '2016-12-31'")
filtered_df16_3 = store_type3.query("DATE >= '2016-01-01'                        & DATE <= '2016-12-31'")
filtered_df17_1 = store_type1.query("DATE >= '2017-01-01'                        & DATE <= '2017-12-31'")
filtered_df17_2 = store_type2.query("DATE >= '2017-01-01'                        & DATE <= '2017-12-31'")
filtered_df17_3 = store_type3.query("DATE >= '2017-01-01'                        & DATE <= '2017-12-31'")


# In[21]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px


# In[22]:


#Group sales by store types in 2016
df1 = filtered_df.groupby('STORE TYPE')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Top Store Types That Make The Most Money in 2016', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[23]:


#Group sales by store types in 2017

df1 = filtered_df1.groupby('STORE TYPE')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Top Store Types That Make The Most Money in 2017', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[24]:


#pie chart to show percentage of sales based on store type in 2016
values = filtered_df['SALES']
names = filtered_df['STORE TYPE']
fig = px.pie(filtered_df, values=values, names=names, title="2016 Total Sales by Store Type")
fig.show()
print(values)


# In[25]:


#pie chart to show percentage of sales based on store type in 2017

values = filtered_df1['SALES']
names = filtered_df1['STORE TYPE']
fig = px.pie(filtered_df1, values=values, names=names, title="2017 Total Sales by Store Type")
fig.show()
print(values)


# In[26]:


##chart to show daily sales based for store type 3 in 2016

fig = go.Figure(go.Scatter(x = filtered_df16_3['DATE'] , y = filtered_df16_3['SALES'], marker_color='rgba(152, 0, 0, .8)'))
fig.update_layout(title = '2016 SALES DATA FOR STORE TYPE 3', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[27]:


##chart to show daily sales based for store type 3 in 2017

fig = go.Figure(go.Scatter(x = filtered_df17_3['DATE'] , y = filtered_df17_3['SALES'], marker_color='rgba(152, 0, 0, .8)'))
fig.update_layout(title = '2017 SALES DATA FOR STORE TYPE 3', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[28]:


#Bar chart to show top 10 total sales by store locations in store type 3 for 2016
df1 = filtered_df16_3.groupby('STORE LOCATION')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Total Slaes in 2016 by Store Locations for Store Type 3', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[29]:


#Bar chart to show top 10 total sales by store locations in store type 3 for 2017

df1 = filtered_df17_3.groupby('STORE LOCATION')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Total Slaes in 2017 by Store Locations for Store Type 3', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[30]:


#pie chart to show daily sales based on store location for store type 3 in 2016

values = filtered_df16_3['SALES']
names = filtered_df16_3['STORE LOCATION']
fig = px.pie(filtered_df16_3, values=values, names=names, title="2016 Total Sales by Store Location in Store Type 3")
fig.show()
print(values)


# In[31]:


#pie chart to show daily sales based on location for store type 3 in 2017

values = filtered_df17_3['SALES']
names = filtered_df17_3['STORE LOCATION']
fig = px.pie(filtered_df17_3, values=values, names=names, title="2017 Total Sales by Store Location in Store Type 3")
fig.show()
print(values)


# In[32]:


#Scatter plot to show recorded daily snow fall in 2016
fig = px.scatter(x = filtered_df16_3['DATE'] , y = filtered_df16_3['SNWD'])
fig.update_layout(title = '2016 SNOW DATA',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[33]:


#Scatter plot to show recorded daily snow fall in 2017

fig = px.scatter(x = filtered_df17_3['DATE'] , y = filtered_df17_3['SNWD'])
fig.update_layout(title = '2017 SNOW DATA',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[34]:


#create new dataframe for sales weather to only filter for store types

filtered_df16_3SL= filtered_df16_3[(filtered_df16_3["STORE LOCATION"] == "1028")]
filtered_df17_3SL= filtered_df17_3[(filtered_df17_3["STORE LOCATION"] == "1028")]
filtered_df16_3SL1= filtered_df16_3[(filtered_df16_3["STORE LOCATION"] == "1003")]
filtered_df17_3SL1= filtered_df17_3[(filtered_df17_3["STORE LOCATION"] == "1003")]


# In[35]:


#chart to show daily sales based on store location 1028 for store type 3 in 2016

fig = go.Figure(go.Scatter(x = filtered_df16_3SL['DATE'] , y = filtered_df16_3SL['SALES']))
fig.update_layout(title = '2016 SALES DATA FOR STORE LOCATION 1028',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[36]:


#chart to show daily sales based on store location 1028 for store type 3 in 2017

fig = go.Figure(go.Scatter(x = filtered_df17_3SL['DATE'] , y = filtered_df17_3SL['SALES']))
fig.update_layout(title = '2017 SALES DATA FOR STORE LOCATION 1028',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[37]:


#chart to show daily sales based on store location 1003 for store type 3 in 2017
fig = go.Figure(go.Scatter(x = filtered_df17_3SL1['DATE'] , y = filtered_df17_3SL1['SALES'], marker_color='rgba(148, 0, 0, .8)'))
fig.update_layout(title = '2017 SALES DATA FOR STORE LOCATION 1003', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[38]:


#chart to show daily sales for all 3 store type in 2017

fig = go.Figure(go.Scatter(x = filtered_df1['DATE'] , y = filtered_df1['SALES']))
fig.update_layout(title = '2017 SALES DATA FOR ALL 3 STORES',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[39]:


#scatter plot to show daily sales for all 3 store type in 2016 

fig = px.scatter(x = filtered_df['DATE'] , y = filtered_df['SALES'])
fig.show()


# In[40]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[41]:


#set columns 
Predictors = ["STORE LOCATION", "STORE TYPE", "SNWD", "TMAX", "TMIN"]


# In[42]:


X_train = filtered_df17_3SL[Predictors]


# In[43]:


X_train.head(5)


# In[44]:


y_train = filtered_df17_3SL["SALES"]


# In[45]:


#X_test = pd.get_dummies(filtered_df17_3SL[Predictors])
X_test = (filtered_df17_3SL[Predictors])


# In[46]:


X_test.head(5)


# In[47]:


#Fit logistic regression model and calculate accuracy using the train data.
my_lr = LogisticRegression(max_iter=4000000).fit(X_train, y_train)
lr_pred_train = my_lr.predict(X_train)
metrics.accuracy_score(y_train, lr_pred_train)


# In[48]:


#create confusion matrix
metrics.confusion_matrix(y_train, lr_pred_train)
metrics.confusion_matrix(y_train, lr_pred_train, normalize ="true")


# In[49]:


#Make predictions using test data.
lr_pred_test = my_lr.predict(X_test)


# In[50]:


#Format predictions for output and write to csv.
lr_output = pd.DataFrame(lr_pred_test,
                      index=X_test.index,
                      columns=["SALES"])
lr_output.to_csv("lr_pred.csv")


# In[51]:


#Fit decision tree model and calculate accuracy on the train data.
my_tree = DecisionTreeClassifier().fit(X_train, y_train)
tree_pred_train = my_tree.predict(X_train)
metrics.accuracy_score(y_train, tree_pred_train)


# In[52]:


#Create confusion matrix for the train data.
pd.DataFrame(metrics.confusion_matrix(y_train, tree_pred_train, normalize="true"))


# In[53]:


#Make predictions using test data.
tree_pred_test = my_tree.predict(X_test)


# In[54]:


#Format predictions for output and write to csv.
tree_output = pd.DataFrame(tree_pred_test,
                      index=X_test.index,
                      columns=["SALES"])
tree_output.to_csv("tree_pred.csv")

