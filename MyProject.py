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


sales = pd.read_csv("sales.csv")
sales.head()


# In[3]:


df1 = pd.DataFrame(sales, columns=["DATE", "STORE LOCATION", "STORE TYPE", "SALES", "CLOSEST WEATHER CENTER LOCATION"])
print(df1)


# In[4]:


del sales['Unnamed: 5']


# In[5]:


sales['STORE LOCATION'] = sales['STORE LOCATION'].astype(str)
sales['STORE TYPE'] = sales['STORE TYPE'].astype(str)


# In[6]:


sales['DATE'] = pd.to_datetime(sales["DATE"])


# In[7]:


print(sales["DATE"])


# In[8]:


sales.dropna()


# In[9]:


sales = sales.astype({'SALES': 'int64'})
print(sales.dtypes)


# In[10]:


weather = pd.read_csv("weather.csv")
weather.head()


# In[11]:


weather['DATE'] = pd.to_datetime(weather["DATE"])


# In[12]:


weather.dropna()


# In[13]:


salesweather = pd.merge(sales, weather, how = "outer", on = "DATE")
salesweather.head()


# In[14]:


weather.dtypes


# In[15]:


print(salesweather.dtypes)


# In[16]:


import plotly.graph_objects as go


# In[17]:


filtered_df = salesweather.query("DATE >= '2016-01-01'                        & DATE <= '2016-12-31'")


# In[18]:


filtered_df1 = salesweather.query("DATE >= '2017-01-01'                        & DATE <= '2017-12-31'")


# In[19]:


store_type1 = salesweather[(salesweather["STORE TYPE"] == "1")]
store_type2 = salesweather[(salesweather["STORE TYPE"] == "2")]
store_type3 = salesweather[(salesweather["STORE TYPE"] == "3")]


# In[20]:


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


df1 = filtered_df.groupby('STORE TYPE')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Top Store Types That Make The Most Money in 2016', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[23]:


df1 = filtered_df1.groupby('STORE TYPE')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Top Store Types That Make The Most Money in 2017', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[24]:


values = filtered_df['SALES']
names = filtered_df['STORE TYPE']
fig = px.pie(filtered_df, values=values, names=names, title="2016 Total Sales by Store Type")
fig.show()
print(values)


# In[25]:


values = filtered_df1['SALES']
names = filtered_df1['STORE TYPE']
fig = px.pie(filtered_df1, values=values, names=names, title="2017 Total Sales by Store Type")
fig.show()
print(values)


# In[26]:


fig = go.Figure(go.Scatter(x = filtered_df16_3['DATE'] , y = filtered_df16_3['SALES'], marker_color='rgba(152, 0, 0, .8)'))
fig.update_layout(title = '2016 SALES DATA FOR STORE TYPE 3', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[27]:


fig = go.Figure(go.Scatter(x = filtered_df17_3['DATE'] , y = filtered_df17_3['SALES'], marker_color='rgba(152, 0, 0, .8)'))
fig.update_layout(title = '2017 SALES DATA FOR STORE TYPE 3', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[28]:


df1 = filtered_df16_3.groupby('STORE LOCATION')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Total Slaes in 2016 by Store Locations for Store Type 3', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[29]:


df1 = filtered_df17_3.groupby('STORE LOCATION')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=df1.values, y=df1.index, palette='viridis')

plt.bar_label(plt.gca().containers[0], fmt='%.2f')
plt.title('Total Slaes in 2017 by Store Locations for Store Type 3', fontsize=16)
plt.xlabel('Total sales', fontsize=12)
plt.ylabel('Store Location', fontsize=12)

plt.show()


# In[30]:


values = filtered_df16_3['SALES']
names = filtered_df16_3['STORE LOCATION']
fig = px.pie(filtered_df16_3, values=values, names=names, title="2016 Total Sales by Store Location in Store Type 3")
fig.show()
print(values)


# In[31]:


values = filtered_df17_3['SALES']
names = filtered_df17_3['STORE LOCATION']
fig = px.pie(filtered_df17_3, values=values, names=names, title="2017 Total Sales by Store Location in Store Type 3")
fig.show()
print(values)


# In[32]:


fig = px.scatter(x = filtered_df16_3['DATE'] , y = filtered_df16_3['SNWD'])
fig.update_layout(title = '2016 SNOW DATA',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[33]:


fig = px.scatter(x = filtered_df17_3['DATE'] , y = filtered_df17_3['SNWD'])
fig.update_layout(title = '2017 SNOW DATA',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[34]:


filtered_df16_3SL= filtered_df16_3[(filtered_df16_3["STORE LOCATION"] == "1028")]
filtered_df17_3SL= filtered_df17_3[(filtered_df17_3["STORE LOCATION"] == "1028")]
filtered_df16_3SL1= filtered_df16_3[(filtered_df16_3["STORE LOCATION"] == "1003")]
filtered_df17_3SL1= filtered_df17_3[(filtered_df17_3["STORE LOCATION"] == "1003")]


# In[35]:


fig = go.Figure(go.Scatter(x = filtered_df16_3SL['DATE'] , y = filtered_df16_3SL['SALES']))
fig.update_layout(title = '2016 SALES DATA FOR STORE LOCATION 1028',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[36]:


fig = go.Figure(go.Scatter(x = filtered_df17_3SL['DATE'] , y = filtered_df17_3SL['SALES']))
fig.update_layout(title = '2017 SALES DATA FOR STORE LOCATION 1028',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[37]:


fig = go.Figure(go.Scatter(x = filtered_df17_3SL1['DATE'] , y = filtered_df17_3SL1['SALES'], marker_color='rgba(148, 0, 0, .8)'))
fig.update_layout(title = '2017 SALES DATA FOR STORE LOCATION 1003', 
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[38]:


fig = go.Figure(go.Scatter(x = filtered_df1['DATE'] , y = filtered_df1['SALES']))
fig.update_layout(title = '2017 SALES DATA FOR ALL 3 STORES',
                 xaxis_tickformat = '%d %B (%a)<br>%Y')

fig.show()


# In[39]:


fig = px.scatter(x = filtered_df['DATE'] , y = filtered_df['SALES'])
fig.show()


# In[40]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[41]:


Predictors = ["STORE LOCATION", "STORE TYPE", "SNWD", "TMAX", "TMIN"]


# In[42]:


X_train = filtered_df17_3SL[Predictors]


# In[43]:


y_train = filtered_df17_3SL["SALES"]


# In[44]:


X_test = pd.get_dummies(filtered_df17_3SL[Predictors])


# In[45]:


my_lr = LogisticRegression(max_iter=4000000).fit(X_train, y_train)
lr_pred_train = my_lr.predict(X_train)
metrics.accuracy_score(y_train, lr_pred_train)


# In[46]:


metrics.confusion_matrix(y_train, lr_pred_train)
metrics.confusion_matrix(y_train, lr_pred_train, normalize ="true")


# In[47]:


lr_pred_test = my_lr.predict(X_test)


# In[48]:


lr_output = pd.DataFrame(lr_pred_test,
                      index=X_test.index,
                      columns=["SALES"])
lr_output.to_csv("lr_pred.csv")


# In[49]:


my_tree = DecisionTreeClassifier().fit(X_train, y_train)
tree_pred_train = my_tree.predict(X_train)
metrics.accuracy_score(y_train, tree_pred_train)


# In[50]:


pd.DataFrame(metrics.confusion_matrix(y_train, tree_pred_train, normalize="true"))


# In[51]:


tree_pred_test = my_tree.predict(X_test)


# In[52]:


tree_output = pd.DataFrame(tree_pred_test,
                      index=X_test.index,
                      columns=["SALES"])
tree_output.to_csv("tree_pred.csv")

