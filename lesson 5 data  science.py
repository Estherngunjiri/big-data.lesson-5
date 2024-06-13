import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
data=pd.read_csv('OnlineRetail.csv',encoding='iso-8859-1')
print(data.info())
print(data.head(20))
#toknow the length of the data;
print(data.shape)
#to know the attributes in the columns
print(data.columns)
#convert the invoice data type to integer or float
#data['InvoiceNo']=data['InvoiceNo'].astype (int64)
print(data.mode())
print(data.describe())
print(data.mean)
print(data.isnull().sum())
#to reduce the sum of null value to morescalable values
#to reduce the sum of null value to morescalable values
data_null=round(100*(data.isnull().sum())/len(data), 2)
print(data_null)
#to drop the null values:
data=data.dropna()
print(data)
#to drop the invoice no row data:
#data=data.drop("InvoiceNO")
#print(data.head())
#to change the customer id data type to string:
data['CustomerID']=data['CustomerID'].astype(str)
print(data.info())
#to introduce a new column Amount:
data['Amount']=data['Quantity']*data['UnitPrice']
print(data.info())
print(data.head())

df=data.groupby('CustomerID')['Amount'].sum
print(df)
# Grouping by Country and calculating total sales
sales_by_country=data.groupby('Country')['Amount'].sum().sort_values(ascending=False)
print(sales_by_country.head())
# introduce a new column frequency to determine the most sold product:
mostsold=data.groupby('Description')['InvoiceNo'].sum()
print(mostsold)
#total last month sales
lastmonth_sales=data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
print(lastmonth_sales.head())
# Convert to datetime to proper datatype
data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
print(data.head())
#to know the last maximum transaction date:
max_date=max(data['InvoiceDate'])
print(max_date)
#to know the last minimum date:
min_date=min(data['InvoiceDate'])
print(min_date)
#difference between maximum and minimum transaction dates:
days=max_date-min_date
print(days)
last_month=(max_date-pd.DateOffset(months=1)).month
last_month_year=(max_date-pd.DateOffset(months=1)).year
last_month_sales=data[(data['InvoiceDate'].dt.month==last_month)&(data['InvoiceDate'].dt.year==last_month_year)
]
print('last month sales data:')
print(last_month_sales)
#last month sales
from datetime import timedelta
new_min_date=max_date-timedelta(days=30)
totalsales = last_month_sales['Quantity']*last_month_sales['UnitPrice']
totalsales=totalsales.sum()
print(f'total sales for last month: {totalsales}')

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

df.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df[['Quantity','UnitPrice']],df[['Description']],test_size=0.3,random_state=0)
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
from sklearn.cluster import KMeans
df_3=KMeans(n_clusters = 3,random_state =0,n_init='auto')
df_3.fit(x_train_norm)
sns.scatterplot(df=x_train,x='Quantity',y='UnitPrice',hue=df_3.labels_)
plt.show()
from sklearn.metrics import silhouette_score
perf=(silhouette_score(x_train_norm,df_3.labels_,metric='euclidean'))
print(perf)
'''Testing a number of clusters to determine how many to use'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    '''Train the model for the current value of k on the training model'''
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train_norm)
    fit.append(model)
    score.append(silhouette_score(x_train_norm,model.labels_,metric='euclidean'))
print(fit)
print(score)
'''Visualize the models for k=2,k=4,k=7,k=5'''
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[0].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[2].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[5].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[0].labels_)
plt.show()
sns.lineplot(x=K,y=score)
plt.show()
sns.scatterplot(data=x_train,x='UnitPrice',y='Quantity',hue=fit[3].labels_)
plt.show()
sns.boxplot(x=fit[3].labels_, y=y_train['Description'])
plt.show()











 
