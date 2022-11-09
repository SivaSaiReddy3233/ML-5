#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data_CC=pd.read_csv("CC.csv")
data_CC


# In[5]:


data_CC.isnull().any()


# In[7]:


data_CC.fillna(data_CC.mean(), inplace=True)
data_CC.isnull().any()


# In[9]:


x = data_CC.iloc[:,1:-1]
y = data_CC.iloc[:,-1]
print(x.shape)
print(y.shape)


# In[54]:


#1.a
pca = PCA(3)
x_pca = pca.fit_transform(x)
principalDf = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, data_CC.iloc[:,-1]], axis = 1)
finalDf


# In[12]:


#1.b
X = finalDf.iloc[:,0:-1]
y = finalDf.iloc[:,-1]


# In[13]:


nclusters = 3 
km = KMeans(n_clusters=nclusters)
km.fit(X)

y_cluster_kmeans = km.predict(X)

print(classification_report(y, y_cluster_kmeans, zero_division=1))
print(confusion_matrix(y, y_cluster_kmeans))

train_accuracy = accuracy_score(y, y_cluster_kmeans)
print("\nAccuracy of Training ds with PCA:", train_accuracy)

score = metrics.silhouette_score(X, y_cluster_kmeans)
print("Sihouette : ",score) 


# In[15]:


#1.c
#Scaling
scaler = StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)


# In[18]:


pca = PCA(3)
x_pca = pca.fit_transform(X_scaled_array)
principalDf = pd.DataFrame(data = x_pca, columns = ['principal component 1', 'principal component 2','principal component 3'])
finalDf = pd.concat([principalDf, data_CC.iloc[:,-1]], axis = 1)
finalDf


# In[19]:


X = finalDf.iloc[:,0:-1]
y = finalDf["TENURE"]
print(X.shape)
print(y.shape)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.34,random_state=0)
nclusters = 3 
km = KMeans(n_clusters=nclusters)
km.fit(X_train,y_train)

y_clus_train = km.predict(X_train)

# predictions made by the classifier
print(classification_report(y_train, y_clus_train, zero_division=1))
print(confusion_matrix(y_train, y_clus_train))

train_accuracy = accuracy_score(y_train, y_clus_train)
print("Accuracy of Training ds with PCA:", train_accuracy)

score = metrics.silhouette_score(X_train, y_clus_train)
print("Sihouette Score: ",score)


# In[21]:


y_clus_test = km.predict(X_test)

#summary
print(classification_report(y_test, y_clus_test, zero_division=1))
print(confusion_matrix(y_test, y_clus_test))

train_accuracy = accuracy_score(y_test, y_clus_test)
print("\nAccuracy of Training dataset with PCA:", train_accuracy)

# sihouette Score
score = metrics.silhouette_score(X_test, y_clus_test)
print("Sihouette Score: ",score) 


# In[22]:


data_pd=pd.read_csv("pd_speech_features.csv")
data_pd


# In[23]:


data_pd.isnull().any()


# In[24]:


X = data_pd.drop('class',axis=1).values
y = data_pd['class'].values


# In[31]:


#2.a Scaling Data
scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)


# In[34]:


#2.b
# Apply PCA with k =3
pca3 = PCA(n_components=3)
principalComponents = pca3.fit_transform(X_Scale)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','Principal Component 3'])

finalDf = pd.concat([principalDf, data_pd[['class']]], axis = 1)
finalDf


# In[35]:


X = finalDf.drop('class',axis=1).values
y = finalDf['class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.34,random_state=0)


# In[36]:


#2.c

from sklearn.svm import SVC

svmClassifier = SVC()
svmClassifier.fit(X_train, y_train)

y_pred = svmClassifier.predict(X_test)

# Summary predictions for classifer
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))


glass_acc_svc = accuracy_score(y_pred,y_test)
print('accuracy is',glass_acc_svc )


score = metrics.silhouette_score(X_test, y_pred)
print("Sihouette Score: ",score) 


# In[46]:


#3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
data_iris = pd.read_csv("Iris.csv")
data_iris.info()
print("=============")
data_iris


# In[48]:


data_iris.isnull().any()


# In[50]:


x = data_iris.iloc[:,1:-1]
y = data_iris.iloc[:,-1]
print(x.shape)
print(y.shape)


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[52]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
le = LabelEncoder()
y = le.fit_transform(y)


# In[53]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


#4

