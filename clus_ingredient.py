# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:23:21 2019

@author: Lenovo
"""

'''Clustering Ingredient data'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('ingredient.csv', na_values = '.')

           '''Hierarchical Clustering Algorithm'''
mergings  = linkage(df.values, method = 'complete')
dendrogram(mergings, leaf_rotation = 90, leaf_font_size = 6)
plt.axhline(y = 7, color = 'r')
plt.show()

hlabels = fcluster(mergings, 7, criterion= 'distance')

hs= pd.Series(hlabels)
dh = pd.concat([hs, df], axis = 1)
dlh = pd.DataFrame(dh)

column = ['labels', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
dlh.columns = column
hlabels_counts = dlh.groupby('labels').size()
print(hlabels_counts)

          
          '''Agglomerative Clustering Algorithm'''
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(df.values)
clabels = cluster.labels_

cs= pd.Series(clabels)
ch = pd.concat([cs, df], axis = 1)
dlc = pd.DataFrame(ch)

column = ['labels', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
dlc.columns = column
clabels_counts = dlc.groupby('labels').size()
print(clabels_counts)

#dendrogram is showing 5 number of clusters 
#with avg distance between clusters = 7  in given dataframe


#Estimation of optimal number of clusters for kmeans
ks = range(1,10)
inertia = []

for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(df.values)
    inertia.append(model.inertia_)

'''no. of optimal clusters = 5 from elbow plot'''
'''plotting Elbow Plot'''
plt.plot(ks,inertia)
plt.title('Elbow Plot')
plt.xlabel('K')
plt.ylabel('inertia')
plt.show()
plt.clf()


            '''KMEANS CLUSTERING ALGORITHM'''
kmeans = KMeans(n_clusters = 5)
kmeans.fit_predict(df.values)

print(kmeans.inertia_)
labels = kmeans.labels_

s= pd.Series(labels)
dl = pd.concat([s, df], axis = 1)
dlf = pd.DataFrame(dl)

column = ['labels', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
dlf.columns = column
labels_counts = dlf.groupby('labels').size()
print(labels_counts)

'''Plotting Labels Count'''
labels_counts.plot(kind = 'bar', color = 'blue', alpha = 0.2)
plt.title('Distinct Number of Formulations Present in Dataset', color = 'red')
plt.xlabel('Distinct Clusters')
plt.ylabel('Formulation Count for Given Data')
plt.show()
