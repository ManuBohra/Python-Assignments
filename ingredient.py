# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:46:39 2019

@author: Lenovo
"""

'''Import Requirede Packages'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from scipy.stats import skew, kurtosis

'''****Every row/observation is considered as a unique formulation of additives****'''

'''Reading csv file'''
df = pd.read_csv('ingredient.csv', na_values = '.')

'''checking and plotting dataframe'''
df.head()
df.info()
df.describe()

''' Descriptive Stats '''
'''stats for every observational units'''

mean = df.mean(axis = 1)
median = df.median(axis = 1)
df_stats = df.apply([np.mean, np.median, np.std], axis = 1)
[plt.hist(df_stats[i]) for i in df_stats.columns if plt.subplots(sharex = False)]
[sns.distplot(df_stats[i]) for i in df_stats.columns.values if plt.subplots(sharex = False)]

'''stats for every additives'''
stats_add = df.apply([np.mean, np.median, np.std], axis = 0)
[plt.hist(stats_add[i]) for i in stats_add.columns if plt.subplots(sharex = False)]
[sns.distplot(stats_add[i]) for i in stats_add.columns.values if plt.subplots(sharex = False)]

'''Percentage of additives a,b,d,e,g present in every single formulation'''
#'''on average a contributes 1.49 percent of total additives in every single formulation'''
#'''on average b contributes 13.11 percent of total additives in every single formulation'''
#'''on average d contributes 1.34 percent of total additives in every single formulation'''
#'''on average e contributes 71.8 percent of total additives in every single formulation'''
#'''on average g contributes 8.48 percent of total additives in every single formulation''' 
#'''e is the major contributor in additives''' 
x = df.sum(axis = 1).mean()
y = stats_add/x
y.sort_values(by = y.loc['median',:])

#''' c is negatively skewed while a,f,g,h,i is positively skewed '''
'''cental tendency of given data is best described by Median'''
skewness = df.skew()

#''' a,f,g,h have high value of kurtosis '''
'''outliers are present in additives observations''' 
kutosis = df.kurt()

'''Plotting Histogram of Given Data'''
[plt.hist(df[i]) for i in df.columns if plt.subplots(sharex = False)]

'''Plotting Density Plots'''
[sns.distplot(df[i]) for i in df.columns.values if plt.subplots(sharex = False)]


'''Plotting Boxplot to check oultiers'''
b = df.b
d = df.d
e = df.e

col = [b,d,e]

[col[i].plot(kind = 'box') for i in range(len(col)) if plt.subplots(sharex = False)]

'''finding 25% & 75% quartile'''
q = df.quantile([.25, .75], axis = 0)

'''selecting b,d,e''' 
x = q[['b', 'd', 'e']] 

'''findig interquartile range'''
y = x.iloc[1]-x.iloc[0]

'''inner outliers range'''
inner_fence = y*1.5
outliers = (x.iloc[0] - inner_fence, x.iloc[1] + inner_fence)

# for additive b
ib = np.where((b < outliers[0][0]) | (b > outliers[1][0]))
b.value_counts(normalize = True)
b.describe()
b.skew()
b.kurt()
'''imputing outlier in b with max repetative sample'''
b.iloc[ib] = b.replace(b.iloc[ib], 13.00)
sns.distplot(b)

# for additive d
id = np.where((d < outliers[0][1]) | (d > outliers[1][1]))
d.value_counts(normalize = True)
d.describe()
d.skew()
d.kurt()
'''imputing outlier in d with max repetative sample'''
d.iloc[id] = d.replace(d.iloc[id], 1.54)
sns.distplot(d)

# for additive e
ie = np.where((e < outliers[0][2]) | (e > outliers[1][2]))
e.value_counts(normalize = True)
e.describe()
e.skew()
e.kurt()
'''imputing outlier in e with max repetative sample'''
e.iloc[ie] = e.replace(e.iloc[ie], 72.86)
sns.distplot(e)
#################

'''Taking Repetation counts of values'''
[df[i].value_counts() for i in df.columns]

#'''c is not present in 42 formulations'''
#'''f is not present in 30 formulations'''
#'''h is not present in 176 formulations'''
#'''i is not present in 144 formulations'''

'''Formulation where c,f,h,i additives are not presents'''
df[(df.c == 0) & (df.f == 0) & (df.h == 0) & (df.i == 0)]

'''Correlation Matrix : finding correlation between additives'''
cor = df.corr()
sns.heatmap(cor,annot = True, linewidths =1, cmap = 'YlGnBu')

'''Empirical Cummulative distribution Function'''
def ecdf(data):
    x = np.sort(data)
    n = len(x)
    y = np.arange(1,n+1)/n
    plt.plot(x,y)
    plt.show()
    return x, y

'''ECDF of Given Data'''
data = [ecdf(df[i]) for i in df.columns if plt.subplots(1,sharex = True)]

'''Mean and st. deviation of Given Data'''
stat = [df[i].apply([np.mean, np.std], axis = 0) for i in df.columns]

'''Theoretical Noraml distribution samples of Given Data from Calculated Mean & std '''
samples = [np.random.normal(stat[i][0],stat[i][1], size = 214) for i in range(len(stat))]

'''Theoretical ECDF of samples'''
theo = [ecdf(samples[i]) for i in range(len(samples))]

'''Function for Plotting ECDF Graphs'''
def twoplot(data1, data2):
    plt.plot(data1[0], data1[1], color = 'red', alpha = 0.3)
    plt.plot(data2[0], data2[1], color = 'blue', marker = '.')
    plt.show()

'''PLotting ECDF of Given Data & Samples'''
edcf_plots = [twoplot(data[i], theo[i]) for i in range(len(data)) if plt.subplots(sharex = True)] 


