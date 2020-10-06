<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:25:33 2020

@author: MrEth
"""

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cbook as cbook
import pandas as pd
from datetime import date
def logdiff(df):
    log = np.log(df.mask(df <= 0))
    logminus1day = log.shift(7, axis = 1)
    return log - logminus1day
today = date.today().strftime("%m/%d/%y")
#import tslearn.clustering as ts
assembledpath = "./Assembled Data/"
data_df = pd.read_excel(assembledpath + 'USA_LockdownStates_Equal.xlsx')
data_df = data_df.set_index('States')
data_df = data_df.rename_axis(axis=1, mapper='Dates')
#temp = data_df.pivot_table(index=['States'], columns = ['Dates'], values = ['Lockdown Effectiveness'])
#pd.pivot_table(data_df, index = ["States"], columns = ["3/17/2020","3/18/2020","3/19/2020","3/20/2020"])
Z=linkage(data_df.iloc[:,:], 'ward', optimal_ordering=True)
from scipy.cluster.hierarchy import inconsistent
depth = 2
incons = inconsistent(Z, depth)
incons[-10:]
#Elbow method
last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)
#get clusters
from scipy.cluster.hierarchy import fcluster
max_d = 4
clusters = fcluster(Z, max_d, criterion='distance')
clusters
# sorted df for clusters
states = data_df.iloc[:,:].index.tolist()
cluster_df = pd.DataFrame({'State' : states, 'Cluster' : clusters}).sort_values('Cluster')

plt.figure(figsize = (25, 10))
plt.title("State Clustering by Lockdown Effectiveness (3/1/20-" + today + ")")
plt.xlabel("State")
plt.ylabel("Distance")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,
           labels=data_df.index.values)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

newcases_df = pd.read_excel('./Assembled Data/USA_States_COVID-19-NewPositivePer100k_2020-08-23.xlsx', index_col = 'States', parse_dates=True)
lockdown_df = pd.read_excel('./Assembled Data/USA_LockdownStates_Equal.xlsx', index_col='States', parse_dates=True)
newcases_df.columns = pd.to_datetime(newcases_df.columns)
lockdown_df.columns = pd.to_datetime(lockdown_df.columns)
newcases_df.fillna(0, inplace=True)
fig1, axes1 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig2, axes2 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig3, axes3 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig4, axes4 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10.), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig5, axes5 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10.), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
i1, i2, i3, i4, i5 = 0, 0, 0, 0, 0
j1, j2, j3, j4, j5 = 0, 0, 0, 0, 0
for state in lockdown_df.index.values:
    combined_df = pd.DataFrame({'LE' : lockdown_df.loc[state], 'New Cases' : newcases_df.loc[state]})
    clusternum = cluster_df[cluster_df['State'] == state]['Cluster'].values[0]
    if clusternum == 1:
        axestouse = axes1[i1, j1]
        figtouse = fig1
        i, j = i1, j1
    elif clusternum == 2:
        axestouse = axes2[i2, j2]
        figtouse = fig2
        i, j = i2, j2
    elif clusternum == 3:
        axestouse = axes3[i3, j3]
        figtouse = fig3
        i, j = i3, j3
    elif clusternum == 4:
        axestouse = axes4[i4, j4]
        figtouse = fig4
        i, j = i4, j4
    else:
        axestouse = axes5[i5, j5]
        figtouse = fig5
        i, j = i5, j5
        print('new cluster detected.')
        #break;
    ax= combined_df[['LE', 'New Cases']].plot(y=["LE", "New Cases"], ax = axestouse, ylim=(0, 1), secondary_y='New Cases', title = state + '(Cluster: ' + str(cluster_df[cluster_df['State'] == state]['Cluster'].values[0]) + ')')
    ax.set_ylim(0,1)
    
    #datemin = np.datetime64(combined_df.index.values[0])
    #datemax = np.datetime64()
    #fig = ax.get_figure()
    ax = figtouse.get_axes()
    ax[i * 5 + j].right_ax.set_ylim(0,35)
    ax[i*5+j].set(xlabel='Date')
    import matplotlib.dates as mdates
    import matplotlib.cbook as cbook
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%m')
    ax[i*5+j].xaxis.set_major_locator(months)
    ax[i*5+j].xaxis.set_major_formatter(months_fmt)
    #combined_df['LE'].plot(ax = axes[i, j], ylim=(0, 1))
    #combined_df.fillna(0, inplace = True)
    #combined_df['New Cases'].plot(secondary_y=True, style='g', ylim=(0, 10000), ax = axes[i, j], title = state)
    if clusternum == 1:
        if j1 == 4:
            i1 += 1
            j1 = 0
        else:
            j1 += 1
        
    elif clusternum == 2:
        if j2 == 4:
            i2 += 1
            j2 = 0
        else:
            j2 += 1
    elif clusternum == 3:
        if j3 == 4:
            i3 += 1
            j3 = 0
        else:
            j3 += 1
    elif clusternum == 4:
        if j4 == 4:
            i4 += 1
            j4 = 0
        else:
            j4 += 1
    elif clusternum == 5:
        if j5 == 5:
            i5 += 1
            j5 = 0
        else:
            j5 += 1





fig, ax = plt.subplots(1,2)
ax[0].scatter(np.arange(len(data_df.iloc[0])), data_df.loc['Michigan', :],marker='x', linewidths=0.1)
ax[1].scatter(np.arange(len(data_df.iloc[0])), data_df.loc['Delaware', :],marker='x',linewidths=0.1)
plt.show()
plt.figure(figsize = (50, 20))





fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle('Lockdown Effectiveness of USA States/Regions')
for ax, name in zip(ax.flatten(), data_df.index.values):
        ax.scatter(np.arange(len(data_df.iloc[0])), data_df.loc[name,:])
        ax.set_ylim([0,1])
        ax.set_xlabel("Day")
        ax.set_ylabel("Lockdown Effectiveness")
        ax.set_title(name)
    
plt.show()
import scipy.stats
import datetime
start = datetime.datetime(2020, 4, 1)
end = datetime.datetime(2020, 7, 17)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = (40, 20))
for clusternum in range(1, 5):
    c_df = cluster_df[cluster_df['Cluster']== clusternum]
    print("Cluster ", clusternum)
    cluster_newcases = newcases_df.loc[c_df.State.tolist()].fillna(0)
    z_scores = scipy.stats.zscore(cluster_newcases, nan_policy='omit')
    abs_z_scores = np.nan_to_num(np.abs(z_scores))
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    new_df = cluster_newcases[filtered_entries]
    new_df.describe().loc['mean'].plot(ax= ax[(clusternum - 1)][0])
    cluster_newcases.describe().loc['mean'].plot(ax=ax[(clusternum - 1)][1])
    #print(newcases_df.loc[c_df.State.tolist()].describe())


=======
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:25:33 2020

@author: MrEth
"""

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cbook as cbook
import pandas as pd
from datetime import date
def logdiff(df):
    log = np.log(df.mask(df <= 0))
    logminus1day = log.shift(7, axis = 1)
    return log - logminus1day
today = date.today().strftime("%m/%d/%y")
#import tslearn.clustering as ts
assembledpath = "./Assembled Data/"
data_df = pd.read_excel(assembledpath + 'USA_LockdownStates_Equal.xlsx')
data_df = data_df.set_index('States')
data_df = data_df.rename_axis(axis=1, mapper='Dates')
#temp = data_df.pivot_table(index=['States'], columns = ['Dates'], values = ['Lockdown Effectiveness'])
#pd.pivot_table(data_df, index = ["States"], columns = ["3/17/2020","3/18/2020","3/19/2020","3/20/2020"])
Z=linkage(data_df.iloc[:,:], 'ward', optimal_ordering=True)
from scipy.cluster.hierarchy import inconsistent
depth = 2
incons = inconsistent(Z, depth)
incons[-10:]
#Elbow method
last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)
#get clusters
from scipy.cluster.hierarchy import fcluster
max_d = 4
clusters = fcluster(Z, max_d, criterion='distance')
clusters
# sorted df for clusters
states = data_df.iloc[:,:].index.tolist()
cluster_df = pd.DataFrame({'State' : states, 'Cluster' : clusters}).sort_values('Cluster')

plt.figure(figsize = (25, 10))
plt.title("State Clustering by Lockdown Effectiveness (3/1/20-" + today + ")")
plt.xlabel("State")
plt.ylabel("Distance")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,
           labels=data_df.index.values)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

newcases_df = pd.read_excel('./Assembled Data/USA_States_COVID-19-NewPositivePer100k_2020-08-23.xlsx', index_col = 'States', parse_dates=True)
lockdown_df = pd.read_excel('./Assembled Data/USA_LockdownStates_Equal.xlsx', index_col='States', parse_dates=True)
newcases_df.columns = pd.to_datetime(newcases_df.columns)
lockdown_df.columns = pd.to_datetime(lockdown_df.columns)
newcases_df.fillna(0, inplace=True)
fig1, axes1 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig2, axes2 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig3, axes3 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig4, axes4 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10.), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
fig5, axes5 = plt.subplots(nrows=5, ncols = 5, figsize=(22,10.), sharex = True)
plt.subplots_adjust(hspace = 0.5, wspace = 0.2)
i1, i2, i3, i4, i5 = 0, 0, 0, 0, 0
j1, j2, j3, j4, j5 = 0, 0, 0, 0, 0
for state in lockdown_df.index.values:
    combined_df = pd.DataFrame({'LE' : lockdown_df.loc[state], 'New Cases' : newcases_df.loc[state]})
    clusternum = cluster_df[cluster_df['State'] == state]['Cluster'].values[0]
    if clusternum == 1:
        axestouse = axes1[i1, j1]
        figtouse = fig1
        i, j = i1, j1
    elif clusternum == 2:
        axestouse = axes2[i2, j2]
        figtouse = fig2
        i, j = i2, j2
    elif clusternum == 3:
        axestouse = axes3[i3, j3]
        figtouse = fig3
        i, j = i3, j3
    elif clusternum == 4:
        axestouse = axes4[i4, j4]
        figtouse = fig4
        i, j = i4, j4
    else:
        axestouse = axes5[i5, j5]
        figtouse = fig5
        i, j = i5, j5
        print('new cluster detected.')
        #break;
    ax= combined_df[['LE', 'New Cases']].plot(y=["LE", "New Cases"], ax = axestouse, ylim=(0, 1), secondary_y='New Cases', title = state + '(Cluster: ' + str(cluster_df[cluster_df['State'] == state]['Cluster'].values[0]) + ')')
    ax.set_ylim(0,1)
    
    #datemin = np.datetime64(combined_df.index.values[0])
    #datemax = np.datetime64()
    #fig = ax.get_figure()
    ax = figtouse.get_axes()
    ax[i * 5 + j].right_ax.set_ylim(0,35)
    ax[i*5+j].set(xlabel='Date')
    import matplotlib.dates as mdates
    import matplotlib.cbook as cbook
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%m')
    ax[i*5+j].xaxis.set_major_locator(months)
    ax[i*5+j].xaxis.set_major_formatter(months_fmt)
    #combined_df['LE'].plot(ax = axes[i, j], ylim=(0, 1))
    #combined_df.fillna(0, inplace = True)
    #combined_df['New Cases'].plot(secondary_y=True, style='g', ylim=(0, 10000), ax = axes[i, j], title = state)
    if clusternum == 1:
        if j1 == 4:
            i1 += 1
            j1 = 0
        else:
            j1 += 1
        
    elif clusternum == 2:
        if j2 == 4:
            i2 += 1
            j2 = 0
        else:
            j2 += 1
    elif clusternum == 3:
        if j3 == 4:
            i3 += 1
            j3 = 0
        else:
            j3 += 1
    elif clusternum == 4:
        if j4 == 4:
            i4 += 1
            j4 = 0
        else:
            j4 += 1
    elif clusternum == 5:
        if j5 == 5:
            i5 += 1
            j5 = 0
        else:
            j5 += 1





fig, ax = plt.subplots(1,2)
ax[0].scatter(np.arange(len(data_df.iloc[0])), data_df.loc['Michigan', :],marker='x', linewidths=0.1)
ax[1].scatter(np.arange(len(data_df.iloc[0])), data_df.loc['Delaware', :],marker='x',linewidths=0.1)
plt.show()
plt.figure(figsize = (50, 20))





fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle('Lockdown Effectiveness of USA States/Regions')
for ax, name in zip(ax.flatten(), data_df.index.values):
        ax.scatter(np.arange(len(data_df.iloc[0])), data_df.loc[name,:])
        ax.set_ylim([0,1])
        ax.set_xlabel("Day")
        ax.set_ylabel("Lockdown Effectiveness")
        ax.set_title(name)
    
plt.show()
import scipy.stats
import datetime
start = datetime.datetime(2020, 4, 1)
end = datetime.datetime(2020, 7, 17)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = (40, 20))
for clusternum in range(1, 5):
    c_df = cluster_df[cluster_df['Cluster']== clusternum]
    print("Cluster ", clusternum)
    cluster_newcases = newcases_df.loc[c_df.State.tolist()].fillna(0)
    z_scores = scipy.stats.zscore(cluster_newcases, nan_policy='omit')
    abs_z_scores = np.nan_to_num(np.abs(z_scores))
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    new_df = cluster_newcases[filtered_entries]
    new_df.describe().loc['mean'].plot(ax= ax[(clusternum - 1)][0])
    cluster_newcases.describe().loc['mean'].plot(ax=ax[(clusternum - 1)][1])
    #print(newcases_df.loc[c_df.State.tolist()].describe())


>>>>>>> 4d36374fde536917603a47ce5a9ad5db89b38ad1
#km = ts.TimeSeriesKMeans(metric="dtw").fit(data_df.iloc[:,:])