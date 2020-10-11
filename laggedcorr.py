# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:30:48 2020

@author: MrEth
"""

#Evaluating the correlation between lockdown effectiveness 
#and Daily new Cases
#per 100k population in a state

import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
today = date.today().strftime("%m/%d/%y")
case_df = pd.read_excel("./Assembled Data/" + 'USA_States_COVID-19-NewPositivePer100k_2020-08-23.xlsx', index_col='States')
data_df = pd.read_excel("./Assembled Data/" + 'USA_LockdownStates_Equal.xlsx')
data_df = data_df.set_index('States')
data_df = data_df.rename_axis(axis=1, mapper='Dates')
newcase_df = pd.read_excel("./Assembled Data/" + 'USA_States_COVID-19-NewPositive_'+ '2020-08-23.xlsx', index_col = 'States')
newcase_df.columns = pd.to_datetime(newcase_df.columns)
case_df.columns = pd.to_datetime(case_df.columns)
data_df.columns = pd.to_datetime(data_df.columns)
case_df.fillna(0, inplace=True)
rolling3 = ((case_df.rolling(3, axis=1).sum())/3).fillna(0)
rolling7 = ((case_df.rolling(7, axis=1).sum())/7).fillna(0)
states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
whynot = np.log(rolling3.mask(rolling3 <= 0)).fillna(0).div(np.log(rolling7.mask(rolling7 <= 0)).fillna(0))

def logdiff(df):
    log = np.log(df.mask(df <= 0))
    logminus7day = log.shift(7, axis = 1)
    return log - logminus7day
fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
j = 0
bestlag = 0
bestmean = 0
beststart = 0
bestend = 0
#for k in range(30, 100):
for i in range(5,25):
    corr_df = pd.DataFrame(columns = states, index = ['R'])
    #i=20
    for state in states:
        startday= 0
        endday = 199
        differenceInDates = (logdiff(rolling7).loc[state].index[-1] - data_df.loc['Pennsylvania'].index[150]).days + 1
        #test_df = pd.DataFrame({'Case': whynot.loc[state][:-14], 'LE': ([0]*i) + (data_df.loc[state][: -i].tolist())}).fillna(0)
        test_df = pd.DataFrame({'Case': logdiff(rolling7).loc[state][:-differenceInDates], 'LE': ([0]*i) + (data_df.loc[state][: 150-i].tolist())}).fillna(0)
        test_df = test_df.iloc[startday:endday]
        overall_pearson_r = test_df.corr().iloc[0,1]
        #print(f"Pandas computed Pearson r: {overall_pearson_r}")
        corr_df[state]['R'] = overall_pearson_r
        #if i == 25 and (state == 'Pennsylvania' or state == 'Florida' or state == 'California'):
            
         #   j+=1
    mean = corr_df.fillna(0).transpose().describe().loc['mean'].values[0]
    if mean < bestmean:
        bestlag = i
        bestmean = mean
        beststart = startday
        bestend= endday
    print("Lag Time: ", i)
    print("Start Day: ", j)
    print('Mean: ' + str(corr_df.fillna(0).transpose().describe().loc['mean'].values[0]))
print(bestlag)
print(bestmean)
i = bestlag
startday = beststart
endday = bestend
state = 'California'
test_df = pd.DataFrame({'Case': logdiff(rolling7).loc[state][:-differenceInDates], 'LE': ([0]*i) + (data_df.loc[state][: 150-i].tolist())}).fillna(0)
test_df = test_df.iloc[startday:endday]
test_df.plot(secondary_y='LE', ax=axes1[0], title='California Lockdown vs 3-Day Infection Growth Rate by Week (Lagged by ' + str(i) + ' Days)')
state = 'Pennsylvania'
test_df = pd.DataFrame({'Case': logdiff(rolling7).loc[state][:-differenceInDates], 'LE': ([0]*i) + (data_df.loc[state][: 150-i].tolist())}).fillna(0)
test_df = test_df.iloc[startday:endday]
test_df.plot(secondary_y='LE', ax=axes1[1], title='Pennsylvania Lockdown vs 3-Day Infection Growth Rate by Week (Lagged by ' + str(i) + ' Days)')
state = 'Florida'
test_df = pd.DataFrame({'Case': logdiff(rolling7).loc[state][:-differenceInDates], 'LE': ([0]*i) + (data_df.loc[state][: 150-i].tolist())}).fillna(0)
test_df = test_df.iloc[startday:endday]
test_df.plot(secondary_y='LE', ax=axes1[2], title='Florida Lockdown vs 3-Day Infection Growth Rate by Week (Lagged by ' + str(i) + ' Days)')

    #print(corr_df.fillna(0).transpose().describe())

# out: Pandas computed Pearson r: 0.2058774513561943

#r, p = stats.pearsonr(df.dropna()['S1_Joy'], df.dropna()['S2_Joy'])
#print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

# Compute rolling window synchrony
#f,ax=plt.subplots(figsize=(7,3))
#df.rolling(window=30,center=True).median().plot(ax=ax)
#ax.set(xlabel='Time',ylabel='Pearson r')
#ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}");