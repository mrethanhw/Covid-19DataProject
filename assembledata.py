
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:23:09 2020

@author: MrEth
"""
import os
import pandas as pd
#basisfilepath = './Basis4/'
convertedfilepath = './ConvertedEqual/'
destinationfilepath = './Assembled Data/'
names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
allstatesdf = pd.DataFrame()
for filename in os.listdir(convertedfilepath):
    if filename.endswith(".xlsx") and '~$' not in filename:
        statename = filename.split('_')[0]
        reader = pd.read_excel(convertedfilepath + filename, sheet_name = 'Converted Data', usecols=['LockdownEffectiveness', 'Policy/Closing'], index_col='Policy/Closing')
        reader = reader.rename(columns={'LockdownEffectiveness': statename})[7:]
        reader = reader.transpose().rename_axis('States')
        reader = reader.rename_axis(axis=1, mapper='Dates')
        allstatesdf = pd.concat([allstatesdf, reader], sort = False)
assembled = allstatesdf.to_excel(destinationfilepath + 'USA_LockdownStates_Equal.xlsx')