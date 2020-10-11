# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:30:46 2020

@author: MrEth
"""

import pandas as pd
import numpy as np
import os
from datetime import date
import datetime
xlsx_path = './Assembled Data'
states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
pop_dict = pd.read_csv("./Web Data/nst-est2019-alldata.csv", usecols=['NAME', 'POPESTIMATE2019'], index_col='NAME', squeeze=True)[states].to_dict()
statesabbrev = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", 
          "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", 
          "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]
statedict = dict(zip(statesabbrev, states))
date1 = datetime.datetime(2020, 3, 1)
DATE_BEGIN = date1;
dates = np.array([])
while date1.date() != date.today() + datetime.timedelta(days=1):
        dates = np.append(dates, date1.date())#.strftime("%#m/%#d/%Y"))
        date1 += datetime.timedelta(days=1)
directory = "./Web Data"
states_mobile = ['US-' + n for n in statesabbrev]
pred = lambda x: x not in states_mobile
for file in os.scandir(directory):
    if "all-states-history" in file.path:
        daily_csv = pd.read_csv(file.path, usecols=['date', 'state', 'hospitalizedCurrently', 'positiveIncrease', 'deathIncrease','totalTestResultsIncrease'],index_col = "state")
    elif "Mobility" in file.path:
        mobility_csv = pd.read_csv(file.path, usecols=['iso_3166_2_code','date', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline'])
        mobility_csv = mobility_csv.loc[(mobility_csv["iso_3166_2_code"] == mobility_csv["iso_3166_2_code"]) & (mobility_csv["iso_3166_2_code"].str.contains('US-'))]
        #change column to abbreviation
        mobility_csv['iso_3166_2_code'] = mobility_csv["iso_3166_2_code"].apply(lambda x: x[3:])
        mobility_csv.rename(columns = {"iso_3166_2_code": "state"}, inplace=True)
        mobility_csv['date'] = pd.to_datetime(mobility_csv['date']).dt.date
        mobility_csv = mobility_csv.loc[(mobility_csv["date"] >= DATE_BEGIN.date()) & (mobility_csv["date"] <= date.today())]
        mobility_csv = mobility_csv.reset_index().drop("index", axis=1)
        
        #states_mobile = states_mobile = ['USA-' + n for n in statesabbrev]
        
daily_csv['date'] = pd.to_datetime(daily_csv['date'])#, format='%Y%m%d')
positiveIncr_all = pd.DataFrame(columns=dates, index=states)
hospitalizedCurr_all = pd.DataFrame(columns=dates, index=states)
testIncr_all = pd.DataFrame(columns=dates, index=states)
deathIncr_all = pd.DataFrame(columns=dates, index=states)
positiveIncr_perhundredk_all = pd.DataFrame(columns=dates, index=states)
positiveIncr_all.index.name = "States"
hospitalizedCurr_all.index.name = "States"
testIncr_all.index.name = "States"
deathIncr_all.index.name = "States"
for ab in statesabbrev:
    state = statedict[ab]
    state_df = daily_csv.loc[ab][::-1]
    state_df.set_index('date', inplace=True)
    positiveIncr_state =  state_df['positiveIncrease']
    hospitalizedCurr_state = state_df['hospitalizedCurrently']
    testIncr_state = state_df['totalTestResultsIncrease']
    deathIncr_state = state_df['deathIncrease']
    positiveIncr_all.loc[state] = positiveIncr_state
    hospitalizedCurr_all.loc[state] = hospitalizedCurr_state
    testIncr_all.loc[state] = testIncr_state
    deathIncr_all.loc[state] = deathIncr_state
positiveIncr_perhundredk_all = positiveIncr_all.apply(lambda x: ((x / pop_dict[x.name]) * 100000.0), axis=1)
deathIncr_perhundredk_all = deathIncr_all.apply(lambda x: ((x / pop_dict[x.name]) * 100000.0), axis=1)
hospitalizedCurr_all.to_excel(xlsx_path + "/USA_States_COVID-19-CurrentlyHospitalized_" + str(date.today()) + ".xlsx")
positiveIncr_all.to_excel(xlsx_path + "/USA_States_COVID-19-NewPositive_" + str(date.today()) + ".xlsx")
testIncr_all.to_excel(xlsx_path + "/USA_States_COVID-19-NewTested_" + str(date.today()) + ".xlsx")
deathIncr_all.to_excel(xlsx_path + "/USA_States_COVID-19-NewDeath_" + str(date.today()) + ".xlsx")
deathIncr_perhundredk_all.to_excel(xlsx_path + "/USA_States_COVID-19-NewDeathPer100k_" + str(date.today()) + ".xlsx")
positiveIncr_perhundredk_all.to_excel(xlsx_path + "/USA_States_COVID-19-NewPositivePer100k_" + str(date.today()) + ".xlsx")

mobilityChange_retailrec = pd.DataFrame(columns=dates, index = states)
mobilityChange_grocerypharma = pd.DataFrame(columns=dates, index = states)
mobilityChange_parks = pd.DataFrame(columns=dates, index = states)
mobilityChange_transit = pd.DataFrame(columns=dates, index = states)
mobilityChange_workplaces = pd.DataFrame(columns=dates, index = states)
mobilityChange_residential = pd.DataFrame(columns=dates, index = states)
mobilitydfs = [mobilityChange_retailrec, mobilityChange_grocerypharma,
               mobilityChange_parks, mobilityChange_transit, mobilityChange_workplaces,
               mobilityChange_residential]
mobilityCols = ['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']
for ab in statesabbrev:
    state = statedict[ab]
    mobile_state = mobility_csv[mobility_csv["state"] == ab]
    mobile_state.set_index('date', inplace=True)
    for i, mobilitytype in enumerate(mobilityCols):
        mobilitydfs[i].loc[state] = mobile_state[mobilitytype]     
mobilityfilenames = ["retailrec_mobility", "grocerypharma_mobility", "parks_mobility", "transit_mobility", "workplace_mobility", "residential_mobility"]
for i, mob in enumerate(mobilitydfs):
    mob.to_excel(xlsx_path + "/" + mobilityfilenames[i] + ".xlsx")    