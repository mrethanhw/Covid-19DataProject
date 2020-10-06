# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:33:01 2020

@author: MrEth
"""

import pandas as pd
import numpy as np
from datetime import date
import datetime 
#import openpyxl as op
#from datetime import datetime
#Enter desired file paths
#file path for converted data
xlsxfilenamepath = './ConvertedEqual/'
#File Path for basis data(data that is entered manually)
basisfilenamepath = './BasisEqual2/'
#Boolean for if you want new web data to update the basis data
updateBasis = False;
updatePolicyEffect = True
#Original Data
class calc:
    @staticmethod
    def getArray(s):
        if not isinstance(s,str):
            return [np.nan]
        else:
            return s.replace(u'["',u'').replace(u'"]',u'').split('","')
    @staticmethod
    def correctDate(p):
        if p in calc.getArray(dfny.loc[n]['details.Retail.names']):
                return 'all_non-ess_business_start_date'
        elif p in calc.getArray(dfny.loc[n]['details.Food and drink.names']):
            return 'all_non-ess_business_start_date'
        elif p in calc.getArray(dfny.loc[n]['details.Personal care.names']):
            return 'all_non-ess_business_start_date'
        elif p in calc.getArray(dfny.loc[n]['details.Entertainment.names']):
            return 'all_non-ess_business_start_date'
        elif p in calc.getArray(dfny.loc[n]['details.Industries.names']):
            return 'all_non-ess_business_start_date'
        else:
            return 'stay_home_start_date'
    @staticmethod
    def findCategory(p, c):
        plower = p.lower()
        if type(c) != str:
            c=c[0]
        if  'librar' in plower or 'museum' in plower:
            return "Libraries/Museums"
        elif 'skating rink' in plower:
            return 'Indoor Recreation'
        elif 'outdoor venue' in plower or 'zoo' in plower or 'outdoor concert' in plower:
            return 'Entertainment(Outdoor)'
        elif 'arcade' in plower or 'gaming' in plower or 'center' in plower or 'casino' in plower or 'event center' in plower or 'theater' in plower or 'galler' in plower or 'bowl' in plower or 'concert' in plower or 'arena' in plower or 'stadium' in plower or 'venue' in plower or 'club' in plower or 'hall' in plower or 'convention' in plower:
            return "Entertainment(Indoor)"
        elif 'outdoor' in plower or 'camp' in plower or 'tennis' in plower or 'playground' in plower or 'pool' in plower or 'beach' in plower or 'park' in plower or 'golf' in plower or 'court' in plower or 'trail' in plower or 'marina' in plower or 'fishing' in plower or 'campground' in plower and 'Recreation' in c:
            return 'Outdoor Recreation'
        elif 'recreation' in c.lower():
            return "Indoor Recreation"
        elif 'mall' in p:
            return "Retail"
        elif c == 'Libraries/Museums' or 'Entertainment' in  c :
            return "Entertainment(Outdoor)"
        else:
            return c
    #string list parser
    @staticmethod
    def lengthListStr(liststr):
        newlist = []
        for i in range(len(liststr)):
            newlist += liststr[i].replace(u' and ',u',').split(',')
        return len(newlist)
def logdiff(df):
    log = np.log(df.mask(df <= 0))
    logminus7day = log.shift(7, axis = 1)
    return log - logminus7day
            
names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    
df = pd.read_csv('./Web Data/Summary_stats_all_locs_0806.csv', index_col = 'location_name')
names = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
maskcustomers = np.array(["California","Virginia","New Mexico","Connecticut", "Hawaii",  "Maryland", "New Jersey", "New York", "Pennsylvania", "Delaware", "District of Columbia", "Illinois", "Maine", "Massachusetts", "Michigan", "Puerto Rico", "Rhode Island"])
mask1dates = ["6/18/2020", "05/29/2020","05/16/2020", "04/20/2020", "04/20/2020", "04/18/2020", "04/13/2020", "04/17/2020", "04/19/2020", "04/25/2020","05/13/2020", "05/01/2020", "05/01/2020", "05/06/2020", "04/24/2020","05/04/2020", "05/08/2020"]
masksomeworkers = np.array(["Alabama", "Arizona", "Arkansas", "Colorado", "Florida", "Indiana", "Louisiana", 
                            "Minnesota", "Nebraska", "New Hampshire", "North Carolina", 
                            "Oregon", "Virginia", "Washington", "West Virginia", "Wyoming"])
mask3dates = ["04/28/2020", "05/08/2020", "05/11/2020", "05/16/2020", "05/11/2020", "05/01/2020", "05/01/2020", 
              "05/03/2020", "05/03/2020", "05/11/2020", "05/10/2020", "05/07/2020", np.nan, "05/04/2020", "05/04/2020", "05/15/2020"]
maskallworkers = np.array(["Utah", "Alaska", "Georgia", "Kentucky", "Mississippi", "Nevada", "Ohio", "Vermont"])
mask2dates = ["04/30/2020","04/24/2020", "04/27/2020", "05/11/2020", "05/12/2020", "05/09/2020", "04/27/2020", "04/17/2020" ]
maskrecc = np.array(["Oklahoma", "Iowa", "Montana", "California", "Kansas", "Missouri", "North Dakota",  "South Carolina", "South Dakota", "Idaho", "Tennessee", "Texas", "Wisconsin"])
mask4dates = np.full(len(maskrecc), np.nan)
all = np.concatenate([maskcustomers, maskallworkers, masksomeworkers, maskrecc])
alldates = np.concatenate([mask1dates, mask2dates, mask3dates, mask4dates])
allenddates = np.full(len(all), np.nan)
mask_df = pd.DataFrame(data = {'State' : all, 'start_date' : alldates, 'end_date': allenddates})
mask_df = mask_df.set_index('State')
categs = ["Food and Drink", "Personal Care", "Retail", "Libraries/Museums", "Entertainment(Indoor)", "Entertainment(Outdoor)", "Industries", "Houses of Worship", "Outdoor Recreation", "Indoor Recreation"]
all_categs = categs + ['Travel', 'Stay-at-Home Order', 'Education', 'Gatherings', 'Business', 'Non-Essential Business']#new mask handler
mask_df = pd.read_excel("./Web Data/mask_data.xlsx", usecols=['states','all_mandate_start', 'all_mandate_end'], index_col='states')
df = df.loc[df['location_id' ] >= 523].reindex(names)
case_df = pd.read_excel("./Assembled Data/" + 'USA_States_COVID-19-NewPositivePer100k_2020-08-23.xlsx', index_col='States')
rolling3 = ((case_df.rolling(3, axis=1).sum())/3).fillna(0)
rolling7 = ((case_df.rolling(7, axis=1).sum())/7).fillna(0)
#fix data

Policy = []
Category = []
start_cols = [col for col in df.columns if 'start' in col]
end_cols = [col for col in df.columns if 'end' in col]

all_states_policyeffect = pd.DataFrame()
for col in df.columns:
    if '_start_date' in col:
        Policy.append(col.split('_start_date')[0])
dfny = pd.read_csv('./Web Data/nytstates8_14.csv', index_col = 'key', parse_dates=True)
reverseclosed_cols = [col for col in dfny.columns if 'details_closed' in col]
reverse_df = dfny[reverseclosed_cols]
df['all_non-ess_business_end_date'] = pd.to_datetime(dfny['reopen_dates_all.0'])
mins = []
    
#-------------------------------------------------------------------------
    #Converted Data
  
date1 = datetime.datetime(2020, 3, 1)
dates = np.array([])
while date1.date() != date.today() + datetime.timedelta(days=1):
        dates = np.append(dates, date1.strftime("%#m/%#d/%Y"))
        date1 += datetime.timedelta(days=1)
today = date.today().strftime("%y-%m-%d")

#Closed parser

for n in names:
    xlsxfilename = xlsxfilenamepath + n + '_Converted' + '.xlsx'
    basisfilename = basisfilenamepath + n + '_Basis.xlsx'
    Policy=[]
    Category = []
    starts = df.loc[n][start_cols].values
    temp = []
    for start in starts:
        if not pd.isnull(start):
            div = start.split('-')
            if div[1][0] == '0':
                div[1] = div[1][1]
            if div[2][0] == '0':
                div[2] = div[2][1]
            temp.append(div[1] + "/" + div[2] + "/" + div[0])
        else:
            temp.append(np.nan)
    starts = temp
    startdict = dict(zip(start_cols, starts))
    for col in df.columns:
        if '_start_date' in col:
            Policy.append(col.split('_start_date')[0])
            if 'travel' in col:
                Category.append('Travel')
            if 'stay_home' in col:
                Category.append('Stay-at-Home Order')
            if 'educational' in col:
                Category.append('Education')
            if 'any_gathering' in col:
                Category.append('Gatherings')
            if 'any_business' in col:
                Category.append('Business')
            if 'all_non' in col:
                Category.append('Non-Essential Business')
    newpolicy = [calc.getArray(dfny.loc[n]['details.Outdoor and recreation.names']),
                calc.getArray(dfny.loc[n]['details.Retail.names']),
                calc.getArray(dfny.loc[n]['details.Food and drink.names']),
                 calc.getArray(dfny.loc[n]['details.Personal care.names']),
                 calc.getArray(dfny.loc[n]['details.Entertainment.names']),
                 calc.getArray(dfny.loc[n]['details.Industries.names']), 
                 [dfny.loc[n]['details.Houses of worship.category']]]
    newcateg = []
    for o in calc.getArray(dfny.loc[n]['details.Outdoor and recreation.names']):
        if o == o:
            o = o.lower()
            if 'outdoor' in o or 'camp' in o or 'tennis' in o or 'playground' in o or 'pool' in o or 'beach' in o or 'park' in o or 'golf' in o or 'court' in o or 'trail' in o or 'marina' in o or 'fishing' in o or 'campground' in o or 'zoo' in o:
                newcateg = np.append(newcateg, 'Outdoor Recreation')
            else:
                newcateg = np.append(newcateg, "Indoor Recreation")
    for o in calc.getArray(dfny.loc[n]['details.Retail.names']):
        if o == o:    
            newcateg = np.append(newcateg, "Retail")
    for o in calc.getArray(dfny.loc[n]['details.Food and drink.names']):
        if o == o:
            newcateg = np.append(newcateg, "Food and Drink")
    for o in calc.getArray(dfny.loc[n]['details.Personal care.names']):
        if o == o:
            newcateg = np.append(newcateg, "Personal Care")
    for o in calc.getArray(dfny.loc[n]['details.Entertainment.names']):
        if o == o:
            o = o.lower()
            if  'librar' in o or 'museum' in o:
                newcateg = np.append(newcateg, "Libraries/Museums")
            elif 'gaming' in o or 'arcade' in o or 'casino' in o or 'theater' in o or 'bowl' in o or 'concert' in o or 'arena' in o or 'stadium' in o or 'venue' in o or 'club' in o:
                newcateg = np.append(newcateg, "Entertainment(Indoor)")
            else:
                newcateg = np.append(newcateg, "Entertainment(Outdoor)")
    for o in calc.getArray(dfny.loc[n]['details.Industries.names']):
        if o == o:
            newcateg = np.append(newcateg, "Industries")
    for o in calc.getArray(dfny.loc[n]['details.Houses of worship.category']):
        if o == o:
            newcateg = np.append(newcateg, "Houses of Worship")
    #Temporary filler columns
    
    tempcount = 0
    templist = []
    for categ in categs:
        if categ not in newcateg:
            newcateg = np.append(newcateg, categ)
            temp = ["Temp Policy_" + categ]
            templist.append(categ)
            tempcount = tempcount + 1
            newpolicy.append(temp)
        #else:
            #if categ == 'Entertainment(Outdoor)':
                #print(n)
            
    #Append new policies(NYT)
    #nytpolicy = []
    for p in newpolicy:
        if p[0] == p[0]:
            Policy = np.concatenate((Policy, p))
            #nytpolicy = np.concatenate((nytpolicy, p))
    for c in newcateg:
        if c[0] == c[0]:
            Category = np.append(Category, c)
            
    
            
    #for p in nytpolicy:
        #same = calc.correctDate(p)
        #starts = np.append(starts, startdict[same])#df.loc[n][start_cols].loc[same])
    
    
    while len(starts) < len(Policy) - tempcount:    
        starts = np.append(starts, np.nan)
    
    
    ends = df.loc[n][end_cols].values
    temp = []
    for end in ends:
        if type(end) is pd.Timestamp:
            end = end.to_pydatetime().strftime("%Y-%m-%d")
        if end == end:
            div = end.split('-')
            if div[1][0] == '0':
                div[1] = div[1][1]
            if div[2][0] == '0':
                div[2] = div[2][1]
            temp.append(div[1] + "/" + div[2] + "/" + div[0])
        else:
            temp.append(np.nan)
    ends = temp
    newreopendates = [calc.getArray(dfny.loc[n]['details.Outdoor and recreation.reopen_dates']),
                calc.getArray(dfny.loc[n]['details.Retail.reopen_dates']),
                calc.getArray(dfny.loc[n]['details.Food and drink.reopen_dates']),
                 calc.getArray(dfny.loc[n]['details.Personal care.reopen_dates']),
                 calc.getArray(dfny.loc[n]['details.Entertainment.reopen_dates']),
                 calc.getArray(dfny.loc[n]['details.Industries.reopen_dates']),
                 [calc.getArray(dfny.loc[n]['details.Houses of worship.reopen_dates'])[-1]]]
    for r in newreopendates:
        if r == r:
            if r[0] == r[0]:
                ends = np.append(ends, r)
    while len(ends) < len(Policy):
        starts = np.append(starts, df.loc[n]['stay_home_start_date'])
        if starts[-1] == 'nan':
            starts[-1] = df.loc[n]['all_non-ess_business_start_date']
        ends = np.append(ends, np.nan)
    Category = np.append(Category, "Face Coverings")
    Policy = np.append(Policy, "Customer Mandate")
    #if n in masksomeworkers:
     #   Category = np.append(Category, "Face Coverings")
      #  Policy = np.append(Policy, "Some Employee Mandate") 
    #if n in maskallworkers:
     #   Category = np.append(Category, "Face Coverings")
      #  Policy = np.append(Policy, "All Employee Mandate")
    #if n in maskrecc:
     #   Category = np.append(Category, "Face Coverings")
      #  Policy = np.append(Policy, "Recommendation Only")
    starts = np.append(starts, mask_df.loc[n]['all_mandate_start'])
    ends = np.append(ends, mask_df.loc[n]['all_mandate_end'])
    #statesdf = pd.DataFrame(data = {'Category': Category, 'Policy/Closing':Policy, 'start_date': starts,'end_dates':ends}, columns=['add_start'])
    statesdf = pd.DataFrame(data = {'Category': Category, 'Policy/Closing':Policy, 'start_date': starts,'end_dates':ends}, columns=['Category', 'Policy/Closing', 'start_date', 'end_dates','add_start', 'add_end'])
    statesdf = statesdf.set_index('Policy/Closing')
    statesdf.loc['stay_home', 'end_dates'] = dfny.loc[n]['stay_at_home.expires']
    #statesdf = statesdf.set_index('Category')
    #-----format file into datetime 'date'
    base_csv = pd.read_excel("./BasisEqual/" + n + '_Basis' + '.xlsx', index_col = 'Policy/Closing', parse_dates = ['start_date','end_dates'])
    base_csv['start_date'] = base_csv['start_date'].dt.date
    base_csv['end_dates'] = base_csv['end_dates'].dt.date
    #statesdf.update(base_csv['start_date'])
    #manualInput = np.array(base_csv['start_date'].values[: -1])
    #while len(manualInput) < len(Policy) - 1:    
        #manualInput = np.append(manualInput, pd.NaT)
    #manualInput = np.append(manualInput, pd.to_datetime(mask_df.loc[n]['start_date']))
    #statesdf['start_date'] = manualInput
    #statesdf['end_dates'] = pd.to_datetime(statesdf['end_dates'], infer_datetime_format = True).dt.date
    #statesdf['start_date'] = pd.to_datetime(statesdf['start_date'], infer_datetime_format = True).dt.date
    multirows = statesdf.reset_index()[(statesdf.reset_index()["Policy/Closing"].str.contains(","))]
    for row in multirows.itertuples():
        policylist = row._1.replace(u' and ',u',').split(',')
        statesdf = statesdf.rename(index = {row._1: policylist[0]})
        #print(n)
        statesdf.loc[policylist[0],'Category'] = calc.findCategory(policylist[0], statesdf.loc[policylist[0]]['Category'])
        #Insert policies in statesdf
        for i, pol in enumerate(policylist):
            if pol != policylist[0] and 'etc' not in pol and pol:
                cat = calc.findCategory(pol, row.Category)
                line = pd.DataFrame({"Policy/Closing": pol.strip(), "Category": cat,"start_date": row.start_date, "end_dates": row.end_dates}, index=[row.Index])
                statesdf = pd.concat([statesdf.reset_index().iloc[:row.Index + 1 + i], line, statesdf.reset_index().iloc[row.Index + 1 + i:]], sort=False).set_index("Policy/Closing")#.reset_index(drop=True)
    
    
    ma = reverse_df.loc[n]
    closedpolicies = np.concatenate([calc.getArray(ma['details_closed.Food and drink.names']), calc.getArray(ma['details_closed.Houses of worship.category']),calc.getArray(ma['details_closed.Industries.names']),calc.getArray(ma['details_closed.Personal care.names']),calc.getArray(ma['details_closed.Retail.names']),calc.getArray(ma['details_closed.Outdoor and recreation.names']),calc.getArray(ma['details_closed.Entertainment.names'])]).tolist()
    closedpolicies[:] = [x for x in closedpolicies if x != 'nan' and x==x]
    closeddates = np.concatenate([calc.getArray(ma['details_closed.Food and drink.reopen_dates']), calc.getArray(ma['details_closed.Houses of worship.category']),calc.getArray(ma['details_closed.Industries.reopen_dates']),calc.getArray(ma['details_closed.Personal care.reopen_dates']),calc.getArray(ma['details_closed.Retail.reopen_dates']),calc.getArray(ma['details_closed.Outdoor and recreation.reopen_dates']),calc.getArray(ma['details_closed.Entertainment.reopen_dates'])]).tolist()
    closeddates[:] = [x for x in closeddates if x != 'nan' and x==x]
    closedcategories = []
    
    tempclosed = []
    tempdates = []
    tempcategories = []
    for p, d in zip(closedpolicies, closeddates):
        addp = p.replace(u' and ', u',').split(',')
        num = len(addp)
        addd = [d] * num
        tempclosed = tempclosed + addp
        tempclosed[:] = [x.strip() for x in tempclosed]
        tempdates = tempdates + addd
    closedpolicies = tempclosed
    closeddates = tempdates
    if ma['details_closed.Food and drink.names']==ma['details_closed.Food and drink.names']:
        closedcategories = closedcategories + ["Food and Drink"] * calc.lengthListStr(calc.getArray(ma['details_closed.Food and drink.names']))
    if ma['details_closed.Houses of worship.names']==ma['details_closed.Houses of worship.names']:
        closedcategories = closedcategories + ["Houses of worship"] * calc.lengthListStr(calc.getArray(ma['details_closed.Houses of worship.names']))
    if ma['details_closed.Industries.names']==ma['details_closed.Industries.names']:
        closedcategories = closedcategories + ["Industries"] * calc.lengthListStr(calc.getArray(ma['Industries']))
    if ma['details_closed.Personal care.names']==ma['details_closed.Personal care.names']:
        closedcategories = closedcategories + ["Personal care"] * calc.lengthListStr(calc.getArray(ma['details_closed.Personal care.names']))
    if ma['details_closed.Retail.names']==ma['details_closed.Retail.names']:
        closedcategories = closedcategories + ["Retail"] * calc.lengthListStr(calc.getArray(ma['details_closed.Retail.names']))
    if ma['details_closed.Outdoor and recreation.names']==ma['details_closed.Outdoor and recreation.names']:
        closedcategories = closedcategories + ["Outdoor and recreation"] * calc.lengthListStr(calc.getArray(ma['details_closed.Outdoor and recreation.names']))
    if ma['details_closed.Entertainment.names']==ma['details_closed.Entertainment.names']:
        closedcategories = closedcategories + ["Entertainment"] * calc.lengthListStr(calc.getArray(ma['details_closed.Entertainment.names']))
        
    for i in range(len(closedpolicies)):
        tempcategories = tempcategories + [calc.findCategory(closedpolicies[i], closedcategories[i])]        
    closedcategories = tempcategories
    if (len(closedpolicies) != len(closeddates) != len(closedcategories)):
        print("Closed not equal")
    #Add policies found in base but not converted, possibly due to NYT not including policies no longer enacted

    newclosed = []
    newcloseddates = []
    for i in range(len(closedpolicies)):
        thisdate = pd.to_datetime(closeddates).date[i].month
        if closedpolicies[i].lower() in statesdf.index.values:
            if not(thisdate == 3 or thisdate == 4 or thisdate == 5):
                statesdf.loc[closedpolicies[i].lower()]['add_start'] = closeddates[i]
                #print("Closed event found in data in " + n + ". Policy: " + closedpolicies[i])
            else:
                statesdf.loc[closedpolicies[i].lower()]['start_date'] = closeddates[i]
                #print("Initial closed event found in data in " + n + ". Policy: " + closedpolicies[i])
        elif closedpolicies[i] in statesdf.index.values:
            if not(thisdate == 3 or thisdate == 4 or thisdate == 5):
                statesdf.loc[closedpolicies[i],'add_start'] = closeddates[i]
                #print("Closed event found in data in " + n + ". Policy: " + closedpolicies[i] )
            else:
                statesdf.loc[closedpolicies[i], 'start_date'] = closeddates[i]
                #print("Initial closed event found in data in " + n + ". Policy: " + closedpolicies[i])
        else:
            if thisdate == 3 or thisdate == 4 or thisdate == 5:
                s = pd.Series({"Category": closedcategories[i], 
                               "start_date": closeddates[i], "end_dates" : np.nan, "add_start": np.nan, "add_end": np.nan})
                s.name = closedpolicies[i]
                statesdf = statesdf.append(s)
                #print("Closed event found in data in " + n + ". Requires initial start date. Policy: " + closedpolicies[i])
            else:
                s = pd.Series({"Category": closedcategories[i], "start_date": np.nan, "end_dates" : np.nan, "add_start" : closeddates[i], "add_end": np.nan})
                s.name = closedpolicies[i]
                statesdf = statesdf.append(s)
                #print("" + n + " has reclosed an institution but initial reopening not specified. Policy: " + closedpolicies[i])
    statesdf.update(base_csv['start_date'].reset_index().drop_duplicates(subset='Policy/Closing').set_index('Policy/Closing'))
    updateend = base_csv[(base_csv['add_start'] == base_csv['add_start'])]['end_dates']
    statesdf.update(updateend)
    base_csv.drop(np.nan, inplace=True, errors='ignore')
    omitted = np.setdiff1d(base_csv.index.values,statesdf.index.values)
    for om in omitted:
        statesdf = statesdf.append(base_csv.loc[om])
        #print('Added Data in ' + n + "Data is: " + om)
    statesdf['end_dates'] = pd.to_datetime(statesdf['end_dates'], errors='coerce', infer_datetime_format = True).dt.date
    statesdf['start_date'] = pd.to_datetime(statesdf['start_date'], errors='coerce', infer_datetime_format = True).dt.date
    statesdf['add_start'] = pd.to_datetime(statesdf['add_start'], errors='coerce', infer_datetime_format=True).dt.date
    statesdf['add_end'] = pd.to_datetime(statesdf['add_end'], errors='coerce', infer_datetime_format=True).dt.date
    #Drop temporary columns (but not empty because may fill out in basis)
    basisdf = statesdf.reset_index()[~(statesdf.reset_index()["Policy/Closing"].str.contains("Temp"))].set_index("Policy/Closing")
    #Drop empty columns
    #Update statesdf by dropping unnecessary temp rows
    droplabels = []
    for c in templist:
        if c in basisdf.Category.values:
            droplabels.append("Temp Policy_" + c)
    statesdf.drop(labels = droplabels, inplace=True)
            #statesdf.drop(statesdf[statesdf['start_date'] != statesdf['start_date']].index)
    
    testdf = statesdf.fillna("None").reset_index()
    #Tests
    #if len(testdf[testdf["end_dates"] == "None"][testdf["start_date"] != "None"]) >= 1:
        #print("Error. Add Start Dates to " + n)
    if len(basisdf[basisdf['Category'].isin(testdf[testdf["Policy/Closing"].str.contains("Temp")]['Category'].values)]) >= 1:
        print("Error. Extra temporary policy in " + n + ". Adjust code.")
    if len(statesdf[(statesdf['end_dates'] != statesdf['end_dates']) & (statesdf['add_start'] == statesdf['add_start'])]) > 0:
        print("Error. Add initial reopening date before reclosing.")
    if len(statesdf[(statesdf['start_date'] != statesdf['start_date']) & (statesdf['end_dates'] == statesdf['end_dates'])]) > 0:
        print("Add start dates in " + n + " for policies: ")
        for i, policyerr in enumerate(statesdf[((statesdf['start_date'] != statesdf['start_date']) & (statesdf['end_dates'] == statesdf['end_dates']))].index.values):
            print(policyerr, end=', ')
        #if len(testdf[testdf["add_start"] != "None"]) >= 1:
        #print ("Error. " + n + "closed down again. Check for correctness")
    statesdf = statesdf.reset_index()[~(statesdf.reset_index()["Category"].str.contains("Face Coverings")) | statesdf.reset_index()["Policy/Closing"].str.contains("Customer Mandate")].set_index("Policy/Closing")
    statesdf.index = statesdf.index + statesdf.groupby(level=0).cumcount().astype(str).replace('0','')
    statesdf.index.name = 'Policy/Closing'
    
    #Add Remaining temporary filler columns
    for categ in categs:
        if categ not in statesdf['Category'].tolist():
            temp_startdate = df.loc[n]['all_non-ess_business_start_date'] if df.loc[n]['stay_home_start_date'] != df.loc[n]['stay_home_start_date'] else df.loc[n]['stay_home_start_date']
            temp_enddate = df.loc[n]['all_non-ess_business_end_date'] if df.loc[n]['stay_home_end_date'] != df.loc[n]['stay_home_end_date'] else df.loc[n]['stay_home_end_date']
            statesdf.loc['Temp Policy_' + categ] = [categ, temp_startdate, temp_enddate, np.nan, np.nan]
    #----------------------------------------------------------------
    #Converted
    writer = pd.ExcelWriter(xlsxfilename , engine='xlsxwriter')
    #filename = './States/' + n + '_' + today + '.xlsx'
    #statesdf2 = pd.read_excel(filename, index_col = 'Policy/Closing')
    #top = statesdf.reset_index().transpose()
    columns = statesdf.reset_index()['Policy/Closing'].values
    df2 = pd.DataFrame({'Dates': dates})
    df2 = df2.set_index('Dates')
    #set the binary truth value for each policy in df2
    count = 0
    listPolicies = list()
    for c in set(columns):
        d=c
        vals = np.full(len(dates), 0)
        start = pd.Series(pd.to_datetime(statesdf.loc[c]['start_date']))
        end = pd.Series(pd.to_datetime(statesdf.loc[c]['end_dates']))
        a_start = pd.Series(pd.to_datetime(statesdf.loc[c]['add_start']))
        a_end = pd.Series(pd.to_datetime(statesdf.loc[c]['add_end']))
        date1 = datetime.datetime(2020, 3, 1)
        #check for duplicate policies and indicate on dataframe if so
        count = 0
        #if len(start) == 1:
        #    count = 0
        #elif len(start) > 1:
            #c = c + '_' + str(count)
        for ind in range(len(start)):
            if len(start) == 1:
                count = 0
            if len(start) >  1:
                c = d + str(count)
            listPolicies.append(c)    
            st = start.iloc[count]
            en = end.iloc[count]
            a_st = a_start.iloc[count]
            a_en = a_end.iloc[count]
            if len(start) > 1:
                count += 1
            #if policy end already occured
            if st == st and en == en:
                stind = (st - date1).days
                eind = (en - date1).days
                vals[stind: (eind + 1)] = 1
            elif st==st:
                stind = (st - date1).days
                vals[stind:] = 1
            elif en == en:
                if 'all_non-ess' not in c:
                    print("No Start Date: " + c + "- In state " + n)
            # if additional closings occur (add_start)
            if a_st == a_st and a_en == a_en:
                stind = (a_st - date1).days
                eind = (a_en - date1).days
                vals[stind: (eind + 1)] = 1
            elif a_st==a_st:
                stind = (a_st - date1).days
                vals[stind:] = 1
            df2.insert(len(df2.columns), ('%s' % c), vals, allow_duplicates = True) 
    #rearrange to group by category: Travel, Stay-at-home, Education, Gatherings, Business, Non-Essential Business, Outdoor Recreation, Indoor Recreation, Retail, Food and Drink, Personal Care, Libraries/Museums, Entertainment(Indoor), Entertainment(Outdoor), Houses of Worship, Industries, Face Coverings
    
    #select weight combination number
    weightsel = 4
    if weightsel == 0:
        colweightname = 'Weights'
    else:
        colweightname = 'Weights.' + str(weightsel)
    weights = pd.read_csv('weights.csv', index_col = 0)
    if weightsel > (len(weights.iloc[0]) - 1):
        print('Weight excel exceeds max. Input new weight select.')
        #return
    weights = weights[colweightname].to_dict()
    statesdfcopy = statesdf.copy()
    stateweights = list()
    pcountdict = dict(zip(statesdf['Category'].values, np.zeros(len(statesdf.index.values))))
    #Count of policies within each category(including within cells) i.e. business1, business2 counts as two policies even if they are contained in a single cell
    for (c, p) in zip(statesdf['Category'].values, statesdf.index.values):
        numincell = len(p.replace(u' and ',u',').split(','))
        pcountdict[c] = pcountdict[c] + numincell
    for (c, p) in zip(statesdf['Category'].values, statesdf.index.values):
        numincell = len(p.replace(u' and ',u',').split(','))
        stateweights.append(round(weights[c]/pcountdict[c] * numincell, 10))
    statesdfcopy['Weights'] = stateweights
    topdf = statesdfcopy.reset_index().transpose()
    topdf[len(topdf.columns)] = ['LockdownEffectiveness', np.nan, np.nan, np.nan, np.nan, np.nan,round(topdf.loc['Weights'].sum(), 2)]
    #topdfsorted = topdf.sort_values(by='Category', axis=1)
    #topdf = topdf[columns]
    topdf.to_excel(writer,sheet_name='Converted Data', header = None)
    a = topdf.loc['Policy/Closing'].tolist()
    a.remove('LockdownEffectiveness')
    df2 = df2[a]
    df2.loc['Weights'] = stateweights
    le = list()
    #create excel file for the "effectiveness" of various policies
    end = 200
    #datestart = 71
    datestart = 0
    datelength = min(len(dates), len(rolling3.columns), end) - 1
    #lagtime = 19
    lagtime = 0
    if updatePolicyEffect:
        categ_effect = pd.DataFrame(columns=all_categs, index=dates[:datelength])
        for cat in all_categs:
            df3 = df2.copy()
            df3.columns = topdf.loc['Category'].tolist()[:-1]
            if isinstance(df3[cat], pd.DataFrame):
                categ_effect[cat] = df3[cat].sum(axis=1).drop('Weights').iloc[:datelength]
            elif isinstance(df3[cat], pd.Series):
                categ_effect[cat] = df3[cat].iloc[:datelength]
            else:
                print("UH....")
        
        categ_effect = categ_effect.apply(lambda x: x * 0 if weights[x.name]==0 else (x / pcountdict[x.name]) * weights[x.name])
        #Lag the lockdown by 'lagtime' days
        categ_effect = categ_effect.shift(lagtime).fillna(0)
        categ_effect = categ_effect.iloc[datestart:datelength]
        categ_effect['Infection'] = logdiff(rolling3).loc[n].values[datestart: datelength]
        categ_effect = categ_effect.fillna(0)
        categ_effect = categ_effect[~(categ_effect['Infection'] > 10)]
        categ_effect = categ_effect[~(categ_effect['Infection'] < -10)]
        weighted_categs = categ_effect.to_excel('./PolicyEffectiveness/' + str(lagtime) + "-DayLag_" + n + 'Policy.xlsx')
        categ_effect = categ_effect.reset_index()
        categ_effect.index = [n] * len(categ_effect)
        all_states_policyeffect = pd.concat([all_states_policyeffect, categ_effect])
        
    for d in dates:
        #calculation for lockdown effectiveness
        le.append(sum(df2.loc[d] * df2.loc['Weights']) / (round(topdf.loc['Weights'].sum(), 2)) * 2)
    df2.drop(df2.tail(1).index, inplace = True)
    df2['LockdownEffectiveness'] = le
    df2 = df2[topdf.loc['Policy/Closing'].values]
    #Check earliest reopen date
    #noness = pd.DataFrame(df['all_non-ess_business_end_date'])
    #noness['dates'] = dfny['reopen_dates_all.0']
    #minimum = statesdf.groupby(statesdf.end_dates, as_index = False).min().end_dates.iloc[0]
    #mins.append(minimum)
    #Convert dataframe to excel document
    df2.to_excel(writer,sheet_name='Converted Data', startrow = (len(topdf) + 1), header = None)            
    statesdf.to_excel(writer, sheet_name = 'Original Data')
    writer.save()
    
    if updateBasis:
        basisdf.to_excel(basisfilename)
    
all_states_policyeffect.to_excel("./Assembled Data/all_states_policyeffect.xlsx")

