# -*- coding: utf-8 -*-
"""
Spyder Editor

"""


import pandas as pd;
import ast; #import the abstracr syntax trees library, to evaluate the string of multiple categories
#from ggplot import *; #https://github.com/yhat/ggpy for documentation
import os; #For exporting, allowing protability and avoiding ambiguity in file paths - https://stackoverflow.com/questions/22872952/set-file-path-for-to-csv-in-pandas

businessesFilePath = "/media/rafi/Data/College/DKEM/3_WS_17_18/Team Project YELP SA/dataset/business.csv";
dfBusinesses = pd.read_csv(businessesFilePath,
                          usecols=['business_id','categories'],
                          engine='python',
                          encoding='utf_8');

                           
#Deceptively enough, categories is of type unicode, not list, so we need to convert it
#lstCategories = dfBusinesses["categories"].values.tolist();
dictCategories = {};
for categoryRecord in dfBusinesses["categories"]:
    lstCategories = ast.literal_eval(categoryRecord);
    for cat in lstCategories:
        dictCategories[cat] = dictCategories.get(cat,0) + 1;

#Convert the dictionary to a DataFrame
dfCategories = pd.DataFrame.from_dict(dictCategories, orient="index");
dfCategories.columns = ["Frequency"];
dfCategories = dfCategories.sort_values(by="Frequency",ascending=False);
#Add the percentage of occurences of a category in the whole dataset
print "Total number of businesses: " + str(dfBusinesses.shape[0]) + "\n";
dfCategories["Category_Total_pct"] = dfCategories["Frequency"] / dfBusinesses.shape[0];


#List the first 5% of business by number of ocurrence
print dfCategories.head(int(0.05 * dfCategories.shape[0]));
#Picking the ones whose pct is >= 5%
dfCandidateCategories = dfCategories[dfCategories["Category_Total_pct"] >= 0.0500];
print dfCandidateCategories;

"""
    Adopting these categories as candidates, we want to retireve the relevant business entities ids to locate their reviews
    i.e. a subset of the original businesses dataset, according to their categories and frequencies
    More on this method here: https://stackoverflow.com/questions/11350770/pandas-dataframe-select-by-partial-string
    
#print dfBusinesses.loc[dfBusinesses["categories"].str.contains("Restaurants")].head(5);
"""
#Build the filter from our candidate categories:
unicodeFilter = "";


for cat in dfCandidateCategories.index:
    unicodeFilter += "|" + str(cat);
    
    
print str(unicodeFilter) + "\n\n" + str(dfCandidateCategories.shape[0]) \
+ " business categories to be considered:\n\n---------------\n\n"


dfCandidateBusinessesIds = dfBusinesses.loc[dfBusinesses["categories"].str.contains(unicodeFilter)];


print "After filtering, " + str(dfCandidateBusinessesIds.shape[0]) + " out of " \
+ str(dfBusinesses.shape[0]) + " " \
"business entities left ("+ str(100*dfCandidateBusinessesIds.shape[0]/dfBusinesses.shape[0]) +"%).\n";


print dfCandidateBusinessesIds.head();

#Since we have 100% coverage, we will split the businesses ids to filter vectors:
path = "../Output_Datasets";
for businessCat in dfCandidateCategories.index:
    dfBusinesses.loc[dfBusinesses["categories"].str.contains(str(businessCat))] \
    .to_csv(os.path.join(path,"Business_"+str(businessCat)));
    
