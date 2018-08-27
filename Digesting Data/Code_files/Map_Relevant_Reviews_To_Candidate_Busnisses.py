#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:39:28 2017

@author: Rafi Trad

This code file will:
1- Load the businesses vector filters to filter reviews by, and load reviews dataset (attention: huge size!)
2- Compute the number of relevant reviews under each candidate business category
3- If desired, split the reviews by the associated business, and save the CSV partial data (attention: time and disk space -GBs- demanding)
"""

import pandas as pd;
import os;



def loadCandidateBusiness(_parentDataDirectory, _fileName):
    return pd.read_csv(os.path.join(_parentDataDirectory, _fileName),usecols=['business_id']);

#We could use panels here, but since they are deprecated, we will resort to dataframes, a dictionary of these honestly:
parentDataDirectory = "../../../dataset/Candidate_businesses/";

dictCandidateBusinesses = {};
for strippedDataset in os.listdir(parentDataDirectory):
    DsName = os.path.splitext(strippedDataset)[0];
    dictCandidateBusinesses[DsName] = loadCandidateBusiness(parentDataDirectory,DsName);

print "Loaded businesses datasets for:\n";
print list(dictCandidateBusinesses.keys());
print "\n*************************\n";


"""
dfCurrentBusId = dictCandidateBusinesses.get("Business_Automotive");
print "*********\n" + str(dfCurrentBusId.shape);
print type(dfCurrentBusId) , dfCurrentBusId.shape;

"""

#Start allocating relevant reviews:
dfReviewsDataset = pd.read_csv("../../../dataset/review.csv", usecols=["review_id","business_id","stars","text"]);
#Converting the join column to string and strip it from spaces:
dfReviewsDataset["business_id"] = dfReviewsDataset["business_id"].astype(str);
dfReviewsDataset.business_id = [bid.strip() for bid in dfReviewsDataset.business_id];
print str(dfReviewsDataset.shape[0]) + " Reviews has been loaded into memory:";
print dfReviewsDataset.info();
print dfReviewsDataset.head();
print "\n*************************\n";


#Create the empty statistics dataframe
dfReviewsByBusinessStatistics = pd.DataFrame(data=None, columns=["Business","cnt_Reviews"]);


for key in dictCandidateBusinesses:
    dfCurrentBusIdVec = pd.DataFrame();
    dfCurrentJoinedReviews = pd.DataFrame();
    
    dfCurrentBusIdVec = dictCandidateBusinesses.get(key);
    #Converting the join column to string:
    dfCurrentBusIdVec["business_id"] = dfCurrentBusIdVec["business_id"].astype(str);
    dfCurrentBusIdVec.business_id = [bid.strip() for bid in dfCurrentBusIdVec.business_id];
    print "joining with the following vector: \n";
    print dfCurrentBusIdVec.info();
    print dfCurrentBusIdVec.head();
    
    dfCurrentJoinedReviews = pd.merge(dfCurrentBusIdVec, dfReviewsDataset,on="business_id",how="inner");
    
    #Save the statistics file, so that we do not need to run the whole code again and load the whole data:
    tempDict = {"Business":str(key),"cnt_Reviews":int(dfCurrentJoinedReviews.shape[0])};
    dfReviewsByBusinessStatistics.loc[dfReviewsByBusinessStatistics.shape[0]+1] = pd.Series(tempDict);
    
    """ Uncomment to save csv files, CAUTION: requires time and disk space! specify where to save below.
    #Save the result in a file to alleviate the size-effects of reviews next time:
    if dfCurrentJoinedReviews.shape[0] > 0:
        dfCurrentJoinedReviews.to_csv(path_or_buf="../../../dataset/Reviews_By_Business/" \
                                      + str(key) + "_reviews.csv");
        print "\nReviews of " + str(key) + " were Saved successfully.\n";
        print "\n-----------------------------\n";
    else:
        print "\nAn error ocurred while trying to save " + str(key) + "...\n\n";
        print "\n-----------------------------\n";
    """
    
#Store the sorted statistics dataframe
dfReviewsByBusinessStatistics.sort_values(by=["cnt_Reviews"],ascending=False)\
.to_csv(path_or_buf="../../../dataset/Reviews_By_Business/Reviews_By_Business_Statistics.csv");

print "\nReviews's statistics by candidate businesses were Saved successfully.";
