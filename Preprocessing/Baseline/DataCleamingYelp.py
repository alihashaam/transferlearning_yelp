# # Data Cleaning: Reviews files

# This .py file deals with cleaning of yelp business reviews. Business reviews are extracted from reviews.json file by mapping according to the business ids of respective business in business.json.

# pass input 'divide' to process the file in chunks or 'direct' to process as whole.

import re, time, string, sys, itertools, unidecode, icu, os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import nltk, shutil
from nltk.corpus import sentiwordnet as swn; #http://www.nltk.org/howto/sentiwordnet.html
from pprint import pprint;
from nltk.stem.wordnet import WordNetLemmatizer;
from nltk.corpus import wordnet;
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from py2casefold import casefold
from nltk.tokenize import word_tokenize


Business = 'Restaurants' 
stemmer = SnowballStemmer("english")
# the path to the main csv file that needs to be processed
mainpath = 'Reviews_By_Business/'
# set directory where to store the chunks of processed data
split_files_path = 'Business_'+Business+'_Baseline/'
if not os.path.exists(split_files_path):
    os.makedirs(split_files_path)
#define regular expressions we will need
regex_slash_newlines = re.compile(r'[\n\r\\]+') 
regex_doublequotes = re.compile(r'\"+')
stop = stopwords.words('english')

def text_preprocess(review):
    """
    Takes in a string of text, then performs the following:
    1. Casefold the string
    2. Remove all stopwords
    3. stem the words
    4. Returns a list of the cleaned text
    """
    casefolded = casefold(unicode(review,'utf-8'))
    tokenize = word_tokenize(casefolded)
    no_stopwards = [item for item in tokenize if item not in stop] 
    stemmed = [stemmer.stem(y) for y in no_stopwards]
    return stemmed

def data_clean(df_business_reviews):
    """
    Takes in a csv file and do following data cleaning steps on that.
    1. Remove if any duplicates are present.
    2. Remove the records with any null values.
    3. Remove all punctuation.
    """
    # Remove duplicates:
    len_df = len(df_business_reviews)
    df_business_reviews.drop_duplicates(keep='first', inplace=True)
    # Check for null values in data
    if df_business_reviews.isnull().values.any():
        rows_with_nan = len(df_business_reviews[df_business_reviews.isnull().values.sum(axis=1)>=1])
        df_business_reviews.dropna(axis=0, how='any', inplace=True)
    # Remove characters like \ and \n and replace " with ' in reviews
    df_business_reviews["text"] = df_business_reviews["text"].str.replace(regex_slash_newlines,' ').replace(regex_doublequotes,"'")
    # Remove punctuation - Can be done with .translate as well, but it's dangerous, as the word "it's" is now "its", 
    # and PoS tagger may fail to tag it correctly as per experiment! for BL, it's OK, but for advanced methods, it's seedy;
    # It's better to expand the English abbreviations before removing all punctuation blindly.
    df_business_reviews["text"] = df_business_reviews["text"].str.replace('[{}]'.format(string.punctuation), '')
    return df_business_reviews

def clean_process_data(df_business_reviews):
	"""Convert a yelp dataset file from json to csv."""
	# So null values removed and no empty strings found.
	start_processingData = time.time()
	# the call of the pre-processing step
	df_business_reviews = data_clean(df_business_reviews)
	df_business_reviews['text'] = df_business_reviews['text'].apply(text_preprocess)
	end_processingData = time.time()
	print("Time to clean and text-process the data: " + str(end_processingData - start_processingData) + " second(s)")
	return df_business_reviews

def divide_and_conquer(TextFileReader):
	"""Divide the bigger file in chunks"""
	for chunk in TextFileReader:
	    df = pd.DataFrame(chunk)
	    df = clean_process_data(df)
	    df.to_csv(split_files_path+'Business_'+Business+'_reviews_'+str(df.first_valid_index())+'_'+str(df.last_valid_index())+'.csv')

def merge_and_blend():
	"""Merge the chunks into one bigger file"""
	main_df = pd.DataFrame(columns=['business_id', 'review_id', 'text', 'stars'])
	for filename in os.listdir(split_files_path):
	    df = pd.read_csv(split_files_path + filename, index_col=0)
	    main_df = pd.concat([main_df, df], axis = 0)
	    print filename +' is done.'

	main_df.to_csv('Preprocessed_'+Business+'_Reviews_Baseline.csv')
	print Business+' Data Cleaned for Baseline.csv created.'

def do_all():
	"""
	preprocess all those chunks with clean_process_data()
	and concatenate all the preprocessed csv(s) with merge_and_blend().
	"""
	# Read csv file in chunks
	TextFileReader = pd.read_csv(mainpath+'Business_'+Business+'_reviews.csv', chunksize=10000, index_col=0)
	divide_and_conquer(TextFileReader)
	merge_and_blend()
	shutil.rmtree(split_files_path)
if __name__ == '__main__':
    do_all()



