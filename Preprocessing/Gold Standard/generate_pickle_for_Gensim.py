
""" 
To detect multiword phrases in sentences within a dataset,
we are going to use gensim's model.phrases class https://radimrehurek.com/gensim/models/phrases.html
to do that first we make have to make an object of the model.phrases class and utilize the method add_vocab
to add all the tokenized sentences in it! and we will store the object with in a pickle file so that when we
want to detect multiwords in sentences within the dataset under consideration, we will read pickle file 
and pass our sentences through the model.phrases object (we just read from pickle file) and it will put 
_ be the multiwords, making new york as new_york (considering if new_york is present frequently inside the dataset)

Developer: Ali Hashaam
Updated: 21 January 2018 

"""

import pandas as pd
import numpy as np
import re, time, string
import nltk
import pickle
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords


# In[3]:


bigram = Phrases()


# In[2]:


def clean_and_format(sentence):
    """tokenize and remove stopwords and then add the sentence in model.phrase object"""
    sentence = unicode(sentence, "utf-8")
    sentence = sentence.translate(string.punctuation)
    tokenize = nltk.word_tokenize(sentence)
    stopWords = set(nltk.corpus.stopwords.words('english'))
    sentence = [x for x in tokenize if x not in stopWords]
    bigram.add_vocab([sentence])
    return sentence


# In[3]:


def pre_process_data(df_restaurants_reviews):
	"""
	Apply pre-processing steps of removing null values and replacein some characters
	"""
	if df_restaurants_reviews.isnull().values.any():
		rows_with_nan = len(df_restaurants_reviews[df_restaurants_reviews.isnull().values.sum(axis=1)>=1])
		print "Number of rows with Null values: ", rows_with_nan
		print "Removing the rows with Null values..."
		df_restaurants_reviews.dropna(axis=0, how='any', inplace=True)
	regex_slash_newlines = re.compile(r'[\n\r\\]+') 
	regex_doublequotes = re.compile(r'\"+')
	df_restaurants_reviews["text"] = df_restaurants_reviews["text"].str.replace(regex_slash_newlines,' ').replace(regex_doublequotes,"'")
	print "cleaning starts"
	df_restaurants_reviews["text"] = df_restaurants_reviews["text"].apply(clean_and_format)
	print "cleaning ends"
	return df_restaurants_reviews


# In[4]:


def divide_and_conquer(TextFileReader):
    """Divide the bigger file in chunks"""
    df_lst = []
    for chunk in TextFileReader:
        start_loadingData1 = time.time()
        df_restaurants_reviews = pd.DataFrame(chunk)
        start_end_index = str(df_restaurants_reviews.first_valid_index())+" - "+str(df_restaurants_reviews.last_valid_index())
        df_restaurants_reviews = pre_process_data(df_restaurants_reviews)
        end_loadingData1 = time.time()
        print("indexes "+start_end_index+" processed' in: " + str(end_loadingData1 - start_loadingData1) + " second(s)") 
    return df_lst


# In[4]:
mainpath = "Reviews_By_Business"
TextFileReader = pd.read_csv(mainpath+"/Business_Bars_reviews.csv", chunksize=100000, index_col=0)
start_loadingData = time.time()
df_list = divide_and_conquer(TextFileReader)
end_loadingData = time.time()
print("Time to Process full data: " + str(end_loadingData - start_loadingData) + " second(s)")


# In[ ]:


with open("gensim_mulitphrases_bars.txt", "wb") as fp:   #Pickling
    pickle.dump(bigram, fp)

