#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Munging Yelp's domains as a preface in order to feed them in to the PAD model.
The PAD model we will use is found here: https://github.com/rpryzant/proxy-a-distance
As per the documentation, we will need to convert the domains to sentences, and find the shared vocabulary among them,
then we can use the PAD model.
-------------------------------
Latest Update:  20180317
Updated By:     Rafi.
"""
import numpy as np
import pandas as pd
#SentiWordNet ontology for computing SO scores
from nltk.tokenize import sent_tokenize
import re, icu;
from string import punctuation;
import os.path;
import random;

# The path to the main output folder, in which the text files are placed, and checks are made to avoid duplicated work:
directory = "./RestaurantsToOthers_Data";


regex_slash_newlines = re.compile(r'[\n\r\\]+');   	
regex_slash_tabs = re.compile(r'\t+');
regex_doublequotes = re.compile(r'\"+');

"""
The py2casefold in not tested enough, slow, and I couldn't install it in anaconda. This lambda function is a replacement.
The function returns a string, not a unicode, because we will use the .translate fast function to remove punct. of strings.
Source: https://stackoverflow.com/a/32838944/3429115
"""
CharsSet = "ascii"; # The Character set to be used as the default one when interpreting texts
casefold = lambda u: unicode(icu.UnicodeString(u).foldCase()).encode(CharsSet,"ignore");


def iter_sample_fast(iterable, samplesize):
    """
    Fast memory-efficient sampling method for pretty large iterables.
    Adopted from: https://stackoverflow.com/a/12583436/3429115
    
    > Parameters:
        * iterable: iterable object | The collection we want to sample from
        
        * samplesize: int           | How many samples to generate
        
    > Returns:
        List of samples, hoewver, since sampling is made without replacement (most probably), they aren't IID; but
        it's not that problem to our case I guess, especially that we do not want to sample the same item twice...
    """
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in xrange(samplesize):
            results.append(iterator.next())
    except StopIteratlion:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results

def elicitDomainSentences(_Filename, _Amount = 0):
    """
    Convert a domain csv resource to a list of sentences, casefolded and stripped from common nuisances. If a sample
    is required, then random sampling is applied after generating the full population.
    
    > Parameters:
        * _Filename: text           | Name of the file within the common dataset directory (datasets)
        
        * _Amount: int              | Sampling size, i.e. how many sentences to elicit? serves for: balancing the data, 
                                    | and for efficiency purposes. if = 0 => no sampling, the full population is returned.
    
    > Returns:
        The list of (_Amount) sentences of the domain in a list. Sentences that contain < 3 words aren't considered.
    """
    df_Reviews = pd.read_csv(filepath_or_buffer="../../dataset/Reviews_By_Business/"+_Filename+".csv",
                              low_memory=False, usecols = ["text"]);
    print "\nData for {} are loaded in order to process and elicite sentences...:".format(_Filename);
                             
    # Drop rows if text is NaN
    df_Reviews.dropna(inplace=True);    
    
    casefolded = np.array(df_Reviews["text"].apply(casefold));

    SentencesReviewsDomain = [];
    counter = 0;
    total = len(casefolded);
    
    for review in casefolded:        
        if (counter % 10000 == 0):
            print "{}% of reviews processed..".format(round(100.0 * counter / total,2));
        # Replace newlines and double quotes  
        review = re.sub(regex_slash_newlines, " ", review);        
        review = re.sub(regex_slash_tabs, " ", review);
        review = re.sub(regex_doublequotes, '"', review);
        #Split sentences:
        currentSentences = sent_tokenize(review);
        #remove punctuation:
        nonPunctuatedSentences = [];
        for s in currentSentences:
            if (len(s.split()) >= 3): # Only consider the sentence if it has at least 3 words.
                nonPunctuatedSentences.append(s.translate(None, punctuation)); # Remove punctuation
                
        if (len(nonPunctuatedSentences) > 0):
            SentencesReviewsDomain.extend(nonPunctuatedSentences);
        counter += 1;
    print "100% of reviews processed.";
        
    if (type(_Amount) == int and _Amount > 0 and _Amount < len(df_Reviews)):
        """ Apply the random sampling if proper and required:
         The population has been generated, and now we want to select _Amount i.i.d samples;
         To choose i.i.d samples, sampling WITH replacement must be carried out.
         Choosing a specific _Amount from all domains guarantees the balance of the U dataset for PAD,
         And enhances the efficiency of the PAD SVM of course: two birds with one stone. """
        # This np.random.choice causes memory problems with large domains, such as restaurants |-_-|..
        # Additionally, it will sometimes sample the same item more than once!
        #return np.random.choice(SentencesReviewsDomain, _Amount).tolist();
        # We will use a more efficient way of sampling without sampling the same item twice:
        return iter_sample_fast(SentencesReviewsDomain, _Amount);
    else:
        # No sampling, return the mere list: 
        return SentencesReviewsDomain;
    
def findCommonVocabularyFromSentences(_domain1, _domain2):
    """
    Find the intersection of vocabulary between the two domains, expressed as lists of sentences, so it is best
    called after `elicitDomainSentences` routine.
    
    > Parameters:
        * _domain1: list      | List of sentences belonging to the first domain;
        
        * _domain2: list      | List of sentences belonging to the second domain.
    
    > Returns:
        The set of common vocabulary between the two domains.
    """
    if ((len(_domain1)==0) or (len (_domain2)==0)):
        print "ATTENTION: One or more empty domain(s) passed to findCommonVocabulary.\n";
        return None;
    
    domain1Vocab = [];
    for s in _domain1:     
        domain1Vocab.extend(s.split());
        
    domain2Vocab = [];
    for s in _domain2:
        domain2Vocab.extend(s.split());
    
    return set.intersection(set(domain1Vocab), set(domain2Vocab));

def saveList2txt(_List, _filePath, IsVocabulary = False):
    """
    Saves a list to a text file, each entry on a new line.
    https://stackoverflow.com/a/13434105/3429115
    
    > Parameters:
      * _List: list             | The list to be saved
      
      * _filePath: string       | The full path, including file name and extension
      
      * IsVocabulary: boolean   | True if we are saving the vocabulary, so that we prepend the special values
    
    > Returns:
        0 if the execution goes well, -1 if the file already exists.
    """
    if (os.path.isfile(_filePath)):
        print "ATTENTION: text file already exists, exiting.";
        return -1;
    
    outfile = open(_filePath, "w");
    if (IsVocabulary):
        outfile.write("<unk>\n<s>\n</s>\n");
    print >> outfile, "\n".join(s.strip() for s in _List);
    outfile.close();
    return 0;

def loadSentencesFromtxt(_Filepath):
    """
    Load sentences list for a specific domain from disk.
    
    > Parameters:
        * _Filepath: the path to the text file, in which each sentence is expected to be on one line. No checks are done
        to ensure the file exists, so be sure it does, or the function will vomit an exception, I guess.
    """
    dataSentences = [];
    for line in open(_Filepath):
        dataSentences.append(line.strip());
    
    return dataSentences;

def ProcessDomains(_Filename1, _Filename2, _NumberOfSentences):
    """
    The main controller; Processes domains into sentences, finds their common vocabulary, and saves the results. Work is
    carried out in a structured way to avoid redundant tasks and unintentional outputs overwriting.
    
    > Parameters:
        * _Filename1 : string       | First domain's name in the common dataset directory -without the extension
        
        * _Filename2 : string       | Second domain's name in the common dataset directory -without the extension
        
        * _NumberOfSentences : int  | The number of sentences to include from both of the domains. 0 means all sentences,
                                    | and > 0 means random sampling will be applied to pick this number of sentences out
                                    | of the domain's population of sentences.
    
    > Returns:
        None. Everything is written to disk in `directory` path.
    """
    sentencesD1 = [];
    sentencesD2 = [];
    
    print "Starting main routine..\n";
    # Is the vocabulary already there?
    IsVocabAlreadyBuilt = os.path.isfile("{}/Vocab_{}_{}.txt".format(directory, _Filename2, _Filename1));
    
    domain1AlreadyDone = os.path.isfile("{}/{}_sentences.txt".format(directory, _Filename1));    
    if (not domain1AlreadyDone):
        sentencesD1 = elicitDomainSentences(_Filename1, _NumberOfSentences);
        print "{} data read and processed..".format(_Filename1);
        saveList2txt(sentencesD1, "{}/{}_sentences.txt".format(directory, _Filename1));
        print "{} data saved ({:d} sentences).\n".format(_Filename1, len(sentencesD1));
    else:
        print "\nATTENTION: Domain {} was found to be already processed.".format(_Filename1);
        if (not IsVocabAlreadyBuilt): # We will have to rebuild the vocabulary then
            # Load sentences
            print "INFO: Domain's sentences loaded as a preface to build the common vocabulary.";
            sentencesD1 = loadSentencesFromtxt("{}/{}_sentences.txt".format(directory, _Filename1));
    
    domain2AlreadyDone = os.path.isfile("{}/{}_sentences.txt".format(directory, _Filename2));
    if (not domain2AlreadyDone):            
        sentencesD2 = elicitDomainSentences(_Filename2, _NumberOfSentences);
        print "\n{} data read and processed..".format(_Filename2);
        saveList2txt(sentencesD2, "{}/{}_sentences.txt".format(directory, _Filename2));
        print "{} data saved ({:d} sentences).\n".format(_Filename2, len(sentencesD2));
    else:
        print "\nATTENTION: Domain {} was found to be already processed.".format(_Filename2);
        if (not IsVocabAlreadyBuilt): # We will have to rebuild the vocabulary then
            # Load sentences
            print "INFO: Domain's sentences loaded as a preface to build the common vocabulary.";
            sentencesD2 = loadSentencesFromtxt("{}/{}_sentences.txt".format(directory, _Filename2));

    if (not IsVocabAlreadyBuilt):
        setCommonVocab = findCommonVocabularyFromSentences(sentencesD1, sentencesD2);
        print "\nCommon vocabulary processed.."
        saveList2txt(setCommonVocab, "{}/Vocab_{}_{}.txt".format(directory, _Filename2, _Filename1), IsVocabulary=True);
        print "Common vocabulary saved ({:d} terms).\n".format(len(setCommonVocab));
        print "All Done.";
    else:
        print "\nATTENTION: Vocabulary was already found on disk, so no need to rebuild it again.\n";


# Filenames of the 11 Yelp domains
FilenameRestaurants = "Business_Restaurants_reviews";
FilenameFood = "Business_Food_reviews";
FilenameBars = "Business_Bars_reviews";
FilenameNightlife = "Business_Nightlife_reviews";
FilenameEventPlanning = "Business_Event Planning_Services_reviews";
FilenameShopping = "Business_Shopping_reviews";
FilenameBeautySpa = "Business_Beauty_Spas_reviews";
FilenameHomeServices = "Business_Home_Services_reviews";
FilenameAutomotive = "Business_Automotive_reviews";
FilenameHealthMedical = "Business_Health_Medical_reviews";
FilenameLocalServices = "Business_Local_Services_reviews";

ProcessDomains(FilenameRestaurants, FilenameLocalServices, 10000);


# =============================================================================
# # Experimentations...
# 
# print "Beta code block here..";
# df_Reviews = pd.read_csv(filepath_or_buffer="../../dataset/Reviews_By_Business/"+FilenameRestaurants+".csv",
#                               low_memory=False, usecols = ["text"]);
#                          
# print "Restaurants data read.";
# #for ix, r in enumerate(df_Reviews["text"]):
# #    print "processing number {:d}\n".format(ix);    
# #    casefold(r);
# =============================================================================


