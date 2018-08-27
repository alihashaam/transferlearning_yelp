#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
A python script as a surrogate for the Semantic Analysis subprocesses.
    To be used within the modular pipeline.
-------------------------------
Latest Update:  20180121-1857
Updated By:     Rafi Trad. 
"""

#import numpy as np;
import pandas as pd;
#SentiWordNet ontology for computing SO scores
import nltk;
#from nltk.corpus import sentiwordnet as swn; #http://www.nltk.org/howto/sentiwordnet.html
#from pprint import pprint;
from nltk.stem.wordnet import WordNetLemmatizer;
from nltk.corpus import wordnet;
import itertools;
import string;
import re;
import unidecode; # To strip text of accents
import icu;
import pmi_cluster as brwn; # Brown Clustering
"""
The py2casefold in not tested enough, slow, and I couldn't install it in anaconda. This lambda function is a replacement.
The function returns a string, not a unicode, because we will use the .translate fast function to remove punct. of strings.
Source: https://stackoverflow.com/a/32838944/3429115
"""
casefold = lambda u: unicode(icu.UnicodeString(u).foldCase()).encode(CharsSet,"replace");

CharsSet = "ascii"; # The Character set to be used as the default one when interpreting texts

def getWordnetPos(_treebank_tag):
    """
    Translate the tree bank PoS tags to the WordNet's
    
    > Parameters:
        _treebank_tag : str     | The tag to be translated
    
    > Returns:
        The relevant WordNet PoS tag
    https://stackoverflow.com/a/15590384/3429115
    """
    if _treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif _treebank_tag.startswith('V'):
        return wordnet.VERB
    elif _treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif _treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return '' # CAUTION! It will remove all the words but the four above! implicit stopwords and punctioation removal somehow


enContractionsDict = {
        "ain't": "am not",# / are not",
        "aren't": "are not",# / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",# / he would",
        "he'd've": "he would have",
        "he'll": "he will",# / he shall",
        "he'll've": "he will have",# / he will have",
        "he's": "he is",# / he has",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",# / how is",
        "i'd": "I would",# / I had",
        "i'd've": "I would have",
        "i'll": "I will ",#/ I will",
        "i'll've": "I will have",# / I shall have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had",# / it would",
        "it'd've": "it would have",
        "it'll": "it will",# / it shall",
        "it'll've": "it shall have",# / it will have",
        "it's": "it is",# / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",# / she had",
        "she'd've": "she would have",
        "she'll": "she will",#/ she shall",
        "she'll've": "she shall have",# / she will have",
        "she's": "she is",# / she has",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",# / so as",
        "that'd": "that would",# / that had",
        "that'd've": "that would have",
        "that's": "that is",# / that has",
        "there'd": "there had",# / there would",
        "there'd've": "there would have",
        "there's": "there is",# / there has",
        "they'd": "they would",# / they had",
        "they'd've": "they would have",
        "they'll": "they shall",# / they will",
        "they'll've": "they shall have",# / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",# / we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall",# / what will",
        "what'll've": "what shall have",#/ what will have",
        "what're": "what are",
        "what's": "what is",# / what has",
        "what've": "what have",
        "when's": "when is",# / when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",# / where has",
        "where've": "where have",
        "who'll": "who shall",# / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who is",# / who is",
        "who've": "who have",
        "why's": "why has",# / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",# / you had",
        "you'd've": "you would have",
        "you'll": "you shall",# / you will",
        "you'll've": "you shall have",# / you will have",
        "you're": "you are",
        "you've": "you have",
        "c'mon": "come on"
        };

def expandEnglishAbbreviations(_text, _contractionsDict):
    """
    Tries to expand the abbreviations of an English text, adopting the most likely case.
    
    > Parameters:
        _text : unicode             | The text which we want to expand, in the lower case
        
        _contractionsDict : dict    | The dictionary which holds the contractions to be resolved, 1-1 mapping.
        
    > Returns:
        The expanded text.
        
    > Comments:
        Sometimes, an abbreviation isn't straightforward to be expanded, like I'd, could be I had or I would. Here, we are using the most likely
    only, as more advanced methods are required to disambiguate the abbreviations' expansions, like what's said here: 
    https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python#comment29447237_19790352.
    
        Originally adopted from https://stackoverflow.com/a/43023503/3429115, and amended to remove ambiguation, keeping the most likely 
    expansions only.
    """
    for w in filter(None, re.split("[ ,!?:/\()*=`|{}#<>]+",_text)):
        # ATTENTION! splitting here, and when PoS tagging! duplicate, should be revisited!
        # Applying NLTK splitting will split on ', which will miss the dictionary keys, so simple splitting is preferred.
        # We do not want to split on ', or -, or anything that deforms the original abbreviation, and splitting should pay attention
        # to the punctuation right after the tokens, like "c'mon, ..", the token should be "c'mon" and not "c'mon,"
        if w in _contractionsDict:
            _text = _text.replace(w, _contractionsDict[w]);    
    return (_text);

# =============================================================================
# enNonEng2EngChars = {
#         u"ă" : "a",
#         u"â" : "a",
#         u"á" : "a",
#         u"é" : "e",
#         u"í" : "i",
#         u"î" : "i",
#         u"ó" : "o",
#         u"ẹ" : "e",
#         u"ị" : "i",
#         u"ọ" : "o",
#         u"ụ" : "u",
#         u"ã" : "a",
#         u"ả" : "a",
#         u"ẻ" : "e",
#         u"ỉ" : "i",
#         u"ỏ" : "o",
#         u"ủ" : "u",
#         u"ñ" : "n",
#         u"č" : "c",
#         u"ď" : "d",
#         u"ě" : "e",
#         u"í" : "i",
#         u"ň" : "n",
#         u"ř" : "r",
#         u"š" : "s",
#         u"ș" : "s",
#         u"ť" : "t",
#         u"ț" : "t",
#         u"ú" : "u",
#         u"ů" : "u",
#         u"ž" : "z",
#         u"đ" : "d",
#         u"ü" : "u",
#         u"ö" : "o",
#         u"Ö" : "o",
#         u"Ü" : "u",
#         u"ä" : "a",        
#         u"Ä" : "a",
#         u"ç" : "c",
#         u"ë" : "e", 
#         u"í" : "i",
#         u"õ" : "o"          
#         };
# 
# =============================================================================
    
def tagPoS(_text, _removePunct = True, _flatten = True):
    """
    TagPoS will find the PoS tags for a review, which is of a multiple sentences, preserving the sentences' boundaries.    
    
    > Parameters:
        _text : str                 | the text of the review in str format.
        
        _removePunct : bool         | [optional] a flag to strip out punctuation if true. Even if it's not smart, the PoS algorithm may be dump when
                                    | a series of punctuation marks is presented, tagging an * as a noun or adjective sometimes.
        
        _flatten : bool             | [optional] if true, output will be flattened, otherwise not. Flattening makes the output a list of tokens with tags,
                                    | otherwise the output will be a list of sentences, which in turn are lists of tokens => a list of lists!
    
    > Returns:
        If _flatten is False, it returns the matrix of tags, each row corresponds to a sentence, and the columns are the words.
    However, if _flatten is True, the returned object is a mere array of words' tags, but the sentences boundaries would have been 
    taken into consideration nevertheless.
    
        Additionally, _removePunt will be taken into consideration as well.
    
    > Comments:
        As said, it is NOT recommended to remove the punctuation blindly. However, this ability is offered. A better way is to remove 
    the punctuation from the output depending on smart PoS tagging, where the tag is "." === a punctuation.
    
        Also, It is not recommended to remove stop words before this stage, the outputs will contain all part of speeches, and afterwards 
    we can remove everything but adjectives, nouns, verbs, and adverbs, a smarter way to get the essence of a sentence.
    
        Flattening is recommended, because we want the Document-Terms matrix to be 2D, not 3D, i.e. the documents/texts should be represented
    as a list of tokens, not a list of lists of tokens (list of sentences). Sentences boundaries are harnessed in PoS tagging, and now can
    be ditched.
    
        The main function here is documented on: http://www.nltk.org/api/nltk.tag.html#nltk.tag.pos_tag_sents
    """
#    if not(isinstance(_text, unicode)):
#        _text = unicode(_text, errors="ignore");        
    listSentences = nltk.sent_tokenize(_text);
    # But even sentences need to be an array of words, so we have to tokenise further, making each sentence array distinguishable by rows
    # Convert the list of sentences to a list of list of words:
    matrixSentences = [];
    for sentence in listSentences:
        if (_removePunct): 
            # Note: if we are dealing with unicodes, not strings, str.translate() will produce a TypeError.            
            # Source: https://stackoverflow.com/a/11066687/3429115 IS TOO SLOW!
#            tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
#                      if unicodedata.category(unichr(i)).startswith('P'))
#            sentence = sentence.translate(tbl);
            # BUT, To cater to the efficiency, I will convert unicode -> str if necessary, and use efficient .translate()
            if not (isinstance(sentence, str)):
                sentence = sentence.encode(CharsSet,"ignore");            
            sentence = sentence.translate(None, string.punctuation);
            # Back to unicode:
#           sentence = unicode(strSentence, errors="ignore");
        # Append non-empty sentences:
        if (len(sentence ) > 0):
            matrixSentences.append(nltk.word_tokenize(sentence));        
    # Now let's try to PoS on the sentences of the text:
    taggedTokens = nltk.pos_tag_sents(matrixSentences);    
    if (_flatten):
        taggedTokens = list(itertools.chain.from_iterable(taggedTokens));
    return taggedTokens;


def lemmatizeTaggedTerms(_tgdSentsMTX, _isFlattened = True, _oddTokensBehaviour = 3):
    """
    Infer the lemmatized form of tokens with their PoS tags.
    
    > Parameters:
        _tgdSentsMTX : collection of pairs: token - PoS tag | The Token/Tag collection from which we want to find lemmas
        
        _isFlattened : bool                                 | True if the collection is an array rather than a matrix
        
        _oddTokensBehaviour : bool                          | What to do when an odd non-English or non-lingual token is encountered,
                                                            | if 1, such token is not lemmatized and included as-is,
                                                            | if 2, such token is  to be in ASCII form, ignoring non-ascii chars,
                                                            | if 3, such token is included after replacing the odd chars with '?'.
                                                            | 0 (and otherwise), such token is discarded.
                                                            
    > Returns:
        The lemmatized list of tokens according to the selected behaviour.
    
    """
    lemmatizer = WordNetLemmatizer();
    lemmatizedSentsMTX = [];
    if (not _isFlattened): # A doc is a list of sentences, which in turn are list of tokens
        for SentencePairs in _tgdSentsMTX:
            currentLemmatizedSentence = [];
            for pair in SentencePairs:
                WordNetTag = getWordnetPos(str(pair[1]));
                if (len(WordNetTag) > 0): # Ensure there is a mapping to WordNet categories, ignore punctuations, propositions, determinants, etc.
                    #Append the lemmatized token to the sentence list after decoding foreign letters:
                    currentLemmatizedSentence.append(lemmatizer.lemmatize(pair[0], WordNetTag));
            # Append the lemmatized sentence tokes to the list of sentences in the doc/review        
            lemmatizedSentsMTX.append(currentLemmatizedSentence);
    else: # A doc is a list of tokens, no boundaries for sentences, which is OK because PoS tags were done already
        for pair in _tgdSentsMTX:
            WordNetTag = getWordnetPos(str(pair[1]));
            if (len(WordNetTag) > 0): # Ensure there is a mapping to WordNet categories, ignore punctuations, propositions, determinants, etc.
                # Append the lemmatized token to the sentence list after decoding foreign letters
                # If lemmatizing fails, often due to non-English characters, kepp the word as it is:
                try:
                    lemmatizedSentsMTX.append(lemmatizer.lemmatize(pair[0], WordNetTag));
                except UnicodeDecodeError: # This code won't be entered at all, since the controlle will convert the unicode to the ASCII chars only
                    if (_oddTokensBehaviour == 1):
                        lemmatizedSentsMTX.append(pair[0]);
                    elif (_oddTokensBehaviour == 2):
                        lemmatizedSentsMTX.append(lemmatizer.lemmatize(unicode(pair[0], errors="ignore").encode(CharsSet,"ignore"), WordNetTag));
                    elif (_oddTokensBehaviour == 3):
                        lemmatizedSentsMTX.append(lemmatizer.lemmatize(unicode(pair[0], errors="replace").encode(CharsSet,"replace"), WordNetTag));
                        
    return lemmatizedSentsMTX;
    
def lemmatizeText(_text, _enAbbrvDict, _oddCharsBehaviour = 3):
    """
        The main controller to lemmatize a text and return the list of lemmatized tokens.
    
        This controller takes care of: replacing foreign accents with most likely letters, casefolding, removing newlines and tabs, replacing " with ', 
    removing punctuation after expanding abbreviations for the sake of a better effectiveness, stopping, tokenizing, generating the PoS tags, and 
    generating lemmas of the text depending on PoS tags.
    
    > Parameters:
        _text : unicode             | The text to tokenize and lemmatize, in unicode, which will be converted to str after proper processing
        
        _enAbbrvDict : dictionary   | Holds the english abbreviations' shorthands and expansions, used when removing punctuation
        
        _oddCharsBehaviour : bool   | What to do when an odd non-English or non-lingual token is encountered,
                                    | if 1, such token is not lemmatized and included as-is,
                                    | if 2, such token is  to be in ASCII form, ignoring non-ascii chars,
                                    | if 3, such token is included after replacing the odd chars with '?'.
                                    | 0 (and otherwise), such token is discarded.
    
    > Returns:
        A list of lemmatized tokens which belong to the inputted _text.
    """    
    # First, strip the unicode of accents and replace them with the closest English ASCII characters
    _text = unidecode.unidecode(_text);
    
    # Then convert to string
    if not(isinstance(_text,str)):
        _text = _text.encode(CharsSet,"ignore"); # Convert the unicode text to the ASCII (CharSet) realm, ignoring non-ascii special characters          
                        
    # Replace newlines, tabs, and double quotes
    regex_slash_newlines = re.compile(r'[\n\r\\]+');     
    _text = re.sub(regex_slash_newlines, " ", _text);
    regex_slash_tabs = re.compile(r'\t+');        
    _text = re.sub(regex_slash_tabs, " ", _text);
    regex_doublequotes = re.compile(r'\"+');
    _text = re.sub(regex_doublequotes, '"', _text);
    
    # Expand the abbreviations before removing the punctuation, in order to pay attention to the linguistic abbreviations
    _text = expandEnglishAbbreviations(casefold(_text), _enAbbrvDict);
    taggedText = tagPoS(_text, _removePunct=True, _flatten=True);
    return lemmatizeTaggedTerms(taggedText, _isFlattened = True, _oddTokensBehaviour = _oddCharsBehaviour);       
        

def brownCluster(DocumentsTokensMatrix, maxVocSize=None, minWordsCount=1, batchSize=500, doLowerCase=True, truncationLevel = None):
    """
    Perform brown clustering on a document/review, and return the bitmap representation of clusters.
    
    The bitmap representation can be cut at a certain level (HAC) and digested as a context-based feature in the NLP models.
    
    To familiarise yourself more with Brown Clustering (aka. IBM clustering): https://en.wikipedia.org/wiki/Brown_clustering
    
    Full documentation of the codefile used to implement the HAC: https://github.com/mheilman/tan-clustering
    
    > Parameters:
        DocumentsTokensMatrix : array of arrays     | An array of documents/reviews, each represented as an array of tokens;
                                                    | It may be better if the tokens are the processed lemmas, but that's open to experimentation.
        
        maxVocSize : int                            | maximum number of words in the vocabulary (a smaller number will be used if 
                                                    | there are ties at the specified level.
                                                    
        minWordsCount : int                         | minimum word count to include a word in the vocabulary (default: 1).
        
        batchSize : int                             | number of clusters to merge at one time (runtime is quadratic in this value - default is 500).
        
        doLowerCase : bool                          | lower-case the input? (True by default)
        
        truncationLevel : int                       | The truncation level to cut the tree at, and produce a set of clusters to be fed as features
    
    > Returns:
        Pandas dataframe with the cluster id, word, and word frequencies. The cluster ids can be then harnessed as features in NLP models. Notice that if
    a truncation level isn't specified, the cluster ids will have multiple lengths and won't be harmonised. batchSize variable appears to increase the quality
    of the output if set higher, but it incurs heavy delay.
        
        Interpretation example is available here: https://stackoverflow.com/a/21006240/3429115
    """
    # The used code is within the file pmi_cluster.py, downloaded from https://github.com/mheilman/tan-clustering, but modified a bit, tailoring it to our needs    
    dfClusters = pd.DataFrame(brwn.brownCluster(DocumentsTokensMatrix, maxVocSize, minWordsCount, batchSize, doLowerCase, passedAsTokens = True),
                              columns=["clusterId", "word", "wordFrequency"]);
    # If a truncation level is specified, then we will cut the tree at that level:
    if (truncationLevel != None):        
        dfClusters["clusterId"] = dfClusters["clusterId"].apply(lambda f : f[0:truncationLevel]);
        
    return dfClusters;
        
    
"""************************************************************************************************
******* Toying area, to check things. Be careful to reclear all variables before execution! *******
************************************************************************************************"""

"""
#PoS Tagging:
review = "AI makes it easier for you to do things every day. whether it is searching for the people you love,\
 breaking down language barriers, or helping you get things done with your own personal digital assistant! \
 AI is used widely, especially in Sudan, Sumalia, Yemen, and especially the south pole. C'mon, isn't it marvellous?";
review = "I am alive!";
""""""
#Trivial on words tokenizer
tokens = nltk.word_tokenize(review);
print "Tokens are:";
pprint(tokens);
taggedTokens = nltk.pos_tag(tokens); #Please run nltk.download('averaged_perceptron_tagger') or even nltk.download("all")!
print "PoS are:";
pprint(taggedTokens);
taggedWeightedTokens = nltk.pos_tag_sents(tokens);
print "-------------------------\n"
""""""
# Now let's tag a review!
PoSed = TagPoS(review, _removePunct=False, _flatten=False);
print "Type: {}".format(type(PoSed));
pprint(PoSed);
"""

# Brown clustering
reviewsDT = [];
reviewsDT.append(re.split(r'\s+' ,"Cats and dogs happen to harbour profound animosity towards each other".lower()));
reviewsDT.append(re.split(r'\s+' , "A cat is an deplorable creature . It will happily devour a soft cute delicious bird on the spot".lower()));
reviewsDT.append(re.split(r'\s+' , "Cats, dogs, and birds are creatures. However, a cat or a dog cannot fly, wherease a bird can.".lower()));
reviewsDT.append(re.split(r'\s+' , "Some people like to cook cats, and even dogs. Personally, eating a dog for lunch isn't an option.".lower()));
reviewsDT.append(re.split(r'\s+' ,"this is one document . it has two sentences but the program only cares about spaces .".lower()));
reviewsDT.append(re.split(r'\s+' ,"here is another document . it also has two sentences .".lower()));
reviewsDT.append(re.split(r'\s+' ,"and here is a third document with one sentence .".lower()));
reviewsDT.append(re.split(r'\s+' ,"this document is short .".lower()));
reviewsDT.append(re.split(r'\s+' ,"the dog ran in the park .".lower()));
reviewsDT.append(re.split(r'\s+' ,"the cat was chased by the dog .".lower()));
reviewsDT.append(re.split(r'\s+' ,"the dog chased the cat .".lower()));
brownClusters = brownCluster(reviewsDT, batchSize=10000000, truncationLevel=2);
#brownClusters = brownCluster(reviewsDT, batchSize=10000000, truncationLevel=None);
