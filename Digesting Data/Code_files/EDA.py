"""
A python script as a surrogate for the EDA and Semantic Analysis subprocesses.
    To be used within the modular pipeline.
-------------------------------
Latest Update:  20171221-1553
Updated By:     Rafi Trad. 
"""

import numpy as np;
import matplotlib.pyplot as plt; 
import wordcloud; #Install it via conda install -c conda-forge wordcloud
from pprint import pprint;


def GenerateWordsCloud(_text, _returnImage = True):
    """
    GenerateWordsCloud will use the word-cloud module to generate a word cloud from the inputted text.
    > Parameters:
        _text : str            | the text from which a cloud shall be generated;
        _returnImage : bool    | [optional] a flag to return an image if true, so that it can be .show()n, or a generated image for matplotlib.
    > Returns:
        If _returnImage is True, it return a mere image that can be simply shown;
        However, if _returnImage is False, the returned object shall be used within matplotlib charts.
    > Comments:
        A good pilot example can be viewed via https://github.com/amueller/word_cloud/blob/master/examples/simple.py
        The full documentation of the word_cloud module is available here: https://github.com/amueller/word_cloud
    """
    #Sentinels:
    assert len(_text) > 0;    
    assert isinstance(_returnImage,bool);
    #Logic:
    wrdCld = wordcloud.WordCloud().generate(_text);
    if (_returnImage):
        retImg = wrdCld.to_image();
        return retImg;
    else:
        return wrdCld;
        

"""************************************************************************************************
******* Toying area, to check things. Be careful to reclear all variables before execution! *******
************************************************************************************************"""

"""
#GenerateWordsCloud trials:
text = "If _returnImage is True, it return a mere image that can be simply shown;\
    However, if _returnImage is False, the returned object shall be used within matplotlib charts.";

#GenerateWordsCloud(text).show();

img = GenerateWordsCloud(text,False);
plt.imshow(img, interpolation='bilinear');
plt.axis("off");
plt.show();
"""
