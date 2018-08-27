#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:54:27 2018
Updated on 20180402
@author: rtrad

Compute the McNemar's significant test between two classifiers/algorithms.
The code is directly based on the paper:
    "SOME STATISTICAL ISSUES IN THE COMPARISON OF SPEECH RECOGNITION ALGORITHMS", by Gillick and Cox.
"""
import numpy as np;
import operator as op;

def nCr(n, r):
    """
    Combinations efficient function, adopted from: https://stackoverflow.com/a/4941932/3429115 (name changed to nCr)
    """
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom




def McNemar(_table):
    """
    Calculate the McNemar's significance P-value

    >Parameters:
       _table: a 2x2 numpy matrix       | The correct and incorrect classifications' frequencies are listed for
                                        | the two classifiers

    >Returns:
        The P-value. It should be < (1- Confidence interval), or greater than a pre-specified threshold, mostly 0.05.
        The interpretation is: we expect (P_value * 100)% of chance in the results, otherwise it would be genuine.
        An elaborated explanation is available in the associated paper.

    NOTICE: Due to technical limitations in floats representation in Python, 0.5**k is considered to be 0, and p-value consequently..
        So, the following online calculator can be used instead: http://vassarstats.net/propcorr.html ; but there: 1 means correct and 0 means incorrect.
        we are interested in McNemar's Two-Tail test result; however, the results aren't that different, presumably.
    """
    N10, N01 = _table[1,0], _table[0,1];
    k = N10 + N01;

    if (N10 == k/2):
        return 1.0;

    P = 0;
    if (N10 > k/2):
        R = range(N10, k+1);
    elif (N10 < k/2):
        R = range(0, N10+1);

    for i in R:
        try:
            expo = (0.5**k);
            if (expo == 0.): # Too much data to have randomness (if k > 1074 empirically)
                break;
            combinations = nCr(k, i);
            accumulation = combinations * expo;
            P = P + accumulation;
        except OverflowError:
            print "Overflow error when i is {:d}.".format(i);
            print "Exiting...";
            return ;
    return (2*P);

#t = np.matrix("1325 3; 13 59"); # Testing example 3.1 from the aforementioned paper
#print "McNemar's test for example 1 - the default- is: ";
#print McNemar(t); #Must produce ~0.0213

#t = np.matrix("36936 57859; 324479 523437"); # Example 2
#print "McNemar's test for example 2 is: ";
#print McNemar(t);

# t = np.matrix("54954 166543; 0 22083"); # Example 3
# print "McNemar's test for example 3 is: ";
# print McNemar(t); 
