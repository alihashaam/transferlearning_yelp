#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:02:33 2018

@author: rtrad
"""

from matplotlib import pyplot as plt;
import numpy as np;

#Store the PAD values as 1-D, i.e. y=0 for all X:
# https://stackoverflow.com/a/14434334/3429115

X = [0, 0.747078, 0.203200, 0.014456, 0.858998, 0.752708, 0.010795, 0.601474, 0.867194, 0.006794];
y = np.zeros(len(X)).tolist();
#labels = ["Restaurants", "Automative", "Bars", "Event Planning", "Food", "Health and Medical", 
#          "Home Services", "Local Services", "Nightlife", "Shopping"];
labels = ["Restaurants", "", "", "", "", "", 
          "", "", "Home Services", "Bars"];          
colors_vals = ["C1", "C4", "C4", "C4", "C4", "C4", "C4", "C4","r","g" ];    
y_label_positions = [-0.004, 0.001, 0.001, -0.002, 0.001, -0.001, 0.001, -0.001, 0.004, 0.004];
fig, ax = plt.subplots();
ax.scatter(X,y, c= colors_vals, marker = "+", s=128);
for i, label in enumerate(labels):
    ax.annotate(label, xy = (X[i], 0), xytext = (X[i] - (0.165 * (i==8)),y_label_positions[i]));
ax.get_yaxis().set_visible(False)
#fig.savefig("PAD.pdf", bbox_inches="tight", dpi=600);
plt.show();

#Plotting the relationship between PAD and Epsilon
def pad(x):
    return 2. * (1. - 2. * x);

X2 = np.linspace(-2,2,24);
vecPad = np.vectorize(pad);
y2 = vecPad(X2);
plt.grid(True);
plt.plot(X2, y2);
plt.xlabel("PAD");
plt.ylabel("Epsilon");

plt.gcf().savefig("EpsilonPADrelation.pdf", bbox_inches="tight", dpi=600);
plt.show();
plt.close();