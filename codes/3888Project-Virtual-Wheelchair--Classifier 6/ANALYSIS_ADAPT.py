import csv
import os
import time
import wave as we
import numpy as np
from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

def stand_and_mean (movementtype,empty_list_stdv,empty_list_mean):        #INSERT AN EMPTY ARRAY AS A VARIABLE WITH A NAME YOU WANT
                                                                          #movementtype NEEDS TO BE A STRING
    i = 1
    while i < 11:
        filename = 'filtereddata/'+(movementtype) + str(i) + '.dat'
        array = np.loadtxt(filename)
        var_direc = np.var(array)
        empty_list_stdv.append(np.sqrt(var_direc))
        empty_list_mean.append(np.mean(array))
        i += 1
        itworked = True
    return empty_list_mean , empty_list_stdv


def timevariance (movementtype,empty_list_diff):                       #movementtype NEEDS TO BE A STRING
    i = 1
    while i < 11:
        filename = 'filtereddata/' + (movementtype) + str(i) + '.dat'
        array = np.loadtxt(filename)
        minarray = np.min(array)
        maxarray = np.max(array)
        empty_list_diff.append(minarray - maxarray)
        return empty_list_diff


##FEATURE 1 STANDARD DEVIATION###

#DOUBLE BLINK TOTALS
doubleblink_total_mean = []
doubleblink_total_sd = []
stand_and_mean("doubleblink",doubleblink_total_sd,doubleblink_total_mean)

#BLINK TOTALS
blink_total_sd = []
blink_total_mean = []
stand_and_mean("left",blink_total_sd,blink_total_mean)

#LEFT TOTALS
left_total_sd = []
left_total_mean = []
stand_and_mean("left",left_total_sd,left_total_mean)

#RIGHT TOTALS
right_total_sd = []
right_total_mean = []
stand_and_mean("left",right_total_sd,right_total_mean)
## END OF FEATURE ONE


##FEATURE 2 TIME##

#LEFT
left_diff = []
timevariance("left",left_diff)

#RIGHT
right_diff = []
timevariance("right",right_diff)

#BLINK
blink_diff = []
timevariance("blink",blink_diff)

#DOUBLEBLINK
doubleblink_diff = []
timevariance("doubleblink",doubleblink_diff)


###PANDAS LABELLING THING
movements1 = ['left']
label = [ele for ele in movements1 for i in range(10)]          #LABELS FOR A BUNCH OF LEFTS
movements2 = ['right']
label2 = [ele for ele in movements2 for i in range(10)]         #LABELS FOR A BUNCH OF RIGHTS ...etc
movements3 = ['blink']
label3 = [ele for ele in movements3 for i in range(10)]
movements4 = ['doubleblink']
label4 = [ele for ele in movements4 for i in range(10)]

#ADDING TO DATASET OR SERIES FOR EACH FEATURE
df_left = pd.concat([pd.Series(left_total_sd), pd.Series(left_total_mean), pd.Series(label)], axis=1)
df_right = pd.concat([pd.Series(right_total_sd), pd.Series(right_total_mean), pd.Series(label2)], axis=1)
df_blink = pd.concat([pd.Series(blink_total_sd), pd.Series(blink_total_mean), pd.Series(label3)], axis=1)
df_doubleblink = pd.concat([pd.Series(doubleblink_total_sd), pd.Series(doubleblink_total_mean), pd.Series(label4)],
                           axis=1)

print("PANDA THING")
print("df_left")
print(df_left)
print("df_right")
print(df_right)


df = df_left.append(df_right)
df = df.append(df_blink)
df = df.append(df_doubleblink)
df.index = range(0, df.shape[0])
df.columns = ['sd', 'mean', 'label']

print("df")
print(df)

## TRAINING DATA
X = df.drop(['label'], axis=1)                   #REMOVES THE LABEL COLUMN
print("X")
print(X)
y = df.label
print("Y")
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)   #Split arrays or matrices into
                                                                                            # random train and test subsets
print("X_TRAIN")
print(X_train)
