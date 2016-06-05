import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib
from sklearn import cross_validation

def run(name, quick=False):   
    print time.asctime(time.localtime()), "Cross Checking"
    
    if quick == True:
        clf = joblib.load("pickle/" + name +"quick.pkl")
        #data = joblib.loag('pickle/all_dataq.pkl')
        #signal = joblib.load('pickle/all_signalq.pkl') 
        data_train = joblib.load("pickle/dataq.pkl")
        output_train = joblib.load("pickle/outputq.pkl")
        data_test = joblib.load("pickle/datatestq.pkl")
        output_test = joblib.load("pickle/outputtestq.pkl")
    else:
        clf = joblib.load("pickle/" + name + ".pkl")
        #signal = joblib.load('pickle/all_signal.pkl')
        #data = joblib.load('pickle/all_data.pkl')
        data_train = joblib.load("pickle/data.pkl")
        output_train = joblib.load("pickle/output.pkl")
        data_test = joblib.load("pickle/datatest.pkl")
        output_test = joblib.load("pickle/outputtest.pkl")

    roc_scores = cross_validation.cross_val_score(clf,
                                          data_test, output_test,
                                          scoring="roc_auc",
                                          n_jobs=6,
                                          cv=3)

    scores = cross_validation.cross_val_score(clf,
                                          data_test, output_test,
                                          n_jobs=6,
                                          cv=3)

    print "ROC Accuracy: %0.5f (+/- %0.5f)"%(roc_scores.mean(), roc_scores.std())
    print "Accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std())