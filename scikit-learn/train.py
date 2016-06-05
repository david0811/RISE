import ROOT, time
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib

def run(name, maxdepth = 3, estimators = 800, learningrate = 0.5, quick = False):
    print time.asctime(time.localtime()), "Training BDT" 
    
    if quick == True:
        data_train = joblib.load("pickle/data.pkl")
        output_train = joblib.load("pickle/outputq.pkl")

    else:
        data_train = joblib.load("pickle/data.pkl")
        output_train = joblib.load("pickle/output.pkl")
    
    clf = ensemble.GradientBoostingClassifier(max_depth = maxdepth, n_estimators = estimators, learning_rate = learningrate)
    #dt = DecisionTreeClassifier(max_depth = maxdepth)
    #clf = ensemble.AdaBoostClassifier(dt, n_estimators = estimators, learning_rate = learningrate)
    clf.fit(data_train, output_train)
    
    if quick == True:
        joblib.dump(clf, 'pickle/' + name + str(maxdepth) + 'quick.pkl')
    else:
        joblib.dump(clf, 'pickle/' + name + str(maxdepth) + '.pkl')
    
    print time.asctime(time.localtime()), "BDT Trained"