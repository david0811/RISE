import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse
import csv
from sklearn.externals import joblib

def run(name, bins = 30, quick = False):
    
    print time.asctime(time.localtime()), "Plotting Response Curves"
    
    v = joblib.load("pickle/variables.pkl")
    
    if quick == True:
        clf = joblib.load("pickle/" + name + "quick.pkl")
        data_train = joblib.load("pickle/dataq.pkl")
        output_train = joblib.load("pickle/outputq.pkl")
        data_test = joblib.load("pickle/datatestq.pkl")
        output_test = joblib.load("pickle/outputtestq.pkl")
    
    else:
        clf = joblib.load("pickle/" + name + ".pkl")
        data_train = joblib.load("pickle/data.pkl")
        output_train = joblib.load("pickle/output.pkl")
        data_test = joblib.load("pickle/datatest.pkl")
        output_test = joblib.load("pickle/outputtest.pkl")
    
    decisions = []
    for X,y in ((data_train, output_train), (data_test, output_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (-20,20)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("Decision Function")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
    if quick == True:
        plt.savefig("output/Response" + name + "quick.pdf")
    else:
        plt.savefig("output/Response" + name + ".pdf")


