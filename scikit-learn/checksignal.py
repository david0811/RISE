import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib

def run(name, quick=False):   
    print time.asctime(time.localtime()), "Making Predictions"
    
    if quick == True:
        clf = joblib.load("pickle/" + name +"quick.pkl")
        data_train = joblib.load("pickle/dataq.pkl")
        output_train = joblib.load("pickle/outputq.pkl")
        data_test = joblib.load("pickle/datatestq.pkl")
        output_test = joblib.load("pickle/outputtestq.pkl")
        #signal_train = joblib.load('pickle/signalq.pkl')
        #signal_test = joblib.load('pickle/signaltestq.pkl')
        #signal_train_output = joblib.load('pickle/signaloutputq.pkl')
        #signal_test_output = joblib.load('pickle/signaloutputtestq.pkl')
        #backgr_train = joblib.load('pickle/backgroundq.pkl')
        #backgr_test = joblib.load('pickle/backgroundtestq.pkl')
        #backgr_train_output= joblib.load('pickle/backgroundoutputq.pkl')
        #backgr_test_output = joblib.load('pickle/backgroundoutputtestq.pkl')
    else:
        clf = joblib.load("pickle/" + name + ".pkl")
        data_train = joblib.load("pickle/data.pkl")
        output_train = joblib.load("pickle/output.pkl")
        data_test = joblib.load("pickle/datatest.pkl")
        output_test = joblib.load("pickle/outputtest.pkl")
        #signal_train = joblib.load('pickle/signal.pkl')
        #signal_test = joblib.load('pickle/signaltest.pkl')
        #signal_train_output = joblib.load('pickle/signaloutput.pkl')
        #signal_test_output = joblib.load('pickle/signaloutputtest.pkl')
        #backgr_train = joblib.load('pickle/background.pkl')
        #backgr_test = joblib.load('pickle/backgroundtest.pkl')
        #backgr_train_output= joblib.load('pickle/backgroundoutput.pkl')
        #backgr_test_output = joblib.load('pickle/backgroundoutputtest.pkl')
    
    print "Score on whole training sample is", clf.score(data_train, output_train)
    print "Score on whole test sample is", clf.score(data_test, output_test)
    #print "Score on training signal is ", clf.score(signal_train, signal_train_output)
    #print "Score on test signal is ", clf.score(signal_test, signal_test_output)
    #print "Score on training background is ", clf.score(backgr_train, backgr_train_output)
    #print "Score on test background is ", clf.score(backgr_test, backgr_test_output)
    
