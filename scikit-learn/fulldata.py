import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib

def run():
    print time.asctime(time.localtime()), "Copying datasets"
    
    f = ROOT.TFile("/net/storage03/data/users/dlafferty/NTuples/SignalMC/2012/combined/Bs2phiphi_MC_2012_13104013_MDMU_combined_TupleA.root")
    t = f.Get("DecayTree")     

    full = []

    v = joblib.load("pickle/variables.pkl")

    tcount = t.GetEntriesFast()

    print time.asctime(time.localtime()), "Real Data contains", tcount, "entries."
    
    for i in range (0, tcount):
        t.GetEntry(i)
        k = []
        for i in range(len(v)):
            k.append(eval("t."+v[i]))
        full.append(k)
    
    print time.asctime(time.localtime()), "Data copied"
    
    fulldata = np.array(full)

    joblib.dump(fulldata, 'pickle/fulldata.pkl')    
    
    print time.asctime(time.localtime()), "Datasets produced!"