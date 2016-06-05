import ROOT, time
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA, PCA
import array
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture 
from root_numpy import root2array, rec2array, array2root

def run(name, source, quick=False):
    print time.asctime(time.localtime()), "Filling BDT Branches"  

    branch_names = joblib.load("pickle/variables.pkl")
    
    if quick == True:
        signal = joblib.load('pickle/all_signalq.pkl')   
        clf = joblib.load("pickle/" + name + "quick.pkl")     
    else:
        signal = joblib.load('pickle/all_signal.pkl')
        clf = joblib.load("pickle/" + name + ".pkl")

    # predict and write probability of each MC event being signal
    bdt_MC_predicted = clf.predict_proba(signal)
    bdt_MC_predicted.dtype = [('GradBoost_prob', np.float64)]
    array2root((np.hsplit(bdt_MC_predicted,2)[1]), "/net/storage03/data/users/dlafferty/NTuples/SignalMC/2012/combined/Bs2phiphi_MC_2012_combined_corrected_TupleA_BDT.root", "DecayTree")

    # predict and write probability of every data event being signal
    all_data = root2array("/net/storage03/data/users/dlafferty/NTuples/data/2012/combined/Bs2phiphi_data_2012_corrected_TupleA_BDT.root", "DecayTree", branch_names)
    all_data = rec2array(all_data)

    bdt_data_predicted = clf.predict_proba(all_data)
    bdt_data_predicted.dtype = [('GradBoost_prob', np.float64)]
    array2root((np.hsplit(bdt_data_predicted,2)[1]), "/net/storage03/data/users/dlafferty/NTuples/data/2012/combined/Bs2phiphi_data_2012_corrected_TupleA_BDT.root", "DecayTree")
        
    print time.asctime(time.localtime()), "Branches Filled!"