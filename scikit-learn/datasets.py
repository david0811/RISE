import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib
from root_numpy import root2array, rec2array, array2root
from sklearn.cross_validation import train_test_split

def run(source, quick=False):
    print time.asctime(time.localtime()), "Copying datasets"

    branch_names = []

    with open(source, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            branch_names.append(row[0])

    if quick == True:
        step_size = 100
    else:
        step_size = 1
        
    if source == "BDTvarBs2phiphi.csv":
        myselection =  "B_s0_MM > 5486.77"
        backgr = root2array("/net/storage03/data/users/dlafferty/NTuples/data/2012/combined/Bs2phiphi_data_2012_corrected_TupleA.root",
                            "DecayTree",
                            branch_names,
                            myselection,
                            step = step_size)  
        backgr = rec2array(backgr)
    
        signal = root2array("/net/storage03/data/users/dlafferty/NTuples/SignalMC/2012/combined/Bs2phiphi_MC_2012_combined_corrected_TupleA.root",
                            "DecayTree",
                            branch_names,
                            step = step_size)
        signal = rec2array(signal)     

    # data contains every data point (later split into evaluation and test)
    data = np.concatenate((signal, backgr))
    # output contains binary class of data (later split into evaluation and test)
    output = np.concatenate((np.ones(signal.shape[0]),
                    np.zeros(backgr.shape[0])))

    frac = 0.5

    data_dev, data_eval, output_dev, output_eval = train_test_split(data, output,
                                              test_size=0.33, random_state=492)
    data_train, data_test, output_train, output_test = train_test_split(data_dev, output_dev,
                                                            test_size=frac, random_state=42)

    joblib.dump(branch_names, 'pickle/variables.pkl')

    print time.asctime(time.localtime()), "Real Data contains", len(data), "entries. Training on ", len(data)*frac, "Entries"
    print time.asctime(time.localtime()), "Monte Carlo contains", len(signal), "entries. Training on ", len(signal)*frac, "Entries"
    
    if quick == True:
        joblib.dump(signal, 'pickle/all_signalq.pkl')
        joblib.dump(data, 'pickle/all_dataq.pkl')
        joblib.dump(data_dev, 'pickle/datadevq.pkl')
        joblib.dump(data_eval, 'pickle/dataevq.pkl')
        joblib.dump(output_dev, 'pickle/outputdevq.pkl')
        joblib.dump(output_eval, 'pickle/outputevq.pkl')
        joblib.dump(data_train, 'pickle/dataq.pkl')
        joblib.dump(data_test, 'pickle/datatestq.pkl')
        joblib.dump(output_train, 'pickle/outputq.pkl')
        joblib.dump(output_test, 'pickle/outputtestq.pkl')
        
    else:
        joblib.dump(signal, 'pickle/all_signal.pkl')
        joblib.dump(data, 'pickle/all_data.pkl')
        joblib.dump(data_dev, 'pickle/datadev.pkl')
        joblib.dump(data_eval, 'pickle/dataev.pkl')
        joblib.dump(output_dev, 'pickle/outputdev.pkl')
        joblib.dump(output_eval, 'pickle/outputev.pkl')
        joblib.dump(data_train, 'pickle/data.pkl')
        joblib.dump(data_test, 'pickle/datatest.pkl')
        joblib.dump(output_train, 'pickle/output.pkl')
        joblib.dump(output_test, 'pickle/outputtest.pkl')   
    
    print time.asctime(time.localtime()), "Datasets produced!"




