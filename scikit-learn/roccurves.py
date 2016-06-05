import ROOT, time
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib

def run(name, quick = False):
    
    print time.asctime(time.localtime()), "Making ROC Curves"
    
    if quick == True:
        clf = joblib.load("pickle/" + name  + "quick.pkl")
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

    probas_ = clf.predict_proba(data_test)
    fpr, tpr, thresholds = roc_curve(output_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print time.asctime(time.localtime()), "Area under the BDT ROC curve : %f" % roc_auc
    title = 'BDT ROC curve (area = %0.2f)' % roc_auc

    pl.clf()

    pl.plot(fpr, tpr, label=title)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curve')
    pl.legend(loc="lower right")
    if quick == True:
        pl.savefig("output/ROC_" + name + "quick.pdf")
    else:
        pl.savefig("output/ROC_3Pres" + name +".pdf")
    
    print time.asctime(time.localtime()), "ROC Curve Saved as ROC_" + name +".pdf !" 