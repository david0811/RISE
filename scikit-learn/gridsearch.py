import ROOT, time
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
import pylab as pl
import argparse
import csv
from sklearn.externals import joblib
from sklearn import grid_search

def run(name, quick = False):
    print time.asctime(time.localtime()), "Training BDT with Grid Search" 
    
    if quick == True:
        data_train = joblib.load('pickle/data.pkl')
        output_train = joblib.load('pickle/outputq.pkl')

    else:
        data_dev = joblib.load('pickle/datadev.pkl')
        data_eval = joblib.load('pickle/dataev.pkl')
        output_dev = joblib.load('pickle/outputdev.pkl')
        output_eval = joblib.load('pickle/outputev.pkl')
        data_train = joblib.load('pickle/data.pkl')
        output_train = joblib.load('pickle/output.pkl')

    #dt = DecisionTreeClassifier()
    #bdt = ensemble.AdaBoostClassifier(dt)
    bdt = ensemble.GradientBoostingClassifier()

    param_grid = {'n_estimators': [50, 200, 400, 800], 'max_depth': [3, 5, 10, 15], 'learning_rate': [0.5,1.0]}

    clf = grid_search.GridSearchCV(bdt, param_grid, n_jobs=6, cv=3)

    clf.fit(data_train, output_train)

    print "Best parameter set found on development set:"
    print clf.best_estimator_
    print
    print "Grid scores on a subset of the development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.6f (+/-%0.06f) for %r"%(mean_score, scores.std(), params)
    print
    print "With the model trained on the full development set:"

    y_true, y_pred = output_dev, clf.decision_function(data_dev)
    print "It scores %0.6f on the full development set"%roc_auc_score(y_true, y_pred)
    y_true, y_pred = output_eval, clf.decision_function(data_eval)
    print "It scores %0.6f on the full evaluation set"%roc_auc_score(y_true, y_pred)
    print
    
    if quick == True:
        joblib.dump(clf.best_estimator_, 'pickle/' + name + str(bdt.max_depth) + 'quick.pkl')
    else:
        joblib.dump(clf.best_estimator_, 'pickle/' + name + str(bdt.max_depth) + '.pkl')
    
    print time.asctime(time.localtime()), "BDT Trained"