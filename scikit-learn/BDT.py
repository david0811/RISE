import ROOT, time, os
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl
import argparse
from subprocess import call

start = time.time()
print time.asctime(time.localtime()), "Starting Code"

parser = argparse.ArgumentParser(description='Train BDT and analyse performance')
parser.add_argument("-d", "--dataset", action="store_true")
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-g", "--gridsearch", action="store_true")
parser.add_argument("-r", "--roc", action="store_true")
parser.add_argument("-f", "--features", action="store_true")
parser.add_argument("-q", "--quick", action="store_true")
parser.add_argument("-c", "--checksignal", action="store_true")
parser.add_argument("-v", "--crossvalidation", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-w", "--write", action="store_true")
parser.add_argument("-n", "--name", default="GradBoost")
parser.add_argument("-m", "--maxdepth", default=3)
parser.add_argument("-s", "--source", default="BDTvarBs2phiphi.csv")
parser.add_argument("-e", "--estimators", default=800)
parser.add_argument("-l", "--learningrate", default = 0.5)
parser.add_argument("-b", "--bins", default = 30)

cfg = parser.parse_args()

if cfg.source == "BDTvarBs2phiphi.csv":
    cfg.name += "Bs2phiphi"
elif cfg.source != "BDTvar.csv":
    raise NameError("Invalid Source Selection!")
    
if cfg.dataset == True:
    import datasets as ds
    if cfg.quick == True:
        ds.run(cfg.source, quick = True)
    else:
        ds.run(cfg.source)

if cfg.train == True:
    import train as tr
    if cfg.quick == True:
        tr.run(cfg.name, int(cfg.maxdepth), int(cfg.estimators), quick = True)
    else:
        tr.run(cfg.name, int(cfg.maxdepth), int(cfg.estimators))

if cfg.gridsearch == True:
    import gridsearch as g
    if cfg.quick == True:
        g.run(cfg.name, quick = True)
    else:
        g.run(cfg.name)

if cfg.roc == True:
    import roccurves as rc
    if cfg.quick == True:
        rc.run(cfg.name + str(cfg.maxdepth), quick = True)
    else:
        rc.run(cfg.name + str(cfg.maxdepth))
    
if cfg.features == True:
    import features as f
    if cfg.quick == True:
        f.run(cfg.name + str(cfg.maxdepth), quick = True)
    else:
        f.run(cfg.name + str(cfg.maxdepth))
        
if cfg.checksignal == True:
    import checksignal as cs
    if cfg.quick == True:
        cs.run(cfg.name + str(cfg.maxdepth), quick = True)
    else:
        cs.run(cfg.name + str(cfg.maxdepth))

if cfg.crossvalidation == True:
    import crossvalidation as cv
    if cfg.quick == True:
        cv.run(cfg.name + str(cfg.maxdepth), quick = True)
    else:
        cv.run(cfg.name + str(cfg.maxdepth))
        
if cfg.plot == True:
    import plot as p
    if cfg.quick == True:
        p.run(cfg.name + str(cfg.maxdepth), int(cfg.bins), quick = True)
    else:
        p.run(cfg.name + str(cfg.maxdepth), int(cfg.bins))

if cfg.write == True:
    import write as w
    if cfg.quick == True:
        raise Exception("Requires full dataset")
    else:
        w.run(cfg.name + str(cfg.maxdepth),cfg.source)

end = time.time()
print time.asctime(time.localtime()), "Code Ended"

pl.show()