## RISE '15 

# scikit-learn BDT training
Allows decision tree training with Gradient/Adaptive boosting through Python's `scikit-learn`. Program is modular to allow user to include/omit features as desired:
* quick - use 1% of events for increased speed
* gridsearch - iterate through many hyper-parameters and select optimal (http://scikit-learn.org/stable/modules/grid_search.html)(Gradient boosting only)
* roc - produce ROC curve (https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* features - list importancies of variables
* checksignal - produce score on training and testing set (http://scikit-learn.org/stable/modules/model_evaluation.html)
* crossvalidation - more thorough evaluation than `checksignal` (http://scikit-learn.org/stable/modules/cross_validation.html)
* plot - check for overtraining 

The probability of each event being signal can be written to a ROOT NTuple, forming the classifier.

# BDT cut optimisation
Searches for optimal cut position through calculating Punzi figure of merit. Fit is made on data which produces signal and background yield also.


 
