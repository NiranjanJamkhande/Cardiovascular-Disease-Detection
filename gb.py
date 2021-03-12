# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:45:22 2021

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.get_dummies(df,drop_first = True)
df.info()

X = df.drop(columns = ['cardio'])
y = df['cardio']

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=1996,
                                                    stratify=y)
###################################################################################
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=1200, n_estimators=500,
                                 verbose=2)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)

################## Tuning with Randomized Grid Search #########################

lr_range = np.linspace(0.1,1)
#n_est_range = np.arange(10,200,10)

depth_range = [3,4,5,6]
minsplit_range = [5,10,20]
minleaf_range = [5,10,15]

parameters = dict(learning_rate=lr_range,
                  #n_estimators=n_est_range,
                  max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=3, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
clf = GradientBoostingClassifier(random_state=1200, n_estimators=500,
                                 verbose=2)
cv = RandomizedSearchCV(clf, param_distributions=parameters,
                  cv=kfold,scoring='roc_auc',random_state=2019,
                  n_iter=5, verbose = 2, n_jobs = -1)

cv.fit(X,y)

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)
############################################################################
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.19183673469387758, loss='deviance',
                           max_depth=4, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=5, min_samples_split=10,
                           min_weight_fraction_leaf=0.0, n_estimators=500,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=1200, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=2,
                           warm_start=False)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)
