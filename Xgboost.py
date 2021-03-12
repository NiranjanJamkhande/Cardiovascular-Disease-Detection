# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:50:38 2021

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
from xgboost import XGBClassifier
from time import time
xgb = XGBClassifier(random_state=2000, n_estimators=500, n_jobs = -1, verbosity=2)
start_time=time()
xgb.fit(X_train,y_train)
end_time=time()
print(end_time-start_time)

y_pred = xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = xgb.predict_proba(X_test)[:,1]

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

##################Tunning using Randomized Search CV ############

lr_range = [0.01,0.2,0.5,0.6,1]

depth_range = [3,4,5]


parameters = dict(learning_rate=lr_range,
                  max_depth=depth_range)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=3, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
clf = XGBClassifier(random_state=2000, n_estimators=500, n_jobs=-1, verbosity=2)
cv = RandomizedSearchCV(clf, param_distributions=parameters,
                  cv=kfold,n_iter=5, scoring='roc_auc', verbose=2, n_jobs = -1)

cv.fit(X,y)
df_cv = pd.DataFrame(cv.cv_results_)

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)
#############################################################################
from xgboost import XGBClassifier

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.2, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=None, n_estimators=500, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=2000,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=2)
xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = xgb.predict_proba(X_test)[:,1]

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
