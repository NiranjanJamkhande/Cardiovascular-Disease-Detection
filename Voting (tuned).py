# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:09:08 2021

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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import SGDClassifier
sgdClass = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.6616734693877552,
              fit_intercept=True, l1_ratio=0.15, learning_rate='constant',
              loss='log', max_iter=1000, n_iter_no_change=5, n_jobs=-1,
              penalty='l2', power_t=0.5, random_state=2019, shuffle=True,
              tol=0.001, validation_fraction=0.1, verbose=1, warm_start=False)
sgdClass.fit(X_train,y_train)
#############################

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=6, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=15, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=2018, splitter='best')
clf.fit(X_train,y_train)
#########################

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.19183673469387758, loss='deviance',
                           max_depth=4, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=5, min_samples_split=10,
                           min_weight_fraction_leaf=0.0, n_estimators=500,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=1200, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=1,
                           warm_start=False)
clf.fit(X_train,y_train)
#######################################################################################

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

Voting = VotingClassifier(estimators=[('sgd',sgdClass),
                                      ('dtree',clf),
                                      ('gb',gb)],voting='soft', n_jobs=-1)

Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
y_pred_prob = Voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

