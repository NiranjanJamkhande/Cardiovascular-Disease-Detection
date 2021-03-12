# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:25:34 2021

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
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=1211, n_estimators=500, oob_score=True,
                                  n_jobs=-1)


from sklearn.ensemble import BaggingClassifier
model_bagging = BaggingClassifier(random_state=1211,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=500,max_samples=X_train.shape[0], n_jobs=-1)


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=1200, n_estimators=500)


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=2000, n_estimators=500, n_jobs=-1)




Voting = VotingClassifier(estimators=[('rf',model_rf),
                                      ('bagging',model_bagging),
                                      ('gb',gb),
                                      ('xgboost', xgb)],
                                       voting='soft', n_jobs=-1)


Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
y_pred_prob = Voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))








