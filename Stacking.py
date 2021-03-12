# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:57:07 2021

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

#####################################################################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression( random_state=1996, verbose=1)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42,max_depth = 4)


from sklearn.linear_model import SGDClassifier
sgdClass = SGDClassifier(loss='log', random_state=2,verbose = 2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)

models_considered = [('Logistic Regression', logreg),
                     ('Dtree', dtc),
                     ('sgd',sgdClass),
                     ('knn',knn)]




from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=2000, n_estimators=500, verbosity=2)



from sklearn.ensemble import StackingClassifier
stack = StackingClassifier(estimators = models_considered,
                           final_estimator=xgb,stack_method="predict_proba",
                           passthrough=True)

stack.fit(X_train,y_train)

y_pred_prob = stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)

