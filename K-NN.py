# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:14:54 2021

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#####################################################################################

##### K- NN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11, n_jobs = -1)
knn.fit( X_train , y_train )
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

############ Plot ROC curve #############
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
#########################################
roc_auc_score(y_test, y_pred_prob)

############### Test for single k at a time ##############
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=42)
log_reg = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
results = cross_val_score(log_reg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

############## Testing all Ks at a time ###################

from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'n_neighbors': np.array([1,3,5,7,9,11])}
print(parameters)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=42)

knn = KNeighborsClassifier(n_jobs=-1)
# Tuned according to accuracy score
cv = GridSearchCV(knn, param_grid=parameters,cv=kfold, scoring='roc_auc')

cv.fit( X , y )

print(cv.cv_results_)

# Table of Grid Search CV Results
df_cv = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)
print(cv.best_score_)
###############################################