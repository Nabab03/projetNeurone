import sys
from pprint import pprint
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.io as sio
from split import splitData, loadData

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix

bands, Y_train, Y_test=loadData()


X_train, X_test, Y_train, Y_test=splitData(Y_train, Y_test, bands)
print(X_train.shape)
print(X_test.shape)

parameters = {
    'n_neighbors': [1, 5, 10],
    'weights': ['uniform', 'distance']
    }



# svc = svm.SVC(gamma='scale')
#n_jobs = nb of processor used. -1 = all available proc
clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=parameters,n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)  
y_predicted=clf.predict(X_test)

print(metrics.classification_report(Y_test, y_predicted))
cm = metrics.confusion_matrix(Y_test, y_predicted)
print(cm)   