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
from split import splitData, loadData, reSplitY, splitData, clusterFromBand ,preTreatment, removeUselessLabel

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

bands, Y_train, Y_test=loadData()

# ------------------ PARAMS Debut----------------------#
#params :  3/5/pixel    0.7/0.8/0.9/none    pt/none

if sys.argv[4]=='50':
    bands=preTreatment(bands,50)
    bands+=-bands.min()
elif sys.argv[4]=='100':
    bands=preTreatment(bands,100)
    bands+=-bands.min()
elif sys.argv[4]=='200':
    bands=preTreatment(bands,200)       
    bands+=-bands.min()

# micro 0.90 de prec  / macro 0.97

if sys.argv[2]=='0.7':
    Y_train, Y_test= reSplitY(Y_train, Y_test, 0.70)
elif sys.argv[2]=='0.8':
    Y_train, Y_test= reSplitY(Y_train, Y_test, 0.80)
elif sys.argv[2]=='0.9':
    Y_train, Y_test= reSplitY(Y_train, Y_test, 0.90)



removeUselessLabel(Y_test,Y_train)



if sys.argv[1]=='3':
    X_train, X_test, Y_train, Y_test=clusterFromBand(Y_train, Y_test, bands,3)
elif sys.argv[1]=='5':
    X_train, X_test, Y_train, Y_test=clusterFromBand(Y_train, Y_test, bands,5)
elif sys.argv[1]=='pixel':
    X_train, X_test, Y_train, Y_test=splitData(Y_train, Y_test, bands)

if sys.argv[1]!='pixel':
    #reshape data to fit Classifier requirements
    nsamples, nx, ny, nw = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny*nw))
    nsamples, nx, ny,nw = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny*nw))

# ------------------ PARAMS Fin----------------------#

parameters = {
    'n_neighbors': [1],
        # 'n_neighbors': [1, 5, 10],
    'weights': ['uniform']
    #     'weights': ['uniform', 'distance']
    }

# svc = svm.SVC(gamma='scale')
#n_jobs = nb of processor used. -1 = all available proc
clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=parameters,n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)  
y_predicted=clf.predict(X_test)

print(metrics.classification_report(Y_test, y_predicted))
cm = metrics.confusion_matrix(Y_test, y_predicted)
print(cm)   
print(clf.cv_results_.keys())


print("Best parameters set :")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))