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
from split import splitData, loadData, addMat ,preTreatment, pixelsToClusters

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix

bands, Y_train, Y_test=loadData()

# bands=preTreatment(bands)

X_train, X_test, Y_train, Y_test=pixelsToClusters(Y_train, Y_test, bands)



parameters = {
    'n_neighbors': [1],
        # 'n_neighbors': [1, 5, 10],
    'weights': ['uniform']
    #     'weights': ['uniform', 'distance']
    }


#reshape data to fit Classifier requirements
nsamples, nx, ny, nw = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny*nw))

nsamples, nx, ny,nw = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny*nw))


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