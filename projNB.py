import sys
from pprint import pprint
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.io as sio
from split import splitData, loadData, addMat,preTreatment,clusterFromBand, reSplitY

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix

bands, Y_train, Y_test=loadData()

print(bands.min())
bands=preTreatment(bands)

#----------
print(bands.min())
bands+=-bands.min()
print(bands.min())
#----------

Y_train, Y_test= reSplitY(bands, Y_train, Y_test, 0.80)
X_train, X_test, Y_train, Y_test=clusterFromBand(Y_train, Y_test, bands,5)


parameters = {
    'alpha': [1.0, 0.1],
    'fit_prior': [True, False]
    }

#n_jobs = nb of processor used. -1 = all available proc
clf = GridSearchCV(naive_bayes.MultinomialNB(), param_grid=parameters,n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)  
y_predicted=clf.predict(X_test)

print(metrics.classification_report(Y_test, y_predicted))
cm = metrics.confusion_matrix(Y_test, y_predicted)
print(cm)   