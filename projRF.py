import sys
from pprint import pprint
from time import time
from libIP import splitData, loadData,pixelByLabel, preTreatment,splitPixel, removeUselessLabel,splitYWeighted,splitBlock,patchFromBand
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

bands, Y_train, Y_test=loadData()

# ------------------ PARAMS Debut----------------------#
#params :  3/5/pixel    0.5/0.7/0.8/0.9   none/50/100   none/rdPix/block/weight

if sys.argv[3]=='50':
    bands=preTreatment(bands,50)
elif sys.argv[3]=='100':
    bands=preTreatment(bands,100)
 
if sys.argv[3]!='none':
    bands+=-bands.min()

ratio=0.5
if sys.argv[2]=='0.5':
    ratio=0.57
if sys.argv[2]=='0.7':
    ratio=0.7
elif sys.argv[2]=='0.8':
    ratio=0.8
elif sys.argv[2]=='0.9':
    ratio=0.9

removeUselessLabel(Y_test,Y_train)

if sys.argv[4]=='rdPix':
    Y_train, Y_test= splitPixel(Y_train, Y_test, ratio)
if sys.argv[4]=='block':
    Y_train, Y_test= splitBlock(Y_train, Y_test, ratio)
if sys.argv[4]=='weight':
    Y_train, Y_test= splitYWeighted(Y_train, Y_test, ratio)

print('pixel train')
print(pixelByLabel(Y_train))
print('pixel test')
print(pixelByLabel(Y_test))

if sys.argv[1]=='3':
    X_train, X_test, Y_train, Y_test=patchFromBand(Y_train, Y_test, bands,3)
elif sys.argv[1]=='5':
    X_train, X_test, Y_train, Y_test=patchFromBand(Y_train, Y_test, bands,5)
elif sys.argv[1]=='pixel':
    X_train, X_test, Y_train, Y_test=splitData(Y_train, Y_test, bands)

if sys.argv[1]!='pixel':
    #reshape data to fit Classifier requirements
    nsamples, nx, ny, nw = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny*nw))
    nsamples, nx, ny,nw = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny*nw))

# ------------------ PARAMS Fin----------------------#

# f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
# ax1.imshow(nX[:,:,60])
# ax2.imshow(nX[:,:,20])
# # # ax3.imshow(mat)
# # plt.imshow(nX[:,:,50])
# plt.show()

parameters = {
        # 'n_estimators':[10, 50, 100],
        # 'max_depth':[10, 20, None],
        'n_estimators':[100],
        'max_depth':[None],
    }

clf = GridSearchCV(RandomForestClassifier(), param_grid=parameters,n_jobs=-1, cv=5)
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
print(Y_test.shape)
print(y_predicted.shape)

# # nsamples, p = y_predicted.shape
# y_predicted = y_predicted.reshape(145*145)
# print(y_predicted.shape)

# plt.imshow(y_predicted)
# plt.show()
