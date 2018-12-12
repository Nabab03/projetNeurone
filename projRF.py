import sys
from pprint import pprint
from time import time
from split import splitData, loadData, clusterFromBand,preTreatment, pixelsToClusters, removeUselessLabel
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
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix


bands, Y_train, Y_test=loadData()
bands=preTreatment(bands)
bands+=-bands.min()
print(bands.min())
print(bands.max())

X_train, X_test, Y_train, Y_test=pixelsToClusters(Y_train, Y_test, bands)
removeUselessLabel(Y_test,Y_train)

# print(X_train.shape)
# print(X_test.shape)
# # print(Y_train.shape)
# # print(Y_test.shape)
# print(X_train.min())
# print(X_train.max())


# print(Y_train)

# print(np.unique(Y_train))
# print(np.unique(Y_test))
# # print(X_train)



# f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
# ax1.imshow(nX[:,:,60])
# ax2.imshow(nX[:,:,20])
# # # ax3.imshow(mat)
# # plt.imshow(nX[:,:,50])
# plt.show()

# reshape data to fit Classifier requirements
nsamples, nx, ny, nw = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny*nw))

nsamples, nx, ny,nw = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny*nw))

parameters = {
        'n_estimators':[10, 50, 100],
        'max_depth':[10, 20, None],
    }

clf = GridSearchCV(RandomForestClassifier(), param_grid=parameters,n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)  
y_predicted=clf.predict(X_test)

print(metrics.classification_report(Y_test, y_predicted))
cm = metrics.confusion_matrix(Y_test, y_predicted)
print(cm)  