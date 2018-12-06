import sys
from pprint import pprint
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.io as sio

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix

bands= sio.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
bands=bands['indian_pines_corrected']
bands=bands.astype('float32')
# bands = [np.array(bands[:,:,i]).flatten() for i in range(bands.shape[-1])]

H=145
L=145
P=200

print("Chargement des donnees d entrainement  :")

#getTrainDat
Y_train=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
Y_test=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')

print(bands.shape)

print(Y_train.shape)
print(Y_test.shape)

def splitData(Y_train,Y_test, bands):
    trainBand=[]
    testBand=[]
    newYtrain=[]
    newYtest=[]
    ytr=0
    yte=0
    for i in range(bands.shape[0]):
        for j in range(bands.shape[1]):
            if Y_train[i][j]!=0:
                trainBand.append(bands[i,j,:])     
                newYtrain=np.insert(newYtrain,ytr, Y_train[i][j])
                ytr+=1
            if Y_test[i][j]!=0:
                testBand.append(bands[i,j,:])
                newYtest=np.insert(newYtest,yte, Y_test[i][j])
                yte+=1
                
    return np.array(trainBand), np.array(testBand), np.array(newYtrain), np.array(newYtest)



X_train, X_test, Y_train, Y_test=splitData(Y_train, Y_test, bands)
print(X_train.shape)
print(X_test.shape)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma='scale')
#n_jobs = nb of processor used. -1 = all available proc
clf = GridSearchCV(svc, parameters,n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)  
y_predicted=clf.predict(X_test)

print(metrics.classification_report(Y_test, y_predicted))
cm = metrics.confusion_matrix(Y_test, y_predicted)
print(cm)   