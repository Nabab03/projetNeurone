import numpy as np
import scipy.io as sio


def loadData():
    bands= sio.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
    bands=bands['indian_pines_corrected']
    bands=bands.astype('float32')
    print("Chargement des donnees d entrainement  :")
    #getTrainDat
    Y_train=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
    Y_test=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')
    print(bands.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    return bands, Y_train, Y_test


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
