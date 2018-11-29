import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
#getpics
bands= scipy.io.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
bands=bands['indian_pines_corrected']

def splitTrainTestSet(X, y, testRatio=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio)
    return X_train, X_test, y_train, y_test


#getTrainDat
trainData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
testData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')

#GLOBAL VAR
height=bands.shape[0]
width=bands.shape[1]
band=bands.shape[2]
windowsSize=1
trainWindow=[]
trainLabels=[]
testWindow=[]
testLabels=[]
classes=[]
count=200
classesNb=16
testRatio=0.1

#Normalize
bands=bands.astype('float')
bands-=np.min(bands)
bands/=np.max(bands)

# print(bands)

arrayMoy=np.ndarray(shape=(band,),dtype=float)

for i in range(band):
    arrayMoy[i]=np.mean(bands[:,:,i])

print(arrayMoy.size)
print(arrayMoy)

def Patch(height_index,width_index):

    transpose_array = np.transpose(bands,(2,0,1))
    height_slice = slice(height_index, height_index+windowsSize)
    width_slice = slice(width_index, width_index+windowsSize)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - arrayMoy[i]) 
    
    return np.array(mean_normalized_patch)


for i in range(classesNb):
    classes.append([])
for i in range(height - windowsSize + 1):
    for j in range(width - windowsSize + 1):
        curr_inp = Patch(i,j)
        curr_tar = trainData[i + int((windowsSize - 1)/2), j + int((windowsSize - 1)/2)]
        if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
            classes[curr_tar-1].append(curr_inp)

for c  in classes:
    print len(c)
