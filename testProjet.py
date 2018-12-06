import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt 
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
#getpics
bands= sio.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
bands=bands['indian_pines_corrected']

def splitTrainTestSet(X, y, testRatio=0.10):
    # X=np.expand_dims(X, axis=0)
    # y=np.expand_dims(y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio)
    return X_train, X_test, y_train, y_test


#getTrainDat
trainData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
testData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')

def addMat(m1,m2):
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            m1[i][j]+=m2[i][j]
    return m1


#shape
# print(bands.shape)
# print(trainData.shape)
# print(testData.shape)
# print(bands[:,:,0])
# print("trainData")
# print(trainData.shape)
# print(trainData)
# print(testData.shape)
# print(testData)
bands=bands.astype('float32')

# # bands/=10000.0
# print(testData.max())
# print(testData.min())


# ID 	Label
# 2 	Maïs- Pas de Technique de Conservation des sols
# 3 	Maïs- Minimum Tillage
# 5 	Herbe-Pâturage
# 6 	Herbe-Arbre
# 10 	Soja - Pas de Technique de Conservation des sols
# 11 	Soja - Minimum Tillage
# 12 	Soja-clean
# 14 	Bois
# 15 	Bâtiment-Herbe-Arbre-drives

# X_train, X_test, y_train, y_test = splitTrainTestSet(bands, trainData, 0.10)

X_train=bands
X_test=bands
y_train=trainData
y_test=testData


# y_train=keras.utils.to_categorical(y=y_train,num_classes=16)
# y_test=keras.utils.to_categorical(y=y_test,num_classes=16)

# print("X_train")
# print(X_train.shape)
# print("X_test")
# print(X_test.shape)
# print("y_train")
# print(y_train.shape)
# print(y_train)



# #flatten array -> List (coorcoef ne marche qu'avec des listes)
# samples = [np.array(bands[:,:,i]).flatten() for i in range(bands.shape[-1])]

# # #
# bands=np.expand_dims(bands, axis=0)
# corMat=np.corrcoef(samples)
# # print("sample shape")
# # print(samples)
# # print("Cormat ?")
# # print(corMat[175][150])
# # print(bands[:,:,175])
# # print(bands[:,:,150])

# # print(corMat[50][125])
# # print(bands[:,:,50])
# # print(bands[:,:,125])


# # #_________________Creation CNN_________________
# mon_shape = (145, 145,1)

# mon_input=Input(shape=mon_shape)
# filters=10

# # #kernel size : spécifie si on traite une région, un pixel etc
# # # Extraction des features
# cnn=Conv2D(filters, kernel_size=(5, 5), strides=(1, 1),activation='relu')(mon_input)
# cnn=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)
# cnn=Conv2D(5, kernel_size=(5, 5), activation='relu')(cnn)
# cnn=MaxPooling2D(pool_size=(2, 2))(cnn)

# # Analyse des features
# cnn=Flatten()(cnn)
# cnn=Dense(16, activation='relu')(cnn)
# # #num_classes ? 10 ? 
# cnn=Dense(16, activation='softmax')(cnn)

# # # Construction du modèle
# model=Model(inputs=[mon_input], outputs=[cnn])

# # # print(len(y_train))
# # #test : ajout de la dimension label ?
# # #pas sur du tout
# # print(bands.shape)
# # bands=np.expand_dims(bands, axis=0)
# # print(bands.shape)
# # #_________________ Compilation  _________________

# # sgd = optimizer.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# # #model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
# # model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# model.compile(loss='mean_squared_error', optimizer='adam')


# # # #_________________Apprentissage_________________

# # #taille échantillon
# mon_batch_size = 128
# # #nb de passage dans le réseau
# #To augment later
# epo = 2

# #X_train, X_test, y_train, y_test
# # print(bands.shape)
# # print(testData.shape)
# X_train=np.expand_dims(X_train, axis=3)
# # y_train=np.expand_dims(y_train, axis=3)
# X_train=np.moveaxis(X_train,2, 0)
# # y_train=np.moveaxis(y_train,2, 0)

# print("_________________________________")
# print(X_train.shape)

# print("_________________________________")
# print(y_train.shape)

# model.fit(X_train, y_train,
# 	batch_size=mon_batch_size,
# 	epochs=epo,
# 	verbose=1,
# 	validation_split=0.1
#     )

# # print(y_train[0])

# score=model.evaluate(bands, testData, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


#TODO : check quelles bandes discriminent quels Labels


# train=trainData
# mat=addMat(testData,trainData)

f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
ax1.imshow(testData)
ax2.imshow(trainData)
# ax3.imshow(mat)
plt.show()


# plt.imshow(bands[:,:,0])
plt.show()

