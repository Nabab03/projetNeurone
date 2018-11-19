import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

#getpics
bands= scipy.io.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
bands=bands['indian_pines_corrected']




#getTrainDat
trainData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
testData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')

#shape
print(bands.shape)
# print(trainData.shape)
# print(testData.shape)
print(bands[:,:,0])
print(trainData)

bands=bands.astype('float32')

# bands/=10000.0
# print(bands.max())
# print(bands.min())


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




#flatten array -> List (coorcoef ne marche qu'avec des listes)
samples = [np.array(bands[:,:,i]).flatten() for i in range(bands.shape[-1])]

# #
corMat=np.corrcoef(samples)

print("Cormat ?")
print(corMat[175][150])
print(bands[:,:,175])
print(bands[:,:,150])

print(corMat[50][125])
print(bands[:,:,50])
print(bands[:,:,125])


# #_________________Creation CNN_________________
# mon_shape = (145, 145, 1)

# mon_input=Input(shape=mon_shape)

# #kernel size : spécifie si on traite une région, un pixel etc
# # Extraction des features
# cnn=Conv2D(5, kernel_size=(5, 5), strides=(1, 1),activation='relu')(mon_input)
# cnn=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)
# cnn=Conv2D(5, kernel_size=(5, 5), activation='relu')(cnn)
# cnn=MaxPooling2D(pool_size=(2, 2))(cnn)

# # Analyse des features
# cnn=Flatten()(cnn)
# cnn=Dense(1000, activation='relu')(cnn)
# #num_classes ? 10 ? 
# cnn=Dense(9, activation='softmax')(cnn)

# # Construction du modèle
# model=Model(inputs=[mon_input], outputs=[cnn])

# # print(len(y_train))
# #test : ajout de la dimension label ?
# #pas sur du tout
# bands=np.expand_dims(bands, axis=0)

# #_________________ Compilation  _________________

# # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# #model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# # model.compile(loss='mean_squared_error', optimizer='sgd')


# # #_________________Apprentissage_________________

# #taille échantillon
# mon_batch_size = 128
# #nb de passage dans le réseau
# epo = 2

# model.fit(bands, trainData,
# 	batch_size=mon_batch_size,
# 	epochs=epo,
# 	verbose=1,
# 	validation_data=(bands, testData))

# # print(y_train[0])




# score=model.evaluate(bands, testData, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


#TODO : check quelles bandes discriminent quels Labels









# f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True)
# ax1.imshow(corMat)
# ax2.imshow(bands[:,:,50])
# ax3.imshow(bands[:,:,175])
# ax4.imshow(trainData)



# plt.imshow(bands[:,:,0])
# plt.show()






