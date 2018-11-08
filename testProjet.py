import numpy as np
import scipy.io
import matplotlib.pyplot as plt 


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

#
corMat=np.corrcoef(samples)

print(corMat.shape)
print(corMat[2][25])
print(corMat[25][2])
# print(len(cormatrix))
# print(len(cormatrix[0][0]))

# print(cor)
# for i in range(0,200) :
# 	print(bands[:,:,i].shape)
# 	cor=np.corrcoef(bands[:,:,i],trainData,False)
# 	print("cor")
# 	print(cor.shape)

# for pic in bands[:,:,]:
# 	print("pic")
# 	print(pic.shape)
# 	print(trainData.shape)
	# print(np.corrcoef(pic,trainData))

# print(bands[:,:,0:5])

# trainData=keras.utils.to_categorical(y=trainData,num_classes=10)
# y_test=keras.utils.to_categorical(y=y_test,num_classes=10)
# print(np.corrcoef(bands,trainData))
# plt.imshow(corMat)
# plt.show()