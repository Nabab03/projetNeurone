import numpy as np
import scipy.io
import matplotlib.pyplot as plt 


#getpics
x_train= scipy.io.loadmat('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/Indian_pines_corrected.mat')
x_train=x_train['indian_pines_corrected']


#getTrainDat
trainData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/train_data.npy')
testData=np.load('/auto_home/aizard/Bureau/Cours/M2/s3/methodesSciencesDonnees/projet/test_data.npy')

#shape
print(x_train.shape )
print(y_train.shape)



trainData=keras.utils.to_categorical(y=trainData,num_classes=10)
y_test=keras.utils.to_categorical(y=y_test,num_classes=10)

# plt.imshow(mat[:,:,50])
# plt.show()