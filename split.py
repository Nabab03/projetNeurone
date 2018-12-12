import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import random

def loadData():
	bands= sio.loadmat('Indian_pines_corrected.mat')
	bands=bands['indian_pines_corrected']
	bands=bands.astype('float32')
	print("Chargement des donnees d entrainement  :")
	#getTrainDat
	Y_train=np.load('train_data.npy')
	Y_test=np.load('test_data.npy')
	# Y_train=addMat(Y_train, Y_test)
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

def pixelsToClusters(Y_train, Y_test, bands):
	trainBand=[]
	testBand=[]
	newYtrain=[]
	newYtest=[]
	ytr=0
	yte=0
	for i in range(1, bands.shape[0]-1):
		for j in range(1, bands.shape[1]-1):
			if Y_train[i][j]!=0:
				newCluster=[]
				for k in range(0, bands.shape[2]):
					newCluster.append([[bands[i-1,j-1,k], bands[i-1, j, k], bands[i-1, j+1, k]],
						[bands[i,j-1,k], bands[i, j, k], bands[i, j+1, k]],
						[bands[i+1,j-1,k], bands[i+1, j, k], bands[i+1, j+1, k]]])
				trainBand.append(newCluster)
				newYtrain=np.insert(newYtrain,ytr, Y_train[i][j])
				ytr+=1
			if Y_test[i][j]!=0:
				newCluster=[]
				for k in range(0, bands.shape[2]):
					newCluster.append([[bands[i-1,j-1,k], bands[i-1, j, k], bands[i-1, j+1, k]],
						[bands[i,j-1,k], bands[i, j, k], bands[i, j+1, k]],
						[bands[i+1,j-1,k], bands[i+1, j, k], bands[i+1, j+1, k]]])
				testBand.append(newCluster)
				newYtest=np.insert(newYtest,yte, Y_test[i][j])
				yte+=1
	return np.array(trainBand), np.array(testBand), np.array(newYtrain), np.array(newYtest)


def clusterFromBand(Y_train, Y_test, bands, cSize=3):
	bord=cSize//2
	trainBand=[]
	testBand=[]
	newYtrain=[]
	newYtest=[]
	ytr=0
	yte=0
	for i in range(bord, bands.shape[0]-(bord+1)):
		for j in range(bord, bands.shape[1]-(bord+1)):
			if Y_train[i][j]!=0:
				trainBand.append(bands[i-bord:i+bord+1, j-bord:j+bord+1, :])
				newYtrain=np.insert(newYtrain,ytr, Y_train[i][j])
				ytr+=1
			if Y_test[i][j]!=0:
				testBand.append(bands[i-bord:i+bord+1, j-bord:j+bord+1, :])
				newYtest=np.insert(newYtest,yte, Y_test[i][j])
				yte+=1
	return np.array(trainBand), np.array(testBand), np.array(newYtrain), np.array(newYtest)



def getCorrelationMatrix(bands):
	samples = [np.array(bands[:,:,i]).flatten() for i in range(bands.shape[-1])]
	corMat=np.corrcoef(samples)
	return corMat


#fusionne les bandes porteuses de la même information
def preTreatment(bands, n_comp=75):
	pca = PCA(n_components=n_comp, whiten=True)
	newBands=np.reshape(bands, (-1,bands.shape[2]))
	newBands=pca.fit_transform(newBands)
	newBands=np.reshape(newBands, (bands.shape[0], bands.shape[1],n_comp))
	return newBands


def addMat(m1,m2):
	m3=np.copy(m1)
	for i in range(m1.shape[0]):
		for j in range(m1.shape[1]):
			m3[i][j]+=m2[i][j]
	return m3

def removeUselessLabel(Y_test,Y_train):
	for i in range(Y_test.shape[0]):
		for j in range(Y_test.shape[1]):
			if Y_test[i][j] not in Y_train:
				Y_test[i][j]=0



def reSplitY(bands, Y_test, Y_train):
	Y_total=addMat(Y_train, Y_test)
	newYTest=np.copy(Y_test)
	newYTrain=np.copy(Y_train)
	total=0

	for i in range(Y_total.shape[0]):
		for j in range(Y_total.shape[1]):
			if Y_total[i,j]!=0:
				total+=1

	toPutInTrain=total*0.80
	for i in range(Y_total.shape[0]):
		for j in range(Y_total.shape[1]):
			if Y_total[i,j]!=0:
				if random.randint(1,total+1)<toPutInTrain:
					newYTrain[i,j]=Y_total[i,j]
					newYTest[i,j]=0
					toPutInTrain-=1
					total-=1
				else:
					newYTest[i,j]=Y_total[i,j]
					newYTrain[i,j]=0
					total-=1

			else:
				newYTest[i,j]=0
				newYTrain[i,j]=0

	return newYTrain, newYTest

	# print(bands.shape)
	# print(Y_total.shape)
	# X_train, X_test, Y_train, Y_test = train_test_split(bands, Y_total, test_size=0.33)
	# return X_train, X_test, Y_train, Y_test