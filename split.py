import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt 

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


#prendre pix si entourage identique
def reSplitY(bands, Y_test, Y_train, ratio):
	Y_total=addMat(Y_train, Y_test)
	newYTest=np.copy(Y_test)
	newYTrain=np.copy(Y_train)
	total=0

	for i in range(Y_total.shape[0]):
		for j in range(Y_total.shape[1]):
			if Y_total[i,j]!=0:
				total+=1

	toPutInTrain=total*ratio
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








def reSplitZoneY(bands, Y_test, Y_train, ratio,zoneSize):
	Y_total=addMat(Y_train, Y_test)
	newYTest=np.copy(Y_test)
	newYTrain=np.copy(Y_train)
	total=0
	rayon=zoneSize//2

	for i in range(Y_total.shape[0]):
		for j in range(Y_total.shape[1]):
			if Y_total[i,j]!=0:
				total+=1

	toPutInTrain=total*ratio
	print(toPutInTrain)
	for i in range(Y_total.shape[0]-(rayon+1)):
		for j in range(Y_total.shape[1]-(rayon+1)):
			if Y_total[i,j]!=0 and j>rayon and i>rayon and total>rayon:
				if random.randint(1,total+1)<toPutInTrain:
					newYTrain[i-rayon:i+rayon+1,j-rayon:j+rayon+1]=Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1]
					newYTest[i-rayon:i+rayon+1,j-rayon:j+rayon+1]= np.zeros((zoneSize,zoneSize))
					toPutInTrain-=Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[0] * Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[1]
					total-=Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[0] * Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[1]
					j+=(zoneSize*zoneSize)-1
				else:
					newYTest[i-rayon:i+rayon+1,j-rayon:j+rayon+1]=Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1]
					newYTrain[i-rayon:i+rayon+1,j-rayon:j+rayon+1]= np.zeros((zoneSize,zoneSize))
					total-=Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[0] * Y_total[i-rayon:i+rayon+1,j-rayon:j+rayon+1].shape[1]
					j+=(zoneSize*zoneSize)-1
			else:
				newYTest[i,j]=0
				newYTrain[i,j]=0
				# print(toPutInTrain)
				# print(total)
		i+=(zoneSize*zoneSize) -1
	return newYTrain, newYTest



def splitY(bands, Y_train,Y_test, ratio):

	Y_total=addMat(Y_train, Y_test)
	newYTrain=np.zeros(Y_train.shape)
	newYTest=np.zeros(Y_test.shape)
	total=labelsNumber(Y_total)
	toPutInTrain=total*ratio
	print("total	toputintrain")
	print(total)
	print(toPutInTrain)
	assign=np.zeros(Y_total.shape)
	cptTrain=0
	cptTest=0
	i=0
	j=0
	print(total)
	for i in range(Y_total.shape[0]):
		for j in range(Y_total.shape[1]):
			if Y_total[i,j]==0:
				assign[i,j]=1
			elif assign[i,j]==0:
				label=Y_total[i,j]
				if total>0 and random.randint(1,total+1)<toPutInTrain:
					added,newYTrain=recursiveAddNeighbour(newYTrain, Y_total, i,j,label,0,assign)
					toPutInTrain-=added
					total-=added
					cptTrain+=added
				elif total>0:
					added,newYTest=recursiveAddNeighbour(newYTest, Y_total, i,j,label,0,assign)
					total-=added
					cptTest+=added
		
	plt.imshow(assign)
	plt.show()
	return newYTrain, newYTest


def recursiveAddNeighbour(Y, Y_T,x,y,l,addAcc, assign):
	added=addAcc
	if Y_T[x,y]==l and assign[x,y]==0:
		assign[x,y]=1
		added+=1
		Y[x,y]=Y_T[x,y]

	#si case de gauche du même label
	tmp=0
	if x-1>0 and Y_T[x-1,y]==l and assign[x-1,y]==0:
		tmp,Y=recursiveAddNeighbour(Y,Y_T,x-1,y,l,added,assign)
		added+=tmp
	elif x+1<Y_T.shape[0] and Y_T[x+1,y]==l and assign[x+1,y]==0:
		tmp,Y=recursiveAddNeighbour(Y,Y_T,x+1,y,l,added,assign)
		added+=tmp
	elif y-1>0 and Y_T[x,y-1]==l and assign[x,y-1]==0:
		tmp,Y=recursiveAddNeighbour(Y,Y_T,x,y-1,l,added,assign)
		added+=tmp
	elif y+1<Y_T.shape[1] and Y_T[x,y+1]==l and assign[x,y+1]==0:
		tmp,Y=recursiveAddNeighbour(Y,Y_T,x,y+1,l,added,assign)
		added+=tmp

	return added,Y
	






def rec(Y, Y_T,x,y,l,addAcc, assign):
	addAcc=addAcc

	#si case de gauche du même label

	if x-1>0 and Y_T[x-1,y]==l and assign[x-1,y]==0:
		Y[x-1,y]=Y_T[x-1,y]
		assign[x-1,y]=1
		addAcc+=1+recursiveAddNeighbour(Y,Y_T,x-1,y,l,addAcc,assign)
	elif x+1<Y_T.shape[0] and Y_T[x+1,y]==l and assign[x+1,y]==0:
		Y[x+1,y]=Y_T[x+1,y]
		assign[x+1,y]=1
		addAcc+=1+recursiveAddNeighbour(Y,Y_T,x+1,y,l,addAcc,assign)
	elif y-1>0 and Y_T[x,y-1]==l and assign[x,y-1]==0:
		Y[x,y-1]=Y_T[x,y-1]
		assign[x,y-1]=1
		addAcc+=1+recursiveAddNeighbour(Y,Y_T,x,y-1,l,addAcc,assign)
	elif y+1<Y_T.shape[1] and Y_T[x,y+1]==l and assign[x,y+1]==0:
		Y[x,y+1]=Y_T[x,y+1]
		assign[x,y+1]=1
		addAcc+=1+recursiveAddNeighbour(Y,Y_T,x,y+1,l,addAcc,assign)
	return addAcc
		








def labelsNumber(Y):
	total=0
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			if Y[i,j]!=0:
				total+=1
	return total
#trainBand.append(bands[i-bord:i+bord+1, j-bord:j+bord+1, :])

# i = 1
# while i < 6:
#   print(i)
#   i += 1







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
