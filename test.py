from split import pixelByLabel, reSplitZoneY, splitYWeighted,patchFromBand,splitY,reSplitY, addMat, loadData,removeUselessLabel, reSplitZoneY
import matplotlib.pyplot as plt 
import numpy as np
bands, Y_train, Y_test=loadData()
import random
import skimage as sk
# removeUselessLabel(Y_test,Y_train)

# X_train, X_test, Y_train, Y_test=clusterFromBand(Y_train, Y_test, bands,3)



Y_train, Y_test= reSplitZoneY(Y_train, Y_test,0.8,20)



# print(X_train.shape)
# print(X_test.shape)
# print(Y_train[0])
# print(Y_test[0])
# splitYWeighted(Y_train, Y_test, 0.8)
# Y_total=addMat(Y_train, Y_test)
# pixelByLabel(Y_total)

f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
ax1.imshow(Y_train)
ax2.imshow(Y_test)
# ax3.imshow(Y_total)

plt.show()


# X_train, X_test, Y_train, Y_test=clusterFromBand(Y_train, Y_test, bands,5)

# print(X_train.shape)
# print(X_test.shape)

# nsamples, nx, ny, nw = X_train.shape
# X_train = X_train.reshape((nsamples,nx*ny*nw))

# nsamples, nx, ny,nw = X_test.shape
# X_test = X_test.reshape((nsamples,nx*ny*nw))


# print(X_train.shape)
# print(X_test.shape)

