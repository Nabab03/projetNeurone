from split import pixelByLabel, splitYWeighted,clusterFromBand,splitY,reSplitY, addMat, loadData,removeUselessLabel, reSplitZoneY
import matplotlib.pyplot as plt 

bands, Y_train, Y_test=loadData()
# removeUselessLabel(Y_test,Y_train)



Y_train, Y_test= reSplitY(Y_train, Y_test, 0.8)



# print(X_train.shape)
# print(X_test.shape)
# print(Y_train[0])
# print(Y_test[0])
# splitYWeighted(Y_train, Y_test, 0.8)
Y_total=addMat(Y_train, Y_test)
# pixelByLabel(Y_total)


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)
ax1.imshow(Y_train)
ax2.imshow(Y_test)
ax3.imshow(Y_total)

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

