import numpy as np

def loadMNIST():
        SHAPEIMAGE = (28,28)
        trainX = np.fromfile("dataset/train_data",dtype=np.uint8)[16:].reshape(60000,28*28)
        trainY = np.fromfile("dataset/train_labels", dtype=np.uint8)[8:]

        indexes = np.random.permutation(len(trainX)) # randomize
        trainX = trainX[indexes]
        trainY = trainY[indexes]
        trainX = trainX/255.0 #- np.mean(trainX,axis=0)[np.newaxis,:]
        YOneHot = np.zeros([len(trainY),10])
        for i in range(10):
            YOneHot[:,i] = np.where(trainY==i,1,0)

        testX = np.fromfile("dataset/test_data",dtype=np.uint8)[16:].reshape(10000,28*28)
        testY = np.fromfile("dataset/test_labels", dtype=np.uint8)[8:]
        testX = testX/255.0 #- np.mean(testX,axis=0)[np.newaxis,:]
        YTestOneHot = np.zeros([len(testY),10])
        for i in range(10):
            YTestOneHot[:,i] = np.where(testY==i,1,0)

        return trainX, trainY, YOneHot, testX, testY, YTestOneHot
