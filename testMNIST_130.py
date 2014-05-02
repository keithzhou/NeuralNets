import loadData
import NeuralNetworkGPU

trainX, trainY, trainYOneHotEncoding, testX, testY, testYOneHotEncoding = loadData.loadMNIST()
nn = NeuralNetworkGPU.NeuralNetworkGPU(layer_shape = [28*28,800,800,10],dropout_probability = [0.0,0.5,0.5,0.0], n_epochs = 3000, l2_max = 15.0)
try:
   nn.fit(trainX,trainYOneHotEncoding, X_validation = testX, y_validation = testY)
except:
   print "exception received" 
