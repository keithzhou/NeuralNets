import numpy as np
import gnumpy as g
import time
def activation_relu(x):
    return (x >= 0) * x

def gradient_relu(x):
    return x > 0

def activation_softmax(x):
    result = x - g.max(x,axis=1)[:,g.newaxis]
    result = g.exp(result)
    result = result / g.sum(result,axis=1)[:,g.newaxis]
    return result

def gradient_output_softmax(y_predicted,y_target):
    return y_target - y_predicted

def score_softmax(y_target,y_predicted):
    assert(type(y_target) == type(y_predicted))
    if type(y_target) is g.garray:
        return g.sum(y_target * g.log(y_predicted + 1e-30))
    else:
        return np.sum(y_target * np.log(y_predicted + 1e-300))

class NeuralNetworkGPU():
    def __init__(self, layer_shape, dropout_probability, n_epochs = 50, l2_max = 15.0, learning_rate = lambda x:1.0 * .998 ** x, doGradientCheck = False):
        assert(len(dropout_probability) == len(layer_shape))
        self.dropout_probability = dropout_probability
        self.activation_hidden = activation_relu
        self.gradient_hidden = gradient_relu
        self.activation_output = activation_softmax
        self.gradient_output = gradient_output_softmax
        self.n_epochs = n_epochs
        self.f_score = score_softmax
        self.learning_rate = learning_rate
        self.mini_batch_size = 100
        self.doGradientCheck = doGradientCheck
        self.l2_max = l2_max

        self.training_score = []
        self.training_validation_error = []
        
        self.weights = []
        self.activation = []
        self.gradient = []
        for i in range(1,len(layer_shape)):
            self.weights.append([g.randn(layer_shape[i-1],layer_shape[i])*0.01, g.zeros(layer_shape[i])])
            self.activation.append(self.activation_hidden)
            self.gradient.append(self.gradient_hidden)
        self.activation[-1] = self.activation_output
        self.gradient[-1] = self.gradient_output
            
    def forward(self, X):
        result = X
        for i in range(len(self.weights)):
            w,b = self.weights[i]
            p = 1.0 - self.dropout_probability[i]
            a = self.activation[i]
            result = g.dot(result,w * p) + b
            result = a(result)
        return result
    
    def backprop(self, X, y_target) :
        # forward
        activity = []
        result = X
        for i in range(len(self.weights)):
            p = self.dropout_probability[i]
            mask = (g.rand(result.shape) >= p)
            result = result * mask
            activity.append(result)
            w,b = self.weights[i]
            result = g.dot(result,w) + b
            result = self.activation[i](result)
            
        # backward
        gradientNodes = []
        lastGradient = self.gradient[-1](result, y_target)
        gradientNodes.append(lastGradient)
        for i in reversed(range(1,len(self.weights))):
            w,b = self.weights[i]
            lastGradient = g.dot(lastGradient, w.T) * self.gradient[i-1](activity[i])
            gradientNodes.append(lastGradient)
                
        # get gradient
        resultGradient = []
        for i in range(len(self.weights)):
            gradW = (g.dot(activity[i].T,gradientNodes[-(i+1)]) / len(X))
            assert(gradW.shape == self.weights[i][0].shape)
            gradB = (g.sum(gradientNodes[-(i+1)],axis=0) / len(X))
            assert(gradB.shape == self.weights[i][1].shape)
            resultGradient.append([gradW,gradB])
        
        return resultGradient
    
    def fit(self,X,y, X_validation = None, y_validation = None):
        batchIndices = [(k, k+self.mini_batch_size) for k in range(0,len(X),self.mini_batch_size)]
        if X_validation is not None:
            X_validation = g.garray(X_validation)
        momentum = []
        for i in range(len(self.weights)):
            momentum.append([None,None])
        for epoch in range(self.n_epochs):
            timeStart = time.clock()
            random_indices = np.random.permutation(len(X))
            X = X[random_indices]
            y = y[random_indices]
            score = 0.0
            p = np.min([epoch/500.0,500]) * (.99-.5) + .5
            lr = self.learning_rate(epoch)
            for batch_start, batch_end in batchIndices:
                    gx = g.garray(X[batch_start:batch_end])
                    gy = g.garray(y[batch_start:batch_end])
                    
                    score += self.f_score(gy,self.predict_proba(gx))
                    gradient = self.backprop(gx,gy)
                    self.doGradientCheck and self.gradient_check(gx,gy,gradient)
                    for i in range(len(self.weights)):
                        w, b = self.weights[i]
                        gw, gb = gradient[i]

                        w += gw*lr
                        b += gb*lr

                        l2 = g.sum(w*w,axis=0)
                        l2 = (l2 >= self.l2_max) * (l2 / self.l2_max) + (l2 < self.l2_max) 
                        w /= l2
            mismatch = ''
            if X_validation is not None:
                mismatch = self.predict(X_validation) 
                mismatch = np.sum(mismatch != y_validation)
                self.training_validation_error.append(mismatch)
                mismatch = "error:%d/%d" % (mismatch,len(y_validation))
                
            self.training_score.append(score)
            print "epoch:", epoch, "score", score,  "lr:", lr, "momentum:", p, "time:", time.clock() - timeStart, mismatch
    
    def predict(self,X):
        if type(X) is g.garray:
            return np.argmax(self.predict_proba(X).as_numpy_array(),axis=1)
        else:
            return np.argmax(self.predict_proba(X),axis=1)
    
    def predict_proba(self,X):
        if type(X) is g.garray:
            return self.forward(X)
        else:
            return self.forward(g.garray(X)).as_numpy_array()
        
    def gradient_check(self,X,y,dweights):
        EPSILON = g.as_garray(1e-4)
        ERRORTHRESHOLD = g.as_garray(1e-2)
        g.GNUMPY_CPU_PRECISION = 64
        g.acceptable_number_types = "no nans or infs"
        for ind in range(len(self.weights)):
            w,b = self.weights[ind]
            dw,db = dweights[ind]
            for i in range(len(b)):
                b[i] = b[i] + EPSILON
                
                fw = self.predict_proba(X)
                op = self.f_score(y,fw)
                b[i] -= 2*EPSILON
                
                fw = self.predict_proba(X)
                om = self.f_score(y,fw)
                b[i] += EPSILON
                rs = (g.as_garray(op) - g.as_garray(om)) / (EPSILON * 2.0) / g.as_garray(len(X))
                if g.abs(rs - g.as_garray(db[i])) > ERRORTHRESHOLD:
                    print ind,i,rs,db[i], type(rs), type(db)
                    assert(0)

            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    w[i,j] += EPSILON
                    fw = self.predict_proba(X)
                    op = self.f_score(y,fw)
                    w[i,j] -= 2*EPSILON
                    fw = self.predict_proba(X)
                    om = self.f_score(y,fw)
                    w[i,j] += EPSILON
                    rs = (g.as_garray(op) - g.as_garray(om)) / (EPSILON * 2.0) / g.as_garray(len(X))
                    if g.abs(rs - g.as_garray(dw[i,j])) > ERRORTHRESHOLD:
                        print ind,i,j,rs,dw[i,j],type(w) , type(dw)
                        assert(0)
        print "gradient_check passed"

if __name__ == "__main__":
    def sanityCheck():
        X = [[2,-1]]
        y = [[1,0]]
        sut = NeuralNetworkGPU([2,2,2,2],[.0,.0,.0,.0])
        w12 = [[.2,.1],[.3,.4]]
        b12 = [.5,.6]
        w23 = [[.3,.2],[.1,.7]]
        b23 = [.5,.4]
        w34 = [[.1,.7],[.2,.1]]
        b34 = [-.6,-.2]
        sut.weights = [[g.garray(w12),g.garray(b12)], [g.garray(w23), g.garray(b23)], [g.garray(w34),g.garray(b34)]]

        result = sut.predict_proba(X)
        assert(np.allclose(result,np.array([[ 0.32038566,  0.67961431]])))
        gradient = sut.backprop(g.garray(X),g.garray(y))
        assert(np.allclose(gradient[2][1].as_numpy_array(),np.array([0.67961431, -0.67961431])))
        assert(np.allclose(gradient[1][1].as_numpy_array(),np.array([-0.40776858,  0.06796143])))
        assert(np.allclose(gradient[0][1].as_numpy_array(),np.array([-0.10873829,  0.00679614])))
        assert(np.allclose(gradient[2][1].as_numpy_array(),np.array([[0.67961431, -0.67961431]])))
        
        assert(np.allclose(gradient[0][0].as_numpy_array(),np.array([[-0.21747658,  0.01359228],[ 0.10873829, -0.00679614]])))
        assert(np.allclose(gradient[1][0].as_numpy_array(),np.array([[-0.24466115,  0.04077686],[-0.16310744,  0.02718458]])))
        assert(np.allclose(gradient[2][0].as_numpy_array(),np.array([[ 0.4893223 , -0.4893223 ],[ 0.54369152, -0.54369152]])))
        print "check passed"
        
    def testGradient():
        g.GNUMPY_CPU_PRECISION = 64
        sut = NeuralNetworkGPU([28*28,2,10],[.0,.0,.0],doGradientCheck=True, n_epochs=5)
        sut.fit(trainX[:400],Y[:400],testX,testY)
        
    sanityCheck()       
    #testGradient()
