import numpy as np
import random
class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.num_layers = len(layers)
        self.mini_batch_size = mini_batch_size
        self.w = []
        self.b = []
        self.count = 0
        for i in xrange(1, len(layers)):
            w = np.asarray(
                np.random.normal(
                    loc = 0, scale = 1 / np.sqrt(layers[i - 1]), size = (layers[i - 1], layers[i])))
            self.w.append(w)
            b = np.asarray(
                np.random.normal(
                    loc = 0, scale = 1.0, size = (1, layers[i])))
            self.b.append(b)

            
    def SGD(self, train_data):
        print self.w
        print self.b
        epochs = 1500
        for j in xrange(epochs):
            random.shuffle(train_data)
            mini_batchs = [train_data[i:i+self.mini_batch_size] for i in xrange(0, len(train_data), self.mini_batch_size)]
            #mini_batchs = [train_data]
            for mini_batch in mini_batchs:    
                self.train_batch(mini_batch)

    def train_batch(self, mini_batch):
    
        total_delta_x = [np.zeros(weight.shape) for weight in self.w]
        total_delta_b = [np.zeros(bias.shape) for bias in self.b]


        for (x, y) in mini_batch:

            deltax, deltab = self.train(x, y)

            for l in xrange(self.num_layers - 1):
                total_delta_x[l] = deltax[l] + total_delta_x[l]
                total_delta_b[l] = deltab[l] + total_delta_b[l]
                
        self.w = [w - (2.0/len(mini_batch)) * nw for w, nw in zip(self.w, total_delta_x)]
        self.b = [b - (2.0/len(mini_batch)) * nb for b, nb in zip(self.b, total_delta_b)]

    def train(self, x, y):
        deltaw = [np.zeros(weight.shape) for weight in self.w]
        deltab = [np.zeros(bias.shape) for bias in self.b]

        actis = [x]
        zs = []
        self.count = self.count + 1
        #forward
        
        for (b, w) in zip(self.b, self.w):
            z = np.dot(actis[-1], w) + b
            zs.append(z)
            activations = sigmoid(z)
            actis.append(activations)

        if (self.count + 1) % 100 == 0:
            print "x=%s, y=%s, z=%s, predict=%s"%(x[0][0],x[0][1],y,actis[-1]) 
        
        #backword propagation
        delta = (actis[-1] - y) * sigmoid_prime(zs[-1])
        deltab[-1] = delta
        a = np.dot(actis[-1 - 1].T, delta)
        deltaw[-1] = a

        for l in xrange(2, self.num_layers):
            prev_layer_w = np.dot(delta, self.w[-l + 1].T) 
            delta = prev_layer_w * sigmoid_prime(zs[-l])
            deltat_w = np.dot(actis[-l - 1].T, delta)

            deltaw[-l] = (deltat_w)
            deltab[-l] = (delta)

        return deltaw, deltab
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


B = Network(layers = [2, 2, 1], mini_batch_size = 12)
minibatch = [(np.asarray([[0, 0]]), 0), (np.asarray([[0, 0]]), 0),(np.asarray([[0, 0]]), 0),(np.asarray([[0, 0]]), 0), 
             (np.asarray([[1, 0]]), 0), (np.asarray([[1, 0]]), 0), (np.asarray([[1, 0]]), 0), (np.asarray([[1, 0]]), 0),
            (np.asarray([[0, 1]]), 0), (np.asarray([[0, 1]]), 0), (np.asarray([[0, 1]]), 0), (np.asarray([[0, 1]]), 0), 
             (np.asarray([[1, 1]]), 1), (np.asarray([[1, 1]]), 1), (np.asarray([[1, 1]]), 1), (np.asarray([[1, 1]]), 1)]
B.SGD(minibatch)





