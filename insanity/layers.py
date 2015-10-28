import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample


class FullyConnectedLayer(object):

	def __init__(self, previousLayer, numNeurons, activation, miniBatchSize, dropout=0.0):
		self.numNeurons = numNeurons
		self.activation = activation
		self.miniBatchSize = miniBatchSize
		self.dropout = dropout
		#Initialize weights and biases.
		self.weights = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(connectionsIn, connectionsOut)),
                dtype=theano.config.floatX),
            name='weights', borrow=True)
        self.biases = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(connectionsOut,)),
                       dtype=theano.config.floatX),
            name='biases', borrow=True)
		#Store weights and biases in self.learningParams so they can be found by NeuralNetwork.
        self.learningParams = [self.weights, self.biases]
		
	def configureInput(self, input, inputDropout, miniBatchSize):
		self.input = input.reshape((miniBatchSize, self.connectionsIn))
		self.inputDropout = dropoutLayer(inputDropout.reshape((miniBatchSize, self.connectionsIn)), self.dropout)
		#Set non-dropout output
		self.output = self.activation((1-self.dropout)*T.dot(self.input, self.weights) + self.biases)
		#Set dropout output
		