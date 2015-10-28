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
		self.dropout = dropout
		
		#Initialize weights and biases.
		self.weights = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(previousLayer.numNeurons, self.numNeurons)),
                dtype=theano.config.floatX),
            name='weights', borrow=True)
        self.biases = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(self.numNeurons,)),
                       dtype=theano.config.floatX),
            name='biases', borrow=True)
            
        self.learningParams = [self.weights, self.biases]
        
        #Configure layer processing procedure.
        something = previousLayer.output
        somethingElse = previousLayer.outputDropout
        self.output = 0
        self.outputDropout = 0
