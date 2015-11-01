import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample



class Layer(object):

	def __init__(self, numInputs, numNeurons, activation, miniBatchSize):
		self.numInputs = numInputs
		self.numNeurons = numNeurons
		self.activation = activation
		self.miniBatchSize = miniBatchSize
		
		#Initialize weights.
		self.weights = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/self.numNeurons), size=(self.numInputs, self.numNeurons)),
                dtype=theano.config.floatX),
            name='weights', borrow=True)
            
        #Initialize biases.
        self.biases = theano.shared(
            np.asarray(
				np.random.normal(
					loc=0.0, scale=1.0, size=(self.numNeurons,)),
				dtype=theano.config.floatX),
            name='biases', borrow=True)
        
    @input.setter
    def input(self, value):
		self.input = value
		#Configure the layer output.
		self.output = something



class FullyConnectedLayer(Layer):
	
	@Layer.input.setter
	def input(self, value):
		self.input = value
		#Configure the layer output.
		self.output = something
