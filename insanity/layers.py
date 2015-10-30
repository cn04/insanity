import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample



class NeuralNetworkLayer(object):

	def __init__(self, numInputs, input, inputDropout, numNeurons, activation, miniBatchSize, dropoutAmount=0.0):
		self.numInputs = numInputs
		self.input = input
		self.inputDropout = inputDropout
		self.numNeurons = numNeurons
		self.activation = activation
		self.miniBatchSize = miniBatchSize
		self.dropoutAmount = dropoutAmount
		
		#Initialize weights.
		self.weights = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(self.numInputs, self.numNeurons)),
                dtype=theano.config.floatX),
            name='weights', borrow=True)
            
        #Initialize biases.
        self.biases = theano.shared(
            np.asarray(
				np.random.normal(
					loc=0.0, scale=1.0, size=(self.numNeurons,)),
                dtype=theano.config.floatX),
            name='biases', borrow=True)
            
        #Store parameters to be learned in an attribute so that they can be externally accessed.
        self.learningParams = [self.weights, self.biases]
        
        #Define layer outputs.
        self.output, self.outputDropout = self.configureProcessing(
			self.input, self.inputDropout, self.weights, self.biases, self.miniBatchSize, self.dropoutAmount)



class FullyConnectedLayer(NeuralNetworkLayer):
	
	def configureProcessing(input, inputDropout, weights, biases, miniBatchSize, dropoutAmount):
        # TODO things go here
        return output, outputDropout
