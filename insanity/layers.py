import numpy as np
import theano
import theano.tensor as T



class NetworkLayer(object):

	def __init__(self, numInputs, numNeurons, activation, miniBatchSize, dropout=0.0):
		self.numInputs = numInputs
		self.numNeurons = numNeurons
		self.activation = activation
		self.dropout = dropout
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
        
        #These are overrided by subclasses to add layer functionality. 
        
	@input.setter
	def input(self, value):
		self.input = value
		self.output = value
		
    	@inputDropout.setter
    	def inputDropout(self, value):
    		self.inputDropout = value
    		self.outputDropout = value



class FullyConnectedLayer(NetworkLayer):
	
	@NetworkLayer.input.setter
	def input(self, value):
		self.input = value
		#Configure the layer's complete neuron inputs.
		neuronInputs = value.reshape((self.miniBatchSize, self.numInputs))
		#Perform the main weight-bias layer output computation.
		self.output = self.activation((1 - self.dropout) * T.dot(neuronInputs, self.weights) + self.biases)
