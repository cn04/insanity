import numpy as np
import theano
import theano.tensor as T


class NeuralNetwork(object):

    def __init__(self, layers, miniBatchSize):
		self.miniBatchSize = miniBatchSize
		#Initialize layers.
		self.layers = layers
		self.numLayers = len(self.layers)
		self.firstLayer = self.layers[0]
		self.lastLayer = self.layers[-1]
		#Populate self.learningParams with a complete list of weights and biases from all layers.
		self.learningParams = []
		for layer in self.layers:
			for param in layer.learningParams:
				self.learningParams.append(param)
		#Connect each layer's input to the previous layer's output.
		for i in xrange(1, self.numLayers):
			nextLayer = layers[i]
			previousLayer = layers[i-1]
			nextLayer.configureInput(previousLayer.output, previousLayer.outputDropout, self.miniBatchSize)