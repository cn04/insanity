import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

def activationLinear(value):
	return value
	
def activationRectified(value):
	return T.maximum(0,0, value)
	
def activationSigmoid(value):
	return sigmoid(value)

def activationTanh(value):
	return tanh(value)
