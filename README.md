# insanity
The simple, no-nonsense machine learning library for Python, intended to be a minimal alternative to libraries such as Caffe and Torch7.  
<br>
##Fast
* Based on [Theano](http://deeplearning.net/software/theano/) and [NumPy](http://www.numpy.org/).
* Can be run on a CPU or an NVIDIA GPU using CUDA.

##Portable
* Supports embedded ARM-based systems such as NVIDIA's [Jetson TK1](http://elinux.org/Jetson_TK1).

##Powerful
* Provides an API for neural networks with several layer types.
  * Fully-connected
  * Convolutional pooling
  * Softmax
* Provides several neuron activation functions.
  * Linear
  * Sigmoid
  * Tanh
  * Rectified linear unit
* Implements stochastic gradient descent for training neural networks.
* Provides methods for serializing and un-serializing networks using JSON.
