# insanity
The simple, no-nonsense machine learning library for Python, intended to be a minimal alternative to libraries such as Caffe and Torch7.  
<br>

##Fast
* Based on [Theano](http://deeplearning.net/software/theano/) and [NumPy](http://www.numpy.org/).
* Can be accelerated using CUDA or OpenCL.

##Portable
* Can run on any computing platform.
 * Traditional CPUs
 * NVIDIA GPUs using CUDA
 * Other processing units using OpenCL
* Supports embedded ARM-based systems.  

##Powerful
* Provides an API for neural networks with several layer types.
  * Fully-connected
  * Convolutional
  * Max-pooling
  * Softmax
* Provides several neuron activation functions.
  * Linear
  * Sigmoid
  * Tanh
  * Rectified linear unit
* Implements stochastic gradient descent for training neural networks.
* Provides a dropout function for increasing the reliability of network layers.
* Provides methods for serializing and un-serializing networks using JSON.
