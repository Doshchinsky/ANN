# coding: utf-8
import numpy as np

def sigmoid(x, derivate = False):
	if(derivate == True):
		   return x * (1 - x)

	return 1 / (1 + np.exp(-x))
	
X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])
				
y = np.array([[0],
			  [1],
			  [1],
			  [0]])

np.random.seed(1)

syn_weight0 = 2 * np.random.random((3, 4)) - 1
syn_weight1 = 2 * np.random.random((4, 1)) - 1

for i in xrange(100000):

	input_layer = X
	hidden_layer = sigmoid(np.dot(input_layer, syn_weight0))
	output_layer = sigmoid(np.dot(hidden_layer, syn_weight1))

	output_layer_error = y - output_layer
	
	if (i % 10000) == 0:
		print "Error:" + str(np.mean(np.abs(output_layer_error)))

	output_layer_delta = output_layer_error * sigmoid(output_layer, derivate = True)

	hidden_layer_error = output_layer_delta.dot(syn_weight1.T)

	hidden_layer_delta = hidden_layer_error * sigmoid(hidden_layer, derivate = True)

	syn_weight1 += hidden_layer.T.dot(output_layer_delta)
	syn_weight0 += input_layer.T.dot(hidden_layer_delta)

#print(output_layer)