import numpy as np 
import pandas as pd 
import matplotlib as mlp 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading dataset
def load_data():
	dataset = load_breast_cancer()

	#Exploring dataset
	print("Features : ", dataset.feature_names)
	print("Target Classes : ",dataset.target_names)

	x = dataset.data
	y = dataset.target
	classes = dataset.target_names
	size = dataset.data.shape
	
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
	return x_train,y_train,x_test,y_test,classes

x_train,y_train,x_test,y_test,classes = load_data()

# Parameters
input_ = x_train
target_ = y_train
num_hidden_layers = 3
num_input_dim = input_.shape[1:]
num_target_dim = 1  # binary classifier
print_loss_ = True
epochs = 80000 # forward pass + backward pass
epsilon = 0.003 # learning rate for gradient descent
reg_lambda = 0.001 # regularization strength 
layer_dims = [30,20,10,5,num_target_dim] # neurons in each layer of the network. Input layer followed by 3 hidden layers and one target layer

def initialize_weights_n_biases(layer_dims):
	# Initialize the parameters to random values. We need to learn these.
	parameters = {}
	L = len(layer_dims)
	for l in range(1,L):
		#weights
		parameters['W'+str(l)] = np.random.randn(layer_dims[l-1],layer_dims[l]) / np.sqrt(layer_dims[l-1])
		#biases
		parameters['b'+str(l)] = np.zeros((1,layer_dims[l]))

	return parameters

def linear_forward(z, w, b):
	Z = np.dot(z,w) + b
	cache = (z,w,b)
	return Z,cache

def activation_function(activation, z):
	if activation == "relu":
		a = np.maximum(0,z)
		cache = z
	elif activation == "sigmoid":
		a = 1/(1+np.exp(-1*z))
		cache = z
	return a,cache
	
def forward_prop(X, params):
	caches = []
	A = X
	L = len(params) // 2
	for l in range(1,L):
		Z_prev = A
		Z,linear_cache = linear_forward(Z_prev,params['W'+str(l)],params['b'+str(l)])
		A,activation_cache = activation_function("relu",Z)
		cache = (linear_cache,activation_cache)
		caches.append(cache)

	Z_,linear_cache = linear_forward(A,params['W'+str(L)],params['b'+str(L)])
	A_,activation_cache = activation_function("sigmoid",Z_)
	cache = (linear_cache,activation_cache)
	caches.append(cache)
	return A_,caches

def compute_loss(loss_func, A, Y,i):
	m = Y.shape[0]
	A = A.flatten()
	if loss_func == "binary_crossentropy":
		cost = (-(Y*np.log(A))-((1-Y)*np.log(1-A)))
		cost = np.sum(cost,axis=0,keepdims = 1)/m 
		cost = np.squeeze(cost)
	return cost

def intermediate_calc(dEA, activation, cache):
	if activation == "relu":
		z = cache
		dAZ = 1
		dEZ = np.array(dEA,copy=True)
		dEZ[z<=0] = 0
	elif activation == "sigmoid":
		z = cache
		tmp = 1/(1+np.exp(-z))
		dAZ = tmp*(1-tmp) # differentiation of output w.r.t input dA/dZ
		# multiply dE/dA * dA/dZ 
		dEZ = dEA*dAZ
	return dEZ

def error_rate_calc(dEZ, cache):
	z,w,b = cache
	m = z.shape[0]
	dZW = z.T # rate of change of input w.r.t weight, dZ/dW = z
	dEW = np.dot(dZW,dEZ)/m # rate of change of error w.r.t weight, dE/dW = dE/dZ * dZ/dW
	dEb = np.sum(dEZ,axis=0,keepdims=1)/m # rate of change of error w.r.t bias, dE/db = dE/dZ * dZ/db
	dA_prev = np.dot(dEZ,w.T) # error propagated backward
	return dA_prev,dEW, dEb

def linear_backward(dEA, cache, activation):
	linear_cache, activation_cache = cache
	dEZ = intermediate_calc(dEA,activation,activation_cache) # dE/dZ
	dA_prev,dW, db = error_rate_calc(dEZ, linear_cache)
	return dA_prev,dW,db

def backward_prop(A,Y,caches):
	grads = {}
	L =len(caches)
	Y = Y.reshape(A.shape)
	dEA = -(np.divide(Y,A)-np.divide(1-Y,1-A)) if (Y != A).any() else 0.0 # differentiation of error w.r.t output dE/dA

	current_cache = caches[L-1]
	grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)] = linear_backward(dEA,current_cache,"sigmoid")
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		grads['dA'+str(l)],grads['dW'+str(l+1)],grads['db'+str(l+1)] = linear_backward(grads['dA'+str(l+1)],current_cache,"relu")

	return grads

def update_parameters(parameters, grads):
	L = len(parameters) // 2
	for l in range(L):
		parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - epsilon * grads['dW'+str(l+1)] # W = W - lr*(dE/dW)
		parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - epsilon * grads['db'+str(l+1)] # b = b - lr*(dE/db)
	return parameters

# Build Sequential model
def build_sequential_model(X, Y, layer_dims, print_loss = False):
	np.random.seed(0)
	params = initialize_weights_n_biases(layer_dims)

	costs = []

	for i in range(0,epochs):

		# Forward Propagation
		A,caches = forward_prop(X, params)
		# Error Calculation 
		cost = compute_loss("binary_crossentropy",A,Y,i)
		# Backward Propagation
		grads = backward_prop(A, Y, caches)
		# Update parameters
		params = update_parameters(params,grads)
		costs.append(cost)

		if i==0 :
			least_cost,j = cost,i
			model = params.copy()
		elif cost < least_cost:
			least_cost,j = cost,i
			model = params.copy()

		if(print_loss == True and i%5000 == 0):
			print("cost at iteration {} is {}".format(i,cost))

	print("Best Model is obtained at {} th iteration with cost as {}".format(j,least_cost))
	return model

model = build_sequential_model(input_, target_, layer_dims, print_loss_)

def predict(X,Y,model):
	m = X.shape[0]
	res = np.zeros(m)
	probabs, caches = forward_prop(X,model)
	for i in range(0,probabs.shape[0]):
		if probabs[i][0] > 0.5:
			res[i] = 1
		else:
			res[i] = 0
	print("Accuracy: "+str(np.sum(res == Y)/m))
	return res

pred_train = predict(x_train,y_train,model)
pred_test = predict(x_test,y_test,model)
