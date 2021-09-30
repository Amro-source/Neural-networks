from scipy import signal
from scipy import misc
import numpy as np
from numpy import zeros

num_layers = 10
 
class layer(object):
    def __init__(self, _m, _n):
        #n: filter size (width)
        #m: filter size (height)
        self.m = _m
        self.n = _n
        self.activation = 'relu'
        #self.activation = 'tanh'
        #self.activation = 'sigmoid'
 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
 
    def forward(self, x, use_bias = False):
       #x is a row vector
        self.W = np.random.normal(0, np.sqrt(2.0/self.m), (self.m,self.n))
        self.b = np.random.normal(0, np.sqrt(2.0/num_layers), self.n)
       # self.b = 1. - 2*np.random.rand(1, 5)
        self.y = np.dot(x, self.W)
        if (use_bias):
            self.y = self.y + self.b
        if (self.activation == 'relu'):
            self.a = np.maximum(0., self.y)
        if (self.activation == 'tanh'):
            self.a = np.tanh(self.y)
        if (self.activation == 'sigmoid'):
            self.a = self.sigmoid(self.y)
        return self.a, self.y
 
 
layers = []
# even numbered layers have a 5*10 weight matrix
# odd numbered layers have a 10*5 weight matrix
for i in range(num_layers):
    layers.append(layer(5 if(i % 2 == 0) else 10, 10 if(i % 2 == 0) else 5))
 
num_trials = 100000
# records the network output (activations of the last layer)
a = np.zeros((num_trials, 5))
# records the network input
i = np.zeros((num_trials, 5))
# record the activations
y = np.zeros((num_trials, 5))
for trial in range(0,num_trials):
    # input to the network is uniformly distributed numbers in (0,1). E(x) != 0.
    # Note that the distribution of the input is different from the distribution of the weights.
    x = 3*np.random.rand(1, 5)
    i[trial, :] = x
    for layer_no in range(0,num_layers):
        x, y_ = layers[layer_no].forward(x, False)
    a[trial, :] = x
    y[trial, :] = y_
 
#E(x^2) (expected value of the square of the input)
E_x2 = np.mean(np.multiply(i,i), 0)
 
# E(a^2) (expected value of the square of the activations of the last layer)
E_a2 = np.mean(np.multiply(a,a), 0)
# verify E_a2 ~ E_x2
 
# var(y): Variance of the output before applying activation function
Var_y = np.var(y,0)
# verify Var_y ~ 2*E_a2