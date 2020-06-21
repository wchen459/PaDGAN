""" 
Test functions

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')
from utils import gen_grid


class Function(object):
    
    def __init__(self):
        pass
    
    def equation(self, x):
        y = 0.
        for i in range(self.n_modes):
            y += tf.exp(-.5*tf.reduce_sum(tf.square(x-self.centers[i]), axis=1)/self.sigma**2)
        return y
    
    def evaluate(self, data):
        x = tf.placeholder(tf.float32, shape=[None, self.dim])
        y = self.equation(x)
        with tf.Session() as sess:
            values = sess.run(y, feed_dict={x: data})
        return values
    
    def entropy(self, data):
        assert hasattr(self, 'sigma')
        N = data.shape[0]
        counts = []
        for center in self.centers:
            distances = np.linalg.norm(data-center, axis=1)
            count = sum(distances<self.sigma)
            counts.append(count)
        counts = np.array(counts)
        rates = counts/N
        entropy = -np.sum(rates*np.log(rates+1e-8))
        return entropy


class Linear(Function):
    
    def __init__(self):
        self.name = 'Linear'
        self.dim = 2
        
    def equation(self, x):
        y = tf.reduce_sum(x, axis=1)
        return y


class MixGrid(Function):
    
    def __init__(self, n_modes=9, lb=-0.5, rb=0.5):
        self.name = 'MixGrid'
        self.dim = 2
        self.n_modes = n_modes
        
        points_per_axis = int(n_modes**0.5)
        self.sigma = (rb-lb)/points_per_axis/4
        
        self.centers = gen_grid(2, points_per_axis, lb=lb+2*self.sigma, rb=rb-2*self.sigma)
    

class MixRing(Function):
    
    def __init__(self, n_modes=4, radius=0.4):
        self.name = 'MixRing'
        self.dim = 2
        self.n_modes = n_modes
        self.radius = radius
        
        thetas = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.centers = np.vstack((xs, ys)).T
        
        self.sigma = np.pi*self.radius/self.n_modes/2
        

class MixRing1(MixRing):
    
    def __init__(self):
        super(MixRing1, self).__init__(n_modes=1)
        self.name = 'MixRing1'
        

class MixRing4(MixRing):
    
    def __init__(self):
        super(MixRing4, self).__init__(n_modes=4)
        self.name = 'MixRing4'
        

class MixRing6(MixRing):
    
    def __init__(self):
        super(MixRing6, self).__init__(n_modes=6)
        self.name = 'MixRing6'
        
