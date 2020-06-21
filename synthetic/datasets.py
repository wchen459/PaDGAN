"""
Generates sparse data

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np

import functions
from visualization import visualize_2d

import sys
sys.path.append('..')
from utils import gen_grid


class SparseGrid2D(object):
    
    def __init__(self, N, lb=-0.5, rb=0.5, perturb=0.02):
        self.name = 'SparseGrid2D'
        points_per_axis = int(N**0.5)
        data = gen_grid(2, points_per_axis, lb=lb, rb=rb)
        # Add noise
        data += np.random.normal(loc=np.zeros_like(data), scale=perturb, size=data.shape)
        self.data = data
        
    
class Ring2D(object):
    
    def __init__(self, N, n_mixture=8, std=0.01, radius=0.5):
        """Gnerate 2D Ring"""
        thetas = np.linspace(0, 2 * np.pi, n_mixture, endpoint=False)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        centers = np.vstack((xs, ys)).T
        data = []
        for i in range(N):
            data.append(np.random.normal(centers[np.random.choice(n_mixture)], std))
        self.data = np.array(data)
        

class Grid2D(object):
    
    def __init__(self, N, n_mixture=9, std=0.01):
        """Generate 2D Grid"""
        centers = SparseGrid2D(n_mixture, lb=-0.4, rb=0.4, perturb=0.).data
        data = []
        for i in range(N):
            data.append(np.random.normal(centers[np.random.choice(centers.shape[0])], std))
        self.data = np.array(data)
        
    
class Donut2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D donut"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
                if norm_x <= 0.5 and norm_x >= 0.25:
                    data.append(x)
                    break
        self.data = np.array(data)
        
    
class ThinDonut2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D donut"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
#                if norm_x <= 0.425 and norm_x >= 0.375:
                if norm_x <= 0.375 and norm_x >= 0.325:
                    data.append(x)
                    break
        self.data = np.array(data)
        
    
class Arc2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D arc"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
                theta = np.arctan2(x[1], x[0])
                if norm_x <= 0.5 and norm_x >= 0.25 and not (theta > -11./12*np.pi and theta < 0):
                    data.append(x)
                    break
        self.data = np.array(data)
    

if __name__ == "__main__":
    
    N = 1000
    data_obj = Donut2D(N)
    data = data_obj.data
    func_obj = functions.MultiModal()
    func = func_obj.evaluate
    print(func)
    visualize_2d(data, func)
    
