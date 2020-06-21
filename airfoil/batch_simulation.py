"""
Airfoil batch simulation

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data.gan import GAN
from simulation import evaluate


if __name__ == "__main__":
    
    xs_train_path = './data/xs_train.npy'
    xs_test_path = './data/xs_test.npy'
    ys_train_path = './data/ys_train.npy'
    ys_test_path = './data/ys_test.npy'
    
    data_exist = os.path.exists(xs_train_path) and \
                os.path.exists(xs_test_path) and \
                os.path.exists(ys_train_path) and \
                os.path.exists(ys_test_path)
                
    if data_exist:
        xs_train = np.load(xs_train_path)
        xs_test = np.load(xs_test_path)
        ys_train = np.load(ys_train_path)
        ys_test = np.load(ys_test_path)
        xs = np.concatenate((xs_train, xs_test), axis=0)
        ys = np.concatenate((ys_train, ys_test), axis=0)
    
    uiuc_airfoils = np.load('./data/airfoil_interp.npy')
    
    latent_dim = 5
    noise_dim = 10
    bezier_degree = 31
    n_points = uiuc_airfoils.shape[1]
    bounds = (0., 1.)
    
    if data_exist:
        new_xs = np.zeros((0, n_points, 2))
    else:
        new_xs = uiuc_airfoils
    for i in range(10):
        print('Model ID: {}'.format(i))
        model = GAN(latent_dim, noise_dim, n_points, bezier_degree, bounds)
        model.restore(directory='./data/trained_gan/{}'.format(i))
        gan_airfoils = model.synthesize(4000)
        new_xs = np.append(new_xs, gan_airfoils, axis=0)
        
    N = new_xs.shape[0]
    
    if not data_exist:
        xs = np.zeros((0, n_points, 2))
        ys = np.zeros(0)
    for i, airfoil in enumerate(new_xs):
        val = evaluate(airfoil, return_CL_CD=False, config_fname='./op_conditions.ini', tmp_dir='./tmp')
        if not np.isnan(val):
            xs = np.append(xs, np.expand_dims(airfoil, axis=0), axis=0)
            ys = np.append(ys, val)
        print('{}/{}: {}'.format(i+1, N, val))
        
    # Split training and test data
    test_split = 0.8
    N = len(ys)
    split = int(N*test_split)
    xs_train = xs[:split]
    ys_train = ys[:split]
    xs_test = xs[split:]
    ys_test = ys[split:]
        
    np.save(xs_train_path, np.array(xs_train))
    np.save(xs_test_path, np.array(xs_test))
    np.save(ys_train_path, ys_train)
    np.save(ys_test_path, ys_test)
    
    xs_train = np.load(xs_train_path)
    xs_test = np.load(xs_test_path)
    ys_train = np.load(ys_train_path)
    ys_test = np.load(ys_test_path)
    
    plt.figure()
    sns.kdeplot(ys_train, color='g', shade=True, Label='Training data')
    sns.kdeplot(ys_test, color='b', shade=True, Label='Test data')
    plt.xlabel('CL/CD') 
    plt.ylabel('Probability density')
    plt.scatter(ys_train, np.zeros(ys_train.shape[0]), color='g', alpha=.5)
    plt.scatter(ys_test, np.zeros(ys_test.shape[0]), color='b', alpha=.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data/data.svg')
    plt.close()