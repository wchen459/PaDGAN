
"""
Trains a BezierGAN, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns

from surrogate.surrogate_model import Model as SM
from bezier_gan import BezierGAN
from shape_plot import plot_samples

import sys
sys.path.append('..')
from utils import ElapsedTimer, create_dir, safe_remove


def read_config(config_fname):
    
    example_name = 'Airfoil'
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    latent_dim = int(Config.get(example_name, 'latent_dim'))
    noise_dim = int(Config.get(example_name, 'noise_dim'))
    bezier_degree = int(Config.get(example_name, 'bezier_degree'))
    train_steps = int(Config.get(example_name, 'train_steps'))
    batch_size = int(Config.get(example_name, 'batch_size'))
    disc_lr = float(Config.get(example_name, 'disc_lr'))
    gen_lr = float(Config.get(example_name, 'gen_lr'))
    lambda0 = float(Config.get(example_name, 'lambda0'))
    lambda1 = float(Config.get(example_name, 'lambda1'))
    save_interval = int(Config.get(example_name, 'save_interval'))
    
    return latent_dim, noise_dim, bezier_degree, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('--naive', help='use naive loss for quality', action='store_true')
    parser.add_argument('--lambda0', type=float, default=None, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=None, help='lambda1')
    parser.add_argument('--id', type=int, default=None, help='experiment ID')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval = read_config(config_fname)
    bounds = (0., 1.)
    
    if args.lambda0 is not None:
        lambda0 = args.lambda0
    if args.naive:
        lambda0 = 'naive'
    if args.lambda1 is not None:
        lambda1 = args.lambda1
        
    print('#################################')
    print('# Airfoil')
    print('# lambda0 = {}'.format(lambda0))
    print('# lambda1 = {}'.format(lambda1))
    print('# disc_lr = {}'.format(disc_lr))
    print('# gen_lr = {}'.format(gen_lr))
    print('# ID: {}'.format(args.id))
    print('#################################')
    
    # Read dataset
    data_fname = './data/xs_train.npy'
    X = np.load(data_fname)
    N = X.shape[0]
    
    # Prepare save directory
    create_dir('./trained_gan')
    create_dir('./trained_gan/{}_{}'.format(lambda0, lambda1))
    save_dir = './trained_gan/{}_{}/{}'.format(lambda0, lambda1, args.id)
    create_dir(save_dir)
    
#    print('Plotting training samples ...')
#    samples = X[np.random.choice(N, size=36, replace=False)]
#    plot_samples(None, samples, scale=1.0, scatter=False, lw=1.2, alpha=.7, c='k', fname='{}/samples'.format(save_dir))
    
    # Train
    surrogate_dir = './surrogate/trained_surrogate'
    model = BezierGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, bounds, lambda0, lambda1)
    if args.mode == 'train':
        safe_remove(save_dir)
        timer = ElapsedTimer()
        model.train(X, batch_size=batch_size, train_steps=train_steps, disc_lr=disc_lr, gen_lr=gen_lr, 
                    save_interval=save_interval, directory=save_dir, surrogate_dir=surrogate_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('{}/runtime.txt'.format(save_dir), 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore(directory=save_dir)
    
    print('Plotting synthesized shapes ...')
    airfoils = model.synthesize(36)
    plot_samples(None, airfoils, scale=1.0, scatter=False, lw=1.2, alpha=.7, c='k', fname='{}/synthesized'.format(save_dir))
    
    # Plot quality distribution
    n = 1000
    ind = np.random.choice(X.shape[0], size=n)
    airfoils_data = np.squeeze(X[ind])
    airfoils_gen = model.synthesize(n)
    with tf.Session() as sess:
        surrogate_model = SM(sess, X.shape[1])
        surrogate_model.restore(directory=surrogate_dir)
        quality_data = surrogate_model.predict(airfoils_data)
        quality_gen = surrogate_model.predict(airfoils_gen)
    
    plt.figure()
    sns.kdeplot(quality_data, color='g', shade=True, Label='Data')
    sns.kdeplot(quality_gen, color='b', shade=True, Label='Generated')
    plt.xlabel('CL/CD') 
    plt.ylabel('Probability density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/quality_dist.svg'.format(save_dir))
    plt.savefig('{}/quality_dist.png'.format(save_dir))
    plt.close()
    
    
