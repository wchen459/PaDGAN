"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import configparser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns

import datasets
import models
import functions
from visualization import visualize_2d

import sys
sys.path.append('..')
from evaluation import diversity_score, quality_score, overall_score, kl_score
from utils import create_dir, ElapsedTimer


def read_config(config_fname, example_name):
    
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    noise_dim = int(Config.get(example_name, 'noise_dim'))
    train_steps = int(Config.get(example_name, 'train_steps'))
    batch_size = int(Config.get(example_name, 'batch_size'))
    disc_lr = float(Config.get(example_name, 'disc_lr'))
    gen_lr = float(Config.get(example_name, 'gen_lr'))
    lambda0 = float(Config.get(example_name, 'lambda0'))
    lambda1 = float(Config.get(example_name, 'lambda1'))
    save_interval = int(Config.get(example_name, 'save_interval'))
    
    return noise_dim, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('data', type=str, default='Grid2D', help='dataset')
    parser.add_argument('func', type=str, default='MixRing1', help='function')
    parser.add_argument('--inf', help='maximize quality but not diversity', action='store_true')
    parser.add_argument('--naive', help='use naive loss for quality', action='store_true')
    parser.add_argument('--lambda0', type=float, default=None, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=None, help='lambda1')
    parser.add_argument('--disc_lr', type=float, default=None, help='learning rate for D')
    parser.add_argument('--gen_lr', type=float, default=None, help='learning rate for G')
    parser.add_argument('--id', type=int, default=None, help='experiment ID')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--train_steps', type=int, default=None, help='training steps')
    parser.add_argument('--save_interval', type=int, default=None, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Data
    N = 10000
    data_obj = getattr(datasets, args.data)(N)
    data = data_obj.data
    
    # Function
    func_obj = getattr(functions, args.func)()
    func = func_obj.evaluate
    
    # Values (>=0)
    values = func(data)
    val_scale = 1./np.max(values)
    
    # Hyperparameters for GAN
    config_fname = 'config.ini'
    example_name = '{}+{}'.format(args.data, args.func)
    noise_dim, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval = read_config(config_fname, example_name)
    if args.lambda0 is not None:
        lambda0 = args.lambda0
    if args.inf:
        lambda0 = 'inf'
    if args.naive:
        lambda0 = 'naive'
    if args.lambda1 is not None:
        lambda1 = args.lambda1
    if args.disc_lr is not None:
        disc_lr = args.disc_lr
    if args.gen_lr is not None:
        gen_lr = args.gen_lr
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.train_steps is not None:
        train_steps = args.train_steps
    if args.save_interval is not None:
        save_interval = args.save_interval
        
    print('#################################')
    print('# {}'.format(args.data))
    print('# {}'.format(args.func))
    print('# lambda0 = {}'.format(lambda0))
    print('# lambda1 = {}'.format(lambda1))
    print('# disc_lr = {}'.format(disc_lr))
    print('# gen_lr = {}'.format(gen_lr))
    print('# ID: {}'.format(args.id))
    print('#################################')
            
    # Prepare save directory
    create_dir('./trained_gan')
    save_dir = './trained_gan/{}_{}_GAN_{}_{}'.format(args.data, args.func, lambda0, lambda1)
    create_dir(save_dir)
    if args.id is not None:
        save_dir += '/{}'.format(args.id)
        create_dir(save_dir)
    
    # Visualize data
    visualize_2d(data, func=func, save_path='{}/data.svg'.format(save_dir), xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    visualize_2d(data, func=func, save_path='{}/data.png'.format(save_dir), xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    
    # Train
    model = getattr(models, 'GAN')(noise_dim, 2, lambda0, lambda1)
    if args.mode == 'train':
        timer = ElapsedTimer()
        model.train(data_obj, func_obj, val_scale, batch_size=batch_size, train_steps=train_steps, 
                    disc_lr=disc_lr, gen_lr=gen_lr, save_interval=save_interval, save_dir=save_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
    else:
        model.restore(save_dir=save_dir)
    
    print('##########################################################################')
    print('Plotting generated samples ...')
        
    # Plot generated samples
    n = 1000
    gen_data = model.synthesize(n)
    visualize_2d(data[:n], func=func, gen_data=gen_data, save_path='{}/synthesized.svg'.format(save_dir), 
                 xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    visualize_2d(data[:n], func=func, gen_data=gen_data, save_path='{}/synthesized.png'.format(save_dir), 
                 xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    
    print('##########################################################################')
    print('Evaluating generated samples ...')
    
    # Evaluate generated samples
    diversity_score = diversity_score(gen_data)
    quality_score = quality_score(gen_data, func_obj)
    overall_score = overall_score(gen_data, func_obj)
    kl_score = kl_score(data[:n], gen_data, func_obj)
    np.save('{}/scores.npy'.format(save_dir), [diversity_score, quality_score, overall_score, kl_score])
    with open('{}/evaluation.txt'.format(save_dir), 'w+') as f:
        f.write('{}, {}, {}'.format(diversity_score, quality_score, overall_score, kl_score))
    
    print('##########################################################################')
    print('Plotting quality distribution ...')
    
    # Plot quality distribution
    n = 1000
    ind = np.random.choice(N, size=n)
    y_data = func(data[ind])
    y_gen = func(gen_data)
    
    plt.figure()
    sns.kdeplot(y_data, color='g', shade=True, Label='Data')
    sns.kdeplot(y_gen, color='b', shade=True, Label='Generated')
    plt.xlabel('CL/CD') 
    plt.ylabel('Probability density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/quality_dist.svg'.format(save_dir))
    plt.savefig('{}/quality_dist.png'.format(save_dir))
    plt.close()
    
    print('Completed!')
