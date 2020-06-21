"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import argparse
from run_experiment import read_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
#    list_data_function = [('ThinDonut2D', 'MixRing6'), ('Donut2D', 'MixRing1'), ('Grid2D', 'MixRing1'), ('Donut2D', 'MixRing6'), ('Grid2D', 'MixRing4')]
    list_data_function = [('Grid2D', 'MixRing4'), ('Donut2D', 'MixRing6'), ('ThinDonut2D', 'MixRing6')]
    list_lambdas = ['GAN', r'GAN$_D$', r'GAN$_Q$', 'PaDGAN', 'naive']
#    list_lambdas = ['GAN', 'PaDGAN']
    n_runs = 10
    
    for (dataset, function) in list_data_function:
        config_fname = 'config.ini'
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
        for lambdas in list_lambdas:
            lambda0, lambda1 = 0., 0.
            if lambdas == r'GAN$_D$':
                lambda0, lambda1 = 0., lambda1_
            if lambdas == r'GAN$_Q$':
                lambda0, lambda1 = 'inf', lambda1_
            elif lambdas == 'PaDGAN':
                lambda0, lambda1 = lambda0_, lambda1_
            elif lambdas == 'naive':
                lambda0, lambda1 = 'naive', lambda1_
            for i in range(n_runs):
                png_path = './trained_gan/{}_{}_GAN_{}_{}/{}/synthesized.png'.format(dataset, function, lambda0, lambda1, i)
                if not os.path.exists(png_path) or args.mode=='evaluate':
                    if lambda0 == 'naive':
                        os.system('python run_experiment.py {} {} {} --naive --lambda1={} --id={}'.format(
                                  args.mode, dataset, function, lambda1, i))
                    elif lambda0 == 'inf':
                        os.system('python run_experiment.py {} {} {} --inf --lambda1={} --id={}'.format(
                                  args.mode, dataset, function, lambda1, i))
                    else:
                        os.system('python run_experiment.py {} {} {} --lambda0={} --lambda1={} --id={}'.format(
                                  args.mode, dataset, function, lambda0, lambda1, i))
                