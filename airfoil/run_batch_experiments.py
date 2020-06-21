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
    
    list_models = ['GAN', 'PaDGAN']
    
    n_runs = 10
    config_fname = 'config.ini'
    _, _, _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname)
    
    for model_name in list_models:
        if model_name == 'GAN':
            lambda0, lambda1 = 0., 0.
        else:
            lambda0, lambda1 = lambda0_, lambda1_
        for i in range(n_runs):
            png_path = './trained_gan/{}_{}/{}/synthesized.png'.format(lambda0, lambda1, i)
            if not os.path.exists(png_path) or args.mode=='evaluate':
                os.system('python run_experiment.py {} --lambda0={} --lambda1={} --id={}'.format(args.mode, lambda0, lambda1, i))
                