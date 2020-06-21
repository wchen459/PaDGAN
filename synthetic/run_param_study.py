"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import datasets
import functions
from run_experiment import read_config

import sys
sys.path.append('..')
from evaluation import diversity_score, quality_score, overall_score


if __name__ == "__main__":
    
#    list_data_function = [('ThinDonut2D', 'MixRing6'), ('Donut2D', 'MixRing1'), ('Grid2D', 'MixRing1'), ('Donut2D', 'MixRing6'), ('Grid2D', 'MixRing4')]
    list_data_function = [('Grid2D', 'MixRing4'), ('Donut2D', 'MixRing6'), ('ThinDonut2D', 'MixRing6')]
    list_lambdas0 = [1.0, 2.0, 5.0]
    list_lambdas1 = [0.2, 0.5, 1.5, 5.0]
    n_runs = 10
    
    def train(dataset, function, lambda0, lambda1, n_runs):
        for i in range(n_runs):
            png_path = './trained_gan/{}_{}_GAN_{}_{}/{}/synthesized.png'.format(dataset, function, lambda0, lambda1, i)
            if not os.path.exists(png_path):
                os.system('python run_experiment.py train {} {} --lambda0={} --lambda1={} --id={}'.format(
                          dataset, function, lambda0, lambda1, i))
            else:
                os.system('python run_experiment.py evaluate {} {} --lambda0={} --lambda1={} --id={}'.format(
                          dataset, function, lambda0, lambda1, i))
    
    for (dataset, function) in list_data_function:
        config_fname = 'config.ini'
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
        for lambda0 in list_lambdas0:
            lambda1 = lambda1_
            train(dataset, function, lambda0, lambda1, n_runs)
        for lambda1 in list_lambdas1:
            lambda0 = lambda0_
            train(dataset, function, lambda0, lambda1, n_runs)
                    
    ###############################################################################
    # Plot diversity and quality scores
    print('Plotting diversity and quality scores ...')
    plt.rcParams.update({'font.size': 20})
    
    N = 1000 # generated sample size for each trained model
    
    n0 = len(list_lambdas0)
    n1 = len(list_lambdas1)
    
    def read_scores(dataset, function, lambda0, lambda1):
    
        example_dir = './trained_gan/{}_{}_GAN_{}_{}'.format(dataset, function, lambda0, lambda1)
            
        list_div = []
        list_qa = []
        list_oa = []
        list_kl = []
        for i in range(10):
            directory = example_dir+'/{}'.format(i)
            npy_path = directory+'/scores.npy'
            div, qa, oa, kl = np.load(npy_path)
            list_div.append(div)
            list_qa.append(qa)
            list_oa.append(oa)
            list_kl.append(kl)
        
        return list_div, list_qa, list_oa, list_kl

    for (dataset, function) in list_data_function:
        
        config_fname = 'config.ini'
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
        
        # Data
        data_obj = getattr(datasets, dataset)(N)
        data = data_obj.data
        # Function
        func_obj = getattr(functions, function)()
        
        div_data = diversity_score(data)
        qa_data = quality_score(data, func_obj)
        oa_data = overall_score(data, func_obj)
            
        list_div_lambda0 = []
        list_qa_lambda0 = []
        list_oa_lambda0 = []
        list_kl_lambda0 = []
        for lambda0 in list_lambdas0:
            lambda1 = lambda1_
            list_div, list_qa, list_oa, list_kl = read_scores(dataset, function, lambda0, lambda1)
            
            list_div_lambda0.append(list_div)
            list_qa_lambda0.append(list_qa)
            list_oa_lambda0.append(list_oa)
            list_kl_lambda0.append(list_kl)
            
        list_div_lambda1 = []
        list_qa_lambda1 = []
        list_oa_lambda1 = []
        list_kl_lambda1 = []
        for lambda1 in list_lambdas1:
            lambda0 = lambda0_
            list_div, list_qa, list_oa, list_kl = read_scores(dataset, function, lambda0, lambda1)
            
            list_div_lambda1.append(list_div)
            list_qa_lambda1.append(list_qa)
            list_oa_lambda1.append(list_oa)
            list_kl_lambda1.append(list_kl)
            
        def subplot(fig, position, values, title, base_val, list_x, n):
            ax = fig.add_subplot(position)
            ax.set_title(title)
            ax.boxplot(values, 0, '')
            ax.hlines(base_val, 0, 1, transform=ax.get_yaxis_transform(), colors='r')
            ax.set_xlim(0.5, n + 0.5)
            ax.set_xticklabels(list_x)
        
        fig = plt.figure(figsize=(20, 5))
        subplot(fig, 141, list_div_lambda0, 'Diversity score', div_data, list_lambdas0, n0)
        subplot(fig, 142, list_qa_lambda0, 'Quality score', qa_data, list_lambdas0, n0)
        subplot(fig, 143, list_oa_lambda0, 'Overall score', qa_data, list_lambdas0, n0)
        subplot(fig, 144, list_kl_lambda0, 'KL divergence', 0, list_lambdas0, n0)
        plt.tight_layout()
        plt.savefig('./trained_gan/{}_{}_scores_lambda0.svg'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores_lambda0.pdf'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores_lambda0.png'.format(dataset, function))
        plt.close()
        
        fig = plt.figure(figsize=(20, 5))
        subplot(fig, 141, list_div_lambda1, 'Diversity score', div_data, list_lambdas1, n1)
        subplot(fig, 142, list_qa_lambda1, 'Quality score', qa_data, list_lambdas1, n1)
        subplot(fig, 143, list_oa_lambda1, 'Overall score', qa_data, list_lambdas1, n1)
        subplot(fig, 144, list_kl_lambda1, 'KL divergence', 0, list_lambdas1, n1)
        plt.tight_layout()
        plt.savefig('./trained_gan/{}_{}_scores_lambda1.svg'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores_lambda1.pdf'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores_lambda1.png'.format(dataset, function))
        plt.close()
                