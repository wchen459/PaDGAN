"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from run_experiment import read_config
from bezier_gan import BezierGAN
from simulation import evaluate

import sys
sys.path.append('..')
from evaluation import diversity_score, kl_score, nearest_dist_score


if __name__ == "__main__":
    
    list_lambdas0 = [1.0, 2.0, 3.0]
    list_lambdas1 = [0.1, 0.2, 0.3]
    n_runs = 10
    
    def train(lambda0, lambda1, n_runs):
        for i in range(n_runs):
            png_path = './trained_gan/{}_{}/{}/synthesized.png'.format(lambda0, lambda1, i)
            if not os.path.exists(png_path):
                os.system('python run_experiment.py train --lambda0={} --lambda1={} --id={}'.format(lambda0, lambda1, i))
            else:
                os.system('python run_experiment.py evaluate --lambda0={} --lambda1={} --id={}'.format(lambda0, lambda1, i))
    
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname)
    for lambda0 in list_lambdas0:
        lambda1 = lambda1_
        train(lambda0, lambda1, n_runs)
    for lambda1 in list_lambdas1:
        lambda0 = lambda0_
        train(lambda0, lambda1, n_runs)
                    
    ###############################################################################
    # Plot diversity and quality scores
    print('Plotting diversity and quality scores ...')
    plt.rcParams.update({'font.size': 20})
    
    # Read dataset
    data_fname = './data/xs_train.npy'
    X = np.load(data_fname)
    N = X.shape[0]
    
    bounds = (0., 1.)
    n = 1000 # generated sample size for each trained model
    subset_size = 10 # for computing DDP
    sample_times = 1000 # for computing DDP
    
    # Scores for training data
    ind = np.random.choice(N, size=n, replace=False)
    airfoils_data = X[ind]
    div_data = diversity_score(airfoils_data, subset_size, sample_times)
    quality_data = np.load('./data/ys_train.npy')[ind]
    qa_data = np.mean(quality_data)
    
    n0 = len(list_lambdas0)
    n1 = len(list_lambdas1)
    
    def read_scores(lambda0, lambda1):
            
        list_div = []
        list_qa = []
        list_kl = []
        list_nd = []
        for i in range(10):
            
            save_dir = './trained_gan/{}_{}/{}'.format(lambda0, lambda1, i)
            npy_path = save_dir+'/scores.npy'
            
            # if os.path.exists(npy_path):
            #     div, qa, kl, nd = np.load(npy_path)
                
            # else:
                
            # Generated airfoils
            # Quality
            ys_path = '{}/ys.npy'.format(save_dir)
            if not os.path.exists(ys_path):
                model = BezierGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, bounds, lambda0, lambda1)
                model.restore(directory=save_dir)
                airfoils_synth = model.synthesize(n)
                # Compute quality statistics
                print('Evaluating new airfoils ...')
                ys = []
                for i, airfoil in enumerate(airfoils_synth):
                    val = evaluate(airfoil, return_CL_CD=False)
                    ys.append(val)
                    print('{}/{}: {}'.format(i+1, n, val))
                ys = np.array(ys)
                np.save(ys_path, ys)
            else:
                ys = np.load(ys_path)
            ys[np.isnan(ys)] = 0.
            qa = np.mean(ys)
            # Diversity
            div = diversity_score(airfoils_synth, subset_size, sample_times)
            # KL divergence
            kl = kl_score(quality_data, ys)
            # Nearest distance score
            nd = nearest_dist_score(airfoils_data, airfoils_synth)
            np.save(npy_path, [div, qa, kl, nd])
                
            list_div.append(div)
            list_qa.append(qa)
            list_kl.append(kl)
            list_nd.append(nd)
                
        return list_div, list_qa, list_oa, list_kl
        
    list_div_lambda0 = []
    list_qa_lambda0 = []
    list_oa_lambda0 = []
    list_kl_lambda0 = []
    for lambda0 in list_lambdas0:
        lambda1 = lambda1_
        list_div, list_qa, list_oa, list_kl = read_scores(lambda0, lambda1)
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
        list_div, list_qa, list_oa, list_kl = read_scores(lambda0, lambda1)
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
    
    fig = plt.figure(figsize=(15, 5))
    subplot(fig, 141, list_div_lambda0, 'Diversity score', div_data, list_lambdas0, n0)
    subplot(fig, 142, list_qa_lambda0, 'Quality score', qa_data, list_lambdas0, n0)
    subplot(fig, 144, list_kl_lambda0, 'KL divergence', 0, list_lambdas0, n0)
    plt.tight_layout()
    plt.savefig('./trained_gan/scores_lambda0.svg')
    plt.savefig('./trained_gan/scores_lambda0.pdf')
    plt.savefig('./trained_gan/scores_lambda0.png')
    plt.close()
    
    fig = plt.figure(figsize=(15, 5))
    subplot(fig, 141, list_div_lambda1, 'Diversity score', div_data, list_lambdas1, n1)
    subplot(fig, 142, list_qa_lambda1, 'Quality score', qa_data, list_lambdas1, n1)
    subplot(fig, 144, list_kl_lambda1, 'KL divergence', 0, list_lambdas1, n1)
    plt.tight_layout()
    plt.savefig('./trained_gan/scores_lambda1.svg')
    plt.savefig('./trained_gan/scores_lambda1.pdf')
    plt.savefig('./trained_gan/scores_lambda1.png')
    plt.close()
                