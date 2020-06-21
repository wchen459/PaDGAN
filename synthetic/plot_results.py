""" 
Plot results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

import datasets
import functions
from models import GAN
from run_experiment import read_config
from visualization import plot_data, plot_density

sys.path.append('..')
from evaluation import diversity_score, quality_score, overall_score


if __name__ == "__main__":
    
#    list_data_function = [('ThinDonut2D', 'MixRing6'), ('Donut2D', 'MixRing1'), ('Grid2D', 'MixRing1'), ('Donut2D', 'MixRing6'), ('Grid2D', 'MixRing4')]
    list_data_function = [('Grid2D', 'MixRing4'), ('Donut2D', 'MixRing6'), ('ThinDonut2D', 'MixRing6')]
#    list_lambdas = ['GAN', r'GAN$_D$', r'GAN$_Q$', 'PaDGAN', 'naive']
    list_lambdas = ['GAN', r'GAN$_D$', r'GAN$_Q$', 'PaDGAN']
    
    m = len(list_data_function)
    n = len(list_lambdas)
    
    config_fname = 'config.ini'
    n_runs = 10
    N = 1000 # generated sample size for each trained model
    subset_size = 10 # for computing DDP
    sample_times = 1000 # for computing DDP
    
    ###############################################################################
    # Plot diversity and quality scores
    print('Plotting diversity and quality scores ...')
    plt.rcParams.update({'font.size': 20})
    
    for (dataset, function) in list_data_function:
        
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
        
        # Data
        data_obj = getattr(datasets, dataset)(N)
        data = data_obj.data
        # Function
        func_obj = getattr(functions, function)()
        func = func_obj.evaluate
        
        div_data = diversity_score(data, subset_size, sample_times)
        qa_data = quality_score(data, func_obj)
        oa_data = overall_score(data, func_obj)
            
        list_div_lambdas = []
        list_qa_lambdas = []
        list_oa_lambdas = []
        for lambdas in list_lambdas:
            lambda0, lambda1 = 0., 0.
            if lambdas == r'GAN$_D$':
                lambda0, lambda1 = 0., lambda1_
            elif lambdas == r'GAN$_Q$':
                lambda0, lambda1 = 'inf', lambda1_
            elif lambdas == 'PaDGAN':
                lambda0, lambda1 = lambda0_, lambda1_
            elif lambdas == 'naive':
                lambda0, lambda1 = 'naive', lambda1_
            example_dir = './trained_gan/{}_{}_GAN_{}_{}'.format(dataset, function, lambda0, lambda1)
            
            list_div = []
            list_qa = []
            list_oa = []
            for i in range(10):
                directory = example_dir+'/{}'.format(i)
                npy_path = directory+'/scores.npy'
                if os.path.exists(npy_path):
                    div, qa, oa = np.load(npy_path)
                else:
                    # Generated data
                    model = GAN(2, 2, lambda0, lambda1)
                    model.restore(save_dir=directory)
                    gen_data = model.synthesize(N)
                    # Compute metrics
                    div = diversity_score(gen_data, subset_size, sample_times)
                    qa = quality_score(gen_data, func_obj)
                    oa = overall_score(gen_data, func_obj)
                    np.save(npy_path, [div, qa, oa])
                list_div.append(div)
                list_qa.append(qa)
                list_oa.append(oa)
            list_div_lambdas.append(list_div)
            list_qa_lambdas.append(list_qa)
            list_oa_lambdas.append(list_oa)
            
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax1.set_title('Diversity score')
        ax1.boxplot(list_div_lambdas, 0, '')
        ax1.hlines(div_data, 0, 1, transform=ax1.get_yaxis_transform(), colors='r')
        ax1.set_xlim(0.5, n + 0.5)
        ax1.set_xticklabels(list_lambdas)
        ax2 = fig.add_subplot(132)
        ax2.set_title('Quality score')
        ax2.boxplot(list_qa_lambdas, 0, '')
        ax2.hlines(qa_data, 0, 1, transform=ax2.get_yaxis_transform(), colors='r')
        ax2.set_xlim(0.5, n + 0.5)
        ax2.set_xticklabels(list_lambdas)
        ax3 = fig.add_subplot(133)
        ax3.set_title('Overall score')
        ax3.boxplot(list_oa_lambdas, 0, '')
        ax3.hlines(oa_data, 0, 1, transform=ax3.get_yaxis_transform(), colors='r')
        ax3.set_xlim(0.5, n + 0.5)
        ax3.set_xticklabels(list_lambdas)
        plt.tight_layout()
        plt.savefig('./trained_gan/{}_{}_scores.svg'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores.pdf'.format(dataset, function))
        plt.savefig('./trained_gan/{}_{}_scores.png'.format(dataset, function))
        plt.close()
        
    ###############################################################################
    # Plot data 
    print('Plotting data and function ...')
    plt.rcParams.update({'font.size': 18})
    # Turn on LaTeX formatting for text    
    plt.rcParams['text.usetex']=True
    # Place the command in the text.latex.preamble using rcParams
    plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'

    fig = plt.figure(figsize=(m*6, 8))
    
    for i, (dataset, function) in enumerate(list_data_function):
    
        # Data
        data_obj = getattr(datasets, dataset)(N)
        data = data_obj.data
        # Function
        func_obj = getattr(functions, function)()
        func = func_obj.evaluate
        
        # Plot data and function
        ax = fig.add_subplot(2, m, i+1)
        ax.set_anchor('W')
        plot_data(ax, data=data, func=func, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
        ax.set_title(r'Example \rom{{{}}}'.format(i+1))
        if i == 0:
            ax.set_xlabel('Data')
        ax = fig.add_subplot(2, m, m+i+1)
        ax.set_anchor('W')
        plot_data(ax, func=func, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
        if i == 0:
            ax.set_xlabel('Function')
            
#    plt.tight_layout()
    plt.savefig('./trained_gan/data.svg', bbox_inches='tight')
    plt.savefig('./trained_gan/data.pdf', bbox_inches='tight')
    plt.savefig('./trained_gan/data.png', bbox_inches='tight')
    plt.close()
        
    ###############################################################################
    # Plot density of generated data 
    print('Plotting density of generated data ...')
    plt.rcParams.update({'font.size': 18})
    
    for i, (dataset, function) in enumerate(list_data_function):
        
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
    
        # Data
        data_obj = getattr(datasets, dataset)(N)
        data = data_obj.data
        # Function
        func_obj = getattr(functions, function)()
        func = func_obj.evaluate
        
        fig = plt.figure(figsize=((n+1)*6, 4))
        
        # Plot data and function
        ax = fig.add_subplot(1, n+1, 1)
        plot_density(ax, data, func, data, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
        plt.title('Data')
        
        for j, lambdas in enumerate(list_lambdas):
            
            lambda0, lambda1 = 0., 0.
            if lambdas == r'GAN$_D$':
                lambda0, lambda1 = 0., lambda1_
            elif lambdas == r'GAN$_Q$':
                lambda0, lambda1 = 'inf', lambda1_
            elif lambdas == 'PaDGAN':
                lambda0, lambda1 = lambda0_, lambda1_
            elif lambdas == 'naive':
                lambda0, lambda1 = 'naive', lambda1_
            example_dir = './trained_gan/{}_{}_GAN_{}_{}/0'.format(dataset, function, lambda0, lambda1)
              
            # Generated data
            model = GAN(2, 2, lambda0, lambda1)
            model.restore(save_dir=example_dir)
            gen_data = model.synthesize(N)
            
            ax = fig.add_subplot(1, n+1, j+2)
            plot_density(ax, data, func, gen_data, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
            plt.title(lambdas)
            
#        plt.tight_layout()
        plt.savefig('./trained_gan/{}_{}_density.svg'.format(dataset, function), bbox_inches='tight')
        plt.savefig('./trained_gan/{}_{}_density.pdf'.format(dataset, function), bbox_inches='tight')
        plt.savefig('./trained_gan/{}_{}_density.png'.format(dataset, function), bbox_inches='tight')
        plt.close()
    
    ###############################################################################
    # Plot the naive approach
    print('Plotting the naive approach ...')
    
    fig = plt.figure(figsize=(m*6, 4))
    for i, (dataset, function) in enumerate(list_data_function):
        
        example_name = '{}+{}'.format(dataset, function)
        _, _, _, _, _, _, lambda1_, _ = read_config(config_fname, example_name)
        
        # Data
        data_obj = getattr(datasets, dataset)(N)
        data = data_obj.data
        # Function
        func_obj = getattr(functions, function)()
        func = func_obj.evaluate
        
        example_dir = './trained_gan/{}_{}_GAN_naive_{}/0'.format(dataset, function, lambda1_)
        
        # Generated data
        model = GAN(2, 2, 'naive', lambda1_)
        model.restore(save_dir=example_dir)
        gen_data = model.synthesize(N)
        
        ax = fig.add_subplot(1, m, i+1)
        plot_density(ax, data, func, gen_data)
        
#    plt.tight_layout()
    plt.savefig('./trained_gan/synthetic_naive.svg', bbox_inches='tight')
    plt.savefig('./trained_gan/synthetic_naive.pdf', bbox_inches='tight')
    plt.savefig('./trained_gan/synthetic_naive.png', bbox_inches='tight')
    plt.close()
    
    ###############################################################################
    # Plot the domain expansion results
    print('Plotting the domain expansion results ...')
    
    dataset = 'ThinDonut2D'
    function = 'MixRing6'
    
    # Data
    data_obj = getattr(datasets, dataset)(N)
    data = data_obj.data
    # Function
    func_obj = getattr(functions, function)()
    func = func_obj.evaluate
    
    example_name = '{}+{}'.format(dataset, function)
    _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
    
    fig = plt.figure(figsize=((n+1)*6, 8))
    grid = plt.GridSpec(2, n+1, figure=fig)
        
    # Plot data and function
    ax = fig.add_subplot(grid[0,0])
    plot_density(ax, data, func, data, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    plt.title('Data')
#    ax.set_anchor('W')
    
    for j, lambdas in enumerate(list_lambdas):
        
        lambda0, lambda1 = 0., 0.
        if lambdas == r'GAN$_D$':
            lambda0, lambda1 = 0., lambda1_
        elif lambdas == r'GAN$_Q$':
            lambda0, lambda1 = 'inf', lambda1_
        elif lambdas == 'PaDGAN':
            lambda0, lambda1 = lambda0_, lambda1_
        elif lambdas == 'naive':
            lambda0, lambda1 = 'naive', lambda1_
        example_dir = './trained_gan/{}_{}_GAN_{}_{}/0'.format(dataset, function, lambda0, lambda1)
          
        # Generated data
        model = GAN(2, 2, lambda0, lambda1)
        model.restore(save_dir=example_dir)
        gen_data = model.synthesize(N)
        norm = np.linalg.norm(gen_data, axis=1)
        gen_data_expand = gen_data[np.logical_or(norm>0.375, norm<0.325)]
        
        ax = fig.add_subplot(grid[0,j+1])
        plot_density(ax, data, func, gen_data, xlim=(-0.5,0.5), ylim=(-0.5,0.5))
        plt.title(lambdas)
        pos1 = ax.get_position()
        
        ax = fig.add_subplot(grid[1,j+1])
        plot_density(ax, data, func, gen_data_expand, xlim=(-0.5,0.5), ylim=(-0.5,0.5), scatter=True)
        circle_in = plt.Circle((0., 0.), 0.325, color='g', fill=False)
        circle_out = plt.Circle((0., 0.), 0.375, color='g', fill=False)
        ax.add_artist(circle_in)
        ax.add_artist(circle_out)
        # Move axes
        pos2 = ax.get_position()
        w = pos2.x1 - pos2.x0
        pos2.x0 = pos1.x0 - 0.013
        pos2.x1 = pos2.x0 + w
        ax.set_position(pos2)
        
#    plt.tight_layout()
    plt.savefig('./trained_gan/{}_{}_expand_density.svg'.format(dataset, function), bbox_inches='tight')
    plt.savefig('./trained_gan/{}_{}_expand_density.pdf'.format(dataset, function), bbox_inches='tight')
    plt.savefig('./trained_gan/{}_{}_expand_density.png'.format(dataset, function), bbox_inches='tight')
    plt.close()
    
    ###############################################################################
    # Print a table of scores
    print('Showing the score tables ...')
    
    for (dataset, function) in list_data_function:
        
        example_name = '{}+{}'.format(dataset, function)
        print(example_name)
        _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
        
        list_div_mean = []
        list_div_std = []
        list_qa_mean = []
        list_qa_std = []
        list_oa_mean = []
        list_oa_std = []
        for lambdas in list_lambdas:
            lambda0, lambda1 = 0., 0.
            if lambdas == r'GAN$_D$':
                lambda0, lambda1 = 0., lambda1_
            elif lambdas == r'GAN$_Q$':
                lambda0, lambda1 = 'inf', lambda1_
            elif lambdas == 'PaDGAN':
                lambda0, lambda1 = lambda0_, lambda1_
            elif lambdas == 'naive':
                lambda0, lambda1 = 'naive', lambda1_
            example_dir = './trained_gan/{}_{}_GAN_{}_{}'.format(dataset, function, lambda0, lambda1)
            
            list_div = []
            list_qa = []
            list_oa = []
            for i in range(10):
                directory = example_dir+'/{}'.format(i)
                npy_path = directory+'/scores.npy'
                div, qa, oa = np.load(npy_path)
                list_div.append(div)
                list_qa.append(qa)
                list_oa.append(oa)
            
            div_mean = np.mean(list_div)
            div_std = np.std(list_div)
            qa_mean = np.mean(list_qa)
            qa_std = np.std(list_qa)
            oa_mean = np.mean(list_oa)
            oa_std = np.std(list_oa)
            list_div_mean.append(div_mean)
            list_div_std.append(div_std)
            list_qa_mean.append(qa_mean)
            list_qa_std.append(qa_std)
            list_oa_mean.append(oa_mean)
            list_oa_std.append(oa_std)
                
        n = len(list_lambdas)
        scores = {'Diversity': ['& ${:.4f} \pm {:.4f}$'.format(list_div_mean[i], 1.96*list_div_std[i]) for i in range(n)],
                  'Quality': ['& ${:.4f} \pm {:.4f}$'.format(list_qa_mean[i], 1.96*list_qa_std[i]) for i in range(n)],
                  'Overall': ['& ${:.4f} \pm {:.4f}$'.format(list_oa_mean[i], 1.96*list_oa_std[i]) for i in range(n)]}
    
        df = pd.DataFrame(scores, columns=['Diversity','Quality','Overall'], index=['GAN','GAN$_D$','GAN$_Q$','PaDGAN'])
        print(df)
    