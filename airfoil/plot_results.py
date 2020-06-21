""" 
Plot results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
import seaborn as sns
import pandas as pd

from run_experiment import read_config
from bezier_gan import BezierGAN
from simulation import evaluate
from evaluation import diversity_score
from shape_plot import plot_shape


if __name__ == "__main__":
    
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, train_steps, batch_size, disc_lr, gen_lr, lambda0_, lambda1_, save_interval = read_config(config_fname)
    bounds = (0., 1.)
    
    # Read dataset
    data_fname = './data/xs_train.npy'
    X = np.load(data_fname)
    N = X.shape[0]
    
    list_models = ['GAN', 'PaDGAN']
    
    ###############################################################################
    # Plot diversity and quality scores
    print('Plotting diversity and quality scores ...')
    plt.rcParams.update({'font.size': 20})
    
    n_runs = 10
    n = 1000 # generated sample size for each trained model
    subset_size = 10 # for computing DDP
    sample_times = 1000 # for computing DDP
    
    # Training data
    ind = np.random.choice(N, size=n, replace=False)
    airfoils_data = X[ind]
    div_data = diversity_score(airfoils_data, subset_size, sample_times)
    quality_data = np.load('./data/ys_train.npy')[ind]
    qa_data = np.mean(quality_data)
    
    list_div_models = []
    list_qa_models = []
    
    for model_name in list_models:
        
        if model_name == 'GAN':
            lambda0, lambda1 = 0., 0.
        elif model_name == 'PaDGAN':
            lambda0, lambda1 = lambda0_, lambda1_
    
        list_div = []
        list_qa = []
        
        for i in range(n_runs):
            save_dir = './trained_gan/{}_{}/{}'.format(lambda0, lambda1, i)
            npy_path = save_dir+'/scores.npy'
            if os.path.exists(npy_path):
                div, qa = np.load(npy_path)
            else:
                # Quality
                ys_path = '{}/ys.npy'.format(save_dir)
                if not os.path.exists(ys_path):
                    # Generated airfoils
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
                np.save(npy_path, [div, qa])
                
            list_qa.append(qa)
            list_div.append(div)
            
        list_div_models.append(list_div)
        list_qa_models.append(list_qa)
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('Diversity')
    ax1.boxplot(list_div_models, 0, '')
    ax1.hlines(div_data, 0, 1, transform=ax1.get_yaxis_transform(), colors='r')
    ax1.set_xlim(0.5, len(list_models) + 0.5)
    ax1.set_xticklabels(list_models)
    ax2 = fig.add_subplot(122)
    ax2.set_title('Quality')
    ax2.boxplot(list_qa_models, 0, '')
    ax2.hlines(qa_data, 0, 1, transform=ax2.get_yaxis_transform(), colors='r')
    ax2.set_xlim(0.5, len(list_models) + 0.5)
    ax2.set_xticklabels(list_models)
    plt.tight_layout()
    plt.savefig('./trained_gan/airfoil_scores.svg')
    plt.savefig('./trained_gan/airfoil_scores.pdf')
    plt.savefig('./trained_gan/airfoil_scores.png')
    plt.close()
    
    ###############################################################################
    # Plot airfoils embedding
    print('Plotting airfoils embedding ...')
    plt.rcParams.update({'font.size': 14})
    
    n = 100
    
    # Compute quality
    def evaluate_airfoils(airfoils):
        ys = []
        for i, airfoil in enumerate(airfoils):
            val = evaluate(airfoil, return_CL_CD=False)
            ys.append(val)
        ys= np.array(ys)
        return ys
    
    # Data
    x_path = './trained_gan/airfoils_data.npy'
    y_path = './trained_gan/ys_data.npy'
    if os.path.exists(x_path) and os.path.exists(y_path):
        airfoils_data = np.load(x_path)
        ys_data = np.load(y_path)
        ind = np.random.choice(airfoils_data.shape[0], size=n, replace=False)
        airfoils_data = airfoils_data[ind]
        ys_data = ys_data[ind]
    else:
        ind = np.random.choice(N, size=n, replace=False)
        airfoils_data = X[ind].astype(np.float32)
        ys_data = evaluate_airfoils(airfoils_data)
        np.save(x_path, airfoils_data)
        np.save(y_path, ys_data)
    # GAN
    x_path = './trained_gan/airfoils_gan.npy'
    y_path = './trained_gan/ys_gan.npy'
    if os.path.exists(x_path) and os.path.exists(y_path):
        airfoils_gan = np.load(x_path)
        ys_gan = np.load(y_path)
        ind = np.random.choice(airfoils_gan.shape[0], size=n, replace=False)
        airfoils_gan = airfoils_gan[ind]
        ys_gan = ys_gan[ind]
    else:
        save_dir = './trained_gan/0.0_0.0/1'
        model = BezierGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, bounds, 0.0, 0.0)
        model.restore(directory=save_dir)
        airfoils_gan = model.synthesize(n)
        ys_gan = evaluate_airfoils(airfoils_gan)
        np.save(x_path, airfoils_gan)
        np.save(y_path, ys_gan)
    # PaDGAN
    x_path = './trained_gan/airfoils_padgan.npy'
    y_path = './trained_gan/ys_padgan.npy'
    if os.path.exists(x_path) and os.path.exists(y_path):
        airfoils_padgan = np.load(x_path)
        ys_padgan = np.load(y_path)
        ind = np.random.choice(airfoils_padgan.shape[0], size=n, replace=False)
        airfoils_padgan = airfoils_padgan[ind]
        ys_padgan = ys_padgan[ind]
    else:
        _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
        save_dir = './trained_gan/{}_{}/3'.format(lambda0, lambda1)
        model = BezierGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, bounds, lambda0, lambda1)
        model.restore(directory=save_dir)
        airfoils_padgan = model.synthesize(n)
        ys_padgan = evaluate_airfoils(airfoils_padgan)
        np.save(x_path, airfoils_padgan)
        np.save(y_path, ys_padgan)
    
    def plot_airfoils(airfoils, ys, zs, ax, norm, cmap, zs_data=None):
        n = airfoils.shape[0]
        ys[np.isnan(ys)] = 0.
        for i in range(n):
            plot_shape(airfoils[i]+np.array([[-.5,0]]), zs[i, 0], zs[i, 1], ax, 1./n**.5, False, None, lw=1.2, alpha=.7, c=cmap(norm(ys[i])))
        if zs_data is not None:
            ax.scatter(zs_data[:,0], zs_data[:,1], s=20, marker='o', edgecolors='none', c='#7be276')
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False)
        ax.set_xlim([-.01-.5/n**.5, 1.01+.5/n**.5])
        ax.set_ylim([-.05, 1.05])
        ax.set_aspect('equal')
            
    xs = np.concatenate([airfoils_data, airfoils_gan, airfoils_padgan])
    xs = xs.reshape(xs.shape[0], -1)
    scaler_x = MinMaxScaler()
    xs = scaler_x.fit_transform(xs)
    tsne = TSNE(n_components=2)
    zs = tsne.fit_transform(xs)
    scaler_z = MinMaxScaler()
    zs = scaler_z.fit_transform(zs)
    
    ys = np.concatenate([ys_data, ys_gan, ys_padgan])
    ys[np.isnan(ys)] = 0.
    y_min = np.min(ys)
    y_max = np.max(ys)
    y_range = y_max-y_min
    y_min -= 0.5*y_range
    norm = matplotlib.colors.Normalize(vmin=y_min, vmax=y_max)
    cmap = cm.Greys
    
    def select_subset(zs, r, y_scale=0.1):
        m = zs.shape[0]
        zs_ = zs.copy()
        zs_[:,1] = zs[:,1]/y_scale
        dists = pairwise_distances(zs_) + np.eye(m) * r
        is_removed = np.zeros(m, dtype=bool)
        for i in range(n):
            if not is_removed[i]:
                is_removed = np.logical_or(is_removed, dists[i]<r)
        not_removed = np.logical_not(is_removed)
        return not_removed
        
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(141)
    ax1.scatter(zs[:n,0], zs[:n,1], s=20, marker='o', edgecolors='none', c='#7be276', label='Data')
    ax1.scatter(zs[n:2*n,0], zs[n:2*n,1], s=20, marker='s', edgecolors='none', c='#fc9e55', label='GAN')
    ax1.scatter(zs[2*n:,0], zs[2*n:,1], s=20, marker='^', edgecolors='none', c='#63b1ed', label='PaDGAN')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False)
    ax1.set_xlim([-.01-.5/n**.5, 1.01+.5/n**.5])
    ax1.set_ylim([-.05, 1.05])
    ax1.set_title('(a) Embedded airfoils')
    r = 2./n**.5
    ax2 = fig.add_subplot(142)
    not_removed = select_subset(zs[:n], r)
    plot_airfoils(airfoils_data[not_removed], ys_data[not_removed], zs[:n][not_removed], ax2, norm, cmap)
    ax2.set_title('(b) Data')
    ax3 = fig.add_subplot(143)
    not_removed = select_subset(zs[n:2*n], r)
    plot_airfoils(airfoils_gan[not_removed], ys_gan[not_removed], zs[n:2*n][not_removed], ax3, norm, cmap, zs[:n])
    ax3.set_title('(c) GAN')
    ax4 = fig.add_subplot(144)
    not_removed = select_subset(zs[2*n:], r)
    plot_airfoils(airfoils_padgan[not_removed], ys_padgan[not_removed], zs[2*n:][not_removed], ax4, norm, cmap, zs[:n])
    ax4.set_title('(d) PaDGAN')
    plt.tight_layout()
    # Add a colorbar
    fig.subplots_adjust(right=0.88)
    ax5 = fig.add_axes([0.9, 0.05, 0.01, 0.9])
    cb = matplotlib.colorbar.ColorbarBase(ax5, cmap=cmap,
                                          norm=norm,
                                          orientation='vertical')
    cb.set_label(r'$C_L/C_D$')
    plt.savefig('./trained_gan/airfoils_tsne.svg')
    plt.savefig('./trained_gan/airfoils_tsne.pdf')
    plt.savefig('./trained_gan/airfoils_tsne.png')
    plt.close()
    
    ###############################################################################
    # Plot ys distribution
    print('Plotting ys distribution ...')
    plt.rcParams.update({'font.size': 18})
    
    ys_gan = np.load('./trained_gan/0.0_0.0/1/ys.npy')
    ys_gan[np.isnan(ys_gan)] = 0.
    _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
    ys_padgan = np.load('./trained_gan/{}_{}/3/ys.npy'.format(lambda0, lambda1))
    ys_padgan[np.isnan(ys_padgan)] = 0.
    ys_data_all = np.load('./data/ys_train.npy')
    N = ys_data.shape[0]
    n = ys_gan.shape[0]
    ind = np.random.choice(N, size=n)
    ys_data = ys_data_all[ind]
    
    plt.figure()
    sns.kdeplot(ys_data, linestyle=':', color='#7be276', shade=True, Label='Data')
    sns.kdeplot(ys_gan, linestyle='--', color='#fc9e55', shade=True, Label='GAN')
    sns.kdeplot(ys_padgan, linestyle='-', color='#63b1ed', shade=True, Label='PaDGAN')
    plt.xlabel(r'$C_L/C_D$') 
    plt.ylabel('Probability density')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('./trained_gan/airfoils_quality_dist.svg')
    plt.savefig('./trained_gan/airfoils_quality_dist.pdf')
    plt.savefig('./trained_gan/airfoils_quality_dist.png')
    plt.close()
    
    ###############################################################################
    # Print a table of scores
    print('Showing the score tables ...')
        
    list_div_mean = []
    list_div_std = []
    list_qa_mean = []
    list_qa_std = []
    for model_name in list_models:
        
        if model_name == 'GAN':
            lambda0, lambda1 = 0., 0.
        elif model_name == 'PaDGAN':
            lambda0, lambda1 = lambda0_, lambda1_
        
        list_div = []
        list_qa = []
        list_oa = []
        for i in range(10):
            save_dir = './trained_gan/{}_{}/{}'.format(lambda0, lambda1, i)
            npy_path = save_dir+'/scores.npy'
            div, qa = np.load(npy_path)
            list_div.append(div)
            list_qa.append(qa)
        
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
            
    n = len(list_models)
    scores = {'Diversity': ['& ${:.4f} \pm {:.4f}$'.format(list_div_mean[i], 1.96*list_div_std[i]) for i in range(n)],
              'Quality': ['& ${:.4f} \pm {:.4f}$'.format(list_qa_mean[i], 1.96*list_qa_std[i]) for i in range(n)]}

    df = pd.DataFrame(scores, columns=['Diversity','Quality'], index=list_models)
    print(df)