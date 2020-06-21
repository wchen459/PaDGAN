import numpy as np
from scipy.stats import kde
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
plt.rcParams.update({'font.size': 18})


def plot_data(ax, data=None, func=None, gen_data=None, axis_off=True, xlim=None, ylim=None):
    
    if data is not None:
        if xlim is None:
            xlim = (np.min(data[:,0]), np.max(data[:,0]))
        if ylim is None:
            ylim = (np.min(data[:,1]), np.max(data[:,1]))
        
    if func is not None:
        n = 100
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n),
                             np.linspace(ylim[0], ylim[1], n))
        grid = np.vstack((xx.ravel(), yy.ravel())).T
        val = func(grid)
        
    if data is not None:
        plt.scatter(data[:,0], data[:,1], marker='o', s=10, c='g', alpha=0.7, edgecolor='none', label='data')
    if func is not None:
        if data is not None or gen_data is not None:
            plt.contour(xx, yy, val.reshape(xx.shape), 15, linewidths=0.3, alpha=0.5, cmap=cm.gray)
        else:
            plt.contourf(xx, yy, val.reshape(xx.shape), 15, linewidths=0.3, cmap=cm.gray)
            plt.colorbar(label='Quality')
    if gen_data is not None:
        assert gen_data.shape[1] == 2
        plt.scatter(gen_data[:,0], gen_data[:,1], marker='+', s=10, c='b', alpha=0.7, edgecolor='none', label='generated')
        plt.legend()
    plt.axis('equal')
    if axis_off:
        plt.axis('off')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
def plot_density(ax, data, func=None, gen_data=None, scatter=False, axis_off=True, xlim=None, ylim=None):
    
    if xlim is None:
        xlim = (np.min(data[:,0]), np.max(data[:,0]))
    if ylim is None:
        ylim = (np.min(data[:,1]), np.max(data[:,1]))
        
    if func is not None:
        n = 100
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n),
                             np.linspace(ylim[0], ylim[1], n))
        grid = np.vstack((xx.ravel(), yy.ravel())).T
        val = func(grid)
        
    if func is not None:
        plt.contour(xx, yy, val.reshape(xx.shape), 15, linewidths=0.3, alpha=0.5)
    if gen_data is not None:
        assert gen_data.shape[1] == 2
        if scatter:
            plt.scatter(gen_data[:,0], gen_data[:,1], marker='o', s=10, c='b', alpha=0.7, edgecolor='none')
        else:
            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            k = kde.gaussian_kde(gen_data.T)
            nbins = 20
            xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='Blues')
            plt.colorbar(label='Density')
    plt.axis('equal')
    if axis_off:
        plt.axis('off')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def visualize_2d(data, func=None, gen_data=None, save_path=None, axis_off=True, xlim=None, ylim=None):
    
    assert data.shape[1] == 2
        
    fig = plt.figure(figsize=(8,10))
    
    # Subplot 1
    ax = fig.add_subplot(211)
    plot_data(ax, data, func, gen_data, axis_off, xlim, ylim)
        
    # Subplot 2
    ax = fig.add_subplot(212)
    plot_density(ax, data, func, gen_data, axis_off, xlim, ylim)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        