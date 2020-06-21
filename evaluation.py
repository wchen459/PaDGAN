""" 
Diversity and quality evaluation

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA


def diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1)
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times

def quality_score(data, func_obj):
    # Average quality
    values = func_obj.evaluate(data)
    mean_val = np.mean(values)
    return mean_val

def overall_score(data, func_obj):
    return func_obj.entropy(data)

def kl_divergence(p, q):
    # KL divergence
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_score(data_X, data_Y, func_obj=None):
    if func_obj is None:
        values_X = data_X
        values_Y = data_Y
    else:
        values_X = func_obj.evaluate(data_X)
        values_Y = func_obj.evaluate(data_Y)
    p = gaussian_kde(values_X)
    q = gaussian_kde(values_Y)
    values_min = np.minimum(values_X.min(), values_Y.min())
    values_max = np.maximum(values_X.max(), values_Y.max())
    xx = np.linspace(values_min, values_max, 1000)
    return kl_divergence(p(xx), q(xx))

def nearest_dist_score(data_X, data_Y):
    nearest_dists = []
    for y in data_Y:
        nearest_dist = np.inf
        for x in data_X:
            dist = directed_hausdorff(x, y)[0]
            if dist < nearest_dist:
                nearest_dist = dist
        nearest_dists.append(nearest_dist)
    return np.mean(nearest_dists)

def get_covex_hull(data, pca_component=7):
    N = data.shape[0]
    data = data.reshape(N, -1)
    pca = PCA(pca_component).fit(data)
    z = pca.transform(data)
    hull = ConvexHull(points=z)
    return hull, pca

def expansion_score(gen_data, hull, pca):
    N = gen_data.shape[0]
    gen_data = gen_data.reshape(N, -1)
    z = pca.transform(gen_data)
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    is_in_hull = np.all((A @ np.transpose(z)) <= np.tile(-b,(1,len(z))),axis=0)
    return np.sum(is_in_hull)/N


if __name__ == "__main__":
    
    # # Test convex hull
    # training_data = np.random.rand(1000,10)
    # gen_data = np.random.rand(10,10)
    # hull, pca = get_covex_hull(training_data, pca_component=7)
    # score = expansion_score(gen_data, hull, pca)
    # print(score)
    
#    import matplotlib.pyplot as plt
#    plt.figure()
#    for simplex in hull.simplices:
#         plt.plot(training_data[simplex, 0], training_data[simplex, 1], 'k-')
#    plt.scatter(training_data[:,0], training_data[:,1], c='b')
#    plt.scatter(gen_data[:,0], gen_data[:,1], c='r')
#    plt.show()

    # Test KL divergence
    values_X = np.random.normal(0, 1, size=10000)
    values_Y = np.random.normal(0, 1, size=10000)
    p = gaussian_kde(values_X)
    q = gaussian_kde(values_Y)
    values_min = np.minimum(values_X.min(), values_Y.min())
    values_max = np.maximum(values_X.max(), values_Y.max())
    xx = np.linspace(values_min, values_max, 1000)
    print(kl_divergence(p(xx), q(xx)))
    
    