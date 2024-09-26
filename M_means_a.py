import pickle
import numpy as np
import matplotlib.pyplot as plt

#!curl -s https://rasmuspagh.net/data/glove.twitter.27B.100d.names.pickle -O

def compute_cost(points, centers):
    distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
    return np.mean(np.min(distances_squared, axis=0))


def k_means(points, k, t, rho):

    initial_assignment = np.random.choice(range(k), n)
    # S_i
    cluster_indexes = [ (initial_assignment == i) for i in range(k) ]
    # g_l
    cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ]
    # compute sigma
    sigma = np.sqrt(3*t/rho)
    for l in range(t):
        #f_l(x)
        cluster_sums = [ np.sum(points[cluster_indexes[i]], axis=0) for i in range(k) ]
        # c_i
        centers = np.array([ (cluster_sums[i]+np.random.normal(0,sigma,d)) / max(1, cluster_sizes[i]) for i in range(k) ])
        distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0)
        cluster_indexes = [ (assignment == i) for i in range(k) ]
        cluster_sizes = [ cluster_indexes[i].sum()+np.random.normal(0,sigma,1) for i in range(k)]

    return centers


if __name__=='__main__':
    # Open file
    input_file = "glove.twitter.27B.100d.names.pickle"
    with open(input_file, 'rb') as f:
        embedding = pickle.load(f)
    names = list(embedding.keys())
    points = np.array([ embedding[x] for x in names ])
    n, d = points.shape
    rho = 0.001
    k = 5 # Number of clusters
    t_range = range(1,10)
    costs = []
    for t in t_range: # number of iterations
        # print(t)
        centers = k_means(points, k, t, rho)
        costs.append(compute_cost(points, centers))
    
    # print(costs[8])
    fig, ax = plt.subplots()
    ax.set_xlabel('t')
    ax.set_ylabel('cost')
    ax.plot(t_range, costs)
    plt.xscale('log')
    plt.show()