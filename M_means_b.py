import pickle
import numpy as np
import matplotlib.pyplot as plt

#!curl -s https://rasmuspagh.net/data/glove.twitter.27B.100d.names.pickle -O

def compute_cost(points, centers):
    distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
    return np.mean(np.min(distances_squared, axis=0))

def k_means(points, k, t, rho):
    # Calculate the sigmas^2 based on rho 
    sigma_sum = (4*t)/rho
    sigma_size = (2*t)/rho
    
    
    initial_assignment = np.random.choice(range(k), n)
    cluster_indexes = [ (initial_assignment == i) for i in range(k) ]
    cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ]

    for l in range(t):
        # Adding noise to each cluster sum
        cluster_sums = [ np.sum(points[cluster_indexes[i] ] , axis=0) + np.random.normal(0,sigma_sum, d) for i in range(k) ]
        centers = np.array([ cluster_sums[i] / max(1, cluster_sizes[i]) for i in range(k) ])
        distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0)
        cluster_indexes = [ (assignment == i) for i in range(k) ]
        # Adding noise to each cluster size. 
        cluster_sizes = [ cluster_indexes[i].sum()+ np.random.normal(0,sigma_size) for i in range(k) ]

    return centers


def M_means(points, k, t, rho):

    initial_assignment = np.random.choice(range(k), n)
    # S_i
    cluster_indexes = [ (initial_assignment == i) for i in range(k) ]
    # g_l
    cluster_sizes = [ cluster_indexes[i].sum() for i in range(k) ]
    # compute sigma
    # sigma = np.sqrt((3*t)/rho)
    # sigma_prime = sigma
    sigma = np.sqrt((4*t)/rho)
    sigma_prime = np.sqrt((2*t)/rho)
    for l in range(t):
        #f_l(x)
        cluster_sums = [ np.sum(points[cluster_indexes[i]], axis=0) for i in range(k) ]
        # c_i
        centers = np.array([ (cluster_sums[i]+np.random.normal(0,sigma,d)) / max(1, cluster_sizes[i]) for i in range(k) ])
        distances_squared = np.sum((points - centers[:,np.newaxis])**2, axis=-1)
        assignment = np.argmin(distances_squared, axis=0)
        cluster_indexes = [ (assignment == i) for i in range(k) ]
        cluster_sizes = [ cluster_indexes[i].sum() for i in range(k)]+ np.random.normal(0,sigma_prime,k)

    return centers


if __name__=='__main__':
    # Open file
    input_file = "glove.twitter.27B.100d.names.pickle"
    with open(input_file, 'rb') as f:
        embedding = pickle.load(f)
    names = list(embedding.keys())
    
    points = np.array([ embedding[x] for x in names ])
    n, d = points.shape
    rho_list = np.linspace(0.001,1,100)
    k = 5 # Number of clusters
    t = 5 # Number of iteration
    costs = []

    for rho in rho_list:
        centers = M_means(points, k, t, rho)
        costs.append(compute_cost(points, centers))
        print(f"rho:{rho:.4f},cost:{costs[-1]}")

    fig, ax = plt.subplots() 
    ax.set_xlabel('rho')
    ax.set_ylabel('cost')
    ax.plot(rho_list, costs)
    plt.xscale('log')
    plt.show()