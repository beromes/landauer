import numpy as np
import matplotlib.pyplot as plt

def plot_pareto_frontier(data, membership, member_value):
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(member_value[:, 0], member_value[:, 1], color='r')
    plt.legend(['Data', 'Fronteira de Pareto'])
    plt.xlabel('Perda de Entropia')
    plt.ylabel('Delay')
    plt.show()

def find_pareto_frontier(input, plot=False):
    out = []
    
    data = np.unique(input, axis=0)
    for i in range(data.shape[0]):
        c_data = np.tile(data[i, :], (data.shape[0], 1))
        t_data = data.copy()
        t_data[i, :] = np.Inf
        smaller_idx = c_data >= t_data
        
        idx = np.sum(smaller_idx, axis=1) == data.shape[1]
        if np.count_nonzero(idx) == 0:
            out.append(data[i, :])
    
    membership = np.all(np.equal(np.expand_dims(input, axis=1), out), axis=2)
    membership = np.any(membership, axis=1)
    member_value = np.array(out)

    if(plot):
        plot_pareto_frontier(input, membership, member_value)
    
    return membership, member_value
