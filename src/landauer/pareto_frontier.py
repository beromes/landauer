import numpy as np
import matplotlib.pyplot as plt


def plot_pareto_frontier(data, membership, member_value, goals):
    goals = np.asarray(goals)

    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(member_value[:, 0], member_value[:, 1], color='r')
    plt.scatter(goals[:, 0], goals[:, 1], color='purple')
    plt.legend(['Data', 'Fronteira de Pareto', 'Naive'])
    plt.xlabel('Perda de Entropia')
    plt.ylabel('Delay')
    plt.show()

def find_pareto_frontier(input, goals, plot=False):
    out = []
    data = np.concatenate((input, goals), axis=0)
    data = np.unique(data, axis=0)
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
        plot_pareto_frontier(input, membership, member_value, goals)
    
    return membership, member_value


def evolution_over_generations(results):
    global_best = np.asarray(results['global_best'])
    generation_best = np.asarray(results['generation_best'])
    generation_worst = np.asarray(results['generation_worst'])

    x_values = range(1, len(global_best) + 1)

    plt.plot(x_values, global_best, color='blue')
    plt.plot(x_values, generation_best, color='green')
    plt.plot(x_values, generation_worst, color='red')
    plt.legend(['Melhor Global', 'Melhor da Geração', 'Pior da geração'])
    plt.xlabel('N° Geração')
    plt.ylabel('Score (Perda de Informação)')
    plt.show()