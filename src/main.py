import landauer.parse as parse
import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.algorithms.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import landauer.genetic_algorithm as ga
import landauer.pareto_frontier as pf
import networkx as nx
import numpy as np
import random
import json
from operator import attrgetter

def get_naive_point(aig, strategy):
    entropy_s = entropy.entropy(aig)
    aig_naive = naive.naive(aig, strategy)
    assignment_naive = framework.assignment(aig_naive)
    forwarding_naive = framework.forwarding(aig_naive, assignment_naive)
    evaluation_naive = evaluate.evaluate(forwarding_naive, entropy_s)
    naive_point = [evaluation_naive['total'], ga.calc_delay(aig_naive)]
    print('Naive - ' + str(strategy))
    print('Energy: ' + str(evaluation_naive['total']))
    print('Delay: ' + str(ga.calc_delay(aig_naive)))
    return naive_point


def find_best(individuals): 
    best_score = min(individuals, key=attrgetter('score')).score
    best = min(filter(lambda x: x.score == best_score, individuals), key=attrgetter('delay'))
    return best

def test_half_adder():

    half_adder = '''
        module half_adder (a, b, sum, cout);
            input a, b;
            output sum, cout;

            assign sum = a ^ b;
            assign cout = a & b;
        endmodule
    '''

    params = {
        'name': 'Teste Com Reprodução',
        'w_energy': 1,
        'w_delay': 0,
        'n_generations': 10,
        'n_initial_individuals': 5,
        'reproduction_rate': 1,
        'mutation_rate': 0.2,
        'mutation_based': False,
        'elitism_rate': 0.1
    }

    aig = parse.parse(half_adder)
    
    best = ga.genetic_algorithm(aig, params)

    result = framework.forwarding(aig, best.assignment)
    framework.colorize(result)

    graph.show(graph.default(result))


n_exec = 1

def test_designs():
    
    designs = [
        'demo.json'
        # 'epfl_testa_ctrl.json',
        # 'epfl_testa_int2float.json',
        # 'epfl_testa_dec.json',
        # 'epfl_testa_cavlc.json'
    ]

    param_map = [
        {
            'name': 'Teste Com Reprodução',
            'w_energy': 1,
            'w_delay': 0,
            'n_generations': 100,
            'n_initial_individuals': 10,
            'reproduction_rate': 1,
            'mutation_rate': 0.2,
            'mutation_based': False,
            'elitism_rate': 0.1
        }
    ]


    for filename in designs:
        f = open('./designs/' + filename)
        aig = parse.deserialize(f.read())
        entropy_s = entropy.entropy(aig)

        initial_energy = evaluate.evaluate(aig, entropy_s)['total']
        initial_delay = ga.calc_delay(aig)

        print(filename)

        naive_points = [ 
            get_naive_point(aig, naive.Strategy.ENERGY_ORIENTED), 
            get_naive_point(aig, naive.Strategy.DELAY_ORIENTED) 
        ]

        for params in param_map:

            for i in range(0, n_exec):
                best, evolution_results, all_solutions = ga.genetic_algorithm(aig, params, returnAll=True)
                best = find_best(all_solutions)

                energy_score = 1 - (best.score / initial_energy)
                delay_score = 1 - (best.delay / initial_delay)

                print(params['name'] + ' - Execução ' + str(i))
                print('Energy: ' + str(best.score) + '(' + str(energy_score) + '%)')
                print('Delay: ' + str(best.delay) + '(' + str(delay_score) + '%)')

                # Pareto
                x = np.array([[i.score, i.delay] for i in all_solutions])
                pf.find_pareto_frontier(x, naive_points, plot=True)
                pf.evolution_over_generations(evolution_results)

                # Desenho
                result = framework.forwarding(aig, best.assignment)
                framework.colorize(result)
                graph.show(graph.default(result))

test_designs()