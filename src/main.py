import landauer.parse as parse
import landauer.simulate as simulate
import landauer.evaluate as evaluate
import landauer.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import landauer.genetic_algorithm as ga
import landauer.pareto_frontier as pf
import networkx as nx
import numpy as np
import random
import json

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
        'mutation_based': True,
        'elitism_rate': 0.1
    }

    aig = parse.parse(half_adder)
    
    best_assignment = ga.genetic_algorithm(aig, params)
    print(best_assignment)

    result = framework.forwarding(aig, best_assignment)
    framework.colorize(result)

    graph.show(graph.default(result))


n_exec = 1

def test_designs():
    
    designs = [
        'epfl_testa_ctrl.json',
        # 'epfl_testa_int2float.json',
        # 'epfl_testa_dec.json',
        # 'epfl_testa_cavlc.json'
    ]

    param_map = [
        # {
        #     'name': 'Teste Com Reprodução',
        #     'w_energy': 1,
        #     'w_delay': 0,
        #     'n_generations': 500,
        #     'n_initial_individuals': 20,
        #     'reproduction_rate': 1,
        #     'mutation_rate': 0.2,
        #     'mutation_based': False,
        #     'elitism_rate': 0.1
        # },
        {
            'name': 'Teste Sem Reprodução',
            'w_energy': 1,
            'w_delay': 0,
            'n_generations': 500,
            'n_initial_individuals': 50,
            'reproduction_rate': 0,
            'mutation_rate': 1,
            'mutation_based': True,
            'elitism_rate': 0.1
        },
    ]


    for filename in designs:
        f = open('./designs/' + filename)
        aig = parse.deserialize(f.read())
        simulation = simulate.simulate(aig)

        initial_energy = evaluate.evaluate(aig, simulation)['total']
        initial_delay = ga.calc_delay(aig)

        print(filename)

        # Naive
        aig_naive = naive.naive(aig, 'ENERGY_ORIENTED')
        assignment_naive = framework.assignment(aig_naive)
        forwarding_naive = framework.forwarding(aig_naive, assignment_naive)
        evaluation_naive = evaluate.evaluate(forwarding_naive, simulation)
        print('Naive')
        print('Energy: ' + str(evaluation_naive['total']))
        print('Delay: ' + str(ga.calc_delay(aig_naive)))

        

        for params in param_map:

            for i in range(0, n_exec):
                best, all_solutions = ga.genetic_algorithm(aig, params, returnAll=True)

                energy_score = 1 - (best.score / initial_energy)
                delay_score = 1 - (best.delay / initial_delay)

                print(params['name'] + ' - Execution ' + str(i))
                print('Energy: ' + str(best.score) + '(' + str(energy_score) + '%)')
                print('Delay: ' + str(best.delay) + '(' + str(delay_score) + '%)')

                # Pareto
                x = np.array([[i.score, i.delay] for i in all_solutions])
                pf.find_pareto_frontier(x, plot=True)

test_designs()