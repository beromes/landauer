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
from enum import Enum, auto

half_adder = '''
    module half_adder (a, b, sum, cout);
        input a, b;
        output sum, cout;

        assign sum = a ^ b;
        assign cout = a & b;
    endmodule
'''

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


class CrossoverStrategy(Enum):
    LEVEL = auto()
    GATE = auto()
    INPUT = auto()

def new_reproduction(strategy = CrossoverStrategy.INPUT):

    aig = parse.parse(half_adder)
    assignment = framework.assignment(aig)
    original = framework.forwarding(aig, assignment)

    def get_random_individual(state):
        if state == 1:
            random_assignment = {(1, 'a'): 2, (2, 'a'): 'a', (4, 'a'): 1, (1, 'b'): 2, (2, 'b'): 'b', (4, 'b'): 2}
        elif state == 2:
            random_assignment = {(1, 'a'): 2, (2, 'a'): 4, (4, 'a'): 'a', (1, 'b'): 4, (2, 'b'): 'b', (4, 'b'): 'b'}
        else:   
            random_assignment = framework.randomize(aig, assignment)

        random_forwarding = framework.forwarding(aig, random_assignment)            
        return { "assignment": random_assignment, "forwarding": random_forwarding }

    def show_graph(forwarding_):
        framework.colorize(forwarding_)
        graph.show(graph.default(forwarding_))

    def split_assignment(i, strategy: CrossoverStrategy):

        if strategy == CrossoverStrategy.INPUT:
            keys = list(i['assignment'].keys())
            inputs = sorted(set(map(lambda k: k[1], keys)))

            leading_inputs = inputs[:len(inputs) // 2]
            trailling_inputs = inputs[len(inputs) // 2:]
        
            return [
                {k: v for k, v in i['assignment'].items() if k[1] in leading_inputs},
                {k: v for k, v in i['assignment'].items() if k[1] in trailling_inputs}
            ]

        elif strategy == CrossoverStrategy.GATE:
            keys = list(i['assignment'].keys())
            gates = sorted(set(map(lambda k: k[0], keys)))

            leading_gates = gates[:len(gates) // 2]
            trailling_gates = gates[len(gates) // 2:]

            return [
                {k: v for k, v in i['assignment'].items() if k[0] in leading_gates},
                {k: v for k, v in i['assignment'].items() if k[0] in trailling_gates}
            ]

        elif strategy == CrossoverStrategy.LEVEL:
            depth = len(nx.dag_longest_path(aig)) - 2

        else:
            raise ValueError("Invalid crossover strategy")




    def is_valid(forwarding, gate, value):
        return value not in nx.descendants(forwarding, gate)
        
    def fix(assignment):
        forwarding = framework.forwarding(aig, assignment)

        for (key, value) in assignment.items():

            if is_valid(forwarding, key[0], value):
                continue

            print('Inválido!', key, value)

            new_value = random.choice(list(framework.candidates(aig, assignment, key[0], key[1])))

            assignment[key] = new_value
            forwarding.remove_edge(value, key[0])
            forwarding.add_edge(new_value, key[0])

        return assignment, forwarding
            
    i1 = get_random_individual(1)
    i2 = get_random_individual(2)

    # show_graph(original)
    # show_graph(i1['forwarding'])
    # show_graph(i2['forwarding'])

    print(i1['assignment'])
    print(i2['assignment'])

    i1 = split_assignment(i1, strategy)
    i2 = split_assignment(i2, strategy)

    child1, child2 = i1[0], i2[0]
    child1.update(i2[1])
    child2.update(i1[1])

    # show_graph(framework.forwarding(aig, child1))
    # show_graph(framework.forwarding(aig, child2))

    print(child1)
    print(child2)

    child1, f1 = fix(child1)
    child2, f2 = fix(child2)

    show_graph(f1)
    show_graph(f2)

    print(child1)
    print(child2)

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
            'n_generations': 10,
            'n_initial_individuals': 10,
            'reproduction_rate': 1,
            'mutation_rate': 0.1,
            'mutation_intensity': 0.1,
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
                best, evolution_results, all_solutions = ga.genetic_algorithm(aig, params)
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

#test_designs()
new_reproduction()