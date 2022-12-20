import landauer.parse as parse
import landauer.simulate as simulate
import landauer.evaluate as evaluate
import landauer.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import landauer.genetic_algorithm as ga
import networkx as nx
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
    aig = parse.parse(half_adder)
    
    best_assignment = ga.genetic_algorithm(aig, 0.5, 0.5)
    print(best_assignment)

    result = framework.forwarding(aig, best_assignment)
    framework.colorize(result)

    graph.show(graph.default(result))


def test_designs():
    
    designs = ['epfl_testa_ctrl.json', 'epfl_testa_int2float.json', 'epfl_testa_dec.json', 'epfl_testa_cavlc.json']

    param_map = [
        {
            'name': 'Energy',
            'w_energy': 1,
            'w_delay': 0,
        },
        {
            'name': 'Mix',
            'w_energy': 0.5,
            'w_delay': 0.5,
        }
    ]


    for filename in designs:
        f = open('./designs/' + filename)
        aig = parse.deserialize(f.read())
        simulation = simulate.simulate(aig)

        initial_energy = evaluate.evaluate(aig, simulation)['total']
        initial_delay = calc_delay(aig)

        print(filename)

        for params in param_map:
            best_assignment = ga.genetic_algorithm(aig, params['w_energy'], params['w_delay'])
            forwarding = framework.forwarding(aig, best_assignment)    
            evaluation = evaluate.evaluate(forwarding, simulation)

            energy_score = 1 - (evaluation['total'] / initial_energy)
            delay_score = 1 - (calc_delay(forwarding) / initial_delay)

            print(params['name'])
            print('Energy: ' + str(energy_score))
            print('Delay: ' + str(delay_score))

test_half_adder()