import landauer.parse as parse
import landauer.framework as framework
import landauer.genetic_algorithm as ga

half_adder = '''
    module half_adder (a, b, sum, cout);
        input a, b;
        output sum, cout;

        assign sum = a ^ b;
        assign cout = a & b;
    endmodule
'''

def get_random_individual(state):
    if state == 1:
        random_assignment = {(1, 'a'): 2, (2, 'a'): 'a', (4, 'a'): 1, (1, 'b'): 2, (2, 'b'): 'b', (4, 'b'): 2}
    elif state == 2:
        random_assignment = {(1, 'a'): 2, (2, 'a'): 4, (4, 'a'): 'a', (1, 'b'): 4, (2, 'b'): 'b', (4, 'b'): 'b'}
    else:   
        random_assignment = framework.randomize(aig, assignment)

    random_forwarding = framework.forwarding(aig, random_assignment)            
    return { "assignment": random_assignment, "forwarding": random_forwarding }

def test_designs(n_exec=1):
    
    designs = [
        'epfl_testa_ctrl.json',
        # 'epfl_testa_int2float.json',
        # 'epfl_testa_dec.json',
        # 'epfl_testa_cavlc.json'
        # 'demo.json'
    ]

    param_map = [
        {
            'name': 'Teste Leve',
            'w_energy': 1,
            'w_delay': 0,
            'n_generations': 50,
            'n_initial_individuals': 10,
            'reproduction_rate': 1,
            'mutation_rate': 0.1,
            'mutation_intensity': 0.1,
            'elitism_rate': 0.1
        },
        {
            'name': 'Teste Maior',
            'w_energy': 1,
            'w_delay': 0,
            'n_generations': 1000,
            'n_initial_individuals': 30,
            'reproduction_rate': 1,
            'mutation_rate': 0.1,
            'mutation_intensity': 0.1,
            'elitism_rate': 0.1
        }
    ]

    for filename in designs:
        print(filename)

        f = open('./designs/' + filename)
        aig = parse.deserialize(f.read())

        for params in param_map:
            for i in range(n_exec):
                print(params['name'] + ' - Execução ' + str(i))
                ga.genetic_algorithm(aig, params, plot_results=True, plot_circuit=False, debug=True)                

test_designs()