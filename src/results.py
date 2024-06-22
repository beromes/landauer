import landauer.algorithms.genetic as genetic
import landauer.algorithms.naive as naive
import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.graph as graph
import landauer.parse as parse
import landauer.plot as plot
import landauer.summary as summary

import csv
import matplotlib.pyplot as plt
import networkx as nx
import pathlib

# Lê informações de entropia e aig a partir dos nomes do benchmark e circuito
def read_benchmark(benchmark, circuit):
    benchmark_tree = pathlib.Path() / '..' / 'benchmark' / 'aig' / benchmark / (circuit + '.json')
    benchmark_entropy = pathlib.Path() / '..' / 'benchmark' / 'entropy' / benchmark / (circuit + '.json')

    with open(benchmark_tree) as f:
        aig = parse.deserialize(f.read())

    with open(benchmark_entropy) as f:
        entropy_data = entropy.deserialize(f.read())

    return aig, entropy_data

# Cria arquivos de entropia
def create_entropy_files(benchmarks):
    for benchmark, circuit in benchmarks:
        aig_file = pathlib.Path() / '..' / 'benchmark' / 'aig' / benchmark / (circuit + '.json')
        entropy_file = pathlib.Path() / '..' / 'benchmark' / 'entropy' / benchmark / (circuit + '.json')

        if aig_file.is_file() == False:
            print(benchmark, circuit, ': não possui arquivo aig')
            return
    
        if entropy_file.is_file() == False:
            print(benchmark, circuit, ': não possui arquivo de entropia')

            # TODO: rodar quando tivermos tempo
            # with open(aig_file) as f:
            #     aig = parse.deserialize(f.read())
            #     benchmark_module.generate_entropy_data(entropy_file, aig, overwrite=False, timeout=30000)
            #     print(benchmark, circuit, ': arquivo de entropia criado com sucesso')

# Obtem resultados do naive
def exec_naive(aig):
    energy_oriented = naive.naive(aig, naive.Strategy.ENERGY_ORIENTED)
    depth_oriented = naive.naive(aig, naive.Strategy.DEPTH_ORIENTED)
    return energy_oriented, depth_oriented

def get_naive_points(aig):
    return genetic.get_naive_point(aig, naive.Strategy.ENERGY_ORIENTED), genetic.get_naive_point(aig, naive.Strategy.DEPTH_ORIENTED)

# Realiza uma execução do GA
def exec_ga(aig, entropy_data):
    results = genetic.genetic(aig, entropy_data, param_map, timeout=10800) # 3 horas
    return results

# Plota fronteira de Pareto
def plot_pareto(samples, aig, entropy_data):    
    p = plot.Plot()
    p.plot_samples(samples, 'Genetic Algorithm', legend=False, size=5, color='black')
    p.plot_naive(aig, entropy_data)
    p.plot_pareto(samples)

# Salva resultados
def save_results(benchmark, circuit, n_exec, results, multi_objective = True):

    results = [{
        'benchmark': benchmark,
        'circuit': circuit,
        'n_execution': n_exec,
        'entropy_loss': results['best_solution'].entropy_loss if multi_objective else results['best_solution'].score,
        'delay': results['best_solution'].delay,
        'execution_time': results['execution_time'],
        'n_invalids': results['n_invalids'],
        'percentual_invalid_items': results['percentual_invalid_items'],
        'n_executed_generations': results['n_executed_generations'],
        'n_discovered_pareto': results['pareto_info']['n_discovered'] if multi_objective else 0,
        'best_energy_entropy_loss': results['pareto_info']['min_entropy_loss'].entropy_loss if multi_objective else None,
        'best_energy_delay': results['pareto_info']['min_entropy_loss'].delay if multi_objective else None,
        'best_delay_entropy_loss': results['pareto_info']['min_delay'].entropy_loss if multi_objective else None,
        'best_delay_delay': results['pareto_info']['min_delay'].delay if multi_objective else None,
        'middle_entropy_loss': results['pareto_info']['middle'].entropy_loss if multi_objective else None,
        'middle_delay': results['pareto_info']['middle'].delay if multi_objective else None,
    }]

    filename = 'output/results/' + ('multi-objective.csv' if multi_objective else 'single-objective.csv')

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writerows(results)


def save_netlists(solutions):
        solutions = list(filter(lambda s: s.delay == 9, solutions))
        solutions.sort(key=lambda x: x.entropy_loss)

        best_energy = solutions[0]
        middle_energy = solutions[len(solutions) // 2]
        worst_energy = solutions[-1]

        print('Best Energy', best_energy.entropy_loss, best_energy.delay)
        print('Middle Energy', middle_energy.entropy_loss, middle_energy.delay)
        print('Worst Energy', worst_energy.entropy_loss, worst_energy.delay)

        graph.save(graph.paper(best_energy.forwarding), 'output/netlists/x2_netlist_best_energy.png')
        graph.save(graph.paper(middle_energy.forwarding), 'output/netlists/x2_netlist_middle_energy.png')
        graph.save(graph.paper(worst_energy.forwarding), 'output/netlists/x2_netlist_worst_energy.png')



# Define benchmarks que serão utilizados
benchmarks = [
    ('mcnc', 'newtag'),
    ('mcnc', 'newtpla'),
    ('mcnc', 'z4ml'),
    ('mcnc', 'x2'),
    ('mcnc', 'm1'),
    #('epfl', 'log2'),
    #('epfl', 'sin'),
    ('epfl', 'cavlc'),
    ('epfl', 'dec'),
    ('epfl', 'int2float'),
    ('epfl', 'ctrl'),
    ('mcnc', 'prom1'),
    ('mcnc', 'mainpla'),
    ('mcnc', 'xparc'),
    ('mcnc', 'bca'),
    ('mcnc', 'prom2'),
    ('mcnc', 'apex4'),
    ('mcnc', 'ex1010'),
    ('mcnc', 'bcb'),
    ('mcnc', 'bcc'),
    ('mcnc', 'C6288'),
    ('mcnc', 'bcd'),
    ('mcnc', 'table3'),
    ('mcnc', 'table5'),
    ('mcnc', 'cps'),
]

# Cria arquivos de entropia
create_entropy_files(benchmarks)


# Lê benchmarks e roda o naive
# for (benchmark, circuit) in benchmarks:
#         print(benchmark, circuit)

#         aig, entropy_data = read_benchmark(benchmark, circuit)

#         # Salva imagens do naive na netlist
#         # energy_oriented, depth_oriented = exec_naive(aig)
#         # graph.save(graph.paper(energy_oriented), 'output/netlists/energy_oriented-' + benchmark + '-' + circuit + '.png')
#         # graph.save(graph.paper(depth_oriented), 'output/netlists/depth_oriented-' + benchmark + '-' + circuit + '.png')
        
#         evaluation = evaluate.evaluate(aig, entropy_data)
#         original = [evaluation['total'], len(nx.dag_longest_path(aig)) - 2]

#         energy, depth = get_naive_points(aig)
#         print('Original', original)
#         print('Energy-oriented', energy)
#         print('Depth-oriented', depth)


# Parametrização padrão
param_map = {
    'name': 'Parametrização Padrão',
    'n_generations': 2000,
    'n_initial_individuals': 40,
    'reproduction_rate': 1,
    'mutation_rate': 0.2,
    'mutation_intensity': 0.1,
    'elitism_rate': 0.05,
    'crossover_strategy': genetic.CrossoverStrategy.GATE
}

# Quantidade de execuções
n_exec = 5

# Para cada execução
for i in range(n_exec):

    # Para cada circuito
    for (benchmark, circuit) in benchmarks:
        print(benchmark, circuit, i)

        aig, entropy_data = read_benchmark(benchmark, circuit)
        solutions = list()
        
        # Executa algoritmo genetico
        results = exec_ga(aig, entropy_data)

        # Armazena individuos produzidos
        solutions = list(map(lambda x: x.forwarding, results['solutions']))

        # Armazena metricas
        save_results(benchmark, circuit, i, results, multi_objective=True)

        # Armazena imagem do pareto
        samples = list()
        for forwarding in solutions:
            details = summary.summary(forwarding, entropy_data)
            samples.append((details['entropy_losses'], details['depth']))

        plot_pareto(samples, aig, entropy_data)
        plt.savefig('output/pareto/multi-objective/' + benchmark.replace('/', '_') + '-' + circuit + '-' + str(i) + '.png')
        plt.cla()
        plt.clf()
    
        # Salva netlits com mesma profundidade e energias diferentes
        #save_netlists(results['solutions'])

        # Salva imagem da netlist 
        middle = results['pareto_info']['middle']
        graph.save(graph.paper(middle.forwarding), 'output/netlists/' + benchmark + '-' + circuit + '-' + str(i) + '.png')