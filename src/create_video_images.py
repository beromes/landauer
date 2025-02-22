import landauer.algorithms.genetic as genetic
import landauer.entropy as entropy
import landauer.parse as parse

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

# Realiza uma execução do GA
def exec_ga(aig, entropy_data):
    results = genetic.genetic(aig, entropy_data, param_map, timeout=10800) # 3 horas
    return results

# Parametrização padrão
param_map = {
    'name': 'Parametrização Padrão',
    'n_generations': 2101,
    'n_initial_individuals': 40,
    'reproduction_rate': 1,
    'mutation_rate': 0.2,
    'mutation_intensity': 0.1,
    'elitism_rate': 0.05,
    'crossover_strategy': genetic.CrossoverStrategy.GATE
}

benchmark = 'mcnc'
circuit = 'newtpla'

aig, entropy_data = read_benchmark(benchmark, circuit)
results = exec_ga(aig, entropy_data)