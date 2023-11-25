'''

Copyright (c) 2023 Marco Diniz Sousa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import landauer.parse as parse
import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.algorithms.naive as naive
import landauer.graph as graph
import landauer.pareto_frontier as pf
import landauer.placement as placement
import networkx as nx
import numpy as np
import random
import json
import time
from operator import attrgetter
from enum import Enum, auto
import pprint

from functools import reduce

'''
Classes/Modelos
'''
class CrossoverStrategy(Enum):
    LEVEL = auto()
    GATE = auto()
    INPUT = auto()

class ParamMap:
    name = 'Padrao'
    w_energy = 1
    w_delay = 0
    n_generations = 600
    n_initial_individuals = 50
    reproduction_rate = 1.0
    mutation_rate = 0.1
    mutation_intensity = 0.15
    elitism_rate = 0.15
    crossover_strategy = CrossoverStrategy.GATE

    def __init__(self, dictt):
        if type(dictt) is dict:          
            for key in dictt:
                setattr(self, key, dictt[key])

class Individual:
    assignment = None
    forwarding = None
    score = 0
    delay = 0

    def __init__(self, assignment, forwarding):
        self.assignment = assignment
        self.forwarding = forwarding

'''
Funcoes auxiliares
'''
def _calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2

def _get_naive_point(aig, strategy):
    entropy_s = entropy.entropy(aig)
    aig_naive = naive.naive(aig, strategy)
    assignment_naive = _assignment(aig_naive)
    forwarding_naive = _forwarding(aig_naive, assignment_naive)
    evaluation_naive = evaluate.evaluate(forwarding_naive, entropy_s)
    naive_point = [evaluation_naive['total'], _calc_delay(aig_naive)]
    print('Naive - ' + str(strategy))
    print('Energy: ' + str(evaluation_naive['total']))
    print('Delay: ' + str(_calc_delay(aig_naive)))
    return naive_point

def _get_single_edge(aig, u, v):
    edges = [key for key in aig.succ[u].get(v, dict()).keys() if not aig.edges[u, v, key].get('forward', False)]
    return edges[0] if len(edges) == 1 else None

def _assignment(aig):
    assignment_ = dict()
    for node in aig.nodes():
        children = set(aig.successors(node))
        if len(children) >= 2:
            assignment_.update({(child, node): node for child in children})
    return assignment_

def _forwarding(aig, assignment):
    return placement.place(aig, assignment)

def _randomize(aig, assignment, forwarding):
    assignment_ = assignment.copy()
    forwarding_ = forwarding.copy()
    assignment_items = list(assignment_.keys())
    random.shuffle(assignment_items)
    for gate, input_ in assignment_items:
        candidates_ = list(_candidates(aig, forwarding_, gate, input_))
        assignment_[(gate, input_)] = random.choice(candidates_)
        forwarding_ = _forwarding(aig, assignment_)
        # old_value = assignment_[(gate, input_)]
        # new_value = random.choice(candidates_)
        # assignment_[(gate, input_)] = new_value
        # forwarding_.remove_edge(old_value, gate)
        # forwarding_.add_edge(new_value, gate)

    return assignment_, forwarding_

def _candidates(aig, forwarding_, gate, input_):
    candidates = set(aig.successors(input_))    
    # Majority support: one node cannot forward more than two inputs
    full = set(c for c in candidates if len(set(key for _, _, key, f in forwarding_.out_edges(c, keys=True, data='forward', default=False) if f) - {input_}) == 2)    
    # Outputs cannot forward information
    outputs = set(p for p in aig.nodes() if len(list(aig.successors(p))) == 0)    
    return (candidates | {input_}) - full - {gate} - nx.descendants(forwarding_, gate) - outputs

def _log(message):
    if debug == False: return

    global prev_time
    print('[TIME] ' + message + ' = ' + str(time.time() - prev_time))
    prev_time = time.time()

'''
Etapas do algoritmo genético
'''
# Retorna populacao inicial
def _init_population(aig, n_individuals):
    assignment = _assignment(aig)
    forwarding = _forwarding(aig, assignment)
    population = []

    for i in range(n_individuals):
        random_assignment, random_forwarding = _randomize(aig, assignment, forwarding)
        new_individual = Individual(random_assignment, random_forwarding)
        population.append(new_individual)        

    return population

# Calcula o score baseado na perda de entropia, também calcula o delay
def _fit(population, entropy_s):
    for p in population:
        evaluation = evaluate.evaluate(p.forwarding, entropy_s)
        p.score = evaluation['total']
        p.delay = _calc_delay(p.forwarding)

# Faz reprodução dos individuos de uma populacao
def _reproduce(aig, population, rate, strategy):
    def split_assignment(i: Individual, strategy: CrossoverStrategy):
        if strategy == CrossoverStrategy.INPUT:
            keys = list(i.assignment.keys())
            inputs = list(set(map(lambda k: str(k[1]), keys)))

            leading_inputs = inputs[:len(inputs) // 2]
            trailing_inputs = inputs[len(inputs) // 2:]

            return [
                {k: v for k, v in i.assignment.items() if str(k[1]) in leading_inputs},
                {k: v for k, v in i.assignment.items() if str(k[1]) in trailing_inputs}
            ]

        elif strategy == CrossoverStrategy.GATE:
            keys = map(lambda k: k[0], i.assignment.keys())
            gates = sorted(set(filter(lambda k: type(k) == int, keys)))
            leading_gates = gates[:len(gates) // 2]

            return [
                {k: v for k, v in i.assignment.items() if k[0] in leading_gates},
                {k: v for k, v in i.assignment.items() if k[0] not in leading_gates}
            ]

        # TODO: implementar divisão por níveis
        else:
            raise ValueError("Invalid crossover strategy")

    # Verifica se determinada atribuição é válida
    def is_valid(forwarding, dest_gate, origin_gate):
        return origin_gate not in nx.descendants(forwarding, dest_gate)

    # Corrige o invidíduo após juntar as duas partes
    def make_individual(aig, assignment):
        forwarding = _forwarding(aig, assignment)
        invalid_edges = list(assignment.items())

        while len(invalid_edges) > 0:
            new_invalid_edges = []

            for (key, value) in invalid_edges:
                
                if is_valid(forwarding, key[0], value):
                    continue
                
                candidates = list(_candidates(aig, forwarding, key[0], key[1]))
                if (len(candidates) == 0):
                    new_invalid_edges.append((key, value))
                    continue                        

                new_value = random.choice(candidates)
                assignment[key] = new_value
                forwarding = _forwarding(aig, assignment)
                # forwarding.remove_edge(value, key[0])
                # forwarding.add_edge(new_value, key[0])

            # Caso entre em loop e não consiga resolver as divergências, retorna nulo
            if (invalid_edges == new_invalid_edges):
                if debug:
                    print('[ERROR] Loop detectado - Indivíduo descartada')
                return None;

            invalid_edges = new_invalid_edges

        return Individual(assignment, forwarding)


    n_children = int(len(population) * rate)
    children = []

    # Ordena a população e define os pesos
    ordered_population = sorted(population, key=lambda p: p.score, reverse=True)
    weights = list(range(1, len(population) + 1)) # Peso é baseado na ordem
    weights = weights / np.sum(weights) # divide pela soma dos pesos para que a soma total seja 1

    # Conta número de soluções inválidas
    n_invalids = 0

    while len(children) < n_children:
        # Escolhe os parentes
        p1, p2 = np.random.choice(ordered_population, 2, replace=False, p=weights) # Escolhe dois parentes sem reposição

        # Separa os genes de acordo com a estratégia
        splitted_p1 = split_assignment(p1, strategy)
        splitted_p2 = split_assignment(p2, strategy)

        # Cria filhos a partir da combinação dos genes dos pais
        # TODO: Explorar estratégia que gera até 4 filhos
        child1, child2 = p1.assignment.copy(), p2.assignment.copy()
        child1.update(splitted_p2[1])
        child2.update(splitted_p1[1])

        # TODO: Remover prints quando ficarem irrelevantes
        for key in p1.assignment.keys():
            if key not in child1.keys():
                print('ERRO: Está em p1 mas não em child1', key)

        for key in child1.keys():
            if key not in p1.assignment.keys():
                print('ERRO: Está em child1 mas não em p1', key)

        for i in range(len(p1.assignment.keys())):
            if list(p1.assignment.keys())[i] != list(child1.keys())[i]:
                print('ERRO: Não estão na mesma ordem!', i)

        i1, i2 = make_individual(aig, child1), make_individual(aig, child2)

        if i1 is not None:
            children.append(i1)
        else:
            n_invalids += 1
        
        if i2 is not None:
            children.append(i2)
        else:
            n_invalids += 1

    if debug:
        print('[ERROR] Soluções inválidas: ', n_invalids, n_invalids / n_children)

    return children[:n_children], n_invalids

# Aplica mutacao nos individuos de uma populacao
def _mutate(aig, population, rate, intensity):
    mutated_pop = []
    for p in population:
        # Copia indivíduo
        i = Individual(p.assignment.copy(), p.forwarding.copy())

        # Sorteia se ele deve sofrer mutação
        [ should_mutate ] = random.choices((True, False), weights=(rate, 1 - rate), k=1)

        # Se não precisa sofrer mutação, adiciona indivíduo sem nenhuma alteração
        if should_mutate == False:
            mutated_pop.append(i)
            continue

        # Seleciona genes que devem mudar
        num_changing_genes = int(len(i.assignment.keys()) * intensity)
        changing_genes = random.choices(list(i.assignment.keys()), k=num_changing_genes)

        for (gate, input_) in changing_genes:
            # Lista os candidatos e escolhe um aleatório                
            candidates = list(_candidates(aig, i.forwarding, gate, input_))
            choosed = random.choice(candidates)
            
            # Se for diferente, atualiza o indivíduo
            if (choosed != i.assignment[(gate, input_)]):
                # Remove aresta antiga do grafo e adiciona a nova
                # i.forwarding.remove_edge(i.assignment[(gate, input_)], gate)
                # i.forwarding.add_edge(choosed, gate)
                # Atualiza assignment
                i.assignment[(gate, input_)] = choosed
                i.forwarding = _forwarding(aig, i.assignment)

        mutated_pop.append(i)
    return mutated_pop

# Seleciona os individuos mais adaptados
def _natural_selection(old_generation, new_generation, elitism_rate):
    old_generation = sorted(old_generation, key=lambda p: p.score, reverse=True)
    new_generation = sorted(new_generation, key=lambda p: p.score, reverse=True)

    n_old_individuals = int(len(old_generation) * elitism_rate)
    n_new_individuals = len(new_generation) - n_old_individuals

    return old_generation[n_old_individuals:] + new_generation[n_new_individuals:]


def genetic(aig, entropy_data, params, seed=None, timeout=300, plot_results=False, plot_circuit=False, show_debug_messages=False):

    # Variável booleana que é define quando serão exibidas as mensagens de debug
    global debug
    debug = show_debug_messages

    # Variável utilizada para calcular tempo gasto em cada etapa
    global prev_time
    prev_time = time.time()
    initial_time = prev_time

    # Converte dicionario de entrada para uma classe mapeada
    params = ParamMap(params)

    # Define semente para randomização
    if seed is None:
        seed = random.randrange(2**32)
    random.seed(seed)
    np.random.seed(seed)

    # Valida entradas
    if params.w_delay + params.w_energy != 1:
        raise ValueError("A soma dos pesos deve ser igual a 1")

    # Simula circuito
    entropy_s = entropy_data

    # Calcula energia e profundidade iniciais
    initial_energy = evaluate.evaluate(aig, entropy_s)['total']
    initial_delay = _calc_delay(aig)

    if show_debug_messages:
        print('Energia e Delay inciais')
        print(initial_energy)
        print(initial_delay)

    # Inicia conjunto com todas as soluções
    all_individuals = set()

    # Inicia retorno com resultados ao longo do tempo
    evolutionary_results = {
        'global_best': [],
        'generation_best': [],
        'generation_worst': [],
        'solutions': []
    }

    # Conta o número total de indivíduos inválidos
    n_invalids = 0

    # Passo 1 - Definir a população inicial
    population = _init_population(aig, params.n_initial_individuals)
    _log('Definir população inicial')

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    _fit(population, entropy_s)
    _log('Avaliar a população inicial')
    
    # Inicia conjunto com todas as soluções
    all_individuals = set(population)
    evolutionary_results['solutions'].append(set(population))

    for i in range(params.n_generations):

        if time.time() - initial_time > timeout:
            _log('Timeout!')
            break

        # Encontra melhor e pior
        best = min(population, key=attrgetter('score'))
        worst = max(population, key=attrgetter('score'))
        evolutionary_results['global_best'].append(best.score)        
        if (i == 0):
            evolutionary_results['generation_best'].append(best.score)
            evolutionary_results['generation_worst'].append(worst.score)

        if show_debug_messages:
            print(str(i) + " - Melhor: " + str(best.score) + " - Pior: " + str(worst.score))

        # Passo 3 - Reprodução
        new_generation, new_invalids = _reproduce(aig, population, params.reproduction_rate, params.crossover_strategy)
        _log('Reprodução')

        # Passo 4 - Mutação
        new_generation = _mutate(aig, new_generation, params.mutation_rate, params.mutation_intensity)
        _log('Mutação')

        # Passo 5 - Fitness
        _fit(new_generation, entropy_s)
        _log('Fitness')

        # Passo 6 - Seleção natural
        population = _natural_selection(population, new_generation, params.elitism_rate)
        _log('Seleção natural')

        # Adiciona novas soluções
        all_individuals = all_individuals.union(set(new_generation))

        # Salva os resultados da geração
        evolutionary_results['generation_worst'].append(max(new_generation, key=attrgetter('score')).score)
        evolutionary_results['generation_best'].append(min(new_generation, key=attrgetter('score')).score)
        evolutionary_results['solutions'].append(set(new_generation))

        # Conta novos indivíduos inválidos
        n_invalids += new_invalids

    # Encontra melhor solução geral
    best = min(population, key=attrgetter('score'))
    evolutionary_results['global_best'].append(best.score)

    print("==== Melhor Solução ====")
    energy_score = 1 - (best.score / initial_energy)
    delay_score = 1 - (best.delay / initial_delay)
    print('Energia: ' + str(best.score) + '(' + str(energy_score) + '%)')
    print('Delay: ' + str(best.delay) + '(' + str(delay_score) + '%)')

    # Plota resultados
    if plot_results:
        points = np.array([[i.score, i.delay] for i in all_individuals])
        naive_points = [_get_naive_point(aig, naive.Strategy.ENERGY_ORIENTED), _get_naive_point(aig, naive.Strategy.DEPTH_ORIENTED)]
        pf.find_pareto_frontier(points, naive_points, plot=True)
        pf.evolution_over_generations(evolutionary_results)

    # Plota circuito
    if plot_circuit:
        graph.show(graph.default(best.forwarding))

    return { 
        'best_solution': best, 
        'solutions': all_individuals,
        'evolutionary_results': evolutionary_results,
        'seed': seed,
        'execution_time': time.time() - initial_time,
        'n_invalids': n_invalids
    }