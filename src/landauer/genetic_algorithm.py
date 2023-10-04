import landauer.parse as parse
import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.framework as framework
import landauer.algorithms.naive as naive
import landauer.graph as graph
import landauer.pareto_frontier as pf
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
class ParamMap:
    name = 'Padrao'
    w_energy = 1
    w_delay = 0
    n_generations = 1000
    n_initial_individuals = 50
    reproduction_rate = 1.0
    mutation_rate = 0.2
    mutation_intensity = 0.8
    elitism_rate = 0.1

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

class CrossoverStrategy(Enum):
    LEVEL = auto()
    GATE = auto()
    INPUT = auto()

'''
Funcoes auxiliares
'''
def calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2

def get_naive_point(aig, strategy):
    entropy_s = entropy.entropy(aig)
    aig_naive = naive.naive(aig, strategy)
    assignment_naive = framework.assignment(aig_naive)
    forwarding_naive = framework.forwarding(aig_naive, assignment_naive)
    evaluation_naive = evaluate.evaluate(forwarding_naive, entropy_s)
    naive_point = [evaluation_naive['total'], calc_delay(aig_naive)]
    print('Naive - ' + str(strategy))
    print('Energy: ' + str(evaluation_naive['total']))
    print('Delay: ' + str(calc_delay(aig_naive)))
    return naive_point

'''
Etapas algoritmo genetico
'''
def genetic_algorithm(aig, params, plot_results=True, plot_circuit=False, debug=False):

    # Variável utilizada para calcular tempo gasto em cada etapa
    global prev_time
    prev_time = time.time()

    # Converte dicionario de entrada para uma classe mapeada
    params = ParamMap(params)

    # Valida entradas
    if params.w_delay + params.w_energy != 1:
        raise ValueError("A soma dos pesos deve ser igual a 1")

    # Simula circuito
    entropy_s = entropy.entropy(aig)

    # Calcula energia e profundidade iniciais
    initial_energy = evaluate.evaluate(aig, entropy_s)['total']
    initial_delay = calc_delay(aig)

    print('Energia e Delay inciais')
    print(initial_energy)
    print(initial_delay)

    # Inicia conjunto com todas as soluções
    all_individuals = set()

    # Inicia retorno com resultados ao longo do tempo
    evolutionary_results = {
        'global_best': [],
        'generation_best': [],
        'generation_worst': []
    }

    # Retorna populacao inicial
    def init_population(aig, n_individuals):
        assignment = framework.assignment(aig)
        population = []

        for i in range(n_individuals):
            random_assignment = framework.randomize(aig, assignment)
            random_forwarding = framework.forwarding(aig, random_assignment)
            new_individual = Individual(random_assignment, random_forwarding)
            population.append(new_individual)

        return population

    # Calcula o score baseado na perda de entropia, também calcula o delay
    def fit(population):
        for p in population:
            p.forwarding = framework.forwarding(aig, p.assignment) # TODO: Entender porque essa linha é necessária!
            evaluation = evaluate.evaluate(p.forwarding, entropy_s)
            p.score = evaluation['total']
            p.delay = calc_delay(p.forwarding)

    # Faz reprodução dos individuos de uma populacao
    def reproduce(population, rate, strategy = CrossoverStrategy.INPUT):

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

        def is_valid(forwarding, gate, value):
            return value not in nx.descendants(forwarding, gate)                    

        # Corrige o invidíduo após juntar partes arbritariamente
        def make_individual(assignment):
            forwarding = framework.forwarding(aig, assignment)
            invalid_edges = list(assignment.items())

            while len(invalid_edges) > 0:
                new_invalid_edges = []

                for (key, value) in invalid_edges:
                    candidates = list(framework.candidates(aig, assignment, key[0], key[1]))

                    if value in candidates:
                        continue
                    
                    # TODO: Remover prints quando ficarem irrelevantes
                    # print('Inválido!', key, value)
                    if (len(candidates) == 0):
                        # print('Nenhum candidato! Voltarei para resolver depois')
                        new_invalid_edges.append((key, value))
                        continue                        

                    new_value = random.choice(candidates)
                    assignment[key] = new_value
                    forwarding.remove_edge(value, key[0])
                    forwarding.add_edge(new_value, key[0])

                # Caso entre em loop e não consiga resolver as divergências, retorna nulo
                if (invalid_edges == new_invalid_edges):
                    if debug:
                        print('Solução Inválida')
                    return None;

                invalid_edges = new_invalid_edges

            return Individual(assignment, forwarding)


        n_children = int(len(population) * rate)
        children = []

        while len(children) < n_children:
            # Escolhe os parentes
            ordered_population = sorted(population, key=lambda p: p.score, reverse=True)
            weights = list(range(1, len(population) + 1)) # Peso é baseado na ordem
            weights = weights / np.sum(weights) # divide pela soma dos pesos para que a soma total seja 1
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

            i1, i2 = make_individual(child1), make_individual(child2)

            if i1 is not None:
                children.append(make_individual(child1))
            
            if i2 is not None:
                children.append(make_individual(child2))

        return children[:n_children]

    # Aplica mutacao nos individuos de uma populacao
    def mutate(population, rate, intensity):
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
                candidates = list(framework.candidates2(aig, i.forwarding, gate, input_)) # TODO: substituir candidates2 por uma alternativa mais eficiente
                choosed = random.choice(candidates)
                
                # Se for diferente, atualiza o indivíduo
                if (choosed != i.assignment[(gate, input_)]):
                    # Remove aresta antiga do grafo e adiciona a nova
                    i.forwarding.remove_edge(i.assignment[(gate, input_)], gate)
                    i.forwarding.add_edge(choosed, gate)
                    # Atualiza assignment
                    i.assignment[(gate, input_)] = choosed

            mutated_pop.append(i)
        return mutated_pop

    # Seleciona os individuos mais adaptados
    def natural_selection(old_generation, new_generation, elitism_rate):
        old_generation = sorted(old_generation, key=lambda p: p.score, reverse=True)
        new_generation = sorted(new_generation, key=lambda p: p.score, reverse=True)

        n_old_individuals = int(len(old_generation) * elitism_rate)
        n_new_individuals = len(new_generation) - n_old_individuals

        return old_generation[n_old_individuals:] + new_generation[n_new_individuals:]

    def log(message):
        if debug == False: return

        global prev_time
        print('[TIME] ' + message + ' = ' + str(time.time() - prev_time))
        prev_time = time.time()


    # Passo 1 - Definir a população inicial
    population = init_population(aig, params.n_initial_individuals)
    log('Definir população inicial')

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    fit(population)
    log('Avaliar a população inicial')
    
    # Inicia conjunto com todas as soluções
    all_individuals = set(population)

    for i in range(params.n_generations):
        # Encontra melhor e pior
        best = min(population, key=attrgetter('score'))
        worst = max(population, key=attrgetter('score'))
        evolutionary_results['global_best'].append(best.score)
        if (i == 0):
            evolutionary_results['generation_best'].append(best.score)
            evolutionary_results['generation_worst'].append(worst.score)

        print("Melhor: " + str(best.score) + " - Pior: " + str(worst.score))

        # Passo 3 - Reprodução
        new_generation = reproduce(population, params.reproduction_rate)
        log('Reprodução')

        # Passo 4 - Mutação
        new_generation = mutate(new_generation, params.mutation_rate, params.mutation_intensity)
        log('Mutação')

        # Passo 5 - Fitness
        fit(new_generation)
        log('Fitness')

        # Passo 6 - Seleção natural
        population = natural_selection(population, new_generation, params.elitism_rate)
        log('Seleção natural')

        # Adiciona novas soluções
        all_individuals = all_individuals.union(set(new_generation))

        # Salva os resultados da geração
        evolutionary_results['generation_worst'].append(max(new_generation, key=attrgetter('score')).score)
        evolutionary_results['generation_best'].append(min(new_generation, key=attrgetter('score')).score)


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
        naive_points = [get_naive_point(aig, naive.Strategy.ENERGY_ORIENTED), get_naive_point(aig, naive.Strategy.DELAY_ORIENTED)]
        pf.find_pareto_frontier(points, naive_points, plot=True)
        pf.evolution_over_generations(evolutionary_results)

    # Plota circuito
    if plot_circuit:
        result = framework.forwarding(aig, best.assignment)
        framework.colorize(result)
        graph.show(graph.default(result))

    return { 
        'best_solution': best, 
        'solutions': all_individuals,
        'evolutionary_results': evolutionary_results
    }