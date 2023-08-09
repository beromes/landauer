import landauer.parse as parse
import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.framework as framework
import landauer.graph as graph
import networkx as nx
import numpy as np
import random
import json
import time
from operator import attrgetter

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

'''
Funcoes auxiliares
'''
def calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2

'''
Etapas algoritmo genetico
'''
def genetic_algorithm(aig, params):

    debug = False
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
    evolution_results = {
        'global_best': [],
        'generation_best': [],
        'generation_worst': []
    }

    # Retorna populacao inicial
    def init_population(aig, n_individuals):
        assignment = framework.assignment(aig)
        population = []

        for i in range(0, n_individuals):
            random_assignment = framework.randomize(aig, assignment)
            random_forwarding = framework.forwarding(aig, random_assignment)
            new_individual = Individual(random_assignment, random_forwarding)
            population.append(new_individual)

        return population

    # Funcao fitness que tenta minimizar a entropia
    def fit(population):
        for p in population:
            forwarding_ = framework.forwarding(aig, p.assignment)
            evaluation = evaluate.evaluate(forwarding_, entropy_s)
            p.score = evaluation['total']
            p.delay = calc_delay(forwarding_)
            #p.score = 1 - (evaluation['total'] / initial_energy)            

    # Faz reprodução dos individuos de uma populacao
    def reproduce(population, rate, min_score):
        n_children = int(len(population) * rate)
        children = []

        for i in range(0, n_children):            

            # Escolhe os parentes
            ordered_population = sorted(population, key=lambda p: p.score, reverse=True)
            weights = list(range(1, len(population) + 1)) # Peso é baseado na ordem
            weights = weights / np.sum(weights) # divide pela soma para probabilidade somar 1
            parent1, parent2 = np.random.choice(ordered_population, 2, replace=False, p=weights)

            # Recombina os genes
            assignment1 = parent1.assignment
            assignment2 = parent2.assignment

            child_assignment = assignment1.copy() # Filho inicialmente é a copia do primeiro pai
            child_forwarding = parent1.forwarding.copy()

            for gate, input_ in list(child_assignment.keys()):
                
                # Nao altera se a informacao for igual em ambos os pais
                if (assignment1[(gate, input_)] == assignment2[(gate, input_)]):
                    continue

                # Lista candidatos para uma determinada tupla
                candidates = list(framework.candidates2(aig, child_forwarding, gate, input_))

                # Verifica se tambem pode puxar o gene do outro parente
                if (assignment2[(gate, input_)] in candidates):
                    options = (assignment1[(gate, input_)], assignment2[(gate, input_)])
                    # Verifica se fará uma mudança (ou seja, se pegará a informação do segundo pai)
                    if (assignment2[(gate, input_)] == random.choice(options)):
                        # Atualiza assignment
                        child_assignment[(gate, input_)] = assignment2[(gate, input_)]
                        # Remove aresta antiga do grafo e adiciona a nova
                        child_forwarding.remove_edge(assignment1[(gate, input_)], gate)
                        child_forwarding.add_edge(assignment2[(gate, input_)], gate)

                        
            children.append(Individual(child_assignment, child_forwarding))

        return children

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
                candidates = list(framework.candidates2(aig, i.forwarding, gate, input_))
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

    # Passo 1 - Definir a população inicial
    population = init_population(aig, params.n_initial_individuals)

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    fit(population)
    if (debug):
        print('Init population = ' + str(time.time() - prev_time))
        prev_time = time.time()
    
    # Inicia conjunto com todas as soluções
    all_individuals = set(population)

    for i in range(0, params.n_generations):
        # Encontra melhor e pior
        best = min(population, key=attrgetter('score'))
        worst = max(population, key=attrgetter('score'))
        evolution_results['global_best'].append(best.score)
        if (i == 0):
            evolution_results['generation_best'].append(best.score)
            evolution_results['generation_worst'].append(worst.score)

        print("Melhor: " + str(best.score) + " - Pior: " + str(worst.score))
        if (debug):
            print('Find best and worst = ' + str(time.time() - prev_time))
            prev_time = time.time()

        # Reprodução
        new_generation = reproduce(population, params.reproduction_rate, worst.score)
        if (debug):
            print('Reproduce = ' + str(time.time() - prev_time))
            prev_time = time.time()

        # Mutação
        new_generation = mutate(new_generation, params.mutation_rate, params.mutation_intensity)
        if (debug):
            print('Mutation = ' + str(time.time() - prev_time))
            prev_time = time.time()

        # Calcula score dos novos indivíduos
        fit(new_generation)
        if (debug):
            print('Fit = ' + str(time.time() - prev_time))
            prev_time = time.time()

        # Adiciona soluções
        all_individsuals = all_individuals.union(set(new_generation))

        # Salva os resultados da geração
        evolution_results['generation_worst'].append(max(new_generation, key=attrgetter('score')).score)
        evolution_results['generation_best'].append(min(new_generation, key=attrgetter('score')).score)

        # Seleciona os mais aptos
        population = natural_selection(population, new_generation, params.elitism_rate)
        if (debug):
            print('Natural selection = ' + str(time.time() - prev_time))
            prev_time = time.time()


    # Encontra melhor e pior
    best = min(population, key=attrgetter('score'))
    worst = max(population, key=attrgetter('score'))
    print("Melhor: " + str(1 - (best.score / initial_energy)) + " - Pior: " + str(1 - (worst.score / initial_energy)))
    evolution_results['global_best'].append(best.score)

    return best, evolution_results, all_individuals