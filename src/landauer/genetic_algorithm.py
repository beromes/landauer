import landauer.parse as parse
import landauer.simulate as simulate
import landauer.evaluate as evaluate
import landauer.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import networkx as nx
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
    mutation_based = False

    def __init__(self, dictt):
        if dictt is dict:          
            for key in dictt:
                setattr(self, key, dictt[key])

class Individual:
    assignment = None
    score = 0

    def __init__(self, assignment):
        self.assignment = assignment

'''
Funcoes auxiliares
'''
def calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2


'''
Etapas algoritmo genetico
'''
def genetic_algorithm(aig, params):

    prev_time = time.time()

    # Converte dicionario de entrada para uma classe mapeada
    params = ParamMap(params)

    # Valida entradas
    if params.w_delay + params.w_energy != 1:
        raise ValueError("A soma dos pesos deve ser igual a 1")

    # Simula circuito
    simulation = simulate.simulate(aig)

    # Calcula energia e profundidade iniciais
    initial_energy = evaluate.evaluate(aig, simulation)['total']
    initial_delay = calc_delay(aig)

    print('Energia e Delay inciais')
    print(initial_energy)
    print(initial_delay)

    # Retorna populacao inicial
    def init_population(aig, n_individuals):
        assignment = framework.assignment(aig)
        population = []

        for i in range(0, n_individuals):
            new_individual = Individual(framework.randomize(aig, assignment))
            population.append(new_individual)

        return population

    # Funcao fitness que tenta minimizar a entropia
    def fit(population):
        for p in population:
            forwarding_ = framework.forwarding(aig, p.assignment)
            evaluation = evaluate.evaluate(forwarding_, simulation)
            p.score = 1 - (evaluation['total'] / initial_energy)

    # Faz reprodução dos individuos de uma populacao
    def reproduce(population, rate, min_score):
        n_children = int(len(population) * rate)
        children = []

        for i in range(0, n_children):            

            # Escolhe os parentes
            weights = list(map(lambda p: p.score / min_score, population))
            parents = random.choices(population, weights=weights, k=2)

            # Recombina os genes
            assignment1 = parents[0].assignment
            assignment2 = parents[1].assignment
            child = assignment1.copy() # Filho inicialmente é a copia do primeiro pai

            for gate, input_ in list(child.keys()):
                
                # Nao altera se a informacao for igual em ambos os pais
                if (assignment1[(gate, input_)] == assignment2[(gate, input_)]):
                    continue

                # Lista candidatos para uma determinada tupla
                candidates = list(framework.candidates(aig, child, gate, input_))

                # Verifica se tambem pode puxar o gene do outro parente
                if (assignment2[(gate, input_)] in candidates):
                    options = (assignment1[(gate, input_)], assignment2[(gate, input_)])
                    child[(gate, input_)] = random.choice(options)
                        
            children.append(Individual(child))

        return children

    # Aplica mutacao nos individuos de uma populacao
    def mutate(population, rate):
        mutated = []
        for p in population:
            i = Individual(p.assignment.copy())
            [ should_mutate ] = random.choices((True, False), weights=(rate, 1 - rate), k=1)
            if should_mutate:
                gate, input_ = random.choice(list(i.assignment.keys()))
                candidates = list(framework.candidates(aig, i.assignment, gate, input_))
                i.assignment[(gate, input_)] = random.choice(candidates)
            mutated.append(i)
        return mutated

    # Seleciona os individuos mais adaptados
    def natural_selection(population):
        half_index = int(len(population) / 2)
        list_ = sorted(population, key=lambda p: p.score)
        return list_[half_index:]

    # Passo 1 - Definir a população inicial
    population = init_population(aig, params.n_initial_individuals)

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    fit(population)

    print('Init population = ' + str(time.time() - prev_time))
    prev_time = time.time()

    for i in range(0, params.n_generations):
        # Encontra melhor e pior
        best = max(population, key=attrgetter('score'))
        worst = min(population, key=attrgetter('score'))
        print("Melhor: " + str(best.score) + " - Pior: " + str(worst.score))

        print('Find best and worst = ' + str(time.time() - prev_time))
        prev_time = time.time()

        # Reprodução
        if (params.mutation_based == False):
            new_generation = reproduce(population, params.reproduction_rate, worst.score)
            print('Reproduce = ' + str(time.time() - prev_time))
            prev_time = time.time()

        # Mutação
        if (params.mutation_based == True):
            new_generation = population
        new_generation = mutate(new_generation, params.mutation_rate)
        print('Mutation = ' + str(time.time() - prev_time))
        prev_time = time.time()

        # Calcula score dos novos indivíduos
        fit(new_generation)
        print('Fit = ' + str(time.time() - prev_time))
        prev_time = time.time()

        # Seleciona os mais aptos

        # Estratégia 1
        # population += new_generation
        # population = natural_selection(population)

        # Estratégia 2
        population = natural_selection(population)
        new_generation = natural_selection(new_generation)
        population += new_generation

        print('Natural selection = ' + str(time.time() - prev_time))
        prev_time = time.time()


    # Encontra melhor e pior
    best = max(population, key=attrgetter('score'))
    worst = min(population, key=attrgetter('score'))
    print("Melhor: " + str(best.score) + " - Pior: " + str(worst.score))

    return best