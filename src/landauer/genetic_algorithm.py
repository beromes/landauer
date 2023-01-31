import landauer.parse as parse
import landauer.simulate as simulate
import landauer.evaluate as evaluate
import landauer.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import networkx as nx
import random
import json

def calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2

def genetic_algorithm(aig, w_energy, w_delay, n_generations = 10, n_initial_individuals = 8, reproduction_rate = 1, mutation_rate = 0.2):

    # Valida entradas
    if w_delay + w_energy != 1:
        raise ValueError("A soma dos pesos deve ser igual a 1")

    # Simula circuito
    simulation = simulate.simulate(aig)

    # Calcula energia e profundidade iniciais
    initial_energy = evaluate.evaluate(aig, simulation)['total']
    # initial_delay = calc_delay(aig)

    # print('Energia e Delay inciais')
    # print(initial_energy)
    # print(initial_delay)

    # Retorna populacao inicial
    def initial_population(aig, n_individuals):
        assignment = framework.assignment(aig)
        population = []

        for i in range(0, n_individuals):
            population.append(framework.randomize(aig, assignment))

        return population

    # Faz reprodução dos individuos de uma populacao
    def reproduce(population, scores, rate):
        n_children = len(population) * rate
        children = []

        for i in range(0, n_children):
            
            # Escolhe os parentes
            min_score = min(scores)
            weights = list(map(lambda s: s - min_score + 1, scores))            
            parents = random.choices(population, weights=weights, k=2)

            # Recombina os genes
            child = parents[0].copy() # Filho inicialmente é a copia do primeiro pai
            for gate, input_ in list(child.keys()):
                # Lista candidatos para uma determinada tupla
                candidates = list(framework.candidates(aig, child, gate, input_))
                
                # Verifica se tambem pode puxar o gene do outro parente
                if (parents[1][(gate, input_)] in candidates):
                    options = (parents[0][(gate, input_)], parents[1][(gate, input_)])
                    child[(gate, input_)] = random.choice(options)

            children.append(child)

        return children

    # Aplica mutacao nos individuos de uma populacao
    def mutate(population, rate):        
        for i in population:
            [ should_mutate ] = random.choices((True, False), weights=(rate, 1 - rate), k=1)
            if should_mutate:
                gate, input_ = random.choice(list(i.keys()))
                candidates = list(framework.candidates(aig, i, gate, input_))
                i[(gate, input_)] = random.choice(candidates)

        return population

    # Funcao fitness
    # def fit(population):
    #     scores = []
    #     for p in population:            
    #         # Avalia individuo
    #         forwarding_ = framework.forwarding(aig, p)
    #         evaluation = evaluate.evaluate(forwarding_, simulation)

    #         # Calcula score de energia e delay
    #         energy_score = 1 - (evaluation['total'] / initial_energy)
    #         delay_score = 1 - (calc_delay(forwarding_) / initial_delay)

    #         # Retorna os scores ponderados com os pesos
    #         scores.append((energy_score * w_energy) + (delay_score * w_delay))
    #     return scores

    def fit(population):
        scores = []
        for p in population:
            forwarding_ = framework.forwarding(aig, p)
            evaluation = evaluate.evaluate(forwarding_, simulation)
            scores.append(1 - (evaluation['total'] / initial_energy))
        return scores

    # Seleciona os individuos mais adaptados
    def natural_selection(population, scores):
        half_index = int(len(population) / 2)
        list_ = sorted(zip(scores, population), key=lambda x: x[0])
        list_ = list_[half_index:]
        return [p for s, p in list_], [s for s, p in list_]

    # Passo 1 - Definir a população inicial
    population = initial_population(aig, n_initial_individuals)

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    scores = fit(population)

    # print('Scores inicais')
    # print(scores)

    for i in range(0, n_generations):
        # Reprodução
        new_generation = reproduce(population, scores, reproduction_rate)

        # Mutação
        new_generation = mutate(new_generation, mutation_rate)

        # Calcula score dos novos indivíduos
        scores += fit(new_generation)
        population += new_generation

        # Seleciona os mais aptos
        population, scores = natural_selection(population, scores)

        print('Pontuação máxima: ' + str(max(scores)))

    max_score = max(scores)
    max_score_i = scores.index(max_score)
    return population[max_score_i]