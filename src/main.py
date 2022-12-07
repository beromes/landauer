import landauer.parse as parse
import landauer.simulate as simulate
import landauer.evaluate as evaluate
import landauer.naive as naive
import landauer.framework as framework
import landauer.graph as graph
import random


def genetic_algorithm(circuit_description, w_energy, w_delay):

    # Valida entradas
    if w_delay + w_energy != 1:
        raise ValueError("A soma dos pesos deve ser igual a 1")

    # Converte circuito em grafo de porta AND
    aig = parse.parse(circuit_description)

    # Simula circuito
    simulation = simulate.simulate(aig)

    # Calcula energia e profundidade iniciais
    initial_energy = evaluate.evaluate(aig, simulation)['total']
    initial_delay = 5 # TODO: Calcular delay da solucao


    # Retorna populacao inicial
    def initial_population(aig, n_individuals = 4):
        assignment = framework.assignment(aig)
        population = []

        for i in range(0, n_individuals):
            population.append(framework.randomize(aig, assignment))

        return population

    # Faz reprodução dos individuos de uma populacao
    def reproduce(population, scores, rate = 1):
        n_children = len(population) * rate
        children = []

        for i in range(0, n_children):
            
            # Escolhe os pais
            total = sum(scores)
            weights = list(map(lambda s: s / total, scores))
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
    def mutate(population, rate = 0.05):        
        for i in population:
            should_mutate = random.choices((True, False), weights=(rate, 1 - rate), k=1)
            if should_mutate:
                gate, input_ = random.choice(list(i.keys()))
                candidates = list(framework.candidates(aig, i, gate, input_))
                i[(gate, input_)] = random.choice(candidates)

        return population

    # Funcao fitness
    def fit(population):
        scores = []
        for p in population:            
            # Avalia individuo
            forwarding_ = framework.forwarding(aig, p)
            evaluation = evaluate.evaluate(forwarding_, simulation)

            # Calcula score de energia e delay
            energy_score = evaluation['total'] #(evaluation['total'] - initial_energy) / initial_energy
            delay_score = (5 - initial_delay) / initial_delay # TODO: Calcular delay da solucao

            # Retorna os scores ponderados com os pesos
            scores.append((energy_score * w_energy) + (delay_score * w_delay))
        return scores

    # Seleciona os individuos mais adaptados
    def natural_selection(population, scores):
        half_index = int(len(population) / 2)
        list_ = sorted(zip(scores, population), key=lambda x: x[0])
        list_ = list_[:half_index]
        return [p for s, p in list_], [s for s, p in list_]

    # Passo 1 - Definir a população inicial
    population = initial_population(aig)

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    scores = fit(population)

    print('Initial scores')
    print(scores)

    for i in range(0, 10):
        # Reprodução
        new_generation = reproduce(population, scores)

        # Mutação
        new_generation = mutate(new_generation)

        # Calcula score dos novos indivíduos
        scores += fit(new_generation)
        population += new_generation

        # Seleciona os mais aptos
        population, scores = natural_selection(population, scores)

        print(scores)


def main():

    half_adder = '''
        module half_adder (a, b, sum, cout);
            input a, b;
            output sum, cout;

            assign sum = a ^ b;
            assign cout = a & b;
        endmodule
    '''
    aig = parse.parse(half_adder)
    simulation = simulate.simulate(aig)

    genetic_algorithm(half_adder, 1, 0)

    # framework.colorize(result)
    # print(result)
    # graph.show(graph.default(result))

main()