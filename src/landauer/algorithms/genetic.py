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

import landauer.entropy as entropy
import landauer.evaluate as evaluate
import landauer.algorithms.naive as naive
import landauer.graph as graph
import landauer.pareto_frontier as pf
import landauer.placement as placement
import networkx as nx
import numpy as np
import random
import time
from operator import attrgetter
from enum import Enum, auto

'''
Classes/Modelos
'''
class CrossoverStrategy(Enum):
    LEVEL = auto()
    GATE = auto()
    INPUT = auto()

class ParamMap:
    name = 'Padrao'
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
    entropy_loss = 0
    delay = 0
    rank: int = None
    domination_count: int = None
    dominates: list['Individual'] = None
    crowdy_distance: float = None

    def __init__(self, assignment, forwarding):
        self.assignment = assignment
        self.forwarding = forwarding

    def is_dominated_by(self, other: 'Individual') -> bool:
        if self.entropy_loss == other.entropy_loss and self.delay == other.delay:
            return False
        return other.entropy_loss <= self.entropy_loss and other.delay <= self.delay

'''
Funcoes auxiliares
'''
def _calc_delay(aig):
    return len(nx.dag_longest_path(aig)) - 2

def get_naive_point(aig, strategy):
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

def _assignment(aig):
    assignment_ = dict()
    for node in aig.nodes():
        children = set(aig.successors(node))
        if len(children) >= 2:
            assignment_.update({(child, node): node for child in children})
    return assignment_

def _forwarding(aig, assignment):
    return placement.place(aig, assignment)

def _replace(aig, assignment, forwarding, slot, new_value):
    return placement.replace(aig, assignment, forwarding, slot, new_value)

def _randomize(aig, assignment, forwarding):
    assignment_ = assignment.copy()
    forwarding_ = forwarding.copy()
    assignment_items = list(assignment_.keys())
    random.shuffle(assignment_items)
    for gate, input_ in assignment_items:
        candidates = placement.candidates(aig, forwarding_, gate, input_)
        new_value = random.choice(candidates)
        assignment_, forwarding_ = _replace(aig, assignment_, forwarding_, (gate, input_), new_value)                

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

# Calcula perda de entropia e delay para os novos indivíduos e computa o rank de toda a população
def _fit_and_selection(cur_pop: list[Individual], new_gen: list[Individual], entropy_s):

    def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:

        # Cria uma lista de ranks. Cada item da lista corresponde a um rank
        ranks = []
        front = []

        # Para cada indivíduo, calcula por quantos indivíduos ele é dominado e quais ele domina
        for ind in population:
            ind.domination_count = 0
            ind.dominates = []
            for other in population:
                if ind.is_dominated_by(other):
                    ind.domination_count += 1
                elif other.is_dominated_by(ind):
                    ind.dominates.append(other)

            if (ind.domination_count == 0):
                ind.rank = 1
                front.append(ind)

        while len(front) > 0:
            ranks.append(front)
            next_front = [] # Proxima fronteira 
            for ind in front:
                for dominated_ind in ind.dominates:
                    dominated_ind.domination_count -= 1
                    if dominated_ind.domination_count == 0:
                        dominated_ind.rank = len(ranks) + 1
                        next_front.append(dominated_ind)
                            
            front = next_front

        return ranks

    # Calcula crowdy distance para o solucoes de um determinado rank
    def crowding_distance_assignment(samples: list[Individual]):
        for s in samples:
            s.crowdy_distance = 0

        objectives = ['entropy_loss', 'delay']

        # Acessa um objetivo de acordo com o parametro
        def objective(i: Individual, obj: str):
            if obj == 'entropy_loss': return i.entropy_loss
            if obj == 'delay': return i.delay
            return None

        for obj in objectives:

            # Ordena pelo objetivo: do menor para o maior
            samples.sort(key=lambda x: objective(x, obj)) # TODO: Testar se ordenacao funciona

            # Identifica diferenca entre os valores extremos para normalizacao
            max_diff = max(objective(samples[-1], obj) - objective(samples[0], obj), 1)

            # Pontos extremos tem distancia infinita
            samples[0].crowdy_distance = float('inf')
            samples[-1].crowdy_distance = float('inf')

            for i in range(1, len(samples) - 1):
                samples[i].crowdy_distance = samples[i].crowdy_distance + ((objective(samples[i+1], obj) - objective(samples[i-1], obj)) / (max_diff))

    # Calcula perda de entropia e delay para os novos indivíduos
    for i in new_gen:
        evaluation = evaluate.evaluate(i.forwarding, entropy_s)
        i.entropy_loss = float(evaluation['total'])
        i.delay = _calc_delay(i.forwarding)

    # Combina populacao antiga com nova geração e separa por ranks
    ranks = fast_non_dominated_sort(cur_pop + new_gen)

    # Calcula crowdy distance para cada rank até preencher o tamanho da população
    pop_size = len(new_gen) # Tamanho da nova população é igual o da anterior    
    pop, i = [], 0
    while len(pop) < pop_size:
        crowding_distance_assignment(ranks[i]) # Calcula crowdy distance

        # Verifica se cabe o novo rank inteiro
        if (len(pop) + len(ranks[i]) <= pop_size):            
            pop += ranks[i] # Adiciona individuos na populacao
            i += 1 # Passa para o próximo rank

        # Completa espacos restantes na populacao com base em crowdy distance
        else:
            ranks[i].sort(key=lambda x: x.crowdy_distance, reverse=True)
            pop += ranks[i][0:pop_size - len(pop)]

    return pop

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
        #return nx.has_path(forwarding, dest_gate, origin_gate) == False

    # Monta um indivíduo a partir das duas partes do crossover
    def make_individual(aig, first_half, second_half):
        assignment = _assignment(aig)
        forwarding = _forwarding(aig, assignment)
        n_invalid_items = 0

        for key in assignment.keys():
            new_value = first_half[key] if key in first_half else second_half[key]
            
            if is_valid(forwarding, key[0], new_value):
                assignment, forwarding = _replace(aig, assignment, forwarding, key, new_value)
            else:
                n_invalid_items += 1

        return Individual(assignment, forwarding), n_invalid_items
    
    def tournament_selection(population: list[Individual]):
        c1, c2 = np.random.choice(population, 2, replace=False)
        if c1.rank == c2.rank:
            return c1 if c1.crowdy_distance >= c2.crowdy_distance else c2
        else:
            return c1 if c1.rank < c2.rank else c2


    n_children = int(len(population) * rate)
    children = []

    # Conta número de soluções inválidas
    n_invalids = 0
    
    # Conta a quantidade de itens que nao foram modificados
    n_items = len(population[0].assignment.items()) * n_children
    n_invalid_items = 0

    while len(children) < n_children:
        # Escolhe os parentes
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        # Separa os genes de acordo com a estratégia
        splitted_p1 = split_assignment(p1, strategy)
        splitted_p2 = split_assignment(p2, strategy)

        child1, n_invalid_items1 = make_individual(aig, splitted_p1[0], splitted_p2[1])
        child2, n_invalid_items2 = make_individual(aig, splitted_p2[0], splitted_p1[1])
        children.extend([child1, child2])

        n_invalid_items += n_invalid_items1 + n_invalid_items2

        if debug:
            invalids = n_invalid_items1 + n_invalid_items2
            n_edges = len(child1.assignment.items()) * 2
            print('[ERROR] Atribuições invalidas na reprodução: ', invalids, (invalids / n_edges) * 100)

    return children[:n_children], n_invalids, n_invalid_items / n_items


# Faz reprodução comparando gene a gene. Evita criação de indivíduos inválidos
def _old_reproduce(aig, population, rate):

    def crossover(p1, p2):
        # Cria filho a partir da cópia dos pais
        child = Individual(p1.assignment.copy(), p1.forwarding.copy())

        for key, current_value in p1.assignment.items():
            incoming_value = p2.assignment[key]

            # Se a informação for a mesma em ambos os pais, não faz nada
            if current_value == incoming_value:
                continue

            # Verifica se é possível trocar a informação
            if placement.is_candidate(child.forwarding, key[0], key[1], incoming_value):
                # Faz a troca do gene pela informação do segundo pai
                child.assignment, child.forwarding = _replace(aig, child.assignment, child.forwarding, key, incoming_value)

        return child

    n_children = int(len(population) * rate)
    children = []

    # Ordena a população e define os pesos
    ordered_population = sorted(population, key=lambda p: p.entropy_loss, reverse=True)
    weights = list(range(1, len(population) + 1)) # Peso é baseado na ordem
    weights = weights / np.sum(weights) # divide pela soma dos pesos para que a soma total seja 1

    while len(children) < n_children:
        # Escolhe os parentes
        p1, p2 = np.random.choice(ordered_population, 2, replace=False, p=weights) # Escolhe dois parentes sem reposição

        # Cria filhos a partir da cópia dos pais
        children.append(crossover(p1, p2))
        children.append(crossover(p2, p1))

    return children[:n_children], 0, 0


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
                i.assignment[(gate, input_)] = choosed
                i.forwarding = _forwarding(aig, i.assignment)

        mutated_pop.append(i)
    return mutated_pop


def _get_pareto_info(pop: list[Individual]):
    pareto = list(filter(lambda p: p.rank == 1, pop))
    n_discovered = len(set(map(lambda p: str(p.entropy_loss) + '-' + str(p.delay), pareto)))

    pareto.sort(key=lambda p: p.entropy_loss)

    return {
        'n_discovered': n_discovered,
        'min_entropy_loss': pareto[0],
        'min_delay': pareto[-1],
        'middle': pareto[len(pareto) // 2]
    }

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
    sum_percentual_invalid_items = 0

    # Passo 1 - Definir a população inicial
    population = _init_population(aig, params.n_initial_individuals)
    _log('Definir população inicial')

    # Passo 2 - Aplicar funcao fitness na populacao inicial
    population = _fit_and_selection([], population, entropy_s)
    _log('Avaliar a população inicial')
    
    # Inicia conjunto com todas as soluções
    all_individuals = set(population)
    evolutionary_results['solutions'].append(set(population))

    for i in range(params.n_generations):

        if time.time() - initial_time > timeout:
            _log('Timeout!')
            break

        # Encontra melhor e pior
        best = min(population, key=attrgetter('entropy_loss'))
        worst = max(population, key=attrgetter('entropy_loss'))
        evolutionary_results['global_best'].append(best.entropy_loss)        
        if (i == 0):
            evolutionary_results['generation_best'].append(best.entropy_loss)
            evolutionary_results['generation_worst'].append(worst.entropy_loss)

        if show_debug_messages:
            print(str(i) + " - Melhor: " + str(best.entropy_loss) + " - Pior: " + str(worst.entropy_loss))

        # Passo 3 - Reprodução
        new_generation, new_invalids, percentual_invalid_items = _reproduce(aig, population, params.reproduction_rate, params.crossover_strategy)
        _log('Reprodução')

        # Passo 4 - Mutação
        new_generation = _mutate(aig, new_generation, params.mutation_rate, params.mutation_intensity)
        _log('Mutação')

        # Passo 5 - Fitness
        population = _fit_and_selection(population, new_generation, entropy_s)
        _log('Fitness & Seleção')

        # Adiciona novas soluções
        all_individuals = all_individuals.union(set(new_generation))

        # Salva os resultados da geração
        evolutionary_results['generation_worst'].append(max(population, key=attrgetter('entropy_loss')).entropy_loss)
        evolutionary_results['generation_best'].append(min(population, key=attrgetter('entropy_loss')).entropy_loss)
        evolutionary_results['solutions'].append(set(population))

        # Conta novos indivíduos inválidos
        n_invalids += new_invalids

        # Soma percentual de atribuicoes invalidas
        sum_percentual_invalid_items += percentual_invalid_items

    # Encontra melhor solução geral
    best = min(population, key=attrgetter('entropy_loss'))
    evolutionary_results['global_best'].append(best.entropy_loss)

    print("==== Melhor Solução ====")
    energy_score = 1 - (best.entropy_loss / initial_energy)
    delay_score = 1 - (best.delay / initial_delay)
    print('Energia: ' + str(best.entropy_loss) + '(' + str(energy_score) + '%)')
    print('Delay: ' + str(best.delay) + '(' + str(delay_score) + '%)')

    # Plota resultados
    if plot_results:
        pf.evolution_over_generations(evolutionary_results)

    # Plota circuito
    if plot_circuit:
        graph.show(graph.default(best.forwarding))

    n_executed_generations = len(evolutionary_results['generation_best'])

    return { 
        'best_solution': best, 
        'solutions': all_individuals,
        'evolutionary_results': evolutionary_results,
        'seed': seed,
        'execution_time': time.time() - initial_time,
        'n_invalids': n_invalids,
        'percentual_invalid_items': sum_percentual_invalid_items / len(evolutionary_results['generation_best']),
        'n_executed_generations': n_executed_generations,
        'pareto_info': _get_pareto_info(evolutionary_results['solutions'][-1])
    }