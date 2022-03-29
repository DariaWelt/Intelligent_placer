# DRAFT FILE!!!!! JUST IDEA FOR IMPROVEMENT
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass()
class OffspringArgs:
    #population_pool_size: int = 3
    mut_min_iter_num: int = 5
    mut_add_weight: float = 0.6
    mut_remove_weight: float = 0.1
    mut_modify_weight: float = 0.3
    mut_add_max_attempt_num: int = 10
    mut_modify_max_attempt_num:int = 3
    small_pos_change_prop: float = 0.2
    small_rotat_change_prop: float = 0.2
    mut_move_num: int = 15
    mut_move_min_dist_proportion: float = 0.03
    mut_rotate_num: int = 8
    mut_intermediate_selection_prob: float = 1.0
    crossover_ignore_mut_prob: float = 0.5
    crossover_max_attempt_num: int = 5
    crossover_shape_min_area_prop: float = 0.1
    crossover_shape_length_prop: Tuple[float, float] = (0.25, 0.75)
    crossover_poly_max_vertex_num: int = 10
    crossover_max_permutation_num: int = 5


class Problem():
    pass


def generate_population(problem, population_size, probability):
    pass


def select_parents(population, offspring_size, pool_size):
    pass


def generate_offspring(parents, offspring_args: OffspringArgs):
    pass


def get_surviving_population(non_elite, size, population_pool_size):
    pass


def get_fittest_solutions(population, elite_size):
    pass


def get_fitness(element):
    pass


def place_objects(polygon_mask: np.ndarray, objects_masks: List[np.ndarray], population_size=100,
                  ini_generation_spec_prop=0.5, offspring_size=200, elite_size=5, parent_pool_size=3,
                  population_pool_size: int = 3,
                  max_generation_num=30, converge_generation_num=15, offspring_args: OffspringArgs = OffspringArgs()):
    problem = Problem(polygon_mask, objects_masks)
    population = generate_population(problem, population_size, ini_generation_spec_prop)
    max_fitness = -np.inf
    not_improved = 0
    elite = list()

    for i in range(max_generation_num):
        parents = select_parents(population, offspring_size, parent_pool_size)
        offspring_result = generate_offspring(parents, offspring_args)
        extended_population = population + offspring_result
        elite = get_fittest_solutions(extended_population, elite_size)
        elite_fitness = get_fitness(elite[0]) if elite else -np.inf
        if elite_fitness > max_fitness:
            max_fitness = elite_fitness
            not_improved = 0
        else:
            not_improved += 1

        if not_improved >= converge_generation_num:
            break

        population = elite + get_surviving_population(
            [individual for individual in extended_population if individual not in elite], population_size - len(elite),
            population_pool_size)


    best_solution = elite[0] if elite else get_fittest_solutions(population, 1)
    return best_solution