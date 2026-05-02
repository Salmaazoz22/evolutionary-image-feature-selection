"""
ga.py — Genetic Algorithm for Binary Feature Selection
Extracted from notebooks/04_GA_Feature_Selection.ipynb
"""

import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# ── Constants ────────────────────────────────────────────────
ALPHA                   = 0.9
MIN_FEATURES            = 10
INIT_FEAT_PROB          = 0.3
CROSSOVER_RATE          = 0.8
DEFAULT_MUT_RATE        = 0.01
UNIFORM_SWAP_PROB       = 0.5
TOURNAMENT_SIZE         = 3
EARLY_STOPPING_PATIENCE = 8
RANDOM_SEED             = 42


# ── Population Initialisation ────────────────────────────────

def initialize_population(population_size, n_features, p_feat=INIT_FEAT_PROB):
    """
    Generate the initial population.
    Each bit is independently set to 1 with probability p_feat.
    Enforces MIN_FEATURES selected per individual.
    """
    population = (np.random.rand(population_size, n_features) < p_feat).astype(int)
    for i in range(population_size):
        if population[i].sum() < MIN_FEATURES:
            idx = np.random.choice(n_features, MIN_FEATURES, replace=False)
            population[i] = 0
            population[i, idx] = 1
    return population


# ── Fitness ──────────────────────────────────────────────────

def evaluate_fitness(individual, X_tr, y_tr, alpha=ALPHA):
    """
    Compute the fitness of a single binary individual.
    fitness = alpha * accuracy + (1 - alpha) * (1 - n_selected / n_total)
    Uses 2-fold CV with LinearSVC (fast; no test leakage).
    Returns (fitness, accuracy, n_selected).
    """
    selected_idx = np.where(individual == 1)[0]
    n_sel = len(selected_idx)
    if n_sel < MIN_FEATURES:
        return 0.0, 0.0, n_sel

    X_sub = X_tr[:, selected_idx]
    svm = LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=1000)
    cv_scores = cross_val_score(svm, X_sub, y_tr, cv=2, scoring='accuracy', n_jobs=-1)
    accuracy = cv_scores.mean()

    n_total = len(individual)
    fitness = alpha * accuracy + (1.0 - alpha) * (1.0 - n_sel / n_total)
    return fitness, accuracy, n_sel


def fitness_sharing(population, raw_fitnesses):
    """
    Distance-based fitness sharing (niching) using Hamming distance.
    Penalises individuals in crowded fitness regions to promote diversity.
    """
    pop_size, n_features = population.shape
    sigma_share = n_features * 0.1
    shared = np.empty(pop_size, dtype=float)
    for i in range(pop_size):
        niche_count = 0.0
        for j in range(pop_size):
            d = np.count_nonzero(population[i] != population[j])
            if d < sigma_share:
                niche_count += 1.0 - (d / sigma_share)
        shared[i] = raw_fitnesses[i] / max(niche_count, 1e-9)
    return shared


def evaluate_population(population, X_tr, y_tr, alpha=ALPHA):
    """
    Evaluate fitness for all individuals.
    Applies fitness sharing after raw evaluation to promote diversity.
    Returns (fitnesses, accuracies, n_selected).
    """
    results = [evaluate_fitness(ind, X_tr, y_tr, alpha) for ind in population]
    fitnesses  = np.array([r[0] for r in results])
    accuracies = np.array([r[1] for r in results])
    n_selected = np.array([r[2] for r in results])
    fitnesses = fitness_sharing(population, fitnesses)
    return fitnesses, accuracies, n_selected


# ── Selection ────────────────────────────────────────────────

def tournament_selection(population, fitnesses, tournament_size=TOURNAMENT_SIZE):
    """Select one individual via tournament selection."""
    candidates = np.random.choice(len(population), tournament_size, replace=False)
    best_idx   = candidates[np.argmax(fitnesses[candidates])]
    return population[best_idx].copy()


def roulette_wheel_selection(population, fitnesses):
    """Select one individual via roulette wheel (fitness-proportionate)."""
    shifted = fitnesses - fitnesses.min() + 1e-6
    probs   = shifted / shifted.sum()
    idx     = np.random.choice(len(population), p=probs)
    return population[idx].copy()


def select_individual(population, fitnesses, method='tournament'):
    """Dispatch to the correct selection operator ('tournament' | 'roulette')."""
    if method == 'tournament':
        return tournament_selection(population, fitnesses)
    elif method == 'roulette':
        return roulette_wheel_selection(population, fitnesses)
    else:
        raise ValueError(f"Unknown selection method: '{method}'. Use 'tournament' or 'roulette'.")


# ── Crossover ────────────────────────────────────────────────

def single_point_crossover(parent1, parent2):
    """Single-point crossover. Returns (child1, child2)."""
    n = len(parent1)
    k = np.random.randint(1, n)
    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])
    return child1, child2


def two_point_crossover(parent1, parent2):
    """Two-point crossover. Returns (child1, child2)."""
    n  = len(parent1)
    k1, k2 = sorted(np.random.choice(np.arange(1, n), size=2, replace=False))
    child1 = np.concatenate([parent1[:k1], parent2[k1:k2], parent1[k2:]])
    child2 = np.concatenate([parent2[:k1], parent1[k1:k2], parent2[k2:]])
    return child1, child2


def uniform_crossover(parent1, parent2, swap_prob=UNIFORM_SWAP_PROB):
    """Uniform crossover. Each gene independently inherited with probability swap_prob."""
    mask   = np.random.rand(len(parent1)) < swap_prob
    child1 = np.where(mask, parent2, parent1)
    child2 = np.where(mask, parent1, parent2)
    return child1, child2


def crossover(parent1, parent2, method='two_point', crossover_rate=CROSSOVER_RATE):
    """
    Apply crossover with probability crossover_rate.
    method: 'single_point' | 'two_point' | 'uniform'
    """
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()
    if method == 'single_point':
        return single_point_crossover(parent1, parent2)
    elif method == 'two_point':
        return two_point_crossover(parent1, parent2)
    elif method == 'uniform':
        return uniform_crossover(parent1, parent2)
    else:
        raise ValueError(f"Unknown crossover method: '{method}'.")


# ── Mutation ─────────────────────────────────────────────────

def swap_mutation(individual):
    """
    Swap mutation: randomly select two positions and swap their values.
    Preserves the total number of selected features.
    """
    mutant = individual.copy()
    n = len(mutant)
    idx1, idx2 = np.random.choice(n, 2, replace=False)
    mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]
    n_sel = mutant.sum()
    if n_sel < MIN_FEATURES:
        zero_idx = np.where(mutant == 0)[0]
        need     = MIN_FEATURES - n_sel
        activate = np.random.choice(zero_idx, need, replace=False)
        mutant[activate] = 1
    return mutant


def mutate(individual, mutation_rate=DEFAULT_MUT_RATE, mutation_type='bitflip'):
    """
    Apply mutation to a binary individual.
    mutation_type: 'bitflip' (per-bit XOR flip) | 'swap' (swap two bits)
    """
    if mutation_type == 'swap':
        return swap_mutation(individual)

    mutant = individual.copy()
    flip_mask = np.random.rand(len(mutant)) < mutation_rate
    mutant[flip_mask] ^= 1

    n_sel = mutant.sum()
    if n_sel < MIN_FEATURES:
        zero_idx = np.where(mutant == 0)[0]
        need     = MIN_FEATURES - n_sel
        activate = np.random.choice(zero_idx, need, replace=False)
        mutant[activate] = 1
    return mutant


# ── Main GA Loop ─────────────────────────────────────────────

def run_ga(X, y, pop_size=20, n_generations=15, mutation_rate=DEFAULT_MUT_RATE,
           mutation_type='bitflip', crossover_type='two_point',
           selection_method='tournament', survivor_method='elitist'):
    """
    Run the Genetic Algorithm for feature selection.

    Parameters
    ----------
    X                : np.ndarray  Feature matrix (samples × features).
    y                : np.ndarray  Labels.
    pop_size         : int         Population size.
    n_generations    : int         Maximum number of generations.
    mutation_rate    : float       Per-bit mutation probability (bitflip only).
    mutation_type    : str         'bitflip' | 'swap'
    crossover_type   : str         'single_point' | 'two_point' | 'uniform'
    selection_method : str         'tournament' | 'roulette'
    survivor_method  : str         'elitist' | 'generational'

    Returns
    -------
    best_mask        : np.ndarray  Best binary feature mask found.
    best_score       : float       CV accuracy of the best individual.
    history          : list[float] Best CV accuracy per generation.
    """
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED, stratify=y)

    n_features = X_train.shape[1]
    np.random.seed(RANDOM_SEED)
    start_time = time.time()

    population = initialize_population(pop_size, n_features)

    history          = []
    best_individual  = None
    best_fitness     = -np.inf
    best_accuracy    = 0.0
    best_n_selected  = 0

    no_improve_count  = 0
    prev_best_fitness = -np.inf

    for gen in range(n_generations):
        fitnesses, accuracies, n_selected_arr = evaluate_population(population, X_train, y_train)

        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness    = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()
            best_accuracy   = accuracies[gen_best_idx]
            best_n_selected = n_selected_arr[gen_best_idx]

        history.append(best_accuracy)

        if best_fitness > prev_best_fitness:
            no_improve_count  = 0
            prev_best_fitness = best_fitness
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            break

        new_population = []
        while len(new_population) < pop_size:
            p1 = select_individual(population, fitnesses, method=selection_method)
            p2 = select_individual(population, fitnesses, method=selection_method)
            c1, c2 = crossover(p1, p2, method=crossover_type)
            c1 = mutate(c1, mutation_rate, mutation_type)
            c2 = mutate(c2, mutation_rate, mutation_type)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        if survivor_method == 'elitist':
            elite_idx = np.argmax(fitnesses)
            new_population[0] = population[elite_idx].copy()

        population = np.array(new_population)

    return best_individual, best_accuracy, history
