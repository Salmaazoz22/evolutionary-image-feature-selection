"""
pso.py — Binary Particle Swarm Optimisation for Feature Selection
Extracted from notebooks/PSO_Feature_Selection.ipynb
"""

import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# ── Constants ────────────────────────────────────────────────
ALPHA                   = 0.9
MIN_FEATURES            = 10
INIT_FEAT_PROB          = 0.3
V_MAX                   = 4.0
X_MIN, X_MAX            = -10.0, 10.0
C1                      = 2.0
C2                      = 2.0
W_DEFAULT               = 0.7
EARLY_STOPPING_PATIENCE = 8
STAGNATION_LIMIT        = 5
RANDOM_SEED             = 42


# ── Transfer Functions ───────────────────────────────────────

def sigmoid(x):
    """S-shaped sigmoid transfer function. Clips x for numerical safety."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def v_shaped(x):
    """V-shaped transfer function based on tanh. Returns values in [0, 1]."""
    return np.abs(np.tanh(x / 2.0))


def apply_transfer(position, current_binary, transfer_type='sigmoid'):
    """
    Map continuous position to a binary vector using a transfer function.

    transfer_type: 'sigmoid' (S-shaped) | 'vshaped' (V-shaped)
    S-shaped: bit = 1 if rand() < sigmoid(x)
    V-shaped: bit flips if rand() < |tanh(x/2)|
    Enforces MIN_FEATURES constraint after binarisation.
    """
    n = len(position)
    binary = current_binary.copy()

    if transfer_type == 'sigmoid':
        prob   = sigmoid(position)
        binary = (np.random.rand(n) < prob).astype(int)
    elif transfer_type == 'vshaped':
        prob      = v_shaped(position)
        flip_mask = np.random.rand(n) < prob
        binary    = binary.copy()
        binary[flip_mask] ^= 1
    else:
        raise ValueError(f"Unknown transfer function: '{transfer_type}'. Use 'sigmoid' or 'vshaped'.")

    n_sel = binary.sum()
    if n_sel < MIN_FEATURES:
        zero_idx = np.where(binary == 0)[0]
        need     = MIN_FEATURES - n_sel
        activate = np.random.choice(zero_idx, need, replace=False)
        binary[activate] = 1

    return binary


# ── Swarm Initialisation ─────────────────────────────────────

def initialise_swarm(swarm_size, n_features, init_strategy='sparse'):
    """
    Initialise a PSO swarm.

    init_strategy: 'sparse' (P(bit=1)=0.3) | 'uniform' (P(bit=1)=0.5)
    Returns (positions, velocities, binary) — all shape (swarm_size, n_features).
    """
    p = INIT_FEAT_PROB if init_strategy == 'sparse' else 0.5

    binary = (np.random.rand(swarm_size, n_features) < p).astype(int)
    for i in range(swarm_size):
        if binary[i].sum() < MIN_FEATURES:
            idx = np.random.choice(n_features, MIN_FEATURES, replace=False)
            binary[i] = 0
            binary[i, idx] = 1

    positions = np.where(
        binary == 1,
        np.random.uniform(0.5, 2.0, (swarm_size, n_features)),
        np.random.uniform(-2.0, -0.5, (swarm_size, n_features))
    )
    velocities = np.random.uniform(-V_MAX / 2, V_MAX / 2, (swarm_size, n_features))

    return positions, velocities, binary


# ── Fitness ──────────────────────────────────────────────────

def evaluate_particle(binary_mask, X_tr, y_tr, alpha=ALPHA):
    """
    Compute the fitness of a single binary particle.
    fitness = alpha * accuracy + (1 - alpha) * (1 - n_selected / n_total)
    Uses 2-fold CV with LinearSVC (fast; no test leakage).
    Returns (fitness, accuracy, n_selected).
    """
    selected_idx = np.where(binary_mask == 1)[0]
    n_sel = len(selected_idx)
    if n_sel < MIN_FEATURES:
        return 0.0, 0.0, n_sel

    X_sub = X_tr[:, selected_idx]
    svm = LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=1000)
    cv_scores = cross_val_score(svm, X_sub, y_tr, cv=2, scoring='accuracy', n_jobs=-1)
    accuracy = cv_scores.mean()

    n_total = len(binary_mask)
    fitness = alpha * accuracy + (1.0 - alpha) * (1.0 - n_sel / n_total)
    return fitness, accuracy, n_sel


# ── Inertia Weight ───────────────────────────────────────────

def get_inertia_weight(iteration, n_iterations, strategy='fixed',
                       w_fixed=W_DEFAULT, w_max=0.9, w_min=0.4):
    """
    Return the inertia weight for the current iteration.
    strategy: 'fixed' | 'linear_decay' | 'random'
    """
    if strategy == 'fixed':
        return w_fixed
    elif strategy == 'linear_decay':
        t = iteration / max(n_iterations - 1, 1)
        return w_max - t * (w_max - w_min)
    elif strategy == 'random':
        return np.random.uniform(0.5, 1.0)
    else:
        raise ValueError(f"Unknown inertia strategy: '{strategy}'.")


# ── Topology / Neighbourhood ─────────────────────────────────

def get_social_best(particle_idx, pbest_positions, pbest_fitness,
                    topology='gbest', ring_k=2):
    """
    Return the social best position for particle particle_idx.
    topology: 'gbest' (global best) | 'lbest' (ring neighbourhood)
    """
    swarm_size = len(pbest_fitness)
    if topology == 'gbest':
        best_idx = np.argmax(pbest_fitness)
        return pbest_positions[best_idx].copy()
    elif topology == 'lbest':
        neighbours = [(particle_idx + j) % swarm_size
                      for j in range(-ring_k, ring_k + 1)]
        best_in_neighbourhood = neighbours[np.argmax(pbest_fitness[neighbours])]
        return pbest_positions[best_in_neighbourhood].copy()
    else:
        raise ValueError(f"Unknown topology: '{topology}'. Use 'gbest' or 'lbest'.")


# ── Particle Reinitialisation ────────────────────────────────

def reinitialise_particle(n_features, init_strategy='sparse'):
    """Reinitialise a single stagnant particle."""
    p = INIT_FEAT_PROB if init_strategy == 'sparse' else 0.5
    binary = (np.random.rand(n_features) < p).astype(int)
    if binary.sum() < MIN_FEATURES:
        idx = np.random.choice(n_features, MIN_FEATURES, replace=False)
        binary = np.zeros(n_features, dtype=int)
        binary[idx] = 1
    pos = np.where(
        binary == 1,
        np.random.uniform(0.5, 2.0, n_features),
        np.random.uniform(-2.0, -0.5, n_features)
    )
    vel = np.random.uniform(-V_MAX / 2, V_MAX / 2, n_features)
    return pos, vel, binary


# ── Velocity & Position Update ───────────────────────────────

def update_particle(position, velocity, binary, pbest_pos, social_best_pos,
                    w, c1=C1, c2=C2, transfer_type='sigmoid'):
    """
    Perform one PSO velocity and position update for a single particle.
    v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(social_best - x)
    x(t+1) = x(t) + v(t+1)  [clipped to position bounds]
    Binary mask updated via transfer function.
    Returns (new_pos, new_vel, new_binary).
    """
    n  = len(position)
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    new_vel = w * velocity + c1 * r1 * (pbest_pos - position) + c2 * r2 * (social_best_pos - position)
    new_vel = np.clip(new_vel, -V_MAX, V_MAX)

    new_pos = position + new_vel
    new_pos = np.clip(new_pos, X_MIN, X_MAX)

    new_binary = apply_transfer(new_pos, binary, transfer_type)
    return new_pos, new_vel, new_binary


# ── Main PSO Loop ────────────────────────────────────────────

def run_pso(X, y, n_particles=20, n_iterations=15, w=W_DEFAULT, c1=C1, c2=C2,
            transfer_type='sigmoid', topology='gbest', inertia_strategy='fixed',
            init_strategy='sparse'):
    """
    Run the Binary PSO for feature selection.

    Parameters
    ----------
    X                : np.ndarray  Feature matrix (samples × features).
    y                : np.ndarray  Labels.
    n_particles      : int         Swarm size.
    n_iterations     : int         Maximum number of iterations.
    w                : float       Inertia weight (used if inertia_strategy='fixed').
    c1               : float       Cognitive (personal best) coefficient.
    c2               : float       Social (global best) coefficient.
    transfer_type    : str         'sigmoid' | 'vshaped'
    topology         : str         'gbest' | 'lbest'
    inertia_strategy : str         'fixed' | 'linear_decay' | 'random'
    init_strategy    : str         'sparse' | 'uniform'

    Returns
    -------
    best_mask        : np.ndarray  Best binary feature mask found.
    best_score       : float       CV accuracy of the best particle.
    history          : list[float] Best CV accuracy per iteration.
    """
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED, stratify=y)

    n_features = X_train.shape[1]
    np.random.seed(RANDOM_SEED)

    positions, velocities, binaries = initialise_swarm(n_particles, n_features, init_strategy)

    fitnesses  = np.zeros(n_particles)
    accuracies = np.zeros(n_particles)
    n_sel_arr  = np.zeros(n_particles, dtype=int)

    for i in range(n_particles):
        f, acc, n = evaluate_particle(binaries[i], X_train, y_train)
        fitnesses[i]  = f
        accuracies[i] = acc
        n_sel_arr[i]  = n

    pbest_positions = positions.copy()
    pbest_fitness   = fitnesses.copy()
    pbest_binary    = binaries.copy()
    pbest_accuracy  = accuracies.copy()
    pbest_n_sel     = n_sel_arr.copy()

    gbest_idx      = np.argmax(pbest_fitness)
    gbest_fitness  = pbest_fitness[gbest_idx]
    gbest_binary   = pbest_binary[gbest_idx].copy()
    gbest_accuracy = pbest_accuracy[gbest_idx]
    gbest_n_sel    = pbest_n_sel[gbest_idx]

    history = [gbest_accuracy]

    stagnation_counters = np.zeros(n_particles, dtype=int)
    no_improve_count    = 0
    prev_best_fitness   = gbest_fitness

    for it in range(n_iterations):
        w_val = get_inertia_weight(it, n_iterations, inertia_strategy, w_fixed=w)

        for i in range(n_particles):
            if stagnation_counters[i] >= STAGNATION_LIMIT:
                pos_new, vel_new, bin_new = reinitialise_particle(n_features, init_strategy)
                positions[i]  = pos_new
                velocities[i] = vel_new
                binaries[i]   = bin_new
                stagnation_counters[i] = 0
                continue

            social_best = get_social_best(i, pbest_positions, pbest_fitness, topology)
            new_pos, new_vel, new_bin = update_particle(
                positions[i], velocities[i], binaries[i],
                pbest_positions[i], social_best,
                w_val, c1, c2, transfer_type
            )
            positions[i]  = new_pos
            velocities[i] = new_vel
            binaries[i]   = new_bin

            f, acc, n = evaluate_particle(new_bin, X_train, y_train)

            if f > pbest_fitness[i]:
                pbest_positions[i] = new_pos.copy()
                pbest_fitness[i]   = f
                pbest_binary[i]    = new_bin.copy()
                pbest_accuracy[i]  = acc
                pbest_n_sel[i]     = n
                stagnation_counters[i] = 0
            else:
                stagnation_counters[i] += 1

        gen_best_idx = np.argmax(pbest_fitness)
        if pbest_fitness[gen_best_idx] > gbest_fitness:
            gbest_fitness  = pbest_fitness[gen_best_idx]
            gbest_binary   = pbest_binary[gen_best_idx].copy()
            gbest_accuracy = pbest_accuracy[gen_best_idx]
            gbest_n_sel    = pbest_n_sel[gen_best_idx]

        history.append(gbest_accuracy)

        if gbest_fitness > prev_best_fitness:
            no_improve_count  = 0
            prev_best_fitness = gbest_fitness
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            break

    return gbest_binary, gbest_accuracy, history
