import cupy as cp
import numpy as np

import lyapunov_dynamics


def test_single_permutations():
    state = np.array([np.random.choice(100, 9, replace=False) for _ in range(100)])
    perm_states = lyapunov_dynamics.generate_single_permutations(state, 5)
    for this_state, this_perm_state in zip(state, perm_states):
        a = np.sum(np.equal(this_state, this_perm_state), axis=-1)
        assert np.all(np.equal(a, [9, 7, 7, 7, 7]))


def _random_energy(state):
    return np.random.uniform(0, 1, state.shape[:-1])


def test_stability_flow():
    state = np.array([[np.random.choice(100, 9, replace=False) for _ in range(100)] for _ in range(3)])
    lyapunov_dynamics.stability_flow(state, _random_energy, 10)


def test_jettison():
    batch_size = tuple([100])
    sample_size = 15
    n_jet = 3
    states = cp.random.randint(0, 256, (*batch_size, sample_size, 9), dtype=np.uint8)
    energies = cp.random.uniform(0, 1, (*batch_size, sample_size))
    new_states = lyapunov_dynamics.jettison_most_unstable(states, energies, n_jet)
    for orig_state, new_state, energy in zip(states, new_states, energies):
        assert np.sum(np.all(np.equal(orig_state, new_state), axis=-1)) == sample_size - n_jet


def test_membrane_physics():
    batch_dim = 100
    n_compare = 15
    window_size = 3
    n_permutations = 10
    n_jetison = 3
    states = cp.random.randint(0, 256, (batch_dim, n_compare, window_size, window_size))
    _ = lyapunov_dynamics.lyapunov_membrane_physics(states, _random_energy, n_permutations, n_jetison)


