import cupy as cp
import numpy as np


def generate_single_permutations(state, n_permutations):
    p_state = np.tile(np.expand_dims(state, -2), [n_permutations, 1])
    for j in range(1, n_permutations):
        swap = np.random.choice(state.shape[-1], 2, replace=False)
        p_state[..., j, swap] =  p_state[..., j, swap[::-1]]
    return p_state


def stability_flow(states, energy_function, n_permutations):
    permuted_states = generate_single_permutations(states, n_permutations)
    energies = energy_function(permuted_states)
    best_idx = np.argmin(energies, axis=-1)
    return np.squeeze(np.take_along_axis(permuted_states, np.expand_dims(np.expand_dims(best_idx, -1), -1), -2), -2),\
        np.squeeze(np.take_along_axis(energies, np.expand_dims(best_idx, -1), -1), -1)


def jettison_most_unstable(states, energies, n_jettison):
    energy_sorted_order = np.argsort(energies, axis=-1)
    undo_order = np.argsort(energy_sorted_order, axis=-1)
    sorted_by_energy = np.take_along_axis(states, np.expand_dims(energy_sorted_order, -1), -2)
    to_jet_states = sorted_by_energy[..., :n_jettison, :]
    perm_to_jet_states = np.swapaxes(cp.random.permutation(np.swapaxes(to_jet_states, -1, 0)), -1, 0)
    new_atoms = perm_to_jet_states.astype(cp.bool_) * cp.random.randint(1, 256, perm_to_jet_states.shape, dtype=cp.uint8)
    sorted_by_energy[..., :n_jettison, :] = new_atoms
    return np.take_along_axis(sorted_by_energy, np.expand_dims(undo_order, -1), -2)


def lyapunov_membrane_physics(state, energy_function, n_permutations, n_jettison):
    flat_state = np.reshape(state, (*state.shape[:-2], -1))
    flat_state, energies = stability_flow(flat_state, energy_function, n_permutations)
    flat_state = jettison_most_unstable(flat_state, energies, n_jettison)
    return np.reshape(flat_state, state.shape)
