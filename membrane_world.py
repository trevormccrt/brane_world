import cupy as cp
import numpy as np

import membrane_core


def batch_random_particle_flux(membrane, n_samples):
    nbatch = np.prod(membrane.shape[:-2]).astype(int)
    flat_membrane = membrane.reshape(-1)
    to_set_zero = cp.random.choice(flat_membrane.shape[-1], n_samples * nbatch, replace=False)
    flat_membrane[to_set_zero] = 0
    to_add_new = cp.random.choice(flat_membrane.shape[-1], n_samples * nbatch, replace=False)
    flat_membrane[to_add_new] = cp.random.randint(0, 256, to_add_new.shape, dtype=cp.uint8)


def apply_physics_random(membrane, physical_model, n_updates, window_size):
    batch_size = np.prod(membrane.shape[:-2]).astype(int)
    rolls = np.random.randint(-1 * window_size, window_size, (2))
    membrane_rolled = np.roll(membrane, tuple(rolls), (-1, -2))
    membrane_blocked = membrane_core.form_submatrices(membrane_rolled, window_size)
    flat_blocked_membrane = membrane_blocked.reshape((-1, *membrane_blocked.shape[-2:]))
    where_apply_physics = cp.random.choice(flat_blocked_membrane.shape[0], n_updates * batch_size, replace=False)
    flat_blocked_membrane[where_apply_physics] = physical_model(flat_blocked_membrane[where_apply_physics])
    return np.roll(membrane_core.merge_submatrices(membrane_blocked), tuple(-1 * rolls), (-1, -2))

