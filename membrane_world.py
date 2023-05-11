import numpy as np
import torch

import membrane_core


def batch_random_particle_flux(membrane: torch.Tensor, n_samples):
    nbatch = np.prod(membrane.shape[:-2])
    flat_membrane = membrane.view(-1)
    to_set_zero = np.random.choice(flat_membrane.shape[-1], n_samples * nbatch, replace=False)
    flat_membrane[to_set_zero] = 0
    to_add_new = np.random.choice(flat_membrane.shape[-1], n_samples * nbatch, replace=False)
    flat_membrane[to_add_new] = torch.randint(0, 256, to_add_new.shape, dtype=membrane.dtype, device=membrane.device)


def apply_physics_random(membrane: torch.Tensor, physical_model: torch.nn.Module, n_updates, window_size):
    batch_size = np.prod(membrane.shape[:-2])
    rolls = np.random.randint(-1 * window_size, window_size, (2))
    membrane_rolled = torch.roll(membrane, tuple(rolls), (-1, -2))
    membrane_blocked = membrane_core.form_submatrices(membrane_rolled, window_size)
    flat_blocked_membrane = membrane_blocked.view((-1, *membrane_blocked.shape[-2:]))
    where_apply_physics = np.random.choice(flat_blocked_membrane.shape[0], n_updates * batch_size, replace=False)
    flat_blocked_membrane[where_apply_physics] = physical_model(flat_blocked_membrane[where_apply_physics])
    return torch.roll(membrane_core.merge_submatrices(membrane_blocked), tuple(-1 * rolls), (-1, -2))

