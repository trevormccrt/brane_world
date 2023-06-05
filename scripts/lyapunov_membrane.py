import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


import membrane_world, nn_permuter, lyapunov_dynamics


window_size = 4
brane_dim = window_size * 200
n_updates_per_tick = int(0.4 * (brane_dim/window_size)**2)

empty_brane = cp.zeros((brane_dim, brane_dim), dtype=cp.uint8)
init_mask = cp.random.binomial(1, 0.05, (brane_dim, brane_dim), dtype=cp.uint8)
brane = empty_brane + (init_mask * cp.random.randint(0, 256, (brane_dim, brane_dim), dtype=cp.uint8))
energy_fn = nn_permuter.make_random_wide_n_deep(window_size**2, 1, 4, 100, nonlin=np.sin)
n_permutations = 10
n_jet = np.max([int(0.001 * n_updates_per_tick), 1])
physics = lambda x: lyapunov_dynamics.lyapunov_membrane_physics(x, lambda y: np.squeeze(energy_fn(y), -1), n_permutations, n_jet)

fig, axs = plt.subplots(ncols=3)
hist_bins = cp.arange(start=0, stop=257, step=1)
cpu_bins = cp.asnumpy(hist_bins)
state = brane
counter = 0
while True:

    state = membrane_world.apply_physics_random(state, physics, n_updates_per_tick, window_size)
    #membrane_world.batch_random_particle_flux(state, int(0.01 * brane_dim**2))
    if counter % 50000:
        hist_data, bins = cp.histogram(state, hist_bins)
        [j.clear() for j in axs]
        axs[2].stairs(cp.asnumpy(hist_data[1:]), cpu_bins[1:])
        axs[0].imshow(cp.asnumpy(state - 128), cmap="bwr")
        axs[1].imshow(cp.asnumpy(state[500:600, 500:600] - 128), cmap="bwr")
        plt.pause(0.000001)
    counter += 1

plt.show()
