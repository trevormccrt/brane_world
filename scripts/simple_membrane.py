import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


import membrane_world, nn_permuter


window_size = 3
brane_dim = window_size * 3000
n_updates_per_tick = int(0.2 * (brane_dim/window_size)**2)

empty_brane = cp.zeros((brane_dim, brane_dim), dtype=cp.uint8)
init_mask = cp.random.binomial(1, 0.001, (brane_dim, brane_dim), dtype=cp.uint8)
brane = empty_brane + (init_mask * cp.random.randint(0, 256, (brane_dim, brane_dim), dtype=cp.uint8))
physics = nn_permuter.make_permuter(nn_permuter.make_random_wide_n_deep(window_size**2, window_size**2, 3, 100))

fig, axs = plt.subplots(ncols=2)
hist_bins = cp.arange(start=0, stop=257, step=1)
cpu_bins = cp.asnumpy(hist_bins)
state = brane
while True:

    state = membrane_world.apply_physics_random(state, physics, n_updates_per_tick, window_size)
    membrane_world.batch_random_particle_flux(state, int(0.01 * brane_dim**2))
    #hist_data, bins = cp.histogram(state, hist_bins)
    #[j.clear() for j in axs]
    #axs[1].stairs(cp.asnumpy(hist_data[1:]), cpu_bins[1:])
    #axs[0].imshow(cp.asnumpy(state) - 128, cmap="seismic")
    #plt.pause(0.000001)

plt.show()
