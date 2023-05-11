import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import membrane_world, nn_permuter


window_size = 3
brane_dim = window_size * 500
n_updates_per_tick = int(0.2 * (brane_dim/window_size)**2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
empty_brane = torch.zeros((brane_dim, brane_dim), dtype=torch.uint8)
init_mask = torch.tensor(np.random.binomial(1, 0.001, (brane_dim, brane_dim)).astype(np.uint8), dtype=torch.uint8)
brane = empty_brane + (init_mask * torch.randint(0, 256, (brane_dim, brane_dim), dtype=torch.uint8))
physics = nn_permuter.FlatNNMatrixPermuter(nn_permuter.wide_n_deep(window_size**2, window_size**2, 3, 200)).to(device)

fig, axs = plt.subplots(ncols=2)
hist_bins = np.arange(start=0, stop=257, step=1)
state = brane.to(device)
while True:
    with torch.no_grad():
        state = membrane_world.apply_physics_random(state, physics, n_updates_per_tick, window_size)
        membrane_world.batch_random_particle_flux(state, int(0.01 * brane_dim**2))
    #state_np = state.cpu().numpy()
    #hist_data, bins = np.histogram(state_np, hist_bins)
    #[j.clear() for j in axs]
    #axs[1].stairs(hist_data[1:], bins[1:])
    #axs[0].imshow(state_np - 128, cmap="seismic")
    #plt.pause(0.000001)

plt.show()
