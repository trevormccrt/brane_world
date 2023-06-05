import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import norm

import symmetry, embedding, conv_energy

batch_size = 10000
N = 10
window_size = (6, 9)

spaces = torch.tensor(np.random.randint(0, N, (batch_size, *window_size)),
                      dtype=torch.get_default_dtype())

vert_len = torch.tensor(window_size[0])
horiz_len = torch.tensor(window_size[1])
distance_mat = symmetry.distance_matrix(vert_len, horiz_len)
mask_mat = symmetry.middle_mask(vert_len, horiz_len)

alpha = 0.5
decay_mat = torch.exp(-alpha * torch.sqrt(torch.sum(distance_mat**2, dim=0)))
a = decay_mat.numpy()

spaces_std = symmetry.standard_orientation(spaces, distance_mat, mask_mat)
directions = symmetry.reaction_direction(spaces_std)
spaces_std_dir = symmetry.standard_reaction_direction(spaces_std, directions)

circular_embedding = embedding.circular_embedding(spaces_std_dir, N)

energy_fn = conv_energy.SimpleConvEnergy2D(2, 2, 1, (3, 3), f_nonlin=torch.nn.Tanh)
with torch.no_grad():
    energies = torch.squeeze(energy_fn(circular_embedding), dim=-1)


def make_cdf(x):
    return sorted(x), np.arange(len(x)) / len(x)

cdf_x, cdf_p = make_cdf(energies.numpy())

mu1,sigma1 = curve_fit(norm.cdf, cdf_x, cdf_p, p0=[0,1])[0]

fig, axs = plt.subplots()
axs.plot(cdf_x, cdf_p)
axs.plot(cdf_x, norm.cdf(cdf_x, mu1, sigma1))
plt.show()
print("")
