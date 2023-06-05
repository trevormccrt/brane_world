import numpy as np


def form_submatrices(large_matrix_batch, split_size):
    n_split = int(large_matrix_batch.shape[-1]/split_size)
    return np.stack(np.split(np.stack(np.split(large_matrix_batch, n_split, axis=-2), axis=-3),
                                   n_split, axis=-1), axis=-3)


def merge_submatrices(blocks):
    return np.reshape(np.swapaxes(blocks, -3, -2),
                      (*blocks.shape[:-4], blocks.shape[-2] * blocks.shape[-4], blocks.shape[-1] * blocks.shape[-3]))


