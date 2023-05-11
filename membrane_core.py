import torch


def form_submatrices(large_matrix_batch, split_size):
    return torch.stack(torch.split(torch.stack(torch.split(large_matrix_batch, split_size, dim=-2), dim=-3),
                                   split_size, dim=-1), dim=-3)


def merge_submatrices(blocks):
    return torch.reshape(torch.swapaxes(blocks, -3, -2),
                      (*blocks.shape[:-4], blocks.shape[-2] * blocks.shape[-4], blocks.shape[-1] * blocks.shape[-3]))


