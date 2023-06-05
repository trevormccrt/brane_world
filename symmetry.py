import numpy as np
import torch


def distance_matrix(vert_len, horiz_len):
    return torch.stack([torch.broadcast_to(torch.unsqueeze((torch.arange(start=0, end=vert_len, step=1) + 0.5) - vert_len/2, dim=-1), (vert_len, horiz_len)),
                     torch.broadcast_to(torch.arange(start=0, end=horiz_len) - torch.floor(horiz_len/2), (vert_len, horiz_len))], dim=0)


def middle_mask(vert_len, horiz_len):
    mask = torch.ones((vert_len, horiz_len))
    mask[[int(vert_len/2), int(vert_len/2 - 1)], int(horiz_len/2)] = 0
    return mask


def compute_centroids(target_mats, distance_mat):
    return torch.einsum("...kj, tkj -> ...t", target_mats, distance_mat)


def standard_orientation(target_mats, distance_mat, middle_mask):
    target_mats = torch.clone(target_mats)
    masked_centroids = compute_centroids(middle_mask * target_mats, distance_mat)
    to_flipud = torch.squeeze(torch.argwhere(masked_centroids[:, 0] < 0), dim=-1)
    to_fliplr = torch.squeeze(torch.argwhere(masked_centroids[:, 1] < 0), dim=-1)
    target_mats[to_flipud, ...] = torch.flip(target_mats[to_flipud, ...], dims=[-2])
    target_mats[to_fliplr, ...] = torch.flip(target_mats[to_fliplr, ...], dims=[-1])
    return target_mats


