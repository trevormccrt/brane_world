import torch


def circular_embedding(data, max_N):
    return torch.stack([torch.cos(2 * torch.pi * data/max_N), torch.sin(2 * torch.pi * data/max_N)])
