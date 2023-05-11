import numpy as np
import torch


def wide_n_deep(input_dimension, output_dimension, n_hidden, hidden_dim, nonlin=torch.nn.Tanh):
    layers = []
    layers.append(torch.nn.Linear(input_dimension, int(hidden_dim / 2)))
    layers.append(nonlin())
    for _ in range(n_hidden):
        layers.append(torch.nn.LazyLinear(hidden_dim))
        layers.append(nonlin())
    layers.append(torch.nn.Linear(hidden_dim, int(hidden_dim / 2)))
    layers.append(nonlin())
    layers.append(torch.nn.Linear(int(hidden_dim / 2), output_dimension))
    return torch.nn.Sequential(*layers)


def sort_inputs_by_output_block(inputs, outputs):
    sort_x = torch.argsort(outputs, dim=-1)
    sort_y = torch.argsort(outputs, dim=-2)
    return torch.take_along_dim(torch.take_along_dim(inputs, sort_x, dim=-1), sort_y, dim=-2)


class FlatNNMatrixPermuter(torch.nn.Module):
    def __init__(self, vector_function):
        super().__init__()
        self.vector_function = vector_function

    def forward(self, input):
        flat_input = torch.reshape(input.type(torch.get_default_dtype()), (*input.shape[:-2], -1))
        forward = self.vector_function(flat_input)
        outputs = torch.reshape(forward, input.shape)
        return sort_inputs_by_output_block(input, outputs)
