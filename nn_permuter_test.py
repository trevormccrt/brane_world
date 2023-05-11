import torch
import numpy as np

import nn_permuter


def test_input_permutation():
    inputs = np.random.randint(0, 100, (10, 4, 4))
    outputs = np.random.uniform(0, 1, (10, 4, 4))
    with torch.no_grad():
        permuted_inputs = nn_permuter.sort_inputs_by_output_block(torch.from_numpy(inputs), torch.from_numpy(outputs))
    permuted_inputs = permuted_inputs.numpy()
    for j in range(inputs.shape[0]):
        assert sorted(list(inputs[j].flatten())) == sorted(list(permuted_inputs[j].flatten()))


def test_nn_permuter():
    wnd = nn_permuter.wide_n_deep(9, 9, 10, 200)
    permuter = nn_permuter.FlatNNMatrixPermuter(wnd)
    inputs = np.random.randint(0, 100, (10, 3, 3), dtype=np.uint8)
    permuter(torch.from_numpy(inputs))


