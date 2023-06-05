import cupy as cp
import numpy as np

import nn_permuter


def test_input_permutation():
    inputs = np.random.randint(0, 100, (10, 4, 4))
    outputs = np.random.uniform(0, 1, (10, 4, 4))
    permuted_inputs = nn_permuter.sort_inputs_by_output_block(inputs, outputs)
    for j in range(inputs.shape[0]):
        assert sorted(list(inputs[j].flatten())) == sorted(list(permuted_inputs[j].flatten()))


def test_nn_permuter():
    wnd = nn_permuter.make_random_wide_n_deep(9, 9, 10, 200)
    permuter = nn_permuter.make_permuter(wnd)
    inputs = cp.random.randint(0, 100, (10, 3, 3)).astype(cp.float16)
    _ = permuter(inputs).astype(cp.uint8)

