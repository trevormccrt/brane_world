import cupy as cp
import numpy as np


def sort_inputs_by_output_block(inputs, outputs):
    sort_x = np.argsort(outputs, axis=-1)
    sort_y = np.argsort(outputs, axis=-2)
    return np.take_along_axis(np.take_along_axis(inputs, sort_x, axis=-1), sort_y, axis=-2)


def apply_linear(state, weights, biases):
    return np.einsum("...j, ij -> ...i", state, weights) + biases


def make_random_wide_n_deep(input_dimension, output_dimension, n_hidden, hidden_dim, nonlin=np.tanh):
    r = np.sqrt(1/input_dimension)
    all_weights = []
    all_biases = []
    all_weights.append(cp.random.uniform(-r, r, (hidden_dim, input_dimension)).astype(cp.float16))
    all_biases.append(cp.random.uniform(-r, r, hidden_dim).astype(cp.float16))
    for _ in range(n_hidden):
        all_weights.append(cp.random.uniform(-r, r, (hidden_dim, hidden_dim)).astype(cp.float16))
        all_biases.append(cp.random.uniform(-r, r, hidden_dim).astype(cp.float16))
    all_weights.append(cp.random.uniform(-r, r, (output_dimension, hidden_dim)).astype(cp.float16))
    all_biases.append(cp.random.uniform(-r, r, output_dimension).astype(cp.float16))

    def apply(state):
        for w, b in zip(all_weights[:-1], all_biases[:-1]):
            state = nonlin(apply_linear(state, w, b))
        return apply_linear(state, all_weights[-1], all_biases[-1])

    return apply


def make_permuter(vector_function):
    def apply(state):
        flat_input = np.reshape(state, (*state.shape[:-2], -1))
        forward = vector_function(flat_input)
        outputs = np.reshape(forward, state.shape)
        return sort_inputs_by_output_block(state, outputs)
    return apply
