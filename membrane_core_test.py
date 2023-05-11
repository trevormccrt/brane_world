import numpy as np
import torch

import membrane_core


def test_blocking():
    a = np.random.uniform(0, 1, (10, 10))
    with torch.no_grad():
        b = membrane_core.form_submatrices(torch.from_numpy(a), 5)
    b = b.numpy()
    np.testing.assert_allclose(b[0, 0], a[:5, :5])
    np.testing.assert_allclose(b[1, 0], a[5:, :5])
    np.testing.assert_allclose(b[0, 1], a[:5, 5:])
    np.testing.assert_allclose(b[1, 1], a[5:, 5:])


def test_block_shaping_inversion():
    matrix = np.random.uniform(0, 1, (3, 7, 100, 100))
    with torch.no_grad():
        split_mat = membrane_core.form_submatrices(torch.from_numpy(matrix), 10)
        merged_mat = membrane_core.merge_submatrices(split_mat)
    merged_mat = merged_mat.numpy()
    np.testing.assert_allclose(matrix, merged_mat)

