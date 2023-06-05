import cupy as cp
import numpy as np

import membrane_world, nn_permuter


def test_random_flux():
    init_membrane = cp.zeros((10, 100, 100), dtype=cp.uint8)
    membrane = np.copy(init_membrane)
    membrane_world.batch_random_particle_flux(membrane, 20)
    assert not np.all(init_membrane == membrane)


def test_identity_physics():
    init_membrane = cp.random.randint(0, 256, (10, 100, 100), dtype=cp.uint8)
    physical_model = lambda x:x
    n_updates = 10
    window_size = 5
    id_phys = membrane_world.apply_physics_random(init_membrane, physical_model, n_updates, window_size)
    assert np.all(np.equal(init_membrane, id_phys))


def test_nn_permute_physics():
    n_updates = 10
    window_size = 5
    init_membrane = cp.random.randint(0, 256, (10, 100, 100), dtype=cp.uint8)
    physical_model = nn_permuter.make_permuter(nn_permuter.make_random_wide_n_deep(window_size**2, window_size**2, 5, 200))
    membrane = init_membrane
    for j in range(10):
        new_membrane = membrane_world.apply_physics_random(membrane, physical_model, n_updates, window_size)
        assert not np.all(np.equal(new_membrane, init_membrane))
        flat_new_membrane = np.sort(np.reshape(new_membrane, (init_membrane.shape[0], -1)), axis=-1)
        flat_membrane = np.sort(np.reshape(membrane, (init_membrane.shape[0], -1)), axis=-1)
        assert np.all(np.equal(flat_new_membrane, flat_membrane))
        membrane = new_membrane
