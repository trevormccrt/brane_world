import numpy as np
import torch

import membrane_world, nn_permuter


def test_random_flux():
    init_membrane = torch.zeros((10, 100, 100), dtype=torch.uint8)
    membrane = torch.clone(init_membrane)
    with torch.no_grad():
        membrane_world.batch_random_particle_flux(membrane, 20)
    assert not np.all(init_membrane.numpy() == membrane.numpy())


def test_identity_physics():
    init_membrane = torch.randint(0, 256, (10, 100, 100), dtype=torch.uint8)
    physical_model = lambda x:x
    n_updates = 10
    window_size = 5
    with torch.no_grad():
        id_phys = membrane_world.apply_physics_random(init_membrane, physical_model, n_updates, window_size)
    np.testing.assert_equal(init_membrane.numpy(), id_phys.numpy())


def test_nn_physics():
    n_updates = 10
    window_size = 5
    init_membrane = torch.randint(0, 256, (10, 100, 100), dtype=torch.uint8)
    physical_model = nn_permuter.FlatNNMatrixPermuter(nn_permuter.wide_n_deep(window_size**2, window_size**2, 5, 1000))
    membrane = init_membrane
    for j in range(10):
        with torch.no_grad():
            new_membrane = membrane_world.apply_physics_random(membrane, physical_model, n_updates, window_size)
        assert not np.all(new_membrane.numpy() == membrane.numpy())
        flat_new_membrane = torch.sort(torch.reshape(new_membrane, (init_membrane.shape[0], -1)), dim=-1)[0]
        flat_membrane = torch.sort(torch.reshape(membrane, (init_membrane.shape[0], -1)), dim=-1)[0]
        assert np.all(flat_new_membrane.numpy() == flat_membrane.numpy())
        membrane = new_membrane
