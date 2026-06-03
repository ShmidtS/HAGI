import numpy as np
import pytest

from hagi.utils.es_noise import noise_tensor, noise_vector, standard_noise


def test_standard_noise_deterministic():
    assert standard_noise(42, 0) == standard_noise(42, 0)
    assert standard_noise(42, 1) == standard_noise(42, 1)
    assert standard_noise(42, 0) != standard_noise(43, 0)


def test_standard_noise_distribution():
    vals = np.array([standard_noise(123, i) for i in range(10000)])
    assert abs(vals.mean()) < 0.1
    assert 0.9 < vals.std() < 1.1


def test_noise_vector():
    v = noise_vector(7, 1000)
    assert v.shape == (1000,)
    assert v.dtype == np.float32


def test_noise_tensor():
    t = noise_tensor(7, (2, 3, 4))
    assert t.shape == (2, 3, 4)
    assert t.dtype == np.float32
