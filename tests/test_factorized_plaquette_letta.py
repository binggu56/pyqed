import numpy as np

from examples.mps.benchmark_factorized_plaquette_letta_4x4 import (
    factorize_plaquette_tensor,
    materialize_plaquette_tensor,
)


def test_untied_plaquette_tt_round_trip_at_full_rank():
    rng = np.random.default_rng(4)
    tensor = rng.normal(size=(3, 2, 16)) + 1.0j * rng.normal(size=(3, 2, 16))
    cores = factorize_plaquette_tensor(tensor, rank=32, tied=False)
    np.testing.assert_allclose(
        materialize_plaquette_tensor(cores, tied=False),
        tensor,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_tied_plaquette_tt_round_trip_at_full_rank():
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(1, 2, 16, 16))
    cores = factorize_plaquette_tensor(tensor, rank=64, tied=True)
    np.testing.assert_allclose(
        materialize_plaquette_tensor(cores, tied=True),
        tensor,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_plaquette_tt_rank_cap_and_physical_core_dimensions():
    rng = np.random.default_rng(9)
    untied = factorize_plaquette_tensor(
        rng.normal(size=(4, 4, 16)),
        rank=3,
        tied=False,
    )
    tied = factorize_plaquette_tensor(
        rng.normal(size=(1, 4, 16, 16)),
        rank=3,
        tied=True,
    )
    assert all(core.shape[2] <= 3 for core in untied[:-1])
    assert all(core.shape[2] <= 3 for core in tied[:-1])
    assert [core.shape[1] for core in untied] == [2, 2, 2, 2]
    assert [core.shape[1] for core in tied] == [4, 4, 4, 4]
