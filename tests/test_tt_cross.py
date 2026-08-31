import numpy as np
from types import SimpleNamespace

from pyqed.mps.cross import tt_cross, tt_cross_tntorch, tt_value


def test_tt_cross_recovers_low_rank_tensor_from_sparse_samples():
    shape = (7,) * 5
    grids = [np.linspace(-1.0, 1.0, size) for size in shape]

    def evaluate(index):
        point = [grids[axis][position] for axis, position in enumerate(index)]
        return (
            np.prod([1.0 + 0.1 * value for value in point])
            + 0.3 * np.prod([value + 0.2 for value in point])
            + 0.1 * np.prod([value**2 + 0.5 for value in point])
        )

    cores, info = tt_cross(
        shape,
        evaluate,
        max_rank=3,
        sweeps=4,
        rtol=1.0e-12,
        validation=100,
        seed=3,
    )
    rng = np.random.default_rng(7)
    probes = [tuple(rng.integers(0, 7, 5)) for _ in range(200)]

    error = max(abs(tt_value(cores, index) - evaluate(index)) for index in probes)
    assert error < 1.0e-11
    assert info["samples"] < 0.05 * np.prod(shape)
    assert max(core.shape[2] for core in cores) <= 3
    assert info["rank_history"][0] == (1, 1, 1, 1, 1, 1)
    assert info["rank_history"][-1] == (1, 3, 3, 3, 3, 1)


def test_native_tt_cross_batches_and_reuses_checkpoint_state():
    shape = (6, 6, 6, 6)
    batch_sizes = []

    def batch_evaluator(indices):
        batch_sizes.append(len(indices))
        coordinates = indices.astype(float)
        return (
            np.prod(1.0 + 0.1 * coordinates, axis=1)
            + 0.2 * np.prod(coordinates + 0.5, axis=1)
        )

    def scalar_evaluator(_index):
        raise AssertionError("the scalar evaluator should not be called")

    cores, info = tt_cross(
        shape,
        scalar_evaluator,
        batch_evaluator=batch_evaluator,
        max_rank=3,
        sweeps=3,
        validation=40,
        rtol=1.0e-11,
        seed=9,
        return_state=True,
    )

    assert info["validation_error"] < 1.0e-11
    assert info["batch_calls"] == len(batch_sizes)
    assert max(batch_sizes) > 1

    warm_cores, warm_info = tt_cross(
        shape,
        scalar_evaluator,
        batch_evaluator=batch_evaluator,
        max_rank=3,
        sweeps=1,
        validation=40,
        rtol=1.0e-11,
        seed=9,
        initial=info["state"],
        return_state=True,
    )

    assert warm_info["new_samples"] < info["samples"]
    assert warm_info["validation_error"] < 1.0e-11
    probes = [(0, 1, 2, 3), (5, 4, 3, 2), (1, 1, 1, 1)]
    np.testing.assert_allclose(
        [tt_value(warm_cores, index) for index in probes],
        [tt_value(cores, index) for index in probes],
        atol=1.0e-11,
    )


def test_tntorch_adapter_batches_indices_and_converts_cores(monkeypatch):
    class FakeTensor:
        def __init__(self, values, device="cpu"):
            self.values = np.asarray(values)
            self.device = SimpleNamespace(type=str(device))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeDevice:
        def __init__(self, name):
            self.type = str(name)

    fake_torch = SimpleNamespace(
        float64=np.float64,
        device=FakeDevice,
        manual_seed=lambda seed: None,
        arange=lambda size, **kwargs: FakeTensor(np.arange(size)),
        as_tensor=lambda values, **kwargs: FakeTensor(values),
    )

    def fake_cross(**kwargs):
        points = FakeTensor([[0.0, 0.0], [1.0, 1.0]])
        values = kwargs["function"](points)
        np.testing.assert_allclose(values.numpy(), [1.0, 5.0])
        tensor = SimpleNamespace(
            cores=[
                FakeTensor(np.ones((1, 2, 1))),
                FakeTensor(np.ones((1, 2, 1))),
            ]
        )
        info = {"nsamples": 6, "val_epss": [0.2, 1.0e-8], "val_eps": 1.0e-8}
        return tensor, info

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tntorch",
        SimpleNamespace(cross=fake_cross),
    )
    cores, info = tt_cross_tntorch(
        (2, 2), lambda index: 1.0 + index[0] + 3.0 * index[1]
    )

    assert [core.shape for core in cores] == [(1, 2, 1), (1, 2, 1)]
    assert info == {
        "backend": "tntorch",
        "samples": 2,
        "function_evaluations": 6,
        "sweeps": 2,
        "validation_error": 1.0e-8,
        "ranks": (1, 1, 1),
    }
