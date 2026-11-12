import numpy as np
import pytest

from pyqed.dvr import DVR
from pyqed.ldr import LDR
from pyqed.ldr.overlap import nearest


@pytest.mark.parametrize(("n", "method"), [(3, "linear"), (5, "cubic")])
def test_linked_interpolation_preserves_gauge_and_refines_nodes(n, method):
    source = DVR([(-2.3, 3.7), (-2.1, 3.4)], (n, n + 1), mass=(2, 3))

    def energies(dvr):
        x, y = np.meshgrid(*dvr.x, indexing="ij")
        lower = 0.3 * x + 0.2 * y
        return np.stack((lower, lower + 1), axis=-1)

    rng = np.random.default_rng(4)
    phases = np.exp(2j * np.pi * rng.random((*source.shape, 2)))
    links = nearest(
        source.shape, lambda i, j: 0.9 * np.diag(phases[i].conj() * phases[j])
    )
    ldr = LDR(source, 2, energies=energies(source), links=links)
    rebuilt = ldr.interpolate(source, method=method)
    np.testing.assert_allclose(
        rebuilt.eigenstates(roots=4)[0], ldr.eigenstates(roots=4)[0], atol=1e-10
    )

    fine = ldr.refine(method=method)
    assert fine.dvr.mass == source.mass
    for x, refined in zip(source.x, fine.dvr.x):
        np.testing.assert_allclose(refined[::2], x, atol=1e-14)
        np.testing.assert_allclose(refined[1::2], (x[:-1] + x[1:]) / 2, atol=1e-14)
    np.testing.assert_allclose(fine.energies, energies(fine.dvr), atol=1e-10)
    singular = np.linalg.svd(list(fine.links.values()), compute_uv=False)
    np.testing.assert_allclose(singular, 0.9**0.25, atol=1e-10)

    shortcut = ldr.interpolate(npts=(9, 11), method=method)
    assert shortcut.shape == (9, 11)
    assert shortcut.dvr.mass == source.mass
    np.testing.assert_allclose(shortcut.energies, energies(shortcut.dvr), atol=1e-10)
