"""Small neural-network regressors for fitting potential energy surfaces.

The module intentionally uses only NumPy.  That keeps PES fitting available in
the base :mod:`pyqed` install, while still giving a familiar ANN workflow:
prepare coordinates and energies, fit, predict energies, and optionally
evaluate numerical gradients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def _require_jax():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is required for EquivariantMLP. Install jax to use this model."
        ) from exc
    return jax, jnp


def _as_2d_coordinates(x: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError("coordinates must have at least one dimension")
    if arr.ndim == 1:
        return arr.reshape(1, -1), ()
    return arr.reshape(-1, arr.shape[-1]), arr.shape[:-1]


def _activation(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "silu":
        return z / (1.0 + np.exp(-z))
    raise ValueError(f"unsupported activation {name!r}")


def _activation_grad(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        a = np.tanh(z)
        return 1.0 - a * a
    if name == "relu":
        return (z > 0.0).astype(z.dtype)
    if name == "silu":
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1.0 + z * (1.0 - sig))
    raise ValueError(f"unsupported activation {name!r}")


def grid_to_samples(
    axes: Sequence[ArrayLike],
    values: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a regular-grid PES into coordinate and energy samples.

    Parameters
    ----------
    axes
        One coordinate axis per nuclear degree of freedom.
    values
        PES values on the tensor-product grid.  Shape can be
        ``(n1, ..., nk)`` for a single state or
        ``(n1, ..., nk, nstates)`` for multiple states.

    Returns
    -------
    coordinates, energies
        ``coordinates`` has shape ``(nsamples, ndim)``.  ``energies`` has shape
        ``(nsamples,)`` for a single-state PES or ``(nsamples, nstates)`` for a
        multi-state PES.
    """

    axes_arr = [np.asarray(axis, dtype=float) for axis in axes]
    if not axes_arr:
        raise ValueError("at least one coordinate axis is required")
    if any(axis.ndim != 1 for axis in axes_arr):
        raise ValueError("all coordinate axes must be one-dimensional")

    grid_shape = tuple(axis.size for axis in axes_arr)
    y = np.asarray(values, dtype=float)
    if y.shape[: len(grid_shape)] != grid_shape:
        raise ValueError(
            "values leading shape must match axes lengths: "
            f"expected {grid_shape}, got {y.shape}"
        )

    meshes = np.meshgrid(*axes_arr, indexing="ij")
    x = np.stack([mesh.ravel() for mesh in meshes], axis=1)
    trailing = y.shape[len(grid_shape) :]
    if not trailing:
        return x, y.reshape(-1)
    return x, y.reshape(x.shape[0], int(np.prod(trailing)))


@dataclass
class PESFitResult:
    """Training diagnostics returned in ``MLP.result_``."""

    epochs: int
    train_loss: float
    validation_loss: float | None = None


class MLP:
    """Fully connected ANN regressor for potential energy surfaces.

    The model standardizes coordinates and energies internally.  It supports
    scalar PESs and vector-valued PESs, for example multiple adiabatic states.
    """

    def __init__(
        self,
        hidden_layers: Iterable[int] = (64, 64),
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        batch_size: int | None = 128,
        max_iter: int = 2000,
        l2: float = 0.0,
        validation_fraction: float = 0.0,
        tol: float = 1e-10,
        patience: int = 200,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.validation_fraction = float(validation_fraction)
        self.tol = float(tol)
        self.patience = int(patience)
        self.random_state = random_state
        self.verbose = bool(verbose)

        if any(width <= 0 for width in self.hidden_layers):
            raise ValueError("hidden layer widths must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        _activation(self.activation, np.zeros(1))

    def fit(self, coordinates: ArrayLike, energies: ArrayLike) -> "MLP":
        """Fit the ANN to PES samples."""

        x = np.asarray(coordinates, dtype=float)
        y = np.asarray(energies, dtype=float)
        if x.ndim != 2:
            raise ValueError("coordinates must have shape (nsamples, ndim)")
        if y.ndim == 1:
            y = y[:, None]
            self._scalar_output = True
        elif y.ndim == 2:
            self._scalar_output = False
        else:
            raise ValueError("energies must have shape (nsamples,) or (nsamples, nstates)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("coordinates and energies must have the same number of samples")
        if x.shape[0] < 2:
            raise ValueError("at least two samples are required")

        rng = np.random.default_rng(self.random_state)
        x_train, y_train, x_val, y_val = self._split_train_validation(x, y, rng)

        self.x_mean_ = x_train.mean(axis=0)
        self.x_scale_ = self._safe_scale(x_train.std(axis=0))
        self.y_mean_ = y_train.mean(axis=0)
        self.y_scale_ = self._safe_scale(y_train.std(axis=0))
        xn_train = (x_train - self.x_mean_) / self.x_scale_
        yn_train = (y_train - self.y_mean_) / self.y_scale_
        xn_val = None if x_val is None else (x_val - self.x_mean_) / self.x_scale_
        yn_val = None if y_val is None else (y_val - self.y_mean_) / self.y_scale_

        self._initialize_parameters(x.shape[1], y.shape[1], rng)
        history = {"train_loss": [], "validation_loss": []}
        best_loss = np.inf
        best_params = self._copy_params()
        stale_epochs = 0

        mw = [np.zeros_like(w) for w in self.weights_]
        vw = [np.zeros_like(w) for w in self.weights_]
        mb = [np.zeros_like(b) for b in self.biases_]
        vb = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        for epoch in range(1, self.max_iter + 1):
            for xb, yb in self._iter_batches(xn_train, yn_train, rng):
                grads_w, grads_b = self._gradients(xb, yb)
                step += 1
                for i in range(len(self.weights_)):
                    mw[i] = beta1 * mw[i] + (1.0 - beta1) * grads_w[i]
                    vw[i] = beta2 * vw[i] + (1.0 - beta2) * (grads_w[i] ** 2)
                    mb[i] = beta1 * mb[i] + (1.0 - beta1) * grads_b[i]
                    vb[i] = beta2 * vb[i] + (1.0 - beta2) * (grads_b[i] ** 2)

                    mwh = mw[i] / (1.0 - beta1**step)
                    vwh = vw[i] / (1.0 - beta2**step)
                    mbh = mb[i] / (1.0 - beta1**step)
                    vbh = vb[i] / (1.0 - beta2**step)
                    self.weights_[i] -= self.learning_rate * mwh / (np.sqrt(vwh) + eps)
                    self.biases_[i] -= self.learning_rate * mbh / (np.sqrt(vbh) + eps)

            train_loss = self._loss(xn_train, yn_train)
            history["train_loss"].append(train_loss)
            if xn_val is None:
                monitor_loss = train_loss
                val_loss = None
            else:
                val_loss = self._loss(xn_val, yn_val)
                history["validation_loss"].append(val_loss)
                monitor_loss = val_loss

            if self.verbose and (epoch == 1 or epoch % 100 == 0):
                msg = f"epoch {epoch:5d} train_loss={train_loss:.6e}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.6e}"
                print(msg)

            if monitor_loss + self.tol < best_loss:
                best_loss = monitor_loss
                best_params = self._copy_params()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.weights_, self.biases_ = best_params
        self.history_ = history
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1]
        final_train_loss = self._loss(xn_train, yn_train)
        final_val_loss = None if xn_val is None else self._loss(xn_val, yn_val)
        self.result_ = PESFitResult(epoch, final_train_loss, final_val_loss)
        return self

    def predict(self, coordinates: ArrayLike) -> np.ndarray:
        """Predict PES values at one or more geometries."""

        self._check_is_fit()
        x, leading_shape = _as_2d_coordinates(coordinates)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} coordinates, got {x.shape[1]}")
        y = self._predict_2d(x)
        if self._scalar_output:
            return y[:, 0].reshape(leading_shape)
        return y.reshape(leading_shape + (self.n_outputs_,))

    energy = predict

    def gradient(self, coordinates: ArrayLike, dx: float = 1e-4) -> np.ndarray:
        """Evaluate numerical PES gradients by central finite differences.

        For a scalar PES the returned shape is ``(..., ndim)``.  For a
        multi-state PES it is ``(..., nstates, ndim)``.
        """

        self._check_is_fit()
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        x, leading_shape = _as_2d_coordinates(coordinates)
        grad = np.empty((x.shape[0], self.n_outputs_, self.n_features_in_), dtype=float)
        for i in range(self.n_features_in_):
            xp = x.copy()
            xm = x.copy()
            xp[:, i] += dx
            xm[:, i] -= dx
            grad[:, :, i] = (self._predict_2d(xp) - self._predict_2d(xm)) / (2.0 * dx)
        if self._scalar_output:
            return grad[:, 0, :].reshape(leading_shape + (self.n_features_in_,))
        return grad.reshape(leading_shape + (self.n_outputs_, self.n_features_in_))

    def save(self, filename: str) -> None:
        """Save model parameters to a NumPy ``.npz`` file."""

        self._check_is_fit()
        data = {
            "config": json.dumps(self._config()),
            "x_mean": self.x_mean_,
            "x_scale": self.x_scale_,
            "y_mean": self.y_mean_,
            "y_scale": self.y_scale_,
        }
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            data[f"W{i}"] = w
            data[f"b{i}"] = b
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename: str) -> "MLP":
        """Load a model saved by :meth:`save`."""

        data = np.load(filename)
        config = json.loads(str(data["config"]))
        model = cls(
            hidden_layers=config["hidden_layers"],
            activation=config["activation"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            max_iter=config["max_iter"],
            l2=config["l2"],
            validation_fraction=config["validation_fraction"],
            tol=config["tol"],
            patience=config["patience"],
            random_state=config["random_state"],
            verbose=config["verbose"],
        )
        n_layers = config["n_layers"]
        model.weights_ = [data[f"W{i}"] for i in range(n_layers)]
        model.biases_ = [data[f"b{i}"] for i in range(n_layers)]
        model.x_mean_ = data["x_mean"]
        model.x_scale_ = data["x_scale"]
        model.y_mean_ = data["y_mean"]
        model.y_scale_ = data["y_scale"]
        model.n_features_in_ = int(config["n_features_in"])
        model.n_outputs_ = int(config["n_outputs"])
        model._scalar_output = bool(config["scalar_output"])
        model.history_ = {"train_loss": [], "validation_loss": []}
        model.result_ = PESFitResult(0, np.nan, None)
        return model

    def _config(self) -> dict[str, object]:
        return {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "l2": self.l2,
            "validation_fraction": self.validation_fraction,
            "tol": self.tol,
            "patience": self.patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "n_layers": len(self.weights_),
            "n_features_in": self.n_features_in_,
            "n_outputs": self.n_outputs_,
            "scalar_output": self._scalar_output,
        }

    @staticmethod
    def _safe_scale(scale: np.ndarray) -> np.ndarray:
        scale = np.asarray(scale, dtype=float)
        return np.where(scale > 0.0, scale, 1.0)

    def _split_train_validation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.validation_fraction == 0.0:
            return x, y, None, None
        n_val = max(1, int(round(x.shape[0] * self.validation_fraction)))
        if n_val >= x.shape[0]:
            raise ValueError("validation split leaves no training samples")
        order = rng.permutation(x.shape[0])
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    def _initialize_parameters(
        self,
        n_features: int,
        n_outputs: int,
        rng: np.random.Generator,
    ) -> None:
        dims = (n_features,) + self.hidden_layers + (n_outputs,)
        self.weights_ = []
        self.biases_ = []
        for din, dout in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / (din + dout))
            self.weights_.append(rng.uniform(-limit, limit, size=(din, dout)))
            self.biases_.append(np.zeros(dout, dtype=float))

    def _iter_batches(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        if self.batch_size is None or self.batch_size >= x.shape[0]:
            yield x, y
            return
        order = rng.permutation(x.shape[0])
        for start in range(0, x.shape[0], self.batch_size):
            idx = order[start : start + self.batch_size]
            yield x[idx], y[idx]

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        a = x
        cache = []
        for w, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = a @ w + b
            cache.append((a, z))
            a = _activation(self.activation, z)
        z = a @ self.weights_[-1] + self.biases_[-1]
        cache.append((a, z))
        return z, cache

    def _gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pred, cache = self._forward(x)
        delta = (2.0 / x.shape[0]) * (pred - y)
        grads_w = [np.empty_like(w) for w in self.weights_]
        grads_b = [np.empty_like(b) for b in self.biases_]

        for layer in reversed(range(len(self.weights_))):
            a_prev = cache[layer][0]
            grads_w[layer] = a_prev.T @ delta + self.l2 * self.weights_[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                z_prev = cache[layer - 1][1]
                delta = (delta @ self.weights_[layer].T) * _activation_grad(
                    self.activation, z_prev
                )
        return grads_w, grads_b

    def _loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred, _ = self._forward(x)
        mse = np.mean((pred - y) ** 2)
        penalty = 0.5 * self.l2 * sum(np.sum(w * w) for w in self.weights_)
        return float(mse + penalty)

    def _predict_2d(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.x_mean_) / self.x_scale_
        yn, _ = self._forward(xn)
        return yn * self.y_scale_ + self.y_mean_

    def _copy_params(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [w.copy() for w in self.weights_], [b.copy() for b in self.biases_]

    def _check_is_fit(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("model is not fit yet")


def fit_pes(
    coordinates: ArrayLike,
    energies: ArrayLike,
    **kwargs: object,
) -> MLP:
    """Convenience function returning a fitted :class:`MLP`."""

    return MLP(**kwargs).fit(coordinates, energies)


class _NumpyEquivariantMLP:
    """MACE-inspired invariant energy model with equivariant forces.

    The model maps Cartesian geometries to atom-type-aware radial and angular
    descriptors, then fits an :class:`MLP` from those descriptors to energies.
    The energy is invariant to translation, rotation, and permutation of atoms
    with the same species.  Forces are obtained as ``-dE/dR`` by central finite
    differences, so they transform equivariantly with the geometry.

    This is intentionally a small NumPy model, not a full MACE implementation.
    """

    def __init__(
        self,
        species: Sequence[object] | None = None,
        n_radial: int = 8,
        radial_centers: Sequence[float] | None = None,
        radial_width: float | None = None,
        cutoff: float | None = None,
        angle_order: int = 2,
        hidden_layers: Iterable[int] = (64, 64),
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        batch_size: int | None = 128,
        max_iter: int = 2000,
        l2: float = 0.0,
        validation_fraction: float = 0.0,
        tol: float = 1e-10,
        patience: int = 200,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if n_radial <= 0:
            raise ValueError("n_radial must be positive")
        if angle_order < 0:
            raise ValueError("angle_order must be non-negative")

        self.species = None if species is None else tuple(str(s) for s in species)
        self.n_radial = int(n_radial)
        self.radial_centers = (
            None if radial_centers is None else np.asarray(radial_centers, dtype=float)
        )
        if self.radial_centers is not None:
            if self.radial_centers.ndim != 1 or self.radial_centers.size == 0:
                raise ValueError("radial_centers must be a non-empty one-dimensional array")
            self.n_radial = int(self.radial_centers.size)
        self.radial_width = radial_width
        self.cutoff = cutoff
        self.angle_order = int(angle_order)
        self.mlp_kwargs = {
            "hidden_layers": tuple(hidden_layers),
            "activation": activation,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "l2": l2,
            "validation_fraction": validation_fraction,
            "tol": tol,
            "patience": patience,
            "random_state": random_state,
            "verbose": verbose,
        }

    def fit(self, geometries: ArrayLike, energies: ArrayLike) -> "_NumpyEquivariantMLP":
        """Fit the model to Cartesian geometries and PES values."""

        xyz, _ = self._as_geometries(geometries)
        if self.species is None:
            self.species_ = tuple("X" for _ in range(xyz.shape[1]))
        else:
            self.species_ = self.species
        if len(self.species_) != xyz.shape[1]:
            raise ValueError("species length must match the number of atoms")

        self._set_radial_grid(xyz)
        self._build_descriptor_index()
        descriptors = self.describe(xyz)
        self.model_ = MLP(**self.mlp_kwargs).fit(descriptors, energies)
        self.n_atoms_ = xyz.shape[1]
        self.n_outputs_ = self.model_.n_outputs_
        self._scalar_output = self.model_._scalar_output
        self.result_ = self.model_.result_
        return self

    def describe(self, geometries: ArrayLike) -> np.ndarray:
        """Return invariant descriptors for one or more geometries."""

        self._check_descriptor_ready()
        xyz, leading_shape = self._as_geometries(geometries)
        if xyz.shape[1] != len(self.species_):
            raise ValueError(f"expected {len(self.species_)} atoms, got {xyz.shape[1]}")

        out = np.empty((xyz.shape[0], self.n_descriptors_), dtype=float)
        for n, geom in enumerate(xyz):
            out[n] = self._describe_one(geom)
        return out.reshape(leading_shape + (self.n_descriptors_,))

    def predict(self, geometries: ArrayLike) -> np.ndarray:
        """Predict energies for one or more Cartesian geometries."""

        self._check_is_fit()
        xyz, leading_shape = self._as_geometries(geometries)
        descriptors = self.describe(xyz)
        y = self.model_.predict(descriptors.reshape(-1, self.n_descriptors_))
        if self._scalar_output:
            return np.asarray(y).reshape(leading_shape)
        return np.asarray(y).reshape(leading_shape + (self.n_outputs_,))

    energy = predict

    def gradient(self, geometries: ArrayLike, dx: float = 1e-4) -> np.ndarray:
        """Return ``dE/dR`` by central finite differences in Cartesian space."""

        self._check_is_fit()
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        xyz, leading_shape = self._as_geometries(geometries)
        grad = np.empty((xyz.shape[0], self.n_outputs_, self.n_atoms_, 3), dtype=float)
        for atom in range(self.n_atoms_):
            for axis in range(3):
                xp = xyz.copy()
                xm = xyz.copy()
                xp[:, atom, axis] += dx
                xm[:, atom, axis] -= dx
                ep = np.asarray(self.model_.predict(self.describe(xp)))
                em = np.asarray(self.model_.predict(self.describe(xm)))
                if self._scalar_output:
                    ep = ep[:, None]
                    em = em[:, None]
                grad[:, :, atom, axis] = (ep - em) / (2.0 * dx)
        if self._scalar_output:
            return grad[:, 0].reshape(leading_shape + (self.n_atoms_, 3))
        return grad.reshape(leading_shape + (self.n_outputs_, self.n_atoms_, 3))

    def forces(self, geometries: ArrayLike, dx: float = 1e-4) -> np.ndarray:
        """Return equivariant Cartesian forces, ``-dE/dR``."""

        return -self.gradient(geometries, dx=dx)

    def _describe_one(self, geom: np.ndarray) -> np.ndarray:
        descriptor = np.zeros(self.n_descriptors_, dtype=float)
        n_atoms = geom.shape[0]
        distances, vectors = self._pair_geometry(geom)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                key = tuple(sorted((self.species_[i], self.species_[j])))
                start = self._pair_slices[key]
                descriptor[start : start + self.n_radial] += self._radial_basis(
                    distances[i, j]
                )

        if self.angle_order == 0 or n_atoms < 3:
            return descriptor

        for center in range(n_atoms):
            neighbors = [i for i in range(n_atoms) if i != center]
            for a, i in enumerate(neighbors[:-1]):
                for k in neighbors[a + 1 :]:
                    si, sk = self.species_[i], self.species_[k]
                    ri = self._radial_basis(distances[center, i])
                    rk = self._radial_basis(distances[center, k])
                    if si > sk:
                        si, sk = sk, si
                        ri, rk = rk, ri
                    radial_product = np.outer(ri, rk)
                    if si == sk:
                        radial_product = 0.5 * (radial_product + radial_product.T)
                    cos_theta = self._cosine(vectors[center, i], vectors[center, k])
                    key = (self.species_[center], si, sk)
                    start = self._triplet_slices[key]
                    width = self.n_radial * self.n_radial
                    for power in range(self.angle_order + 1):
                        lo = start + power * width
                        hi = lo + width
                        descriptor[lo:hi] += (cos_theta**power) * radial_product.ravel()
        return descriptor

    def _set_radial_grid(self, xyz: np.ndarray) -> None:
        if self.radial_centers is not None:
            self.radial_centers_ = self.radial_centers.copy()
        else:
            distances = []
            for geom in xyz:
                d, _ = self._pair_geometry(geom)
                distances.extend(d[np.triu_indices(geom.shape[0], k=1)])
            distances = np.asarray(distances)
            d_min = max(1e-8, float(np.min(distances)))
            d_max = float(np.max(distances))
            if self.cutoff is not None:
                d_max = min(d_max, float(self.cutoff))
            if d_max <= d_min:
                d_max = d_min + 1.0
            self.radial_centers_ = np.linspace(d_min, d_max, self.n_radial)
        if self.radial_width is None:
            if self.radial_centers_.size == 1:
                self.radial_width_ = 1.0
            else:
                spacing = np.mean(np.diff(self.radial_centers_))
                self.radial_width_ = float(max(spacing, 1e-8))
        else:
            self.radial_width_ = float(self.radial_width)
            if self.radial_width_ <= 0.0:
                raise ValueError("radial_width must be positive")

    def _build_descriptor_index(self) -> None:
        species_set = tuple(sorted(set(self.species_)))
        self._pair_slices = {}
        offset = 0
        for a, sa in enumerate(species_set):
            for sb in species_set[a:]:
                self._pair_slices[(sa, sb)] = offset
                offset += self.n_radial

        self._triplet_slices = {}
        if self.angle_order > 0:
            width = (self.angle_order + 1) * self.n_radial * self.n_radial
            for center_species in species_set:
                for a, sa in enumerate(species_set):
                    for sb in species_set[a:]:
                        self._triplet_slices[(center_species, sa, sb)] = offset
                        offset += width
        self.n_descriptors_ = offset

    def _radial_basis(self, r: float) -> np.ndarray:
        basis = np.exp(-0.5 * ((r - self.radial_centers_) / self.radial_width_) ** 2)
        if self.cutoff is None:
            return basis
        rc = float(self.cutoff)
        if r >= rc:
            return np.zeros_like(basis)
        return basis * 0.5 * (np.cos(np.pi * r / rc) + 1.0)

    @staticmethod
    def _pair_geometry(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = geom[None, :, :] - geom[:, None, :]
        distances = np.linalg.norm(vectors, axis=-1)
        return distances, vectors

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _as_geometries(geometries: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
        xyz = np.asarray(geometries, dtype=float)
        if xyz.ndim < 2 or xyz.shape[-1] != 3:
            raise ValueError("geometries must have shape (..., natoms, 3)")
        if xyz.ndim == 2:
            return xyz.reshape(1, xyz.shape[0], 3), ()
        return xyz.reshape(-1, xyz.shape[-2], 3), xyz.shape[:-2]

    def _check_descriptor_ready(self) -> None:
        if not hasattr(self, "n_descriptors_"):
            if self.species is None or self.radial_centers is None:
                raise RuntimeError(
                    "descriptor grid is not initialized; call fit first or provide "
                    "species and radial_centers"
                )
            self.species_ = self.species
            self.radial_centers_ = self.radial_centers.copy()
            self.radial_width_ = (
                float(self.radial_width)
                if self.radial_width is not None
                else float(np.mean(np.diff(self.radial_centers_)))
                if self.radial_centers_.size > 1
                else 1.0
            )
            self._build_descriptor_index()

    def _check_is_fit(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("model is not fit yet")


class EquivariantMLP(_NumpyEquivariantMLP):
    """JAX-backed invariant energy model with autodiff equivariant forces.

    This has the same descriptor construction as the NumPy fallback, but
    trains the neural network in JAX and computes Cartesian gradients/forces by
    automatic differentiation.  If force targets are supplied to :meth:`fit`,
    the loss is ``energy_mse + force_weight * force_mse``.
    """

    def __init__(
        self,
        species: Sequence[object] | None = None,
        n_radial: int = 8,
        radial_centers: Sequence[float] | None = None,
        radial_width: float | None = None,
        cutoff: float | None = None,
        angle_order: int = 2,
        hidden_layers: Iterable[int] = (64, 64),
        activation: str = "tanh",
        learning_rate: float = 1e-3,
        batch_size: int | None = 128,
        max_iter: int = 2000,
        l2: float = 0.0,
        validation_fraction: float = 0.0,
        tol: float = 1e-10,
        patience: int = 200,
        force_weight: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            species=species,
            n_radial=n_radial,
            radial_centers=radial_centers,
            radial_width=radial_width,
            cutoff=cutoff,
            angle_order=angle_order,
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_iter=max_iter,
            l2=l2,
            validation_fraction=validation_fraction,
            tol=tol,
            patience=patience,
            random_state=random_state,
            verbose=verbose,
        )
        if force_weight < 0.0:
            raise ValueError("force_weight must be non-negative")
        self.force_weight = float(force_weight)

    def fit(
        self,
        geometries: ArrayLike,
        energies: ArrayLike,
        forces: ArrayLike | None = None,
    ) -> "EquivariantMLP":
        """Fit to Cartesian geometries, energies, and optional forces."""

        jax, jnp = _require_jax()
        xyz, _ = self._as_geometries(geometries)
        y = np.asarray(energies, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
            self._scalar_output = True
        elif y.ndim == 2:
            self._scalar_output = False
        else:
            raise ValueError("energies must have shape (nsamples,) or (nsamples, nstates)")
        if xyz.shape[0] != y.shape[0]:
            raise ValueError("geometries and energies must have the same number of samples")

        force_targets = None
        if forces is not None:
            force_targets = np.asarray(forces, dtype=float)
            if self._scalar_output and force_targets.shape == xyz.shape:
                force_targets = force_targets[:, None, :, :]
            expected_shape = (xyz.shape[0], y.shape[1], xyz.shape[1], 3)
            if force_targets.shape != expected_shape:
                raise ValueError(
                    "forces must have shape (nsamples, natoms, 3) for scalar energies "
                    "or (nsamples, nstates, natoms, 3) for vector energies"
                )

        if self.species is None:
            self.species_ = tuple("X" for _ in range(xyz.shape[1]))
        else:
            self.species_ = self.species
        if len(self.species_) != xyz.shape[1]:
            raise ValueError("species length must match the number of atoms")

        self._set_radial_grid(xyz)
        self._build_descriptor_index()
        self._prepare_jax_descriptor_arrays(jnp)
        descriptors = np.asarray(jax.vmap(self._descriptor_one_jax)(jnp.asarray(xyz)))

        rng = np.random.default_rng(self.mlp_kwargs["random_state"])
        x_train, y_train, f_train, x_val, y_val, f_val = self._split_jax_training_data(
            xyz, y, force_targets, rng
        )

        train_desc = descriptors[: x_train.shape[0]]
        if self.mlp_kwargs["validation_fraction"] > 0.0:
            # The split shuffles samples, so recompute descriptors for the selected arrays.
            train_desc = np.asarray(jax.vmap(self._descriptor_one_jax)(jnp.asarray(x_train)))
            val_desc = np.asarray(jax.vmap(self._descriptor_one_jax)(jnp.asarray(x_val)))
        else:
            val_desc = None

        self.x_mean_ = train_desc.mean(axis=0)
        self.x_scale_ = MLP._safe_scale(train_desc.std(axis=0))
        self.y_mean_ = y_train.mean(axis=0)
        self.y_scale_ = MLP._safe_scale(y_train.std(axis=0))
        self._j_x_mean = jnp.asarray(self.x_mean_)
        self._j_x_scale = jnp.asarray(self.x_scale_)
        self._j_y_mean = jnp.asarray(self.y_mean_)
        self._j_y_scale = jnp.asarray(self.y_scale_)

        self.n_atoms_ = xyz.shape[1]
        self.n_outputs_ = y.shape[1]
        self.params_ = self._initialize_jax_params(jnp, rng)

        has_forces = f_train is not None
        x_train_j = jnp.asarray(x_train)
        y_train_j = jnp.asarray(y_train)
        f_train_j = None if f_train is None else jnp.asarray(f_train)
        x_val_j = None if x_val is None else jnp.asarray(x_val)
        y_val_j = None if y_val is None else jnp.asarray(y_val)
        f_val_j = None if f_val is None else jnp.asarray(f_val)

        def force_one(params, geom):
            if self.n_outputs_ == 1:
                grad = jax.grad(lambda g: self._energy_one_jax(params, g)[0])(geom)
                return -grad[None, :, :]
            jac = jax.jacrev(lambda g: self._energy_one_jax(params, g))(geom)
            return -jac

        def loss_fn(params, batch_x, batch_y, batch_f):
            pred_y = jax.vmap(lambda g: self._energy_one_jax(params, g))(batch_x)
            loss = jnp.mean((pred_y - batch_y) ** 2)
            if has_forces:
                pred_f = jax.vmap(lambda g: force_one(params, g))(batch_x)
                loss = loss + self.force_weight * jnp.mean((pred_f - batch_f) ** 2)
            if self.mlp_kwargs["l2"] > 0.0:
                penalty = sum(jnp.sum(w * w) for w, _ in params)
                loss = loss + 0.5 * self.mlp_kwargs["l2"] * penalty
            return loss

        @jax.jit
        def train_step(params, m, v, step, batch_x, batch_y, batch_f):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y, batch_f)
            step = step + 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m = jax.tree_util.tree_map(
                lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads
            )
            v = jax.tree_util.tree_map(
                lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), v, grads
            )
            mhat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta1**step), m)
            vhat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta2**step), v)
            params = jax.tree_util.tree_map(
                lambda p, mh, vh: p
                - self.mlp_kwargs["learning_rate"] * mh / (jnp.sqrt(vh) + eps),
                params,
                mhat,
                vhat,
            )
            return params, m, v, step, loss

        m = jax.tree_util.tree_map(jnp.zeros_like, self.params_)
        v = jax.tree_util.tree_map(jnp.zeros_like, self.params_)
        step = jnp.asarray(0)
        best_params = self.params_
        best_loss = np.inf
        stale_epochs = 0
        history = {"train_loss": [], "validation_loss": []}
        batch_size = self.mlp_kwargs["batch_size"] or x_train.shape[0]

        for epoch in range(1, self.mlp_kwargs["max_iter"] + 1):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                idx = order[start : start + batch_size]
                bx = x_train_j[idx]
                by = y_train_j[idx]
                bf = None if f_train_j is None else f_train_j[idx]
                self.params_, m, v, step, _ = train_step(
                    self.params_, m, v, step, bx, by, bf
                )

            train_loss = float(loss_fn(self.params_, x_train_j, y_train_j, f_train_j))
            history["train_loss"].append(train_loss)
            if x_val_j is None:
                monitor_loss = train_loss
                val_loss = None
            else:
                val_loss = float(loss_fn(self.params_, x_val_j, y_val_j, f_val_j))
                history["validation_loss"].append(val_loss)
                monitor_loss = val_loss

            if self.mlp_kwargs["verbose"] and (epoch == 1 or epoch % 100 == 0):
                msg = f"epoch {epoch:5d} train_loss={train_loss:.6e}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.6e}"
                print(msg)

            if monitor_loss + self.mlp_kwargs["tol"] < best_loss:
                best_loss = monitor_loss
                best_params = jax.tree_util.tree_map(lambda p: p.copy(), self.params_)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.mlp_kwargs["patience"]:
                    break

        self.params_ = best_params
        final_train_loss = float(loss_fn(self.params_, x_train_j, y_train_j, f_train_j))
        final_val_loss = None
        if x_val_j is not None:
            final_val_loss = float(loss_fn(self.params_, x_val_j, y_val_j, f_val_j))
        self.history_ = history
        self.result_ = PESFitResult(epoch, final_train_loss, final_val_loss)
        self._force_one_jax = force_one
        return self

    def describe(self, geometries: ArrayLike) -> np.ndarray:
        """Return JAX descriptor values as a NumPy array."""

        jax, jnp = _require_jax()
        self._check_descriptor_ready()
        if not hasattr(self, "_j_pair_i"):
            self._prepare_jax_descriptor_arrays(jnp)
        xyz, leading_shape = self._as_geometries(geometries)
        desc = jax.vmap(self._descriptor_one_jax)(jnp.asarray(xyz))
        return np.asarray(desc).reshape(leading_shape + (self.n_descriptors_,))

    def predict(self, geometries: ArrayLike) -> np.ndarray:
        """Predict energies with the JAX model."""

        jax, jnp = _require_jax()
        self._check_is_fit()
        xyz, leading_shape = self._as_geometries(geometries)
        y = jax.vmap(lambda g: self._energy_one_jax(self.params_, g))(jnp.asarray(xyz))
        y = np.asarray(y)
        if self._scalar_output:
            return y[:, 0].reshape(leading_shape)
        return y.reshape(leading_shape + (self.n_outputs_,))

    energy = predict

    def gradient(self, geometries: ArrayLike) -> np.ndarray:
        """Return autodiff Cartesian gradients ``dE/dR``."""

        jax, jnp = _require_jax()
        self._check_is_fit()
        xyz, leading_shape = self._as_geometries(geometries)

        if self.n_outputs_ == 1:
            grad_fn = jax.grad(lambda g: self._energy_one_jax(self.params_, g)[0])
            grad = jax.vmap(grad_fn)(jnp.asarray(xyz))
            return np.asarray(grad).reshape(leading_shape + (self.n_atoms_, 3))

        jac_fn = jax.jacrev(lambda g: self._energy_one_jax(self.params_, g))
        grad = jax.vmap(jac_fn)(jnp.asarray(xyz))
        return np.asarray(grad).reshape(
            leading_shape + (self.n_outputs_, self.n_atoms_, 3)
        )

    def forces(self, geometries: ArrayLike) -> np.ndarray:
        """Return autodiff Cartesian forces, ``-dE/dR``."""

        return -self.gradient(geometries)

    def save(self, filename: str) -> None:
        """Save the fitted JAX equivariant MLP to a NumPy ``.npz`` file."""

        self._check_is_fit()
        config = {
            "species": self.species,
            "n_radial": self.n_radial,
            "radial_width": self.radial_width,
            "cutoff": self.cutoff,
            "angle_order": self.angle_order,
            "mlp_kwargs": self.mlp_kwargs,
            "force_weight": self.force_weight,
            "n_atoms": self.n_atoms_,
            "n_outputs": self.n_outputs_,
            "scalar_output": self._scalar_output,
            "n_descriptors": self.n_descriptors_,
        }
        data = {
            "config": json.dumps(config),
            "radial_centers": np.asarray(self.radial_centers_),
            "radial_width_value": np.asarray(self.radial_width_),
            "x_mean": np.asarray(self.x_mean_),
            "x_scale": np.asarray(self.x_scale_),
            "y_mean": np.asarray(self.y_mean_),
            "y_scale": np.asarray(self.y_scale_),
        }
        for i, (w, b) in enumerate(self.params_):
            data[f"W{i}"] = np.asarray(w)
            data[f"b{i}"] = np.asarray(b)
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename: str) -> "EquivariantMLP":
        """Load a model saved by :meth:`save`."""

        _, jnp = _require_jax()
        data = np.load(filename)
        config = json.loads(str(data["config"]))
        model = cls(
            species=config["species"],
            n_radial=config["n_radial"],
            radial_centers=data["radial_centers"],
            radial_width=config["radial_width"],
            cutoff=config["cutoff"],
            angle_order=config["angle_order"],
            force_weight=config["force_weight"],
            **config["mlp_kwargs"],
        )
        model.species_ = tuple(str(s) for s in config["species"])
        model.n_atoms_ = int(config["n_atoms"])
        model.n_outputs_ = int(config["n_outputs"])
        model._scalar_output = bool(config["scalar_output"])
        model.radial_centers_ = np.asarray(data["radial_centers"])
        model.radial_width_ = float(np.asarray(data["radial_width_value"]))
        model.x_mean_ = np.asarray(data["x_mean"])
        model.x_scale_ = np.asarray(data["x_scale"])
        model.y_mean_ = np.asarray(data["y_mean"])
        model.y_scale_ = np.asarray(data["y_scale"])
        model._j_x_mean = jnp.asarray(model.x_mean_)
        model._j_x_scale = jnp.asarray(model.x_scale_)
        model._j_y_mean = jnp.asarray(model.y_mean_)
        model._j_y_scale = jnp.asarray(model.y_scale_)
        model._build_descriptor_index()
        model._prepare_jax_descriptor_arrays(jnp)
        n_layers = len(config["mlp_kwargs"]["hidden_layers"]) + 1
        model.params_ = [
            (jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"]))
            for i in range(n_layers)
        ]
        model.history_ = {"train_loss": [], "validation_loss": []}
        model.result_ = PESFitResult(0, np.nan, None)
        return model

    def _split_jax_training_data(
        self,
        xyz: np.ndarray,
        y: np.ndarray,
        forces: np.ndarray | None,
        rng: np.random.Generator,
    ):
        if self.mlp_kwargs["validation_fraction"] == 0.0:
            return xyz, y, forces, None, None, None
        n_val = max(1, int(round(xyz.shape[0] * self.mlp_kwargs["validation_fraction"])))
        if n_val >= xyz.shape[0]:
            raise ValueError("validation split leaves no training samples")
        order = rng.permutation(xyz.shape[0])
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        f_train = None if forces is None else forces[train_idx]
        f_val = None if forces is None else forces[val_idx]
        return xyz[train_idx], y[train_idx], f_train, xyz[val_idx], y[val_idx], f_val

    def _initialize_jax_params(self, jnp, rng: np.random.Generator):
        dims = (self.n_descriptors_,) + self.mlp_kwargs["hidden_layers"] + (
            self.n_outputs_,
        )
        params = []
        for din, dout in zip(dims[:-1], dims[1:]):
            limit = np.sqrt(6.0 / (din + dout))
            w = rng.uniform(-limit, limit, size=(din, dout))
            b = np.zeros(dout, dtype=float)
            params.append((jnp.asarray(w), jnp.asarray(b)))
        return params

    def _prepare_jax_descriptor_arrays(self, jnp) -> None:
        n_atoms = len(self.species_)
        pair_i = []
        pair_j = []
        pair_feature_idx = []
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                key = tuple(sorted((self.species_[i], self.species_[j])))
                start = self._pair_slices[key]
                pair_i.append(i)
                pair_j.append(j)
                pair_feature_idx.append(start + np.arange(self.n_radial))

        triplet_center = []
        triplet_left = []
        triplet_right = []
        triplet_same = []
        triplet_feature_idx = []
        if self.angle_order > 0:
            width = (self.angle_order + 1) * self.n_radial * self.n_radial
            for center in range(n_atoms):
                neighbors = [i for i in range(n_atoms) if i != center]
                for a, i in enumerate(neighbors[:-1]):
                    for k in neighbors[a + 1 :]:
                        si, sk = self.species_[i], self.species_[k]
                        left, right = i, k
                        if si > sk:
                            si, sk = sk, si
                            left, right = right, left
                        start = self._triplet_slices[(self.species_[center], si, sk)]
                        triplet_center.append(center)
                        triplet_left.append(left)
                        triplet_right.append(right)
                        triplet_same.append(si == sk)
                        triplet_feature_idx.append(start + np.arange(width))

        self._j_pair_i = jnp.asarray(pair_i, dtype=int)
        self._j_pair_j = jnp.asarray(pair_j, dtype=int)
        self._j_pair_feature_idx = jnp.asarray(pair_feature_idx, dtype=int)
        self._j_triplet_center = jnp.asarray(triplet_center, dtype=int)
        self._j_triplet_left = jnp.asarray(triplet_left, dtype=int)
        self._j_triplet_right = jnp.asarray(triplet_right, dtype=int)
        self._j_triplet_same = jnp.asarray(triplet_same, dtype=bool)
        self._j_triplet_feature_idx = jnp.asarray(triplet_feature_idx, dtype=int)
        self._j_radial_centers = jnp.asarray(self.radial_centers_)
        self._j_radial_width = jnp.asarray(self.radial_width_)
        self._j_cutoff = None if self.cutoff is None else jnp.asarray(self.cutoff)

    def _activation_jax(self, x):
        _, jnp = _require_jax()
        if self.mlp_kwargs["activation"] == "tanh":
            return jnp.tanh(x)
        if self.mlp_kwargs["activation"] == "relu":
            return jnp.maximum(x, 0.0)
        if self.mlp_kwargs["activation"] == "silu":
            return x / (1.0 + jnp.exp(-x))
        raise ValueError(f"unsupported activation {self.mlp_kwargs['activation']!r}")

    def _forward_jax(self, params, x):
        a = x
        for w, b in params[:-1]:
            a = self._activation_jax(a @ w + b)
        w, b = params[-1]
        return a @ w + b

    def _energy_one_jax(self, params, geom):
        desc = self._descriptor_one_jax(geom)
        x = (desc - self._j_x_mean) / self._j_x_scale
        y = self._forward_jax(params, x)
        return y * self._j_y_scale + self._j_y_mean

    def _descriptor_one_jax(self, geom):
        _, jnp = _require_jax()
        vectors = geom[None, :, :] - geom[:, None, :]
        distances = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1) + 1e-24)
        desc = jnp.zeros((self.n_descriptors_,), dtype=geom.dtype)

        pair_r = distances[self._j_pair_i, self._j_pair_j]
        pair_values = self._radial_basis_jax(pair_r)
        desc = desc.at[self._j_pair_feature_idx].add(pair_values)

        if self._j_triplet_center.size == 0:
            return desc

        c = self._j_triplet_center
        left = self._j_triplet_left
        right = self._j_triplet_right
        ri = self._radial_basis_jax(distances[c, left])
        rk = self._radial_basis_jax(distances[c, right])
        outer = ri[:, :, None] * rk[:, None, :]
        sym_outer = 0.5 * (outer + jnp.swapaxes(outer, -1, -2))
        outer = jnp.where(self._j_triplet_same[:, None, None], sym_outer, outer)

        vl = vectors[c, left]
        vr = vectors[c, right]
        denom = jnp.linalg.norm(vl, axis=1) * jnp.linalg.norm(vr, axis=1)
        cos_theta = jnp.clip(jnp.sum(vl * vr, axis=1) / (denom + 1e-24), -1.0, 1.0)
        powers = jnp.stack([cos_theta**p for p in range(self.angle_order + 1)], axis=1)
        features = powers[:, :, None, None] * outer[:, None, :, :]
        features = features.reshape((features.shape[0], -1))
        return desc.at[self._j_triplet_feature_idx].add(features)

    def _radial_basis_jax(self, r):
        _, jnp = _require_jax()
        basis = jnp.exp(
            -0.5 * ((r[..., None] - self._j_radial_centers) / self._j_radial_width) ** 2
        )
        if self._j_cutoff is None:
            return basis
        envelope = jnp.where(
            r[..., None] < self._j_cutoff,
            0.5 * (jnp.cos(jnp.pi * r[..., None] / self._j_cutoff) + 1.0),
            0.0,
        )
        return basis * envelope

    def _check_is_fit(self) -> None:
        if not hasattr(self, "params_"):
            raise RuntimeError("model is not fit yet")


class MPNN:
    """Small scalar-vector equivariant message-passing PES model.

    This is a compact JAX implementation of the core message-passing idea:
    atoms carry learned scalar and vector features, directed pair messages are
    built from neighbor features and the unit vector between atoms, and the
    final scalar atom features are summed into a total energy.  The energy is
    invariant to translations, rotations, and identical-atom permutations; the
    autodiff forces are equivariant.
    """

    def __init__(
        self,
        species: Sequence[object],
        features: int = 32,
        n_layers: int = 3,
        n_radial: int = 8,
        radial_centers: Sequence[float] | None = None,
        radial_width: float | None = None,
        cutoff: float | None = None,
        readout_hidden: int = 32,
        learning_rate: float = 1e-3,
        batch_size: int | None = 16,
        max_iter: int = 1000,
        l2: float = 0.0,
        validation_fraction: float = 0.0,
        tol: float = 1e-10,
        patience: int = 100,
        force_weight: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if features <= 0:
            raise ValueError("features must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if n_radial <= 0:
            raise ValueError("n_radial must be positive")
        if force_weight < 0.0:
            raise ValueError("force_weight must be non-negative")

        self.species = tuple(str(s) for s in species)
        self.features = int(features)
        self.n_layers = int(n_layers)
        self.n_radial = int(n_radial)
        self.radial_centers = (
            None if radial_centers is None else np.asarray(radial_centers, dtype=float)
        )
        if self.radial_centers is not None:
            if self.radial_centers.ndim != 1 or self.radial_centers.size == 0:
                raise ValueError("radial_centers must be a non-empty one-dimensional array")
            self.n_radial = int(self.radial_centers.size)
        self.radial_width = radial_width
        self.cutoff = cutoff
        self.readout_hidden = int(readout_hidden)
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.validation_fraction = float(validation_fraction)
        self.tol = float(tol)
        self.patience = int(patience)
        self.force_weight = float(force_weight)
        self.random_state = random_state
        self.verbose = bool(verbose)

    def fit(
        self,
        geometries: ArrayLike,
        energies: ArrayLike,
        forces: ArrayLike | None = None,
    ) -> "MPNN":
        """Fit the message-passing potential to energies and optional forces."""

        jax, jnp = _require_jax()
        xyz, _ = _NumpyEquivariantMLP._as_geometries(geometries)
        if len(self.species) != xyz.shape[1]:
            raise ValueError("species length must match the number of atoms")
        y = np.asarray(energies, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
            self._scalar_output = True
        elif y.ndim == 2:
            self._scalar_output = False
        else:
            raise ValueError("energies must have shape (nsamples,) or (nsamples, nstates)")
        if xyz.shape[0] != y.shape[0]:
            raise ValueError("geometries and energies must have the same number of samples")

        force_targets = None
        if forces is not None:
            force_targets = np.asarray(forces, dtype=float)
            if self._scalar_output and force_targets.shape == xyz.shape:
                force_targets = force_targets[:, None, :, :]
            expected_shape = (xyz.shape[0], y.shape[1], xyz.shape[1], 3)
            if force_targets.shape != expected_shape:
                raise ValueError(
                    "forces must have shape (nsamples, natoms, 3) for scalar energies "
                    "or (nsamples, nstates, natoms, 3) for vector energies"
                )

        self.n_atoms_ = xyz.shape[1]
        self.n_outputs_ = y.shape[1]
        self._prepare_species()
        self._prepare_edges(jnp)
        self._set_radial_grid(xyz)
        self._prepare_radial_jax(jnp)

        rng = np.random.default_rng(self.random_state)
        x_train, y_train, f_train, x_val, y_val, f_val = self._split_training_data(
            xyz, y, force_targets, rng
        )
        self.y_mean_ = y_train.mean(axis=0)
        self.y_scale_ = MLP._safe_scale(y_train.std(axis=0))
        self._j_y_mean = jnp.asarray(self.y_mean_)
        self._j_y_scale = jnp.asarray(self.y_scale_)
        self.params_ = self._initialize_params(jnp, rng)

        has_forces = f_train is not None
        x_train_j = jnp.asarray(x_train)
        y_train_j = jnp.asarray(y_train)
        f_train_j = None if f_train is None else jnp.asarray(f_train)
        x_val_j = None if x_val is None else jnp.asarray(x_val)
        y_val_j = None if y_val is None else jnp.asarray(y_val)
        f_val_j = None if f_val is None else jnp.asarray(f_val)

        def force_one(params, geom):
            if self.n_outputs_ == 1:
                grad = jax.grad(lambda g: self._energy_one(params, g)[0])(geom)
                return -grad[None, :, :]
            return -jax.jacrev(lambda g: self._energy_one(params, g))(geom)

        def loss_fn(params, batch_x, batch_y, batch_f):
            pred_y = jax.vmap(lambda g: self._energy_one(params, g))(batch_x)
            loss = jnp.mean((pred_y - batch_y) ** 2)
            if has_forces:
                pred_f = jax.vmap(lambda g: force_one(params, g))(batch_x)
                loss = loss + self.force_weight * jnp.mean((pred_f - batch_f) ** 2)
            if self.l2 > 0.0:
                leaves = jax.tree_util.tree_leaves(params)
                loss = loss + 0.5 * self.l2 * sum(jnp.sum(x * x) for x in leaves)
            return loss

        @jax.jit
        def train_step(params, m, v, step, batch_x, batch_y, batch_f):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y, batch_f)
            step = step + 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m = jax.tree_util.tree_map(
                lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads
            )
            v = jax.tree_util.tree_map(
                lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), v, grads
            )
            mhat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta1**step), m)
            vhat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta2**step), v)
            params = jax.tree_util.tree_map(
                lambda p, mh, vh: p - self.learning_rate * mh / (jnp.sqrt(vh) + eps),
                params,
                mhat,
                vhat,
            )
            return params, m, v, step, loss

        m = jax.tree_util.tree_map(jnp.zeros_like, self.params_)
        v = jax.tree_util.tree_map(jnp.zeros_like, self.params_)
        step = jnp.asarray(0)
        best_params = self.params_
        best_loss = np.inf
        stale_epochs = 0
        history = {"train_loss": [], "validation_loss": []}
        batch_size = self.batch_size or x_train.shape[0]

        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                idx = order[start : start + batch_size]
                bf = None if f_train_j is None else f_train_j[idx]
                self.params_, m, v, step, _ = train_step(
                    self.params_,
                    m,
                    v,
                    step,
                    x_train_j[idx],
                    y_train_j[idx],
                    bf,
                )

            train_loss = float(loss_fn(self.params_, x_train_j, y_train_j, f_train_j))
            history["train_loss"].append(train_loss)
            if x_val_j is None:
                monitor_loss = train_loss
                val_loss = None
            else:
                val_loss = float(loss_fn(self.params_, x_val_j, y_val_j, f_val_j))
                history["validation_loss"].append(val_loss)
                monitor_loss = val_loss

            if self.verbose and (epoch == 1 or epoch % 100 == 0):
                msg = f"epoch {epoch:5d} train_loss={train_loss:.6e}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.6e}"
                print(msg)

            if monitor_loss + self.tol < best_loss:
                best_loss = monitor_loss
                best_params = jax.tree_util.tree_map(lambda p: p.copy(), self.params_)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.params_ = best_params
        final_train_loss = float(loss_fn(self.params_, x_train_j, y_train_j, f_train_j))
        final_val_loss = None
        if x_val_j is not None:
            final_val_loss = float(loss_fn(self.params_, x_val_j, y_val_j, f_val_j))
        self.history_ = history
        self.result_ = PESFitResult(epoch, final_train_loss, final_val_loss)
        return self

    def predict(self, geometries: ArrayLike) -> np.ndarray:
        """Predict total energies."""

        jax, jnp = _require_jax()
        self._check_is_fit()
        xyz, leading_shape = _NumpyEquivariantMLP._as_geometries(geometries)
        y = jax.vmap(lambda g: self._energy_one(self.params_, g))(jnp.asarray(xyz))
        y = np.asarray(y)
        if self._scalar_output:
            return y[:, 0].reshape(leading_shape)
        return y.reshape(leading_shape + (self.n_outputs_,))

    energy = predict

    def gradient(self, geometries: ArrayLike) -> np.ndarray:
        """Return autodiff Cartesian gradients ``dE/dR``."""

        jax, jnp = _require_jax()
        self._check_is_fit()
        xyz, leading_shape = _NumpyEquivariantMLP._as_geometries(geometries)
        if self.n_outputs_ == 1:
            grad_fn = jax.grad(lambda g: self._energy_one(self.params_, g)[0])
            grad = jax.vmap(grad_fn)(jnp.asarray(xyz))
            return np.asarray(grad).reshape(leading_shape + (self.n_atoms_, 3))
        jac_fn = jax.jacrev(lambda g: self._energy_one(self.params_, g))
        grad = jax.vmap(jac_fn)(jnp.asarray(xyz))
        return np.asarray(grad).reshape(
            leading_shape + (self.n_outputs_, self.n_atoms_, 3)
        )

    def forces(self, geometries: ArrayLike) -> np.ndarray:
        """Return autodiff Cartesian forces, ``-dE/dR``."""

        return -self.gradient(geometries)

    def save(self, filename: str) -> None:
        """Save the fitted message-passing model to a NumPy ``.npz`` file."""

        self._check_is_fit()
        config = {
            "species": self.species,
            "features": self.features,
            "n_layers": self.n_layers,
            "n_radial": self.n_radial,
            "radial_width": self.radial_width,
            "cutoff": self.cutoff,
            "readout_hidden": self.readout_hidden,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "l2": self.l2,
            "validation_fraction": self.validation_fraction,
            "tol": self.tol,
            "patience": self.patience,
            "force_weight": self.force_weight,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "n_atoms": self.n_atoms_,
            "n_outputs": self.n_outputs_,
            "scalar_output": self._scalar_output,
        }
        data = {
            "config": json.dumps(config),
            "radial_centers": np.asarray(self.radial_centers_),
            "radial_width_value": np.asarray(self.radial_width_),
            "y_mean": np.asarray(self.y_mean_),
            "y_scale": np.asarray(self.y_scale_),
            "embedding": np.asarray(self.params_["embedding"]),
            "readout_w1": np.asarray(self.params_["readout_w1"]),
            "readout_b1": np.asarray(self.params_["readout_b1"]),
            "readout_w2": np.asarray(self.params_["readout_w2"]),
            "readout_b2": np.asarray(self.params_["readout_b2"]),
        }
        for i, layer in enumerate(self.params_["layers"]):
            for name, value in layer.items():
                data[f"layer{i}_{name}"] = np.asarray(value)
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename: str) -> "MPNN":
        """Load a model saved by :meth:`save`."""

        _, jnp = _require_jax()
        data = np.load(filename)
        config = json.loads(str(data["config"]))
        model = cls(
            species=config["species"],
            features=config["features"],
            n_layers=config["n_layers"],
            n_radial=config["n_radial"],
            radial_centers=data["radial_centers"],
            radial_width=config["radial_width"],
            cutoff=config["cutoff"],
            readout_hidden=config["readout_hidden"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            max_iter=config["max_iter"],
            l2=config["l2"],
            validation_fraction=config["validation_fraction"],
            tol=config["tol"],
            patience=config["patience"],
            force_weight=config["force_weight"],
            random_state=config["random_state"],
            verbose=config["verbose"],
        )
        model.n_atoms_ = int(config["n_atoms"])
        model.n_outputs_ = int(config["n_outputs"])
        model._scalar_output = bool(config["scalar_output"])
        model.radial_centers_ = np.asarray(data["radial_centers"])
        model.radial_width_ = float(np.asarray(data["radial_width_value"]))
        model.y_mean_ = np.asarray(data["y_mean"])
        model.y_scale_ = np.asarray(data["y_scale"])
        model._j_y_mean = jnp.asarray(model.y_mean_)
        model._j_y_scale = jnp.asarray(model.y_scale_)
        model._prepare_species()
        model._prepare_edges(jnp)
        model._prepare_radial_jax(jnp)
        model.params_ = {
            "embedding": jnp.asarray(data["embedding"]),
            "layers": [],
            "readout_w1": jnp.asarray(data["readout_w1"]),
            "readout_b1": jnp.asarray(data["readout_b1"]),
            "readout_w2": jnp.asarray(data["readout_w2"]),
            "readout_b2": jnp.asarray(data["readout_b2"]),
        }
        for i in range(model.n_layers):
            model.params_["layers"].append(
                {
                    "w_hh": jnp.asarray(data[f"layer{i}_w_hh"]),
                    "w_vh": jnp.asarray(data[f"layer{i}_w_vh"]),
                    "w_hv": jnp.asarray(data[f"layer{i}_w_hv"]),
                    "w_vv": jnp.asarray(data[f"layer{i}_w_vv"]),
                    "b_h": jnp.asarray(data[f"layer{i}_b_h"]),
                }
            )
        model.history_ = {"train_loss": [], "validation_loss": []}
        model.result_ = PESFitResult(0, np.nan, None)
        return model

    def _energy_one(self, params, geom):
        raw = self._raw_energy_one(params, geom)
        return raw * self._j_y_scale + self._j_y_mean

    def _raw_energy_one(self, params, geom):
        _, jnp = _require_jax()
        h = params["embedding"][self._j_species_ids]
        v = jnp.zeros((self.n_atoms_, self.features, 3), dtype=geom.dtype)
        vectors = geom[self._j_dst] - geom[self._j_src]
        distances = jnp.sqrt(jnp.sum(vectors * vectors, axis=1) + 1e-24)
        unit = vectors / distances[:, None]
        radial = self._radial_basis(distances)

        for layer in params["layers"]:
            sender_h = h[self._j_dst]
            sender_v = v[self._j_dst]
            proj_v = jnp.sum(sender_v * unit[:, None, :], axis=2)

            whh = jnp.einsum("er,rfg->efg", radial, layer["w_hh"])
            wvh = jnp.einsum("er,rfg->efg", radial, layer["w_vh"])
            whv = jnp.einsum("er,rfg->efg", radial, layer["w_hv"])
            wvv = jnp.einsum("er,rfg->efg", radial, layer["w_vv"])

            msg_h = jnp.einsum("efg,eg->ef", whh, sender_h)
            msg_h = msg_h + jnp.einsum("efg,eg->ef", wvh, proj_v)
            coeff_v = jnp.einsum("efg,eg->ef", whv, sender_h)
            msg_v = coeff_v[:, :, None] * unit[:, None, :]
            msg_v = msg_v + jnp.einsum("efg,egc->efc", wvv, sender_v)

            agg_h = jnp.zeros_like(h).at[self._j_src].add(msg_h)
            agg_v = jnp.zeros_like(v).at[self._j_src].add(msg_v)
            h = jnp.tanh(h + agg_h + layer["b_h"])
            v = v + agg_v

        atom_hidden = jnp.tanh(h @ params["readout_w1"] + params["readout_b1"])
        atom_e = atom_hidden @ params["readout_w2"] + params["readout_b2"]
        return jnp.sum(atom_e, axis=0)

    def _split_training_data(
        self,
        xyz: np.ndarray,
        y: np.ndarray,
        forces: np.ndarray | None,
        rng: np.random.Generator,
    ):
        if self.validation_fraction == 0.0:
            return xyz, y, forces, None, None, None
        n_val = max(1, int(round(xyz.shape[0] * self.validation_fraction)))
        if n_val >= xyz.shape[0]:
            raise ValueError("validation split leaves no training samples")
        order = rng.permutation(xyz.shape[0])
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        f_train = None if forces is None else forces[train_idx]
        f_val = None if forces is None else forces[val_idx]
        return xyz[train_idx], y[train_idx], f_train, xyz[val_idx], y[val_idx], f_val

    def _prepare_species(self) -> None:
        species_set = tuple(sorted(set(self.species)))
        species_index = {species: i for i, species in enumerate(species_set)}
        self.species_set_ = species_set
        self.species_ids_ = np.array([species_index[s] for s in self.species], dtype=int)
        _, jnp = _require_jax()
        self._j_species_ids = jnp.asarray(self.species_ids_, dtype=int)

    def _prepare_edges(self, jnp) -> None:
        src = []
        dst = []
        for i in range(self.n_atoms_):
            for j in range(self.n_atoms_):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self._j_src = jnp.asarray(src, dtype=int)
        self._j_dst = jnp.asarray(dst, dtype=int)

    def _set_radial_grid(self, xyz: np.ndarray) -> None:
        if self.radial_centers is not None:
            self.radial_centers_ = self.radial_centers.copy()
        else:
            distances = []
            for geom in xyz:
                d = np.linalg.norm(geom[None, :, :] - geom[:, None, :], axis=-1)
                distances.extend(d[np.triu_indices(geom.shape[0], k=1)])
            distances = np.asarray(distances)
            d_min = max(1e-8, float(np.min(distances)))
            d_max = float(np.max(distances))
            if self.cutoff is not None:
                d_max = min(d_max, float(self.cutoff))
            if d_max <= d_min:
                d_max = d_min + 1.0
            self.radial_centers_ = np.linspace(d_min, d_max, self.n_radial)
        if self.radial_width is None:
            if self.radial_centers_.size == 1:
                self.radial_width_ = 1.0
            else:
                self.radial_width_ = float(max(np.mean(np.diff(self.radial_centers_)), 1e-8))
        else:
            self.radial_width_ = float(self.radial_width)
            if self.radial_width_ <= 0.0:
                raise ValueError("radial_width must be positive")

    def _prepare_radial_jax(self, jnp) -> None:
        self._j_radial_centers = jnp.asarray(self.radial_centers_)
        self._j_radial_width = jnp.asarray(self.radial_width_)
        self._j_cutoff = None if self.cutoff is None else jnp.asarray(self.cutoff)

    def _radial_basis(self, r):
        _, jnp = _require_jax()
        basis = jnp.exp(
            -0.5 * ((r[:, None] - self._j_radial_centers) / self._j_radial_width) ** 2
        )
        if self._j_cutoff is None:
            return basis
        envelope = jnp.where(
            r[:, None] < self._j_cutoff,
            0.5 * (jnp.cos(jnp.pi * r[:, None] / self._j_cutoff) + 1.0),
            0.0,
        )
        return basis * envelope

    def _initialize_params(self, jnp, rng: np.random.Generator):
        def glorot(*shape):
            fan_in, fan_out = shape[-2], shape[-1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return jnp.asarray(rng.uniform(-limit, limit, size=shape))

        scale = 1.0 / np.sqrt(max(self.features, 1))
        params = {
            "embedding": jnp.asarray(
                rng.normal(scale=scale, size=(len(self.species_set_), self.features))
            ),
            "layers": [],
            "readout_w1": glorot(self.features, self.readout_hidden),
            "readout_b1": jnp.zeros(self.readout_hidden),
            "readout_w2": glorot(self.readout_hidden, self.n_outputs_),
            "readout_b2": jnp.zeros(self.n_outputs_),
        }
        for _ in range(self.n_layers):
            params["layers"].append(
                {
                    "w_hh": glorot(self.n_radial, self.features, self.features),
                    "w_vh": glorot(self.n_radial, self.features, self.features),
                    "w_hv": glorot(self.n_radial, self.features, self.features),
                    "w_vv": glorot(self.n_radial, self.features, self.features),
                    "b_h": jnp.zeros(self.features),
                }
            )
        return params

    def _check_is_fit(self) -> None:
        if not hasattr(self, "params_"):
            raise RuntimeError("model is not fit yet")


class H3PES:
    """Permutation-invariant H3+ PES wrapper using sorted H-H distances.

    This is the compact model that performed best on the AM1/MECI H3+ benchmark:
    the Cartesian geometry is reduced to the three sorted pair distances, then
    an :class:`MLP` predicts one or more electronic-state energies.
    """

    def __init__(
        self,
        hidden_layers: Iterable[int] = (64, 64),
        activation: str = "tanh",
        learning_rate: float = 3e-3,
        batch_size: int | None = 32,
        max_iter: int = 500,
        l2: float = 0.0,
        validation_fraction: float = 0.1,
        tol: float = 1e-10,
        patience: int = 200,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.mlp_kwargs = {
            "hidden_layers": tuple(hidden_layers),
            "activation": activation,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "l2": l2,
            "validation_fraction": validation_fraction,
            "tol": tol,
            "patience": patience,
            "random_state": random_state,
            "verbose": verbose,
        }

    def fit(self, geometries: ArrayLike, energies: ArrayLike) -> "H3PES":
        """Fit the H3 PES model from Cartesian geometries and energies."""

        xyz, _ = _NumpyEquivariantMLP._as_geometries(geometries)
        if xyz.shape[1] != 3:
            raise ValueError("H3PES expects geometries with exactly three atoms")
        self.model_ = MLP(**self.mlp_kwargs).fit(self.describe(xyz), energies)
        self.n_outputs_ = self.model_.n_outputs_
        self._scalar_output = self.model_._scalar_output
        self.result_ = self.model_.result_
        return self

    def describe(self, geometries: ArrayLike) -> np.ndarray:
        """Return sorted pair-distance descriptors."""

        xyz, leading_shape = _NumpyEquivariantMLP._as_geometries(geometries)
        if xyz.shape[1] != 3:
            raise ValueError("H3PES expects geometries with exactly three atoms")
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        descriptors = np.sort(np.stack((d01, d02, d12), axis=1), axis=1)
        return descriptors.reshape(leading_shape + (3,))

    def predict(self, geometries: ArrayLike) -> np.ndarray:
        """Predict energies for one or more H3 geometries."""

        self._check_is_fit()
        xyz, leading_shape = _NumpyEquivariantMLP._as_geometries(geometries)
        y = self.model_.predict(self.describe(xyz).reshape(-1, 3))
        if self._scalar_output:
            return np.asarray(y).reshape(leading_shape)
        return np.asarray(y).reshape(leading_shape + (self.n_outputs_,))

    energy = predict

    def gradient(self, geometries: ArrayLike, dx: float = 1e-5) -> np.ndarray:
        """Return Cartesian gradients ``dE/dR`` by central differences."""

        self._check_is_fit()
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        xyz, leading_shape = _NumpyEquivariantMLP._as_geometries(geometries)
        grad = np.empty((xyz.shape[0], self.n_outputs_, 3, 3), dtype=float)
        for atom in range(3):
            for axis in range(3):
                xp = xyz.copy()
                xm = xyz.copy()
                xp[:, atom, axis] += dx
                xm[:, atom, axis] -= dx
                ep = np.asarray(self.predict(xp))
                em = np.asarray(self.predict(xm))
                if self._scalar_output:
                    ep = ep[:, None]
                    em = em[:, None]
                grad[:, :, atom, axis] = (ep - em) / (2.0 * dx)
        if self._scalar_output:
            return grad[:, 0].reshape(leading_shape + (3, 3))
        return grad.reshape(leading_shape + (self.n_outputs_, 3, 3))

    def forces(self, geometries: ArrayLike, dx: float = 1e-5) -> np.ndarray:
        """Return Cartesian forces, ``-dE/dR``."""

        return -self.gradient(geometries, dx=dx)

    def save(self, filename: str) -> None:
        """Save the fitted H3 PES model."""

        self._check_is_fit()
        data = {"h3pes_config": json.dumps({"mlp_kwargs": self.mlp_kwargs})}
        tmp_data = {}
        # Reuse MLP serialization into an in-memory-like dict by mirroring fields.
        config = self.model_._config()
        data["mlp_config"] = json.dumps(config)
        data["x_mean"] = self.model_.x_mean_
        data["x_scale"] = self.model_.x_scale_
        data["y_mean"] = self.model_.y_mean_
        data["y_scale"] = self.model_.y_scale_
        for i, (w, b) in enumerate(zip(self.model_.weights_, self.model_.biases_)):
            tmp_data[f"W{i}"] = w
            tmp_data[f"b{i}"] = b
        data.update(tmp_data)
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename: str) -> "H3PES":
        """Load a model saved by :meth:`save`."""

        data = np.load(filename)
        h3_config = json.loads(str(data["h3pes_config"]))
        mlp_config = json.loads(str(data["mlp_config"]))
        model = cls(**h3_config["mlp_kwargs"])
        mlp = MLP(
            hidden_layers=mlp_config["hidden_layers"],
            activation=mlp_config["activation"],
            learning_rate=mlp_config["learning_rate"],
            batch_size=mlp_config["batch_size"],
            max_iter=mlp_config["max_iter"],
            l2=mlp_config["l2"],
            validation_fraction=mlp_config["validation_fraction"],
            tol=mlp_config["tol"],
            patience=mlp_config["patience"],
            random_state=mlp_config["random_state"],
            verbose=mlp_config["verbose"],
        )
        n_layers = int(mlp_config["n_layers"])
        mlp.weights_ = [data[f"W{i}"] for i in range(n_layers)]
        mlp.biases_ = [data[f"b{i}"] for i in range(n_layers)]
        mlp.x_mean_ = data["x_mean"]
        mlp.x_scale_ = data["x_scale"]
        mlp.y_mean_ = data["y_mean"]
        mlp.y_scale_ = data["y_scale"]
        mlp.n_features_in_ = int(mlp_config["n_features_in"])
        mlp.n_outputs_ = int(mlp_config["n_outputs"])
        mlp._scalar_output = bool(mlp_config["scalar_output"])
        mlp.history_ = {"train_loss": [], "validation_loss": []}
        mlp.result_ = PESFitResult(0, np.nan, None)
        model.model_ = mlp
        model.n_outputs_ = mlp.n_outputs_
        model._scalar_output = mlp._scalar_output
        model.result_ = mlp.result_
        return model

    def _check_is_fit(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("model is not fit yet")


__all__ = [
    "EquivariantMLP",
    "H3PES",
    "MLP",
    "MPNN",
    "PESFitResult",
    "fit_pes",
    "grid_to_samples",
]
