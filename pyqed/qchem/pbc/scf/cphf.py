"""Static coupled-perturbed Hartree-Fock for periodic KRHF references."""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def _kpoint_vectors(values, name):
    if isinstance(values, (list, tuple)):
        out = [np.asarray(value) for value in values]
    else:
        array = np.asarray(values)
        if array.ndim == 1:
            out = [array]
        elif array.ndim == 2:
            out = [array[index] for index in range(len(array))]
        else:
            raise ValueError(f"{name} must be a vector or one vector per k-point.")
    if not out or any(value.ndim != 1 for value in out):
        raise ValueError(f"{name} must contain one-dimensional arrays.")
    return out


def _kpoint_response_blocks(values, nkpts, name):
    if isinstance(values, (list, tuple)):
        blocks = [np.asarray(value, dtype=np.complex128) for value in values]
    else:
        array = np.asarray(values, dtype=np.complex128)
        if nkpts == 1 and array.ndim in (2, 3):
            blocks = [array]
        elif array.ndim in (3, 4) and len(array) == nkpts:
            blocks = [array[index] for index in range(nkpts)]
        else:
            raise ValueError(
                f"{name} must provide one response block per k-point."
            )
    if len(blocks) != nkpts:
        raise ValueError(f"{name} must provide {nkpts} k-point blocks.")
    normalized = []
    for block in blocks:
        if block.ndim == 2:
            block = block[None, :, :]
        if block.ndim != 3:
            raise ValueError(
                f"Each {name} block must have shape (nmo, nocc) or "
                "(ncomp, nmo, nocc)."
            )
        normalized.append(np.ascontiguousarray(block))
    ncomp = normalized[0].shape[0]
    if any(block.shape[0] != ncomp for block in normalized):
        raise ValueError(f"All {name} blocks must have the same component count.")
    return normalized


class CPHF:
    """Matrix-free static periodic CPHF solver in the canonical MO basis.

    ``fvind`` receives one ``(ncomp, nmo, nocc)`` first-order orbital block
    per k-point and returns induced Fock blocks with the same layout.
    """

    def __init__(
        self,
        fvind,
        mo_energy,
        mo_occ,
        h1,
        s1=None,
        *,
        mo_energy_left=None,
        mo_occ_left=None,
        max_cycle=50,
        tol=1.0e-9,
        level_shift=0.0,
    ):
        self.fvind = fvind
        self.mo_energy = _kpoint_vectors(mo_energy, "mo_energy")
        self.mo_occ = _kpoint_vectors(mo_occ, "mo_occ")
        self.nkpts = len(self.mo_energy)
        if len(self.mo_occ) != self.nkpts:
            raise ValueError("mo_energy and mo_occ have different k-point counts.")
        self.mo_energy_left = (
            self.mo_energy
            if mo_energy_left is None
            else _kpoint_vectors(mo_energy_left, "mo_energy_left")
        )
        self.mo_occ_left = (
            self.mo_occ
            if mo_occ_left is None
            else _kpoint_vectors(mo_occ_left, "mo_occ_left")
        )
        if (
            len(self.mo_energy_left) != self.nkpts
            or len(self.mo_occ_left) != self.nkpts
        ):
            raise ValueError(
                "Left and right references must have the same k-point count."
            )
        self.h1 = _kpoint_response_blocks(h1, self.nkpts, "h1")
        self.s1 = (
            [np.zeros_like(block) for block in self.h1]
            if s1 is None
            else _kpoint_response_blocks(s1, self.nkpts, "s1")
        )
        self.max_cycle = int(max_cycle)
        self.tol = float(tol)
        self.level_shift = float(level_shift)
        if self.max_cycle <= 0:
            raise ValueError("max_cycle must be positive.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be a positive finite number.")
        if not np.isfinite(self.level_shift) or self.level_shift < 0.0:
            raise ValueError("level_shift must be a non-negative finite number.")

        self.mo1 = None
        self.mo_e1 = None
        self.converged = False
        self.success = False
        self.message = "not run"
        self.niter = 0
        self.gmres_info = None
        self.residual_norm = None
        self.seconds = None

    def _layout(self):
        ncomp = self.h1[0].shape[0]
        layouts = []
        offset = 0
        for k_index, (
            energy,
            occupation,
            left_energy,
            left_occupation,
            h1,
            s1,
        ) in enumerate(
            zip(
                self.mo_energy,
                self.mo_occ,
                self.mo_energy_left,
                self.mo_occ_left,
                self.h1,
                self.s1,
            )
        ):
            if energy.shape != occupation.shape:
                raise ValueError(
                    f"mo_energy and mo_occ shapes differ at k-point {k_index}."
                )
            if left_energy.shape != left_occupation.shape:
                raise ValueError(
                    f"Left mo_energy/mo_occ shapes differ at k-point {k_index}."
                )
            occupied = occupation > 1.0e-10
            right_virtual = occupation < 1.0e-10
            left_occupied = left_occupation > 1.0e-10
            virtual = left_occupation < 1.0e-10
            if np.any(~(occupied | right_virtual)) or np.any(
                np.abs(occupation[occupied] - 2.0) > 1.0e-10
            ) or np.any(~(left_occupied | virtual)) or np.any(
                np.abs(left_occupation[left_occupied] - 2.0) > 1.0e-10
            ):
                raise NotImplementedError(
                    "Static periodic CPHF currently requires integer closed-shell "
                    "occupations; metallic occupation response is not implemented."
                )
            nocc = int(np.count_nonzero(occupied))
            nvir = int(np.count_nonzero(virtual))
            nmo = len(left_energy)
            expected = (ncomp, nmo, nocc)
            if h1.shape != expected or s1.shape != expected:
                raise ValueError(
                    f"h1/s1 block {k_index} must have shape {expected}."
                )
            e_occ = np.asarray(energy[occupied], dtype=float)
            left_e_occ = np.asarray(left_energy[left_occupied], dtype=float)
            e_vir = np.asarray(left_energy[virtual], dtype=float)
            denominator = e_vir[:, None] - e_occ[None, :] + self.level_shift
            if denominator.size and np.min(np.abs(denominator)) < 1.0e-12:
                raise np.linalg.LinAlgError(
                    "The occupied-virtual CPHF denominator is singular."
                )
            size = ncomp * nvir * nocc
            layouts.append(
                {
                    "occupied": occupied,
                    "left_occupied": left_occupied,
                    "virtual": virtual,
                    "e_occ": e_occ,
                    "left_e_occ": left_e_occ,
                    "denominator": denominator,
                    "shape": (ncomp, nvir, nocc),
                    "slice": slice(offset, offset + size),
                    "nmo": nmo,
                    "nocc": nocc,
                }
            )
            offset += size
        return layouts, offset

    @staticmethod
    def _pack(blocks, layouts):
        if not blocks:
            return np.zeros(0, dtype=np.complex128)
        return np.concatenate(
            [np.asarray(block).reshape(-1) for block in blocks]
        ).astype(np.complex128, copy=False)

    @staticmethod
    def _unpack(vector, layouts):
        return [
            np.asarray(vector[layout["slice"]]).reshape(layout["shape"])
            for layout in layouts
        ]

    @staticmethod
    def _full_blocks(vo_blocks, fixed_blocks, layouts):
        full = [np.array(block, copy=True) for block in fixed_blocks]
        for block, target, layout in zip(vo_blocks, full, layouts):
            target[:, layout["virtual"], :] = block
        return full

    def kernel(self):
        started = time.perf_counter()
        layouts, size = self._layout()
        ncomp = self.h1[0].shape[0]
        fixed = []
        for s1, layout in zip(self.s1, layouts):
            block = np.zeros(
                (ncomp, layout["nmo"], layout["nocc"]),
                dtype=np.complex128,
            )
            block[:, layout["left_occupied"], :] = (
                -0.5 * s1[:, layout["left_occupied"], :]
            )
            fixed.append(block)

        if size == 0:
            self.mo1 = fixed
            self.mo_e1 = [
                np.zeros((ncomp, layout["nocc"], layout["nocc"]), dtype=np.complex128)
                for layout in layouts
            ]
            self.converged = self.success = True
            self.message = "no occupied-virtual response amplitudes"
            self.niter = 0
            self.residual_norm = 0.0
            self.seconds = float(time.perf_counter() - started)
            return self.mo1, self.mo_e1

        fixed_induced = (
            [np.zeros_like(block) for block in fixed]
            if all(not np.any(block) for block in fixed)
            else _kpoint_response_blocks(
                self.fvind(fixed), self.nkpts, "fvind output"
            )
        )
        rhs_blocks = []
        for h1, s1, induced, layout in zip(
            self.h1, self.s1, fixed_induced, layouts
        ):
            driving = (
                h1[:, layout["virtual"], :]
                - s1[:, layout["virtual"], :]
                * layout["e_occ"][None, None, :]
                + induced[:, layout["virtual"], :]
            )
            rhs_blocks.append(-driving / layout["denominator"][None, :, :])
        rhs_complex = self._pack(rhs_blocks, layouts)

        def matvec_complex(vector):
            vo = self._unpack(vector, layouts)
            variable = self._full_blocks(
                vo,
                [np.zeros_like(block) for block in fixed],
                layouts,
            )
            induced = _kpoint_response_blocks(
                self.fvind(variable), self.nkpts, "fvind output"
            )
            out = []
            for amplitudes, response, layout in zip(vo, induced, layouts):
                out.append(
                    amplitudes
                    + response[:, layout["virtual"], :]
                    / layout["denominator"][None, :, :]
                )
            return self._pack(out, layouts)

        def to_real(vector):
            vector = np.asarray(vector, dtype=np.complex128)
            return np.concatenate((vector.real, vector.imag))

        def from_real(vector):
            vector = np.asarray(vector, dtype=float)
            return vector[:size] + 1.0j * vector[size:]

        def matvec(vector):
            return to_real(matvec_complex(from_real(vector)))

        iterations = []
        operator = LinearOperator(
            (2 * size, 2 * size),
            matvec=matvec,
            dtype=float,
        )
        rhs = to_real(rhs_complex)
        solution_real, info = gmres(
            operator,
            rhs,
            rtol=self.tol,
            atol=0.0,
            restart=min(40, max(1, size)),
            maxiter=self.max_cycle,
            callback=iterations.append,
            callback_type="pr_norm",
        )
        residual = np.linalg.norm(matvec(solution_real) - rhs) / max(
            np.linalg.norm(rhs), 1.0
        )
        solution = from_real(solution_real)
        vo_solution = self._unpack(solution, layouts)
        self.mo1 = self._full_blocks(vo_solution, fixed, layouts)
        induced = _kpoint_response_blocks(
            self.fvind(self.mo1), self.nkpts, "fvind output"
        )
        self.mo_e1 = []
        for h1, s1, mo1, v1, layout in zip(
            self.h1, self.s1, self.mo1, induced, layouts
        ):
            occupied = layout["left_occupied"]
            e_occ = layout["e_occ"]
            e1 = (
                h1[:, occupied, :]
                - s1[:, occupied, :] * e_occ[None, None, :]
                + v1[:, occupied, :]
            )
            e1 += mo1[:, occupied, :] * (
                layout["left_e_occ"][:, None] - e_occ[None, :]
            )[None, :, :]
            self.mo_e1.append(e1)

        self.niter = int(len(iterations))
        self.gmres_info = int(info)
        self.residual_norm = float(residual)
        self.converged = self.success = bool(
            residual <= max(10.0 * self.tol, 1.0e-12)
        )
        self.message = (
            "converged"
            if self.converged
            else f"GMRES did not converge (info={int(info)}, residual={residual:.3e})"
        )
        self.seconds = float(time.perf_counter() - started)
        if not self.converged:
            raise RuntimeError(self.message)
        return self.mo1, self.mo_e1

    run = kernel


def solve(fvind, mo_energy, mo_occ, h1, s1=None, **kwargs):
    """Solve static periodic CPHF and return ``(mo1, mo_e1)`` k-point lists."""

    return CPHF(
        fvind,
        mo_energy,
        mo_occ,
        h1,
        s1=s1,
        **kwargs,
    ).kernel()


def _as_kpoint_reference(values, nkpts, name):
    if nkpts == 1 and np.asarray(values).ndim == 1:
        return [np.asarray(values)]
    if nkpts == 1 and np.asarray(values).ndim == 2 and name == "mo_coeff":
        return [np.asarray(values)]
    if isinstance(values, (list, tuple)):
        out = [np.asarray(value) for value in values]
    else:
        array = np.asarray(values)
        if len(array) != nkpts:
            raise ValueError(f"{name} must provide one array per k-point.")
        out = [array[index] for index in range(nkpts)]
    if len(out) != nkpts:
        raise ValueError(f"{name} must provide one array per k-point.")
    return out


def _ao_response_blocks(values, nkpts, nao, name):
    if isinstance(values, (list, tuple)):
        blocks = [np.asarray(value, dtype=np.complex128) for value in values]
    else:
        array = np.asarray(values, dtype=np.complex128)
        if nkpts == 1 and array.ndim in (2, 3):
            blocks = [array]
        elif array.ndim == 3 and array.shape == (nkpts, nao, nao):
            blocks = [array[index] for index in range(nkpts)]
        elif array.ndim == 4 and array.shape[0] == nkpts:
            blocks = [array[index] for index in range(nkpts)]
        else:
            raise ValueError(
                f"{name} must have shape (nao, nao), (ncomp, nao, nao), "
                "or provide one such block per k-point."
            )
    if len(blocks) != nkpts:
        raise ValueError(f"{name} must provide {nkpts} k-point blocks.")
    normalized = []
    for block in blocks:
        if block.ndim == 2:
            block = block[None, :, :]
        if block.ndim != 3 or block.shape[1:] != (nao, nao):
            raise ValueError(
                f"Each {name} block must have shape (ncomp, {nao}, {nao})."
            )
        normalized.append(np.ascontiguousarray(block))
    ncomp = normalized[0].shape[0]
    if any(block.shape[0] != ncomp for block in normalized):
        raise ValueError(f"All {name} blocks must have the same component count.")
    return normalized


class KRHFResponse:
    """Static periodic CPHF response attached to a converged KRHF object."""

    def __init__(self, mean_field):
        self.base = mean_field
        self.mo1 = None
        self.mo_coeff1 = None
        self.mo_e1 = None
        self.dm1 = None
        self.cphf_solver = None
        self.converged = False
        self.success = False
        self.message = "not run"
        self.niter = 0
        self.gmres_info = None
        self.residual_norm = None
        self.seconds = None
        self.qpoint = None
        self.q_index = None
        self.kq_indices = None
        self.minus_q_index = None
        self.mo1_minus_q = None
        self.mo_coeff1_minus_q = None
        self.mo_e1_minus_q = None

    def _reference(self):
        mf = self.base
        if not getattr(mf, "converged", False):
            raise RuntimeError("Run and converge KRHF before requesting CPHF response.")
        nkpts = int(mf.nkpts)
        mo_energy = _as_kpoint_reference(mf.mo_energy, nkpts, "mo_energy")
        mo_coeff = _as_kpoint_reference(mf.mo_coeff, nkpts, "mo_coeff")
        mo_occ = _as_kpoint_reference(mf.mo_occ, nkpts, "mo_occ")
        for occupation in mo_occ:
            occupied = occupation > 1.0e-10
            virtual = occupation < 1.0e-10
            if np.any(~(occupied | virtual)) or np.any(
                np.abs(occupation[occupied] - 2.0) > 1.0e-10
            ):
                raise NotImplementedError(
                    "Periodic CPHF currently requires integer closed-shell occupations."
                )
        return mo_energy, mo_coeff, mo_occ

    def _kernel_general_q(
        self,
        h1,
        s1,
        qpoint,
        q_index,
        kq_indices,
        mo_energy,
        mo_coeff,
        mo_occ,
        *,
        max_cycle,
        tol,
        level_shift,
    ):
        mf = self.base
        nkpts = int(mf.nkpts)
        nao = int(mf.cell.nao)
        h1_q = _ao_response_blocks(h1, nkpts, nao, "h1")
        s1_q = (
            [np.zeros_like(block) for block in h1_q]
            if s1 is None
            else _ao_response_blocks(s1, nkpts, nao, "s1")
        )

        minus_q_index = int(mf.with_df.find_qpoint_index(-np.asarray(qpoint)))
        minus_pair_by_k = {
            int(k): int(kmq)
            for k, kmq in mf.with_df.pair_keys(minus_q_index)
        }
        if len(minus_pair_by_k) != nkpts:
            raise RuntimeError("The -q block does not map every SCF k point.")
        kmq_indices = [minus_pair_by_k[k_index] for k_index in range(nkpts)]
        if any(
            kmq_indices[kq_indices[k_index]] != k_index
            for k_index in range(nkpts)
        ):
            raise RuntimeError("The +q and -q k-point maps are not inverses.")

        h1_minus = [np.zeros_like(block) for block in h1_q]
        s1_minus = [np.zeros_like(block) for block in s1_q]
        for k_index, kq_index in enumerate(kq_indices):
            h1_minus[kq_index] = h1_q[k_index].conj().transpose(0, 2, 1)
            s1_minus[kq_index] = s1_q[k_index].conj().transpose(0, 2, 1)

        right_energy = list(mo_energy) + list(mo_energy)
        right_occ = list(mo_occ) + list(mo_occ)
        left_indices = list(kq_indices) + list(kmq_indices)
        left_energy = [mo_energy[index] for index in left_indices]
        left_occ = [mo_occ[index] for index in left_indices]
        left_coeff = [mo_coeff[index] for index in left_indices]
        right_coeff = list(mo_coeff) + list(mo_coeff)
        occupied_masks = [occupation > 1.0e-10 for occupation in right_occ]
        h1_ao = list(h1_q) + list(h1_minus)
        s1_ao = list(s1_q) + list(s1_minus)
        h1_mo = []
        s1_mo = []
        for coefficients_left, coefficients, occupied, h1_block, s1_block in zip(
            left_coeff,
            right_coeff,
            occupied_masks,
            h1_ao,
            s1_ao,
        ):
            cocc = coefficients[:, occupied]
            h1_mo.append(
                np.einsum(
                    "pa,xpq,qi->xai",
                    coefficients_left.conj(),
                    h1_block,
                    cocc,
                    optimize=True,
                )
            )
            s1_mo.append(
                np.einsum(
                    "pa,xpq,qi->xai",
                    coefficients_left.conj(),
                    s1_block,
                    cocc,
                    optimize=True,
                )
            )

        def forward_density(blocks, channel_left_coeff, channel_offset):
            out = []
            for k_index in range(nkpts):
                occupation = mo_occ[k_index]
                occupied = occupation > 1.0e-10
                cocc = mo_coeff[k_index][:, occupied]
                c1occ = (
                    channel_left_coeff[k_index]
                    @ blocks[channel_offset + k_index]
                )
                weights = occupation[occupied]
                out.append((c1occ * weights[None, :]) @ cocc.conj().T)
            return out

        def add_madelung(veff, densities, target_indices):
            if mf.madelung is None:
                return veff
            for k_index, target in enumerate(target_indices):
                veff[k_index] -= 0.5 * mf.madelung * (
                    mf._overlap_k[target]
                    @ densities[k_index]
                    @ mf._overlap_k[k_index]
                )
            return veff

        def fvind(mo1_blocks):
            ncomp = mo1_blocks[0].shape[0]
            induced = [np.zeros_like(block) for block in mo1_blocks]
            q_left_coeff = [mo_coeff[index] for index in kq_indices]
            minus_left_coeff = [mo_coeff[index] for index in kmq_indices]
            for component in range(ncomp):
                component_blocks = [block[component] for block in mo1_blocks]
                forward_q = forward_density(component_blocks, q_left_coeff, 0)
                forward_minus = forward_density(
                    component_blocks,
                    minus_left_coeff,
                    nkpts,
                )
                density_q = [
                    forward_q[k_index]
                    + forward_minus[kq_indices[k_index]].conj().T
                    for k_index in range(nkpts)
                ]
                density_minus = [
                    forward_minus[k_index]
                    + forward_q[kmq_indices[k_index]].conj().T
                    for k_index in range(nkpts)
                ]
                vj_q, vk_q = mf.with_df.get_jk_response(density_q, q_index)
                vj_minus, vk_minus = mf.with_df.get_jk_response(
                    density_minus,
                    minus_q_index,
                )
                veff_q = add_madelung(
                    [vj_q[k] - 0.5 * vk_q[k] for k in range(nkpts)],
                    density_q,
                    kq_indices,
                )
                veff_minus = add_madelung(
                    [vj_minus[k] - 0.5 * vk_minus[k] for k in range(nkpts)],
                    density_minus,
                    kmq_indices,
                )
                for targets, potentials, offset in (
                    (kq_indices, veff_q, 0),
                    (kmq_indices, veff_minus, nkpts),
                ):
                    for k_index, target in enumerate(targets):
                        occupied = mo_occ[k_index] > 1.0e-10
                        cocc = mo_coeff[k_index][:, occupied]
                        induced[offset + k_index][component] = np.einsum(
                            "pa,pq,qi->ai",
                            mo_coeff[target].conj(),
                            potentials[k_index],
                            cocc,
                            optimize=True,
                        )
            return induced

        solver = CPHF(
            fvind,
            right_energy,
            right_occ,
            h1_mo,
            s1=s1_mo,
            mo_energy_left=left_energy,
            mo_occ_left=left_occ,
            max_cycle=max_cycle,
            tol=tol,
            level_shift=level_shift,
        )
        mo1_all, mo_e1_all = solver.kernel()
        mo1_q = mo1_all[:nkpts]
        mo1_minus = mo1_all[nkpts:]
        coeff1_q = []
        coeff1_minus = []
        forward_q = []
        forward_minus = []
        for k_index in range(nkpts):
            occupied = mo_occ[k_index] > 1.0e-10
            cocc = mo_coeff[k_index][:, occupied]
            weights = mo_occ[k_index][occupied]
            cq1 = np.einsum(
                "pa,xai->xpi",
                mo_coeff[kq_indices[k_index]],
                mo1_q[k_index],
                optimize=True,
            )
            cm1 = np.einsum(
                "pa,xai->xpi",
                mo_coeff[kmq_indices[k_index]],
                mo1_minus[k_index],
                optimize=True,
            )
            coeff1_q.append(cq1)
            coeff1_minus.append(cm1)
            forward_q.append(
                np.einsum(
                    "xpi,i,qi->xpq",
                    cq1,
                    weights,
                    cocc.conj(),
                    optimize=True,
                )
            )
            forward_minus.append(
                np.einsum(
                    "xpi,i,qi->xpq",
                    cm1,
                    weights,
                    cocc.conj(),
                    optimize=True,
                )
            )
        density_q = [
            forward_q[k_index]
            + forward_minus[kq_indices[k_index]].conj().transpose(0, 2, 1)
            for k_index in range(nkpts)
        ]

        self.cphf_solver = solver
        self.mo1 = mo1_q
        self.mo1_minus_q = mo1_minus
        self.mo_coeff1 = coeff1_q
        self.mo_coeff1_minus_q = coeff1_minus
        self.mo_e1 = mo_e1_all[:nkpts]
        self.mo_e1_minus_q = mo_e1_all[nkpts:]
        self.dm1 = density_q
        self.converged = self.success = solver.converged
        self.message = solver.message
        self.niter = solver.niter
        self.gmres_info = solver.gmres_info
        self.residual_norm = solver.residual_norm
        self.seconds = solver.seconds
        self.qpoint = np.array(qpoint, copy=True)
        self.q_index = int(q_index)
        self.minus_q_index = int(minus_q_index)
        self.kq_indices = tuple(int(index) for index in kq_indices)
        return self

    def kernel(
        self,
        h1,
        s1=None,
        *,
        qpoint=None,
        max_cycle=50,
        tol=1.0e-9,
        level_shift=0.0,
    ):
        mf = self.base
        qpoint = np.zeros(3) if qpoint is None else np.asarray(qpoint, dtype=float)
        if qpoint.shape != (3,):
            raise ValueError("qpoint must contain three Cartesian components.")
        mo_energy, mo_coeff, mo_occ = self._reference()
        nkpts = int(mf.nkpts)
        nao = int(mf.cell.nao)
        q_is_zero = np.linalg.norm(qpoint) <= 1.0e-12
        if q_is_zero:
            q_index = None
            kq_indices = list(range(nkpts))
        else:
            if str(mf.jk_builder) != "gdf" or mf.with_df is None:
                raise NotImplementedError(
                    "Nonzero-q periodic CPHF currently requires jk_builder='gdf'."
                )
            try:
                q_index = int(mf.with_df.find_qpoint_index(qpoint))
            except ValueError as exc:
                raise ValueError(
                    "qpoint must belong to the SCF k-point difference mesh."
                ) from exc
            pair_keys = mf.with_df.pair_keys(q_index)
            pair_by_k = {int(k): int(kq) for k, kq in pair_keys}
            if len(pair_by_k) != nkpts:
                raise RuntimeError("The selected q point does not map every k point.")
            kq_indices = [pair_by_k[k_index] for k_index in range(nkpts)]
            if any(
                kq_indices[kq_indices[k_index]] != k_index
                for k_index in range(nkpts)
            ):
                return self._kernel_general_q(
                    h1,
                    s1,
                    qpoint,
                    q_index,
                    kq_indices,
                    mo_energy,
                    mo_coeff,
                    mo_occ,
                    max_cycle=max_cycle,
                    tol=tol,
                    level_shift=level_shift,
                )
        left_mo_energy = [mo_energy[index] for index in kq_indices]
        left_mo_coeff = [mo_coeff[index] for index in kq_indices]
        left_mo_occ = [mo_occ[index] for index in kq_indices]
        h1_ao = _ao_response_blocks(h1, nkpts, nao, "h1")
        s1_ao = (
            [np.zeros_like(block) for block in h1_ao]
            if s1 is None
            else _ao_response_blocks(s1, nkpts, nao, "s1")
        )
        if q_is_zero:
            for name, blocks in (("h1", h1_ao), ("s1", s1_ao)):
                for block in blocks:
                    residual = np.max(
                        np.abs(block - block.conj().transpose(0, 2, 1))
                    )
                    scale = max(float(np.max(np.abs(block))), 1.0)
                    if residual > 1.0e-10 * scale:
                        raise ValueError(
                            f"Static q=0 {name} perturbations must be Hermitian."
                        )
        occupied_masks = [occupation > 1.0e-10 for occupation in mo_occ]
        h1_mo = []
        s1_mo = []
        for left_coefficients, coefficients, occupied, h1_block, s1_block in zip(
            left_mo_coeff, mo_coeff, occupied_masks, h1_ao, s1_ao
        ):
            cocc = coefficients[:, occupied]
            h1_mo.append(
                np.einsum(
                    "pa,xpq,qi->xai",
                    left_coefficients.conj(),
                    h1_block,
                    cocc,
                    optimize=True,
                )
            )
            s1_mo.append(
                np.einsum(
                    "pa,xpq,qi->xai",
                    left_coefficients.conj(),
                    s1_block,
                    cocc,
                    optimize=True,
                )
            )

        def fvind(mo1_blocks):
            ncomp = mo1_blocks[0].shape[0]
            induced_mo = [
                np.zeros_like(block, dtype=np.complex128)
                for block in mo1_blocks
            ]
            for component in range(ncomp):
                forward_dm1 = []
                for left_coefficients, coefficients, occupation, occupied, mo1_block in zip(
                    left_mo_coeff,
                    mo_coeff,
                    mo_occ,
                    occupied_masks,
                    mo1_blocks,
                ):
                    cocc = coefficients[:, occupied]
                    c1occ = left_coefficients @ mo1_block[component]
                    weights = occupation[occupied]
                    dm1 = (c1occ * weights[None, :]) @ cocc.conj().T
                    if q_is_zero:
                        dm1 += (
                            (cocc * weights[None, :]) @ c1occ.conj().T
                        )
                        dm1 = 0.5 * (dm1 + dm1.conj().T)
                    forward_dm1.append(dm1)
                if q_is_zero:
                    dm1_k = forward_dm1
                else:
                    dm1_k = [
                        forward_dm1[k_index]
                        + forward_dm1[kq_indices[k_index]].conj().T
                        for k_index in range(nkpts)
                    ]
                if q_is_zero:
                    fock1_k = mf._build_fock_k(dm1_k)
                    veff1_k = [
                        fock1 - hcore
                        for fock1, hcore in zip(fock1_k, mf._hcore_k)
                    ]
                else:
                    vj1, vk1 = mf.with_df.get_jk_response(dm1_k, q_index)
                    veff1_k = [
                        np.asarray(vj1[k]) - 0.5 * np.asarray(vk1[k])
                        for k in range(nkpts)
                    ]
                    if mf.madelung is not None:
                        for k_index, kq_index in enumerate(kq_indices):
                            veff1_k[k_index] -= 0.5 * mf.madelung * (
                                mf._overlap_k[kq_index]
                                @ dm1_k[k_index]
                                @ mf._overlap_k[k_index]
                            )
                for k_index, (
                    left_coefficients,
                    coefficients,
                    occupied,
                ) in enumerate(
                    zip(left_mo_coeff, mo_coeff, occupied_masks)
                ):
                    cocc = coefficients[:, occupied]
                    induced_mo[k_index][component] = np.einsum(
                        "pa,pq,qi->ai",
                        left_coefficients.conj(),
                        veff1_k[k_index],
                        cocc,
                        optimize=True,
                    )
            return induced_mo

        solver = CPHF(
            fvind,
            mo_energy,
            mo_occ,
            h1_mo,
            s1=s1_mo,
            mo_energy_left=left_mo_energy,
            mo_occ_left=left_mo_occ,
            max_cycle=max_cycle,
            tol=tol,
            level_shift=level_shift,
        )
        mo1, mo_e1 = solver.kernel()
        mo_coeff1 = []
        forward_dm1 = []
        for left_coefficients, coefficients, occupation, occupied, mo1_block in zip(
            left_mo_coeff, mo_coeff, mo_occ, occupied_masks, mo1
        ):
            cocc = coefficients[:, occupied]
            coeff1 = np.einsum(
                "pa,xai->xpi", left_coefficients, mo1_block, optimize=True
            )
            weights = occupation[occupied]
            density1 = np.empty(
                (len(coeff1), nao, nao), dtype=np.complex128
            )
            for component, c1occ in enumerate(coeff1):
                value = (c1occ * weights[None, :]) @ cocc.conj().T
                if q_is_zero:
                    value += (cocc * weights[None, :]) @ c1occ.conj().T
                    value = 0.5 * (value + value.conj().T)
                density1[component] = value
            mo_coeff1.append(coeff1)
            forward_dm1.append(density1)
        if q_is_zero:
            dm1_k = forward_dm1
        else:
            dm1_k = [
                forward_dm1[k_index]
                + forward_dm1[kq_indices[k_index]].conj().transpose(0, 2, 1)
                for k_index in range(nkpts)
            ]

        self.cphf_solver = solver
        self.mo1 = mo1[0] if nkpts == 1 else mo1
        self.mo_coeff1 = mo_coeff1[0] if nkpts == 1 else mo_coeff1
        self.mo_e1 = mo_e1[0] if nkpts == 1 else mo_e1
        self.dm1 = dm1_k[0] if nkpts == 1 else dm1_k
        self.converged = self.success = solver.converged
        self.message = solver.message
        self.niter = solver.niter
        self.gmres_info = solver.gmres_info
        self.residual_norm = solver.residual_norm
        self.seconds = solver.seconds
        self.qpoint = np.array(qpoint, copy=True)
        self.q_index = q_index
        self.kq_indices = tuple(int(index) for index in kq_indices)
        return self

    run = kernel


__all__ = ["CPHF", "KRHFResponse", "solve"]
