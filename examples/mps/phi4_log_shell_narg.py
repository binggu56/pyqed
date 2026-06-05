"""Log-discretized momentum-shell phi4 NARG."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4LogShellNARG


def main():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=1,
        amplitude_npoints=4,
        field_range=4.5,
        mass2=0.5,
        coupling=0.8,
        quadrature_order=128,
    )
    exact = toy.exact_energies(4)
    active_labels = [toy.mode_labels[index] for index in toy.active_modes]
    environment_labels = [toy.mode_labels[index] for index in toy.environment_modes]

    print("k-log-discretized phi4 momentum-shell NARG")
    print(f"Lambda                 : {toy.cutoff:.6f}")
    print(f"log factor             : {toy.log_factor:.6f}")
    print(f"IR cutoff              : {toy.ir_cutoff:.6f}")
    print(f"shell edges            : {' '.join(f'{value:.6f}' for value in toy.shell_edges)}")
    print(f"representative k       : {' '.join(f'{value:.6f}' for value in toy.shell_representatives)}")
    print(f"shell weights          : {' '.join(f'{value:.6f}' for value in toy.shell_weights)}")
    print(f"mode labels            : {toy.mode_labels}")
    print(f"active modes           : {active_labels}")
    print(f"environment modes      : {environment_labels}")
    print(f"full Hilbert dimension : {toy.amplitude_npoints ** toy.nmodes}")
    print(f"exact E0               : {exact[0]: .12f}")
    print()
    print("Branch convergence after keeping the lowest-k shell active")
    print("branches  dim(Heff)        E0              E0-exact")
    for branches in (1, 2, 4, toy.environment_configs.shape[0]):
        result = toy.narg_effective_hamiltonian(branches)
        error = result.effective_energies[0] - exact[0]
        print(
            f"{branches:8d}  {result.hamiltonian.shape[0]:9d}  "
            f"{result.effective_energies[0]: .12f}  {error: .3e}"
        )

    print()
    print("Slow-sector cutoff flow (lowest-k shell pairs kept active)")
    print("active shells  branches  dim(Heff)        E0              E0-exact")
    for row in toy.shell_flow_summary(nbranches=4):
        print(
            f"{row['active_shells']:13d}  {row['branches']:8d}  {row['dimension']:9d}  "
            f"{row['energy']: .12f}  {row['error']: .3e}"
        )

    larger = Phi4LogShellNARG(
        cutoff=8.0,
        log_factor=2.0,
        nshells=4,
        active_shells=0,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.5,
        quadrature_order=160,
    )
    print()
    print("Iterative shell NARG with larger cutoff stack")
    print(f"shell edges            : {' '.join(f'{value:.6f}' for value in larger.shell_edges)}")
    print(f"full formal dimension  : {larger.amplitude_npoints ** larger.nmodes}")
    print("kept D   final E0        largest projected dim")
    for kept_dim in (8, 16, 32):
        result = larger.iterative_shell_narg(kept_dim=kept_dim, max_exact_dim=1000)
        largest_projected = max(row["projected_dim"] for row in result.records)
        print(f"{kept_dim:6d}  {result.energies[0]: .12f}  {largest_projected:21d}")

    result = larger.iterative_shell_narg(kept_dim=16, max_exact_dim=1000)
    print()
    print("UV-to-IR shell records for D=16")
    print("step  shell  modes added                         kept  projected dim   E0          discarded gap")
    for row in result.records:
        labels = ",".join(f"{kind}{index}" for kind, index in row["mode_labels"])
        gap = row["discarded_gap"]
        gap_text = "inf" if gap == float("inf") else f"{gap:.3e}"
        print(
            f"{row['step']:4d}  {row['shell']:5d}  {labels:32s}  "
            f"{row['kept_dim']:4d}  {row['projected_dim']:13d}  {row['energy']: .8f}  {gap_text:>13s}"
        )

    print()
    print("Low-mode moments from the final iterative ground state")
    moments2 = larger.iterative_mode_moments(result, power=2)
    for label, value in moments2.items():
        print(f"{label!s:16s}  <q^2> = {value:.8f}")

    fit = toy.fit_ir_shell_effective_potential()
    coeff = fit["coefficients"]
    print()
    print("IR-shell effective potential fit after integrating UV shells")
    print(f"c0         = {coeff['c0']: .8f}")
    print(f"omega2_eff = {coeff['omega2_eff']: .8f}")
    print(f"lambda_eff = {coeff['lambda_eff']: .8f}")
    print(f"c6         = {coeff['c6']: .8e}")
    print(f"rms error  = {fit['rms_error']:.3e}")

    print()
    print("Log-factor scan at fixed shell count")
    print("b       E0(D=16)       IR cutoff")
    for row in toy.log_factor_scan([1.5, 2.0, 2.5], kept_dim=16):
        print(f"{row['log_factor']:<7.3f} {row['energy']: .12f}  {row['ir_cutoff']:.6f}")

    print()
    print("UV cutoff scan at fixed log factor")
    print("Lambda  E0(D=16)       IR cutoff")
    for row in toy.cutoff_scan([3.0, 4.0, 5.0], kept_dim=16):
        print(f"{row['cutoff']:<7.3f} {row['energy']: .12f}  {row['ir_cutoff']:.6f}")


if __name__ == "__main__":
    main()
