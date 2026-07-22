"""Scalable autoregressive VMC for a Heisenberg spin chain.

This example never constructs a dense Hamiltonian or a ``2**N`` state vector.
The RNN draws independent samples and evaluates only the ``O(N)`` states
connected to each sample by the nearest-neighbor Hamiltonian.
"""

from functools import partial

from pyqed.ml import ARNN, heisenberg_connections


def main() -> None:
    n_sites = 20
    n_steps = 1000
    state = ARNN(
        n_sites,
        hidden_size=64,
        seed=8,
        init_scale=0.2,
    )
    connectivity = partial(heisenberg_connections, periodic=False)

    for step in range(1, n_steps + 1):
        learning_rate = 3.0e-3 if step <= 600 else 1.0e-3
        state.train_step(
            connectivity,
            n_samples=2048,
            learning_rate=learning_rate,
        )
        if step == 1 or step % 50 == 0:
            print(
                f"step {step:4d} | E {state.energy.real: .10f} "
                f"| E/site {state.energy.real / n_sites: .10f} "
                f"| variance {state.energy_variance:.3e}"
            )

    print("five independent autoregressive samples:")
    for spins in state.sample(5):
        print(" ", tuple(int(spin) for spin in spins))


if __name__ == "__main__":
    main()
