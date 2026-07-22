"""Explicit autoregressive wavefunction for three spin-1/2 sites.

The dictionaries below are the conditional functions that a neural network
would approximate.  Keeping them explicit makes the chain rule visible.
"""

from itertools import product

import numpy as np


# Each key contains the preceding spins followed by the spin being predicted.
CONDITIONAL_PROBABILITIES = (
    {
        (-1,): 0.4,
        (+1,): 0.6,
    },
    {
        (-1, -1): 0.8,
        (-1, +1): 0.2,
        (+1, -1): 0.3,
        (+1, +1): 0.7,
    },
    {
        (-1, -1, -1): 0.2,
        (-1, -1, +1): 0.8,
        (-1, +1, -1): 0.7,
        (-1, +1, +1): 0.3,
        (+1, -1, -1): 0.6,
        (+1, -1, +1): 0.4,
        (+1, +1, -1): 0.1,
        (+1, +1, +1): 0.9,
    },
)

# Phases are conditional too.  Unlisted branches have zero phase.
CONDITIONAL_PHASES = (
    {(+1,): 0.1},
    {(+1, -1): -0.2},
    {(+1, -1, -1): 0.4},
)


def conditional_amplitude(prefix: tuple[int, ...], spin: int) -> complex:
    """Return sqrt(p) exp(i phi) for the next spin after ``prefix``."""

    site = len(prefix)
    key = prefix + (spin,)
    probability = CONDITIONAL_PROBABILITIES[site][key]
    phase = CONDITIONAL_PHASES[site].get(key, 0.0)
    return np.sqrt(probability) * np.exp(1j * phase)


def amplitude(configuration: tuple[int, int, int]) -> complex:
    """Multiply the three conditional amplitudes."""

    value = 1.0 + 0.0j
    for site, spin in enumerate(configuration):
        value *= conditional_amplitude(configuration[:site], spin)
    return value


def sample(rng: np.random.Generator) -> tuple[int, int, int]:
    """Generate one independent configuration from |psi|**2."""

    configuration: list[int] = []
    for _ in range(3):
        prefix = tuple(configuration)
        p_minus = abs(conditional_amplitude(prefix, -1)) ** 2
        p_plus = abs(conditional_amplitude(prefix, +1)) ** 2
        spin = int(rng.choice((-1, +1), p=(p_minus, p_plus)))
        configuration.append(spin)
    return tuple(configuration)


def main() -> None:
    configurations = list(product((+1, -1), repeat=3))
    probabilities = np.array([abs(amplitude(spins)) ** 2 for spins in configurations])

    print("configuration       psi(configuration)        probability")
    for spins, probability in zip(configurations, probabilities):
        print(f"{spins!s:18} {amplitude(spins):>20.12f}  {probability:.6f}")
    print(f"\ntotal probability = {np.sum(probabilities):.12f}")

    spins = (+1, -1, -1)
    factors = [conditional_amplitude(spins[:site], spin) for site, spin in enumerate(spins)]
    print("\nFor configuration (+1, -1, -1):")
    print(f"  psi_1(+1)           = {factors[0]:.12f}")
    print(f"  psi_2(-1 | +1)      = {factors[1]:.12f}")
    print(f"  psi_3(-1 | +1, -1)  = {factors[2]:.12f}")
    print(f"  product              = {np.prod(factors):.12f}")

    rng = np.random.default_rng(3)
    print("\nfive independent samples:")
    for _ in range(5):
        print(" ", sample(rng))


if __name__ == "__main__":
    main()
