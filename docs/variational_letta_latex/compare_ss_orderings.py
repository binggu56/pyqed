#!/usr/bin/env python3
"""Compare 1D coupling ranges for Shastry-Sutherland site orderings."""

from __future__ import annotations

from collections import Counter


def snake_order(lx: int, ly: int):
    order = []
    for x in range(lx):
        ys = range(ly) if x % 2 == 0 else range(ly - 1, -1, -1)
        for y in ys:
            order.append((x, y))
    return order


def ss_dimers(lx: int, ly: int):
    dimers = []
    used = set()
    for x in range(0, lx - 1, 2):
        for y in range(ly):
            if y % 2 == 0:
                pair = ((x, y), (x + 1, (y + 1) % ly))
            else:
                pair = ((x, y), (x + 1, (y - 1) % ly))
            if pair[0] in used or pair[1] in used:
                raise ValueError("dimer pattern overlaps sites")
            used.update(pair)
            dimers.append(pair)
    return dimers


def dimer_first_order(lx: int, ly: int):
    order = []
    for pair in ss_dimers(lx, ly):
        order.extend(pair)
    return order


def square_edges(lx: int, ly: int):
    edges = []
    for x in range(lx):
        for y in range(ly):
            if x + 1 < lx:
                edges.append(((x, y), (x + 1, y)))
            edges.append(((x, y), (x, (y + 1) % ly)))
    return edges


def bond_stats(order, edges):
    index = {site: i for i, site in enumerate(order)}
    distances = [abs(index[a] - index[b]) for a, b in edges]
    counts = Counter(distances)
    mean = sum(distances) / len(distances)
    return {
        "max": max(distances),
        "mean": mean,
        "adjacent_fraction": counts[1] / len(distances),
        "hist": counts,
    }


def summarize(lx: int, ly: int):
    dimer_edges = ss_dimers(lx, ly)
    jprime_edges = square_edges(lx, ly)
    rows = []
    for name, order_fn in [
        ("snake", snake_order),
        ("dimer-first", dimer_first_order),
    ]:
        order = order_fn(lx, ly)
        dimer = bond_stats(order, dimer_edges)
        jprime = bond_stats(order, jprime_edges)
        rows.append((name, dimer, jprime))
    return rows


def main() -> None:
    print("Shastry-Sutherland cylinder ordering diagnostics")
    print("open x, periodic y; ranges are 1D index distances")
    print()
    for ly in (4, 6, 8, 10, 12, 14):
        lx = 2 * ly
        print(f"Lx={lx:2d}, Ly={ly:2d}, N={lx * ly:3d}")
        for name, dimer, jprime in summarize(lx, ly):
            print(
                f"  {name:11s} | "
                f"J max={dimer['max']:3d}, J mean={dimer['mean']:5.1f}, J adj={dimer['adjacent_fraction']:.2f} | "
                f"J' max={jprime['max']:3d}, J' mean={jprime['mean']:5.1f}, J' adj={jprime['adjacent_fraction']:.2f}"
            )
        print()


if __name__ == "__main__":
    main()
