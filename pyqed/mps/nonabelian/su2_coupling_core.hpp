#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pyqed::su2 {

inline double sqrt_ratio(double numerator, double denominator) noexcept {
    return (
        numerator <= 0.0
        ? 0.0
        : std::sqrt(numerator / denominator)
    );
}

inline double phase_from_doubled(std::int32_t exponent2) noexcept {
    return ((exponent2 / 2) & 1) ? -1.0 : 1.0;
}

inline bool valid_doubled_spin_value(
    std::int32_t two_j,
    std::int32_t two_m
) noexcept {
    return (
        two_j >= 0 &&
        std::abs(two_m) <= two_j &&
        ((two_j - two_m) & 1) == 0
    );
}

inline bool valid_doubled_fusion(
    std::int32_t two_j_left,
    std::int32_t two_j_right,
    std::int32_t two_j_fused
) noexcept {
    return (
        two_j_fused >= std::abs(two_j_left - two_j_right) &&
        two_j_fused <= two_j_left + two_j_right &&
        ((two_j_left + two_j_right - two_j_fused) & 1) == 0
    );
}

inline double factorial_nonnegative(double value) noexcept {
    return (
        value < -1.0e-12
        ? std::numeric_limits<double>::infinity()
        : std::tgamma(value + 1.0)
    );
}

inline double cg_right_spin_half(
    std::int32_t left_j2,
    std::int32_t left_m2,
    std::int32_t right_m2,
    std::int32_t coupled_j2
) noexcept {
    const double j2 = static_cast<double>(left_j2);
    const double m2 = static_cast<double>(left_m2);
    if (right_m2 == 1) {
        if (coupled_j2 == left_j2 + 1) {
            return sqrt_ratio(j2 + m2 + 2.0, 2.0 * (j2 + 1.0));
        }
        if (coupled_j2 == left_j2 - 1) {
            return -sqrt_ratio(j2 - m2, 2.0 * (j2 + 1.0));
        }
    } else if (right_m2 == -1) {
        if (coupled_j2 == left_j2 + 1) {
            return sqrt_ratio(j2 - m2 + 2.0, 2.0 * (j2 + 1.0));
        }
        if (coupled_j2 == left_j2 - 1) {
            return sqrt_ratio(j2 + m2, 2.0 * (j2 + 1.0));
        }
    }
    return 0.0;
}

inline double cg_right_spin_one(
    std::int32_t left_j2,
    std::int32_t left_m2,
    std::int32_t right_m2,
    std::int32_t coupled_j2
) noexcept {
    const double j2 = static_cast<double>(left_j2);
    const double m2 = static_cast<double>(left_m2);
    if (coupled_j2 == left_j2 + 2) {
        const double denominator = 4.0 * (j2 + 1.0) * (j2 + 2.0);
        if (right_m2 == 2) {
            return sqrt_ratio(
                (j2 + m2 + 2.0) * (j2 + m2 + 4.0),
                denominator
            );
        }
        if (right_m2 == 0) {
            return sqrt_ratio(
                (j2 - m2 + 2.0) * (j2 + m2 + 2.0),
                0.5 * denominator
            );
        }
        if (right_m2 == -2) {
            return sqrt_ratio(
                (j2 - m2 + 2.0) * (j2 - m2 + 4.0),
                denominator
            );
        }
    }
    if (coupled_j2 == left_j2 && left_j2 > 0) {
        const double denominator = 2.0 * j2 * (j2 + 2.0);
        if (right_m2 == 2) {
            return -sqrt_ratio(
                (j2 - m2) * (j2 + m2 + 2.0),
                denominator
            );
        }
        if (right_m2 == 0) {
            return m2 / std::sqrt(j2 * (j2 + 2.0));
        }
        if (right_m2 == -2) {
            return sqrt_ratio(
                (j2 + m2) * (j2 - m2 + 2.0),
                denominator
            );
        }
    }
    if (coupled_j2 == left_j2 - 2 && left_j2 >= 2) {
        const double denominator = 4.0 * j2 * (j2 + 1.0);
        if (right_m2 == 2) {
            return sqrt_ratio(
                (j2 - m2) * (j2 - m2 - 2.0),
                denominator
            );
        }
        if (right_m2 == 0) {
            return -sqrt_ratio(
                (j2 - m2) * (j2 + m2),
                0.5 * denominator
            );
        }
        if (right_m2 == -2) {
            return sqrt_ratio(
                (j2 + m2) * (j2 + m2 - 2.0),
                denominator
            );
        }
    }
    return 0.0;
}

inline double clebsch_gordan_doubled(
    std::int32_t two_j_left,
    std::int32_t two_j_right,
    std::int32_t two_j_fused,
    std::int32_t two_m_left,
    std::int32_t two_m_right,
    std::int32_t two_m_fused
) noexcept {
    if (
        two_m_left + two_m_right != two_m_fused ||
        !valid_doubled_spin_value(two_j_left, two_m_left) ||
        !valid_doubled_spin_value(two_j_right, two_m_right) ||
        !valid_doubled_spin_value(two_j_fused, two_m_fused) ||
        !valid_doubled_fusion(
            two_j_left,
            two_j_right,
            two_j_fused
        )
    ) {
        return 0.0;
    }
    if (two_j_right == 0) {
        return 1.0;
    }
    if (two_j_right == 1) {
        return cg_right_spin_half(
            two_j_left,
            two_m_left,
            two_m_right,
            two_j_fused
        );
    }
    if (two_j_right == 2) {
        return cg_right_spin_one(
            two_j_left,
            two_m_left,
            two_m_right,
            two_j_fused
        );
    }
    if (two_j_left <= 2) {
        return (
            phase_from_doubled(
                two_j_left + two_j_right - two_j_fused
            ) *
            clebsch_gordan_doubled(
                two_j_right,
                two_j_left,
                two_j_fused,
                two_m_right,
                two_m_left,
                two_m_fused
            )
        );
    }
    const double j1 = 0.5 * static_cast<double>(two_j_left);
    const double j2 = 0.5 * static_cast<double>(two_j_right);
    const double fused = 0.5 * static_cast<double>(two_j_fused);
    const double m1 = 0.5 * static_cast<double>(two_m_left);
    const double m2 = 0.5 * static_cast<double>(two_m_right);
    const double fused_m = 0.5 * static_cast<double>(two_m_fused);
    double prefactor = std::sqrt(
        (2.0 * fused + 1.0) *
        factorial_nonnegative(fused + j1 - j2) *
        factorial_nonnegative(fused - j1 + j2) *
        factorial_nonnegative(j1 + j2 - fused) /
        factorial_nonnegative(j1 + j2 + fused + 1.0)
    );
    prefactor *= std::sqrt(
        factorial_nonnegative(fused + fused_m) *
        factorial_nonnegative(fused - fused_m) *
        factorial_nonnegative(j1 - m1) *
        factorial_nonnegative(j1 + m1) *
        factorial_nonnegative(j2 - m2) *
        factorial_nonnegative(j2 + m2)
    );
    const auto exact_int = [](double value) noexcept {
        return static_cast<std::int32_t>(std::llround(value));
    };
    const std::int32_t k_min = std::max({
        std::int32_t{0},
        exact_int(j2 - fused - m1),
        exact_int(j1 + m2 - fused),
    });
    const std::int32_t k_max = std::min({
        exact_int(j1 + j2 - fused),
        exact_int(j1 - m1),
        exact_int(j2 + m2),
    });
    double total = 0.0;
    for (std::int32_t k = k_min; k <= k_max; ++k) {
        const double denominator =
            factorial_nonnegative(static_cast<double>(k)) *
            factorial_nonnegative(j1 + j2 - fused - k) *
            factorial_nonnegative(j1 - m1 - k) *
            factorial_nonnegative(j2 + m2 - k) *
            factorial_nonnegative(fused - j2 + m1 + k) *
            factorial_nonnegative(fused - j1 - m2 + k);
        total += (k & 1 ? -1.0 : 1.0) / denominator;
    }
    return prefactor * total;
}

}  // namespace pyqed::su2
