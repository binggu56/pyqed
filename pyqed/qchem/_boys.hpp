#ifndef PYQED_QCHEM_BOYS_HPP
#define PYQED_QCHEM_BOYS_HPP

#include <cmath>
#include <algorithm>
#include <limits>

namespace pyqed::qchem {

inline constexpr double kBoysPi =
    3.141592653589793238462643383279502884;

inline constexpr double kBoys0MidChebyshev[3][17] = {
    {3.69260892593684109e-01, -6.28317191907165201e-02, 7.78147807755485196e-03, -1.01824922931021138e-03, 1.30508527081563824e-04, -1.58459023097007741e-05, 1.79378499513903238e-06, -1.88060978338879438e-07, 1.82309276136738246e-08, -1.63602207972775239e-09, 1.36246942215674343e-10, -1.05627704833339070e-11, 7.64920512606895655e-13, -5.19020980460245112e-14, 3.10927261055600145e-15, -2.16717292100600925e-16, -1.91834944614729702e-17},
    {2.82392962887930754e-01, -2.85562567876372145e-02, 2.16121319327928184e-03, -1.80974042560373282e-04, 1.57609803527738631e-05, -1.38719093334998265e-06, 1.21011061292869219e-07, -1.03077047929483520e-08, 8.47686398613829223e-10, -6.67675966377318255e-11, 5.01093146865571330e-12, -3.57236219061710461e-13, 2.43404256846936807e-14, -1.57750782602794538e-15, -4.30489072102299092e-17, -8.38146365290142540e-18, 1.80395600448499878e-16},
    {2.37770624901000233e-01, -1.70820914178422857e-02, 9.19985072678147688e-04, -5.50362676255404418e-05, 3.45464478138237826e-06, -2.22644470115022074e-07, 1.45569210486204908e-08, -9.56969674338625626e-10, 6.27465433914607864e-11, -4.07166502220521936e-12, 2.59717592016691244e-13, -1.61581557658677476e-14, 1.13365440188229370e-15, -6.79821661128639535e-17, -7.01062679852013498e-17, 1.59364317989742982e-18, 1.86498993798541971e-16},
};

inline constexpr double kBoys0SmallChebyshev[19] = {
    6.57186257657690676e-01, -2.66817291710163607e-01,
    6.11450766743829172e-02, -1.23329742967011637e-02,
    2.14537771790367093e-03, -3.24238500012818173e-04,
    4.30892255459287019e-05, -5.09514596310239750e-06,
    5.41772106229890224e-07, -5.22794262476461671e-08,
    4.61459185483047427e-09, -3.75137727347970943e-10,
    2.82540313420055434e-11, -1.98183387666075457e-12,
    1.30438207295877616e-13, -8.22014661804220726e-15,
    9.20457102772300290e-16, -1.65470754326506321e-16,
    2.87011308337681401e-16,
};

inline double boys0_small_chebyshev(double value) {
    const double x = 0.5 * value - 1.0;
    double next = 0.0;
    double next_next = 0.0;
    for (int degree = 18; degree >= 1; --degree) {
        const double current = 2.0 * x * next - next_next
            + kBoys0SmallChebyshev[degree];
        next_next = next;
        next = current;
    }
    return x * next - next_next + kBoys0SmallChebyshev[0];
}

inline double boys0_mid_chebyshev(double value) {
    const int interval = value < 8.0 ? 0 : (value < 12.0 ? 1 : 2);
    const double center = interval == 0 ? 6.0 : (interval == 1 ? 10.0 : 14.0);
    const double x = 0.5 * (value - center);
    const double* coefficients = kBoys0MidChebyshev[interval];
    double next = 0.0;
    double next_next = 0.0;
    for (int degree = 16; degree >= 1; --degree) {
        const double current = 2.0 * x * next - next_next + coefficients[degree];
        next_next = next;
        next = current;
    }
    return x * next - next_next + coefficients[0];
}

inline double boys0(double value) {
    if (value < 1.0e-14) {
        return 1.0;
    }
    if (value < 4.0) {
        return boys0_small_chebyshev(value);
    }
    if (value < 16.0) {
        return boys0_mid_chebyshev(value);
    }
    const double root = std::sqrt(value);
    double series = 1.0;
    double term = 1.0;
    double previous = std::numeric_limits<double>::infinity();
    for (int order = 1; order < 40; ++order) {
        term *= -(2.0 * order - 1.0) / (2.0 * value);
        if (std::abs(term) >= previous) {
            break;
        }
        series += term;
        previous = std::abs(term);
    }
    const double erfc = std::exp(-value) * series
        / (std::sqrt(kBoysPi) * root);
    return 0.5 * std::sqrt(kBoysPi / value) * (1.0 - erfc);
}

inline void short_range_boys(int max_order, double value, double width, double* out) {
    // Positive moments integral_{1-width}^1 t^(2m) exp(-value*t*t) dt.
    // Retaining width separately avoids losing a very narrow interval to rounding.
    constexpr double nodes[8] = {
        .095012509837637440, .28160355077925891, .45801677765722737,
        .61787624440264377, .75540440835500303, .86563120238783176,
        .94457502307323258, .98940093499164994};
    constexpr double weights[8] = {
        .18945061045506850, .18260341504492359, .16915651939500265,
        .14959598881657673, .12462897125553399, .09515851168249272,
        .06225352393864782, .02715245941175390};
    std::fill(out, out + max_order + 1, 0.0);
    const double lower = 1.0 - width;
    if (max_order == 0 && value > 1.0 && width * std::sqrt(value) > 0.1) {
        const double root = std::sqrt(value);
        out[0] = 0.5 * std::sqrt(kBoysPi / value)
            * (std::erfc(lower * root) - std::erfc(root));
        return;
    }
    double length = width;
    if (value > 0.0) {
        // The omitted positive gamma tail is far below double precision for
        // the supported angular orders, even relative to the retained tail.
        const double end_argument = std::max(value * lower * lower,
                                             static_cast<double>(max_order))
            + 80.0 + 4.0 * max_order;
        if (value > end_argument) {
            length = std::sqrt(end_argument / value) - lower;
        }
    }
    const int panels = std::max(1, static_cast<int>(std::ceil(
        value * length * (2.0 * lower + length) / 8.0)));
    const double half = length / (2.0 * panels);
    for (int panel = 0; panel < panels; ++panel) {
        const double center = lower + (2.0 * panel + 1.0) * half;
        for (int node = 0; node < 8; ++node) {
            for (int sign : {-1, 1}) {
                const double t = center + sign * half * nodes[node];
                const double t2 = t * t;
                double term = half * weights[node] * std::exp(-value * t2);
                for (int m = 0; m <= max_order; ++m) {
                    out[m] += term;
                    term *= t2;
                }
            }
        }
    }
}

}  // namespace pyqed::qchem

#endif
