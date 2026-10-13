// Bracket-first diagonal Dyson continuation; independent orbital OpenMP work.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static bool openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

struct Equation {
    const double *poles, *residue;
    py::ssize_t count;
    double energy, shift, eta, coupling;
    double residual(double w) const {
        double value = shift;
        const double eta2 = eta*eta;
        for (py::ssize_t k = 0; k < count; ++k) {
            if (residue[k] == 0.) continue;
            const double x = w-poles[k];
            value += residue[k]*x/(x*x+eta2);
        }
        return w-energy-coupling*value;
    }
    void evaluate(double w, double &residual, double &slope) const {
        double value = shift, derivative = 0.;
        const double eta2 = eta*eta;
        for (py::ssize_t k = 0; k < count; ++k) {
            if (residue[k] == 0.) continue;
            const double x = w-poles[k], denominator = x*x+eta2;
            value += residue[k]*x/denominator;
            derivative += residue[k]*(eta2-x*x)/(denominator*denominator);
        }
        residual = w-energy-coupling*value;
        slope = 1.-coupling*derivative;
    }
};

static bool bracket_root(const Equation &eq, double left, double right, double &root) {
    double x = .5*(left+right);
    for (int iteration = 0; iteration < 120; ++iteration) {
        double f, slope;
        eq.evaluate(x, f, slope);
        if (!std::isfinite(f)) return false;
        if (std::abs(f) <= 1e-11) {
            root = x;
            return slope > 0.;
        }
        if (f < 0.) left = x; else right = x;
        const double trial = x-f/slope;
        // A bracket safeguards Newton; a central step guarantees contraction.
        const double margin = .1*(right-left);
        x = slope > 0. && trial > left+margin && trial < right-margin
            ? trial : .5*(left+right);
    }
    return false;
}

static bool continue_root(Equation &eq, double &root, double &weight, double &error) {
    root = eq.energy;
    double coupling = 0., increment = .125;
    while (coupling < 1.) {
        eq.coupling = std::min(1., coupling+increment);
        double f, slope;
        eq.evaluate(root, f, slope);
        bool valid = std::abs(f) <= 1e-9 && slope > 0.;
        double candidate = root, left = root, right = root;
        double fl = f, fr = f;
        const double scale = std::log(2.*std::max(1., std::abs(root))/1e-7);
        for (int shell = 0; shell < 120 && !valid; ++shell) {
            const double radius = 1e-7*std::exp(scale*shell/119.);
            const double far_left = eq.residual(root-radius), far_right = eq.residual(root+radius);
            double a, b;
            const bool found_left = far_left <= 0. && fl >= 0. && bracket_root(eq, root-radius, left, a);
            const bool found_right = fr <= 0. && far_right >= 0. && bracket_root(eq, right, root+radius, b);
            if (found_left || found_right) {
                candidate = found_left && (!found_right || std::abs(a-root) <= std::abs(b-root)) ? a : b;
                valid = true;
            }
            left = root-radius; right = root+radius;
            fl = far_left; fr = far_right;
        }
        if (valid) {
            root = candidate; coupling = eq.coupling;
            increment = std::min(.125, 2.*increment);
        } else {
            increment *= .5;
            if (increment < 1e-4) return false;
        }
    }
    double f, slope;
    eq.evaluate(root, f, slope);
    weight = 1./slope; error = std::abs(f);
    return std::isfinite(weight) && weight > 0. && error <= 1e-9;
}

static py::tuple solve(Array poles, Array residues, Array energies, Array shifts,
                       double eta, int threads) {
    if (poles.ndim() != 1 || residues.ndim() != 2 || energies.ndim() != 1 || shifts.ndim() != 1
        || residues.shape(1) != poles.size() || residues.shape(0) != energies.size()
        || shifts.size() != energies.size() || threads < 1 || !std::isfinite(eta) || eta < 0.)
        throw std::invalid_argument("Invalid Dyson batch dimensions, eta, or threads");
    if (threads > 1 && !openmp_available())
        throw std::runtime_error("GW Dyson extension was built without OpenMP");
    const auto n = energies.size(), count = poles.size();
    const double *p = poles.data(), *r = residues.data(), *e = energies.data(), *s = shifts.data();
    for (py::ssize_t k = 0; k < count; ++k)
        if (!std::isfinite(p[k])) throw std::invalid_argument("Nonfinite Dyson pole");
    for (py::ssize_t k = 0; k < residues.size(); ++k)
        if (!std::isfinite(r[k]) || r[k] < 0.) throw std::invalid_argument("Dyson residues must be finite and nonnegative");
    for (py::ssize_t k = 0; k < n; ++k)
        if (!std::isfinite(e[k]) || !std::isfinite(s[k])) throw std::invalid_argument("Nonfinite Dyson energy or shift");
    Array roots(n), weights(n), residuals(n);
    double *out = roots.mutable_data(), *z = weights.mutable_data(), *err = residuals.mutable_data();
    std::vector<int> success(n);
    int used = 1;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(std::min<int>(threads, std::max<py::ssize_t>(1, n))) if(threads > 1 && n > 1)
        {
#pragma omp single
            used = omp_get_num_threads();
#pragma omp for schedule(dynamic, 1)
#endif
            for (py::ssize_t orbital = 0; orbital < n; ++orbital) {
                Equation equation{p, r+orbital*count, count, e[orbital], s[orbital], eta, 0.};
                success[orbital] = continue_root(equation, out[orbital], z[orbital], err[orbital]);
            }
#ifdef _OPENMP
        }
#endif
    }
    for (py::ssize_t orbital = 0; orbital < n; ++orbital)
        if (!success[orbital]) throw std::runtime_error("Could not track a positive-weight quasiparticle root in batch orbital " + std::to_string(orbital));
    return py::make_tuple(roots, weights, residuals, used);
}

PYBIND11_MODULE(_dyson, module) {
    module.doc() = "Bracket-first spectral Dyson continuation with orbital OpenMP";
    module.def("openmp_available", &openmp_available);
    module.def("solve", &solve, py::arg("poles"), py::arg("residues"), py::arg("energies"),
               py::arg("shifts"), py::arg("eta"), py::arg("threads") = 1);
}
