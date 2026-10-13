// Complete auxiliary contour contractions with disjoint OpenMP output tiles.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "../linalg/row_product.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
using Complex = std::complex<double>;
using Real = py::array_t<double, py::array::forcecast>;
template <typename T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

static bool openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

static int team_size(int threads, int tasks, size_t work) {
    if (threads < 1) throw std::invalid_argument("threads must be positive");
    if (threads > 1 && !openmp_available()) throw std::runtime_error("Contour contractions were built without OpenMP");
    return work < 262144 ? 1 : std::min(threads, std::max(1, tasks));
}

struct Pairs {
    Real owner;
    int rows, cols, stride;
    const double *data;
    explicit Pairs(Real array) : owner(array) {
        if (array.ndim() != 2 || array.shape(0) < 1 || array.shape(1) < 1
            || array.shape(0) > INT_MAX || array.shape(1) > INT_MAX
            || array.strides(1) != sizeof(double) || array.strides(0) < array.shape(1)*py::ssize_t(sizeof(double))
            || array.strides(0)/sizeof(double) > INT_MAX)
            throw std::invalid_argument("Contour pairs require positive row-contiguous auxiliary/transition storage");
        rows = array.shape(0); cols = array.shape(1); stride = array.strides(0)/sizeof(double); data = array.data();
    }
};

static void project(const Pairs &p, const Complex *vector, Complex *out, int team, int &used) {
    const int tile = 128, tasks = (p.cols+tile-1)/tile;
#ifdef _OPENMP
#pragma omp parallel num_threads(team) if(team > 1)
    {
#pragma omp single
        used = std::max(used, omp_get_num_threads());
#pragma omp for schedule(static)
#endif
        for (int task = 0; task < tasks; ++task) {
            const int first = task*tile, stop = std::min(p.cols, first+tile);
            std::fill(out+first, out+stop, Complex{});
            for (int row = 0; row < p.rows; ++row) {
                const double *pairs = p.data+size_t(row)*p.stride;
                for (int col = first; col < stop; ++col) out[col] += pairs[col]*vector[row];
            }
        }
#ifdef _OPENMP
    }
#endif
}

static py::tuple projection(Real pairs, Array<Complex> vector, int threads) {
    Pairs p(pairs);
    if (vector.ndim() != 1 || vector.size() != p.rows) throw std::invalid_argument("Invalid contour projection vector");
    const int team = team_size(threads, (p.cols+127)/128, size_t(p.rows)*p.cols);
    Array<Complex> result(p.cols);
    int used = 1;
    {
        py::gil_scoped_release release;
        project(p, vector.data(), result.mutable_data(), team, used);
    }
    return py::make_tuple(result, used);
}

static py::tuple action(Real pairs, Array<Complex> weights, Array<Complex> vector, int threads) {
    Pairs p(pairs);
    if (weights.ndim() != 1 || weights.size() != p.cols || vector.ndim() != 1 || vector.size() != p.rows)
        throw std::invalid_argument("Invalid dielectric action weights/vector");
    const int forward = team_size(threads, (p.cols+127)/128, size_t(p.rows)*p.cols);
    [[maybe_unused]] const int back = team_size(threads, (p.rows+31)/32, size_t(p.rows)*p.cols);
    Array<Complex> result(p.rows);
    int used = 1;
    {
        py::gil_scoped_release release;
        std::vector<Complex> projected(p.cols);
        project(p, vector.data(), projected.data(), forward, used);
        for (int col = 0; col < p.cols; ++col) projected[col] *= weights.data()[col];
        const int tasks = (p.rows+31)/32;
#ifdef _OPENMP
#pragma omp parallel num_threads(back) if(back > 1)
        {
#pragma omp single
            used = std::max(used, omp_get_num_threads());
#pragma omp for schedule(static)
#endif
            for (int task = 0; task < tasks; ++task)
                for (int row = 32*task; row < std::min(p.rows, 32*(task+1)); ++row) {
                    Complex value{};
                    const double *data = p.data+size_t(row)*p.stride;
                    for (int col = 0; col < p.cols; ++col) value += data[col]*projected[col];
                    result.mutable_data()[row] = vector.data()[row]-value;
                }
#ifdef _OPENMP
        }
#endif
    }
    return py::make_tuple(result, used);
}

template <typename Scalar>
static py::tuple dielectric(Real pairs, Array<Scalar> weights, int threads) {
    Pairs p(pairs);
    if (weights.ndim() != 1 || weights.size() != p.cols) throw std::invalid_argument("Invalid dielectric weights");
    const int row_tile = 16, transition_tile = 64, tasks = (p.rows+row_tile-1)/row_tile;
    const int team = team_size(threads, tasks, size_t(p.rows)*p.rows*p.cols);
    py::array_t<Scalar> result({p.rows, p.rows}, {py::ssize_t(sizeof(Scalar)), py::ssize_t(p.rows*sizeof(Scalar))});
    Scalar *out = result.mutable_data();
    int used = 1;
    {
        py::gil_scoped_release release;
        std::fill(out, out+size_t(p.rows)*p.rows, Scalar{});
        for (int i = 0; i < p.rows; ++i) out[size_t(i)*p.rows+i] = 1.;
        std::vector<double> right(size_t(transition_tile)*p.rows);
        std::vector<std::vector<double>> left(team, std::vector<double>(row_tile*transition_tile));
        std::vector<std::vector<double>> products(team, std::vector<double>(size_t(row_tile)*p.rows));
        for (int first = 0; first < p.cols; first += transition_tile) {
            const int count = std::min(transition_tile, p.cols-first);
            for (int row = 0; row < p.rows; ++row)
                for (int k = 0; k < count; ++k) right[size_t(k)*p.rows+row] = p.data[size_t(row)*p.stride+first+k];
#ifdef _OPENMP
#pragma omp parallel num_threads(team) if(team > 1)
            {
#pragma omp single
                used = std::max(used, omp_get_num_threads());
#pragma omp for schedule(dynamic, 1)
#endif
                for (int task = 0; task < tasks; ++task) {
                    int worker = 0;
#ifdef _OPENMP
                    worker = omp_get_thread_num();
#endif
                    const int row = task*row_tile, rows = std::min(row_tile, p.rows-row), width = row+rows;
                    auto &a = left[worker]; auto &value = products[worker];
                    constexpr int components = std::is_same_v<Scalar, Complex> ? 2 : 1;
                    for (int component = 0; component < components; ++component) {
                        for (int i = 0; i < rows; ++i)
                            for (int k = 0; k < count; ++k) {
                                double w;
                                if constexpr (std::is_same_v<Scalar, Complex>)
                                    w = component ? weights.data()[first+k].imag() : weights.data()[first+k].real();
                                else w = weights.data()[first+k];
                                a[i*count+k] = p.data[size_t(row+i)*p.stride+first+k]*w;
                            }
                        pyqed::linalg::row_product(a.data(), count, right.data(), value.data(), rows, width, count, p.rows);
                        for (int i = 0; i < rows; ++i)
                            for (int j = 0; j <= row+i; ++j) {
                                auto &target = out[size_t(j)*p.rows+row+i];
                                const double update = value[size_t(i)*width+j];
                                if constexpr (std::is_same_v<Scalar, Complex>) {
                                    if (component) target.imag(target.imag()-update);
                                    else target.real(target.real()-update);
                                } else target -= update;
                            }
                    }
                }
#ifdef _OPENMP
            }
#endif
        }
        // Complex symmetric, not Hermitian: mirror without conjugation.
        for (int i = 0; i < p.rows; ++i)
            for (int j = 0; j < i; ++j) out[size_t(i)*p.rows+j] = out[size_t(j)*p.rows+i];
    }
    return py::make_tuple(result, used);
}

static py::tuple quadrature(Array<double> energy, Array<double> w0, Array<double> imag,
                            Array<double> frequencies, Array<double> weights,
                            int no, double omega, int threads) {
    if (energy.ndim() != 1 || w0.ndim() != 1 || imag.ndim() != 2 || frequencies.ndim() != 1
        || weights.ndim() != 1 || w0.size() != energy.size() || imag.shape(0) != energy.size()
        || imag.shape(1) != frequencies.size() || frequencies.size() != weights.size()
        || no < 0 || no > energy.size() || !std::isfinite(omega))
        throw std::invalid_argument("Invalid contour quadrature arrays");
    const int n = energy.size(), nw = frequencies.size(), tile = 16, tasks = (n+tile-1)/tile;
    [[maybe_unused]] const int team = team_size(threads, tasks, size_t(n)*nw*32);
    std::vector<std::array<double, 3>> parts(tasks);
    int used = 1;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(team) if(team > 1)
        {
#pragma omp single
            used = omp_get_num_threads();
#pragma omp for schedule(static)
#endif
            for (int task = 0; task < tasks; ++task) {
                auto &sum = parts[task]; sum = {0., 0., 0.};
                for (int m = task*tile; m < std::min(n, (task+1)*tile); ++m) {
                    const double delta = omega-energy.data()[m], delta2 = delta*delta;
                    sum[0] += (.5-(m < no))*w0.data()[m];
                    for (int k = 0; k < nw; ++k) {
                        const double f2 = frequencies.data()[k]*frequencies.data()[k], denom = delta2+f2;
                        const double difference = imag.data()[size_t(m)*nw+k]-w0.data()[m];
                        sum[1] += difference*weights.data()[k]*delta/denom;
                        sum[2] += difference*weights.data()[k]*(f2-delta2)/(denom*denom);
                    }
                }
            }
#ifdef _OPENMP
        }
#endif
    }
    std::array<double, 3> total{0., 0., 0.};
    for (const auto &sum : parts) for (int k = 0; k < 3; ++k) total[k] += sum[k];
    const double pi = std::acos(-1.);
    return py::make_tuple(total[0]-total[1]/pi, -total[2]/pi, used);
}

PYBIND11_MODULE(_contour_contractions, module) {
    module.doc() = "Complete contour-GW contractions and quadrature with deterministic OpenMP output tiles";
    module.def("openmp_available", &openmp_available);
    module.def("project", &projection);
    module.def("action", &action);
    module.def("imaginary_dielectric", &dielectric<double>);
    module.def("dielectric", &dielectric<Complex>);
    module.def("quadrature", &quadrature);
}
