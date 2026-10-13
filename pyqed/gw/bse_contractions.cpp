// Streamed factorized static BSE action; fixed-order auxiliary-tile reduction.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../linalg/row_product.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
using Array = py::array_t<double, py::array::forcecast>;
using Block = py::array_t<double, py::array::c_style | py::array::forcecast>;

static bool openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// SciPy exports a stable Cython BLAS entry point using C int dimensions.
// Calling the same BLAS as scipy.linalg lets threadpoolctl prevent nested teams.
using Dgemm = void (*)(char*, char*, int*, int*, int*, double*, double*, int*,
                      double*, int*, double*, double*, int*);
static Dgemm dgemm = nullptr;

static void row_product(const double *a, int stride, const double *b, double *out,
                        int m, int n, int k) {
    if (size_t(m)*n*k < 32768) {
        pyqed::linalg::row_product(a, stride, b, out, m, n, k);
        return;
    }
    // Row-major C=A B is column-major C^T=B^T A^T, without copies.
    char trans = 'N';
    double one = 1., zero = 0.;
    dgemm(&trans, &trans, &n, &m, &k, &one, const_cast<double*>(b), &n,
          const_cast<double*>(a), &stride, &zero, out, &n);
}

struct Scratch {
    std::vector<double> left, tmp, packed, value, projection;
    Scratch(int no, int nv, int columns)
        : left(size_t(no)*std::max(no, nv)),
          tmp(size_t(no)*std::max(no, nv)*columns),
          packed(size_t(no)*std::max(no, nv)*columns), value(size_t(no)*nv*columns),
          projection(columns) {}
};

static void add_factor(const double *factor, int stride, const double *screen,
                       int screen_stride, const double *x, const double *xp,
                       double *out, int no, int nv, int columns, bool b,
                       double sign, Scratch &work) {
    const double *vo = factor+size_t(no)*stride;
    // The two blocks have different Hartree pair orientations.
    std::fill(work.projection.begin(), work.projection.end(), 0.);
    for (int j = 0; j < no; ++j)
        for (int v = 0; v < nv; ++v) {
            const double f = b ? factor[size_t(j)*stride+no+v] : vo[size_t(v)*stride+j];
            const double *row = x+(size_t(j)*nv+v)*columns;
            for (int k = 0; k < columns; ++k) work.projection[k] += f*row[k];
        }
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a) {
            const double f = sign*2.*vo[size_t(a)*stride+i];
            double *row = out+(size_t(i)*nv+a)*columns;
            for (int k = 0; k < columns; ++k) row[k] += f*work.projection[k];
        }
    const int width = b ? nv : no;
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < width; ++j)
            work.left[size_t(i)*width+j] = -factor[size_t(i)*stride+(b ? no+j : j)]
                +4.*screen[size_t(i)*screen_stride+j];
    if (b) {
        row_product(work.left.data(), nv, xp, work.tmp.data(), no, no*columns, nv);
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < columns; ++k)
                    work.packed[(size_t(j)*no+i)*columns+k] = work.tmp[(size_t(i)*no+j)*columns+k];
        row_product(vo, stride, work.packed.data(), work.value.data(), nv, no*columns, no);
    } else {
        row_product(work.left.data(), no, x, work.tmp.data(), no, nv*columns, no);
        for (int i = 0; i < no; ++i)
            for (int v = 0; v < nv; ++v)
                for (int k = 0; k < columns; ++k)
                    work.packed[(size_t(v)*no+i)*columns+k] = work.tmp[(size_t(i)*nv+v)*columns+k];
        row_product(vo+no, stride, work.packed.data(), work.value.data(), nv, no*columns, nv);
    }
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a)
            for (int k = 0; k < columns; ++k)
                out[(size_t(i)*nv+a)*columns+k] += sign*work.value[(size_t(a)*no+i)*columns+k];
}

static void add_seed_factor(const double *f, int stride, const double *o, int os,
                            const double *v, int vs, const double *x, double *out,
                            const std::vector<int> &seeds, int no, int nv,
                            int columns, const std::string &kind) {
    const size_t dim = size_t(no)*nv, size = dim*columns;
    for (int k=0; k<columns; ++k) {
        if (seeds[k]<0) continue;
        const int half = seeds[k]/dim, pair = seeds[k]%dim;
        const int j = pair/nv, b = pair%nv;
        const double scale = x[size_t(seeds[k])*columns+k];
        for (int a=0; a<nv; ++a) {
            const double ab = f[size_t(no+a)*stride+no+b];
            const double aj = f[size_t(no+a)*stride+j];
            for (int i=0; i<no; ++i) {
                const double ai = f[size_t(no+a)*stride+i];
                double av=0., bv=0.;
                if (kind != "b") av = 2*ai*f[size_t(no+b)*stride+j]
                    +(4*o[size_t(i)*os+j]-f[size_t(i)*stride+j])*ab;
                if (kind != "a") bv = 2*ai*f[size_t(j)*stride+no+b]
                    +(4*v[size_t(i)*vs+b]-f[size_t(i)*stride+no+b])*aj;
                const size_t index = (size_t(i)*nv+a)*columns+k;
                if (kind == "full") {
                    out[index] += scale*(half ? bv : av);
                    out[size+index] -= scale*(half ? av : bv);
                } else out[index] += scale*(kind == "a" ? av : bv);
            }
        }
    }
}

static py::tuple action(Array factors, Array occupied, Array ov, Block vectors,
                        int no, std::string kind, int threads) {
    if (factors.ndim() != 3 || occupied.ndim() != 3 || ov.ndim() != 3 || vectors.ndim() != 2
        || factors.shape(1) != factors.shape(2) || no < 1 || no >= factors.shape(1)
        || threads < 1 || (kind != "a" && kind != "b" && kind != "full"))
        throw std::invalid_argument("Invalid factorized BSE action dimensions or controls");
    if (threads > 1 && !openmp_available())
        throw std::runtime_error("BSE contractions were built without OpenMP");
    const int nmo = factors.shape(1), nv = nmo-no;
    const auto count = factors.shape(0);
    const bool full = kind == "full";
    const size_t dimension = size_t(no)*nv;
    const auto columns = vectors.shape(1);
    if (count < 1 || columns > INT_MAX/std::max(no, nv) || dimension > INT_MAX
        || vectors.shape(0) != py::ssize_t(dimension*(full ? 2 : 1))
        || occupied.shape(0) != count || occupied.shape(1) != no || occupied.shape(2) != no
        || ov.shape(0) != count || ov.shape(1) != no || ov.shape(2) != nv)
        throw std::invalid_argument("BSE factors, static screening and trials must match");
    std::vector<Array> checked{factors, occupied};
    if (kind != "a") checked.push_back(ov);
    for (const auto &array : checked)
        if (array.strides(2) != sizeof(double) || array.strides(1) < array.shape(2)*py::ssize_t(sizeof(double))
            || array.strides(0) < array.shape(1)*array.strides(1))
            throw std::invalid_argument("BSE factors require positive row-contiguous storage");
    const auto fs = factors.strides(0)/sizeof(double), os = occupied.strides(0)/sizeof(double), vs = ov.strides(0)/sizeof(double);
    const int fr = factors.strides(1)/sizeof(double), ort = occupied.strides(1)/sizeof(double), vr = ov.strides(1)/sizeof(double);
    const double *f = factors.data(), *o = occupied.data(), *v = ov.data(), *x = vectors.data();
    Block result({vectors.shape(0), columns});
    double *out = result.mutable_data();
    const size_t size = dimension*columns, total = size*(full ? 2 : 1);
    std::vector<int> seeds(columns, -1);
    bool sparse = true;
    for (int k=0; k<columns && sparse; ++k)
        for (py::ssize_t i=0; i<vectors.shape(0); ++i)
            if (x[size_t(i)*columns+k] != 0.) {
                if (seeds[k] >= 0) { sparse=false; break; }
                seeds[k]=i;
            }
    const bool active_x = std::any_of(x, x+size, [](double a){return a != 0.;});
    const bool active_y = full && std::any_of(x+size, x+total, [](double a){return a != 0.;});
    const bool fused = full && active_x && active_y && !sparse;
    const int tile = 16;
    const int tasks = (count+tile-1)/tile;
    const int team = std::min(threads, tasks), wave = std::min(2*team, tasks);
    int used = 1;
    if (!columns) return py::make_tuple(result, used, tile);
    {
        py::gil_scoped_release release;
        std::vector<Scratch> scratch;
        for (int i = 0; i < team; ++i)
            scratch.emplace_back(sparse ? 0 : no, nv, columns*(fused ? 2 : 1));
        const size_t partial_size = total*(fused ? 2 : 1);
        std::vector<double> partial(size_t(wave)*partial_size), packed(total), joined(fused ? total : 0);
        std::fill(out, out+total, 0.);
        if (fused) {
            for (int i=0; i<no; ++i)
                for (int a=0; a<nv; ++a)
                    for (int k=0; k<columns; ++k)
                        for (int half=0; half<2; ++half) {
                            const double z=x[half*size+(size_t(i)*nv+a)*columns+k];
                            joined[(size_t(i)*nv+a)*2*columns+half*columns+k]=z;
                            packed[(size_t(a)*no+i)*2*columns+half*columns+k]=z;
                        }
        } else if (!sparse) for (int half = 0; half < (full ? 2 : 1); ++half)
            for (int i = 0; i < no; ++i)
                for (int a = 0; a < nv; ++a)
                    for (int k = 0; k < columns; ++k)
                        packed[half*size+(size_t(a)*no+i)*columns+k] = x[half*size+(size_t(i)*nv+a)*columns+k];
        for (int first = 0; first < tasks; first += wave) {
            const int current = std::min(wave, tasks-first);
#ifdef _OPENMP
#pragma omp parallel num_threads(team) if(team > 1)
            {
#pragma omp single
                used = std::max(used, omp_get_num_threads());
#pragma omp for schedule(static)
#endif
                for (int task = 0; task < current; ++task) {
                    int worker = 0;
#ifdef _OPENMP
                    worker = omp_get_thread_num();
#endif
                    double *target = partial.data()+size_t(task)*partial_size;
                    std::fill(target, target+partial_size, 0.);
                    for (py::ssize_t p = (first+task)*tile; p < std::min(count, py::ssize_t((first+task+1)*tile)); ++p) {
                        auto &work = scratch[worker];
                        if (sparse) {
                            add_seed_factor(f+p*fs, fr, o+p*os, ort, v+p*vs, vr,
                                            x, target, seeds, no, nv, columns, kind);
                        } else if (fused) {
                            // Apply A and B to [X,Y] together: wider GEMMs reuse
                            // each factor while preserving the full XY operator.
                            add_factor(f+p*fs, fr, o+p*os, ort, joined.data(), packed.data(), target, no, nv, 2*columns, false, 1., work);
                            add_factor(f+p*fs, fr, v+p*vs, vr, joined.data(), packed.data(), target+total, no, nv, 2*columns, true, 1., work);
                        } else if (full) {
                            if (active_x) add_factor(f+p*fs, fr, o+p*os, ort, x, packed.data(), target, no, nv, columns, false, 1., work);
                            if (active_y) add_factor(f+p*fs, fr, v+p*vs, vr, x+size, packed.data()+size, target, no, nv, columns, true, 1., work);
                            if (active_x) add_factor(f+p*fs, fr, v+p*vs, vr, x, packed.data(), target+size, no, nv, columns, true, -1., work);
                            if (active_y) add_factor(f+p*fs, fr, o+p*os, ort, x+size, packed.data()+size, target+size, no, nv, columns, false, -1., work);
                        } else {
                            const bool b = kind == "b";
                            add_factor(f+p*fs, fr, b ? v+p*vs : o+p*os, b ? vr : ort,
                                x, packed.data(), target, no, nv, columns, b, 1., work);
                        }
                    }
                }
#ifdef _OPENMP
            }
#endif
            // Reduction order follows factor tiles, independent of team size.
            for (int task = 0; task < current; ++task) {
                const double *av = partial.data()+size_t(task)*partial_size;
                if (fused) {
                    const double *bv=av+total;
                    for (size_t i=0; i<dimension; ++i)
                        for (int k=0; k<columns; ++k) {
                            const size_t j=i*2*columns+k, q=i*columns+k;
                            out[q] += av[j]+bv[j+columns];
                            out[size+q] -= bv[j]+av[j+columns];
                        }
                } else for (size_t i=0; i<total; ++i) out[i] += av[i];
            }
        }
    }
    return py::make_tuple(result, used, tile);
}

PYBIND11_MODULE(_bse_contractions, module) {
    py::capsule capsule = py::module_::import("scipy.linalg.cython_blas")
        .attr("__pyx_capi__")["dgemm"].cast<py::capsule>();
    dgemm = reinterpret_cast<Dgemm>(capsule.get_pointer());
    module.doc() = "Streamed static BSE contractions with deterministic auxiliary-tile OpenMP";
    module.def("openmp_available", &openmp_available);
    module.def("action", &action, py::arg("factors"), py::arg("occupied"), py::arg("ov"),
               py::arg("vectors"), py::arg("nocc"), py::arg("kind"), py::arg("threads") = 1);
}
