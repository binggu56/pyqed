#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cmath>
#include <new>
#include <thread>
#include <vector>

#include "../_dop853.hpp"

namespace {

using complex128 = std::complex<double>;
constexpr complex128 I(0.0, 1.0);

struct ArrayRef {
    PyArrayObject* obj = nullptr;

    ArrayRef() = default;
    ArrayRef(PyObject* value, int typenum, int flags)
        : obj(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(value, typenum, flags))) {}

    ~ArrayRef() {
        Py_XDECREF(obj);
    }

    ArrayRef(const ArrayRef&) = delete;
    ArrayRef& operator=(const ArrayRef&) = delete;

    void reset(PyObject* value, int typenum, int flags) {
        Py_XDECREF(obj);
        obj = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(value, typenum, flags));
    }

    explicit operator bool() const {
        return obj != nullptr;
    }
};

bool require_ndim(PyArrayObject* array, int ndim, const char* name) {
    if (PyArray_NDIM(array) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name, ndim);
        return false;
    }
    return true;
}

inline npy_intp ddos_index(npy_intp nsys, npy_intp iado, npy_intp i, npy_intp j) {
    return (iado * nsys + i) * nsys + j;
}

inline npy_intp op_index(npy_intp nsys, npy_intp i, npy_intp j) {
    return i * nsys + j;
}

inline npy_intp q_index(npy_intp nsys, npy_intp mode, npy_intp i, npy_intp j) {
    return (mode * nsys + i) * nsys + j;
}

constexpr npy_intp kParallelRhsMinAdos = 4096;

int normalize_thread_count(int requested) {
    if (requested == 0) {
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : static_cast<int>(hardware);
    }
    return requested < 1 ? 1 : requested;
}

void rhs_range_raw(
    const complex128* ddos,
    complex128* out,
    const std::int64_t* keys,
    const std::int64_t* minus_index,
    const std::int64_t* plus_index,
    const complex128* expn,
    const complex128* etal,
    const complex128* etar,
    const complex128* etaa,
    const std::int64_t* mode,
    const complex128* h,
    const complex128* q,
    npy_intp nmax,
    npy_intp nsys,
    npy_intp nind,
    npy_intp begin,
    npy_intp end,
    const complex128* decay_rates) {

    for (npy_intp iado = begin; iado < end; ++iado) {
        complex128 decay = decay_rates == nullptr
            ? complex128(0.0, 0.0)
            : decay_rates[iado];
        if (decay_rates == nullptr) {
            for (npy_intp mp = 0; mp < nind; ++mp) {
                decay += static_cast<double>(keys[iado * nind + mp]) * expn[mp];
            }
        }

        for (npy_intp i = 0; i < nsys; ++i) {
            for (npy_intp j = 0; j < nsys; ++j) {
                complex128 value = -decay * ddos[ddos_index(nsys, iado, i, j)];

                complex128 left_h(0.0, 0.0);
                complex128 right_h(0.0, 0.0);
                for (npy_intp k = 0; k < nsys; ++k) {
                    left_h += h[op_index(nsys, i, k)] * ddos[ddos_index(nsys, iado, k, j)];
                    right_h += ddos[ddos_index(nsys, iado, i, k)] * h[op_index(nsys, k, j)];
                }
                value -= I * (left_h - right_h);
                out[ddos_index(nsys, iado, i, j)] = value;
            }
        }

        for (npy_intp mp = 0; mp < nind; ++mp) {
            const std::int64_t occupation = keys[iado * nind + mp];
            const std::int64_t m = mode[mp];

            const std::int64_t minus_pos = minus_index[iado * nind + mp];
            if (occupation > 0 && minus_pos >= 0) {
                const complex128 scale = std::sqrt(static_cast<double>(occupation)) / std::sqrt(etaa[mp]);
                const complex128 left_coef = -I * scale * etal[mp];
                const complex128 right_coef = I * scale * etar[mp];
                for (npy_intp i = 0; i < nsys; ++i) {
                    for (npy_intp j = 0; j < nsys; ++j) {
                        complex128 left_q(0.0, 0.0);
                        complex128 right_q(0.0, 0.0);
                        for (npy_intp k = 0; k < nsys; ++k) {
                            left_q += q[q_index(nsys, m, i, k)] *
                                      ddos[ddos_index(nsys, minus_pos, k, j)];
                            right_q += ddos[ddos_index(nsys, minus_pos, i, k)] *
                                       q[q_index(nsys, m, k, j)];
                        }
                        out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                    }
                }
            }

            const std::int64_t plus_pos = plus_index[iado * nind + mp];
            if (plus_pos >= 0) {
                const complex128 scale = std::sqrt(static_cast<double>(occupation + 1)) * std::sqrt(etaa[mp]);
                const complex128 left_coef = -I * scale;
                const complex128 right_coef = I * scale;
                for (npy_intp i = 0; i < nsys; ++i) {
                    for (npy_intp j = 0; j < nsys; ++j) {
                        complex128 left_q(0.0, 0.0);
                        complex128 right_q(0.0, 0.0);
                        for (npy_intp k = 0; k < nsys; ++k) {
                            left_q += q[q_index(nsys, m, i, k)] *
                                      ddos[ddos_index(nsys, plus_pos, k, j)];
                            right_q += ddos[ddos_index(nsys, plus_pos, i, k)] *
                                       q[q_index(nsys, m, k, j)];
                        }
                        out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                    }
                }
            }
        }
    }
}

void rhs_into_raw(
    const complex128* ddos,
    complex128* out,
    const std::int64_t* keys,
    const std::int64_t* minus_index,
    const std::int64_t* plus_index,
    const complex128* expn,
    const complex128* etal,
    const complex128* etar,
    const complex128* etaa,
    const std::int64_t* mode,
    const complex128* h,
    const complex128* q,
    npy_intp nmax,
    npy_intp nsys,
    npy_intp nind,
    const complex128* decay_rates = nullptr,
    int requested_threads = 1,
    pyqed::dop853::ThreadPool* pool = nullptr) {

    const int thread_count = normalize_thread_count(requested_threads);
    if (thread_count <= 1 || nmax < kParallelRhsMinAdos) {
        rhs_range_raw(
            ddos,
            out,
            keys,
            minus_index,
            plus_index,
            expn,
            etal,
            etar,
            etaa,
            mode,
            h,
            q,
            nmax,
            nsys,
            nind,
            0,
            nmax,
            decay_rates);
        return;
    }

    if (pool != nullptr && pool->thread_count() > 1) {
        pool->for_each(static_cast<std::size_t>(nmax), [&](std::size_t begin, std::size_t end) {
            rhs_range_raw(
                ddos,
                out,
                keys,
                minus_index,
                plus_index,
                expn,
                etal,
                etar,
                etaa,
                mode,
                h,
                q,
                nmax,
                nsys,
                nind,
                static_cast<npy_intp>(begin),
                static_cast<npy_intp>(end),
                decay_rates);
        });
        return;
    }

    const npy_intp worker_count = std::min<npy_intp>(thread_count, nmax);
    const npy_intp chunk = (nmax + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count - 1));

    auto run_range = [&](npy_intp begin, npy_intp end) {
        rhs_range_raw(
            ddos,
            out,
            keys,
            minus_index,
            plus_index,
            expn,
            etal,
            etar,
            etaa,
            mode,
            h,
            q,
            nmax,
            nsys,
            nind,
            begin,
            end,
            decay_rates);
    };

    try {
        for (npy_intp worker = 1; worker < worker_count; ++worker) {
            const npy_intp begin = worker * chunk;
            const npy_intp end = std::min<npy_intp>(nmax, begin + chunk);
            if (begin >= end) {
                break;
            }
            workers.emplace_back(run_range, begin, end);
        }
        run_range(0, std::min<npy_intp>(nmax, chunk));
        for (auto& worker : workers) {
            worker.join();
        }
    } catch (...) {
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        rhs_range_raw(
            ddos,
            out,
            keys,
            minus_index,
            plus_index,
            expn,
            etal,
            etar,
            etaa,
            mode,
            h,
            q,
            nmax,
            nsys,
            nind,
            0,
            nmax,
            decay_rates);
    }
}

void rhs_edges_range_raw(
    const complex128* ddos,
    complex128* out,
    const std::int64_t* lower_offset,
    const std::int64_t* lower_src,
    const std::int64_t* lower_mode,
    const complex128* lower_left,
    const complex128* lower_right,
    const std::int64_t* upper_offset,
    const std::int64_t* upper_src,
    const std::int64_t* upper_mode,
    const complex128* upper_left,
    const complex128* upper_right,
    const complex128* decay_rates,
    const complex128* h,
    const complex128* q,
    npy_intp nsys,
    npy_intp begin,
    npy_intp end) {

    for (npy_intp iado = begin; iado < end; ++iado) {
        const complex128 decay = decay_rates[iado];
        for (npy_intp i = 0; i < nsys; ++i) {
            for (npy_intp j = 0; j < nsys; ++j) {
                complex128 value = -decay * ddos[ddos_index(nsys, iado, i, j)];

                complex128 left_h(0.0, 0.0);
                complex128 right_h(0.0, 0.0);
                for (npy_intp k = 0; k < nsys; ++k) {
                    left_h += h[op_index(nsys, i, k)] * ddos[ddos_index(nsys, iado, k, j)];
                    right_h += ddos[ddos_index(nsys, iado, i, k)] * h[op_index(nsys, k, j)];
                }
                value -= I * (left_h - right_h);
                out[ddos_index(nsys, iado, i, j)] = value;
            }
        }

        for (std::int64_t e = lower_offset[iado]; e < lower_offset[iado + 1]; ++e) {
            const npy_intp src = static_cast<npy_intp>(lower_src[e]);
            const npy_intp m = static_cast<npy_intp>(lower_mode[e]);
            const complex128 left_coef = lower_left[e];
            const complex128 right_coef = lower_right[e];
            for (npy_intp i = 0; i < nsys; ++i) {
                for (npy_intp j = 0; j < nsys; ++j) {
                    complex128 left_q(0.0, 0.0);
                    complex128 right_q(0.0, 0.0);
                    for (npy_intp k = 0; k < nsys; ++k) {
                        left_q += q[q_index(nsys, m, i, k)] *
                                  ddos[ddos_index(nsys, src, k, j)];
                        right_q += ddos[ddos_index(nsys, src, i, k)] *
                                   q[q_index(nsys, m, k, j)];
                    }
                    out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                }
            }
        }

        for (std::int64_t e = upper_offset[iado]; e < upper_offset[iado + 1]; ++e) {
            const npy_intp src = static_cast<npy_intp>(upper_src[e]);
            const npy_intp m = static_cast<npy_intp>(upper_mode[e]);
            const complex128 left_coef = upper_left[e];
            const complex128 right_coef = upper_right[e];
            for (npy_intp i = 0; i < nsys; ++i) {
                for (npy_intp j = 0; j < nsys; ++j) {
                    complex128 left_q(0.0, 0.0);
                    complex128 right_q(0.0, 0.0);
                    for (npy_intp k = 0; k < nsys; ++k) {
                        left_q += q[q_index(nsys, m, i, k)] *
                                  ddos[ddos_index(nsys, src, k, j)];
                        right_q += ddos[ddos_index(nsys, src, i, k)] *
                                   q[q_index(nsys, m, k, j)];
                    }
                    out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                }
            }
        }
    }
}

void rhs_edges_range_nsys2_raw(
    const complex128* ddos,
    complex128* out,
    const std::int64_t* lower_offset,
    const std::int64_t* lower_src,
    const std::int64_t* lower_mode,
    const complex128* lower_left,
    const complex128* lower_right,
    const std::int64_t* upper_offset,
    const std::int64_t* upper_src,
    const std::int64_t* upper_mode,
    const complex128* upper_left,
    const complex128* upper_right,
    const complex128* decay_rates,
    const complex128* h,
    const complex128* q,
    npy_intp begin,
    npy_intp end) {

    const complex128 h00 = h[0];
    const complex128 h01 = h[1];
    const complex128 h10 = h[2];
    const complex128 h11 = h[3];

    for (npy_intp iado = begin; iado < end; ++iado) {
        const complex128* rho = ddos + static_cast<std::size_t>(iado) * 4;
        complex128* dst = out + static_cast<std::size_t>(iado) * 4;
        const complex128 decay = decay_rates[iado];
        const complex128 r00 = rho[0];
        const complex128 r01 = rho[1];
        const complex128 r10 = rho[2];
        const complex128 r11 = rho[3];

        dst[0] = -decay * r00 - I * ((h00 * r00 + h01 * r10) - (r00 * h00 + r01 * h10));
        dst[1] = -decay * r01 - I * ((h00 * r01 + h01 * r11) - (r00 * h01 + r01 * h11));
        dst[2] = -decay * r10 - I * ((h10 * r00 + h11 * r10) - (r10 * h00 + r11 * h10));
        dst[3] = -decay * r11 - I * ((h10 * r01 + h11 * r11) - (r10 * h01 + r11 * h11));

        for (std::int64_t e = lower_offset[iado]; e < lower_offset[iado + 1]; ++e) {
            const complex128* src = ddos + static_cast<std::size_t>(lower_src[e]) * 4;
            const complex128* qq = q + static_cast<std::size_t>(lower_mode[e]) * 4;
            const complex128 q00 = qq[0];
            const complex128 q01 = qq[1];
            const complex128 q10 = qq[2];
            const complex128 q11 = qq[3];
            const complex128 s00 = src[0];
            const complex128 s01 = src[1];
            const complex128 s10 = src[2];
            const complex128 s11 = src[3];
            const complex128 left_coef = lower_left[e];
            const complex128 right_coef = lower_right[e];

            dst[0] += left_coef * (q00 * s00 + q01 * s10) + right_coef * (s00 * q00 + s01 * q10);
            dst[1] += left_coef * (q00 * s01 + q01 * s11) + right_coef * (s00 * q01 + s01 * q11);
            dst[2] += left_coef * (q10 * s00 + q11 * s10) + right_coef * (s10 * q00 + s11 * q10);
            dst[3] += left_coef * (q10 * s01 + q11 * s11) + right_coef * (s10 * q01 + s11 * q11);
        }

        for (std::int64_t e = upper_offset[iado]; e < upper_offset[iado + 1]; ++e) {
            const complex128* src = ddos + static_cast<std::size_t>(upper_src[e]) * 4;
            const complex128* qq = q + static_cast<std::size_t>(upper_mode[e]) * 4;
            const complex128 q00 = qq[0];
            const complex128 q01 = qq[1];
            const complex128 q10 = qq[2];
            const complex128 q11 = qq[3];
            const complex128 s00 = src[0];
            const complex128 s01 = src[1];
            const complex128 s10 = src[2];
            const complex128 s11 = src[3];
            const complex128 left_coef = upper_left[e];
            const complex128 right_coef = upper_right[e];

            dst[0] += left_coef * (q00 * s00 + q01 * s10) + right_coef * (s00 * q00 + s01 * q10);
            dst[1] += left_coef * (q00 * s01 + q01 * s11) + right_coef * (s00 * q01 + s01 * q11);
            dst[2] += left_coef * (q10 * s00 + q11 * s10) + right_coef * (s10 * q00 + s11 * q10);
            dst[3] += left_coef * (q10 * s01 + q11 * s11) + right_coef * (s10 * q01 + s11 * q11);
        }
    }
}

void rhs_into_edges_raw(
    const complex128* ddos,
    complex128* out,
    const std::int64_t* lower_offset,
    const std::int64_t* lower_src,
    const std::int64_t* lower_mode,
    const complex128* lower_left,
    const complex128* lower_right,
    const std::int64_t* upper_offset,
    const std::int64_t* upper_src,
    const std::int64_t* upper_mode,
    const complex128* upper_left,
    const complex128* upper_right,
    const complex128* decay_rates,
    const complex128* h,
    const complex128* q,
    npy_intp nmax,
    npy_intp nsys,
    int requested_threads = 1,
    pyqed::dop853::ThreadPool* pool = nullptr) {

    auto run_range = [&](npy_intp begin, npy_intp end) {
        if (nsys == 2) {
            rhs_edges_range_nsys2_raw(
                ddos,
                out,
                lower_offset,
                lower_src,
                lower_mode,
                lower_left,
                lower_right,
                upper_offset,
                upper_src,
                upper_mode,
                upper_left,
                upper_right,
                decay_rates,
                h,
                q,
                begin,
                end);
        } else {
            rhs_edges_range_raw(
                ddos,
                out,
                lower_offset,
                lower_src,
                lower_mode,
                lower_left,
                lower_right,
                upper_offset,
                upper_src,
                upper_mode,
                upper_left,
                upper_right,
                decay_rates,
                h,
                q,
                nsys,
                begin,
                end);
        }
    };

    const int thread_count = normalize_thread_count(requested_threads);
    if (thread_count <= 1 || nmax < kParallelRhsMinAdos) {
        run_range(0, nmax);
        return;
    }

    if (pool != nullptr && pool->thread_count() > 1) {
        pool->for_each(static_cast<std::size_t>(nmax), [&](std::size_t begin, std::size_t end) {
            run_range(static_cast<npy_intp>(begin), static_cast<npy_intp>(end));
        });
        return;
    }

    const npy_intp worker_count = std::min<npy_intp>(thread_count, nmax);
    const npy_intp chunk = (nmax + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count - 1));
    for (npy_intp worker = 1; worker < worker_count; ++worker) {
        const npy_intp begin = worker * chunk;
        const npy_intp end = std::min<npy_intp>(nmax, begin + chunk);
        if (begin >= end) {
            break;
        }
        workers.emplace_back(run_range, begin, end);
    }
    run_range(0, std::min<npy_intp>(nmax, chunk));
    for (auto& worker : workers) {
        worker.join();
    }
}

bool call_pulse(PyObject* pulse, double t, complex128& value) {
    if (pulse == Py_None) {
        value = complex128(0.0, 0.0);
        return true;
    }
    PyObject* argument = PyFloat_FromDouble(t);
    if (argument == nullptr) {
        return false;
    }
    PyObject* result = PyObject_CallOneArg(pulse, argument);
    Py_DECREF(argument);
    if (result == nullptr) {
        return false;
    }
    const Py_complex scalar = PyComplex_AsCComplex(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
        return false;
    }
    value = complex128(scalar.real, scalar.imag);
    return true;
}

struct HeomRhsEvaluator {
    const std::int64_t* keys;
    const std::int64_t* minus_index;
    const std::int64_t* plus_index;
    const complex128* expn;
    const complex128* etal;
    const complex128* etar;
    const complex128* etaa;
    const std::int64_t* mode;
    const complex128* h;
    const complex128* h_dipole;
    const complex128* q;
    const complex128* q_dipole;
    PyObject* pulse_system;
    PyObject* pulse_coupling;
    npy_intp nmax;
    npy_intp nsys;
    npy_intp nind;
    npy_intp nmode;
    int threads;
    pyqed::dop853::ThreadPool* pool;
    const std::int64_t* lower_offset;
    const std::int64_t* lower_src;
    const std::int64_t* lower_mode;
    const complex128* lower_left;
    const complex128* lower_right;
    const std::int64_t* upper_offset;
    const std::int64_t* upper_src;
    const std::int64_t* upper_mode;
    const complex128* upper_left;
    const complex128* upper_right;
    std::vector<complex128> h_work;
    std::vector<complex128> q_work;
    std::vector<complex128> decay_rates;

    bool has_python_callbacks() const {
        return pulse_system != Py_None || pulse_coupling != Py_None;
    }

    bool evaluate(double t, const complex128* state, complex128* derivative) {
        const complex128* h_t = h;
        const complex128* q_t = q;

        if (pulse_system != Py_None) {
            complex128 pulse;
            if (!call_pulse(pulse_system, t, pulse)) {
                return false;
            }
            const npy_intp size = nsys * nsys;
            for (npy_intp i = 0; i < size; ++i) {
                h_work[static_cast<std::size_t>(i)] = h[i] + pulse * h_dipole[i];
            }
            h_t = h_work.data();
        }
        if (pulse_coupling != Py_None) {
            complex128 pulse;
            if (!call_pulse(pulse_coupling, t, pulse)) {
                return false;
            }
            const npy_intp size = nmode * nsys * nsys;
            for (npy_intp i = 0; i < size; ++i) {
                q_work[static_cast<std::size_t>(i)] = q[i] + pulse * q_dipole[i];
            }
            q_t = q_work.data();
        }

        if (lower_offset != nullptr) {
            rhs_into_edges_raw(
                state,
                derivative,
                lower_offset,
                lower_src,
                lower_mode,
                lower_left,
                lower_right,
                upper_offset,
                upper_src,
                upper_mode,
                upper_left,
                upper_right,
                decay_rates.data(),
                h_t,
                q_t,
                nmax,
                nsys,
                threads,
                pool);
        } else {
            rhs_into_raw(
                state,
                derivative,
                keys,
                minus_index,
                plus_index,
                expn,
                etal,
                etar,
                etaa,
                mode,
                h_t,
                q_t,
                nmax,
                nsys,
                nind,
                decay_rates.data(),
                threads,
                pool);
        }
        return true;
    }
};

PyObject* available(PyObject*, PyObject*) {
    Py_RETURN_TRUE;
}

PyObject* rhs_by_index(PyObject*, PyObject* args) {
    PyObject* ddos_obj = nullptr;
    PyObject* keys_obj = nullptr;
    PyObject* minus_obj = nullptr;
    PyObject* plus_obj = nullptr;
    PyObject* expn_obj = nullptr;
    PyObject* etal_obj = nullptr;
    PyObject* etar_obj = nullptr;
    PyObject* etaa_obj = nullptr;
    PyObject* mode_obj = nullptr;
    PyObject* h_obj = nullptr;
    PyObject* q_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOO",
            &ddos_obj,
            &keys_obj,
            &minus_obj,
            &plus_obj,
            &expn_obj,
            &etal_obj,
            &etar_obj,
            &etaa_obj,
            &mode_obj,
            &h_obj,
            &q_obj)) {
        return nullptr;
    }

    ArrayRef ddos_ref(ddos_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef keys_ref(keys_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef minus_ref(minus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef plus_ref(plus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef expn_ref(expn_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etal_ref(etal_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etar_ref(etar_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etaa_ref(etaa_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef mode_ref(mode_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_ref(h_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_ref(q_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!ddos_ref || !keys_ref || !minus_ref || !plus_ref || !expn_ref || !etal_ref ||
        !etar_ref || !etaa_ref || !mode_ref || !h_ref || !q_ref) {
        return nullptr;
    }

    if (!require_ndim(ddos_ref.obj, 3, "ddos") ||
        !require_ndim(keys_ref.obj, 2, "keys") ||
        !require_ndim(minus_ref.obj, 2, "minus_index") ||
        !require_ndim(plus_ref.obj, 2, "plus_index") ||
        !require_ndim(expn_ref.obj, 1, "expn") ||
        !require_ndim(etal_ref.obj, 1, "etal") ||
        !require_ndim(etar_ref.obj, 1, "etar") ||
        !require_ndim(etaa_ref.obj, 1, "etaa") ||
        !require_ndim(mode_ref.obj, 1, "mode") ||
        !require_ndim(h_ref.obj, 2, "H") ||
        !require_ndim(q_ref.obj, 3, "Q")) {
        return nullptr;
    }

    const npy_intp nmax = PyArray_DIM(ddos_ref.obj, 0);
    const npy_intp nsys = PyArray_DIM(ddos_ref.obj, 1);
    const npy_intp nind = PyArray_DIM(keys_ref.obj, 1);
    const npy_intp nmode = PyArray_DIM(q_ref.obj, 0);

    if (PyArray_DIM(ddos_ref.obj, 2) != nsys ||
        PyArray_DIM(keys_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 1) != nind ||
        PyArray_DIM(plus_ref.obj, 0) != nmax ||
        PyArray_DIM(plus_ref.obj, 1) != nind ||
        PyArray_DIM(expn_ref.obj, 0) != nind ||
        PyArray_DIM(etal_ref.obj, 0) != nind ||
        PyArray_DIM(etar_ref.obj, 0) != nind ||
        PyArray_DIM(etaa_ref.obj, 0) != nind ||
        PyArray_DIM(mode_ref.obj, 0) != nind ||
        PyArray_DIM(h_ref.obj, 0) != nsys ||
        PyArray_DIM(h_ref.obj, 1) != nsys ||
        PyArray_DIM(q_ref.obj, 1) != nsys ||
        PyArray_DIM(q_ref.obj, 2) != nsys) {
        PyErr_SetString(PyExc_ValueError, "HEOM native RHS input shapes are inconsistent");
        return nullptr;
    }

    npy_intp dims[3] = {nmax, nsys, nsys};
    PyObject* out_obj = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    if (out_obj == nullptr) {
        return nullptr;
    }
    auto* out_array = reinterpret_cast<PyArrayObject*>(out_obj);

    const auto* ddos = reinterpret_cast<const complex128*>(PyArray_DATA(ddos_ref.obj));
    const auto* expn = reinterpret_cast<const complex128*>(PyArray_DATA(expn_ref.obj));
    const auto* etal = reinterpret_cast<const complex128*>(PyArray_DATA(etal_ref.obj));
    const auto* etar = reinterpret_cast<const complex128*>(PyArray_DATA(etar_ref.obj));
    const auto* etaa = reinterpret_cast<const complex128*>(PyArray_DATA(etaa_ref.obj));
    const auto* h = reinterpret_cast<const complex128*>(PyArray_DATA(h_ref.obj));
    const auto* q = reinterpret_cast<const complex128*>(PyArray_DATA(q_ref.obj));
    const auto* keys = reinterpret_cast<const std::int64_t*>(PyArray_DATA(keys_ref.obj));
    const auto* minus_index = reinterpret_cast<const std::int64_t*>(PyArray_DATA(minus_ref.obj));
    const auto* plus_index = reinterpret_cast<const std::int64_t*>(PyArray_DATA(plus_ref.obj));
    const auto* mode = reinterpret_cast<const std::int64_t*>(PyArray_DATA(mode_ref.obj));
    auto* out = reinterpret_cast<complex128*>(PyArray_DATA(out_array));

    for (npy_intp iado = 0; iado < nmax; ++iado) {
        complex128 decay(0.0, 0.0);
        for (npy_intp mp = 0; mp < nind; ++mp) {
            decay += static_cast<double>(keys[iado * nind + mp]) * expn[mp];
        }

        for (npy_intp i = 0; i < nsys; ++i) {
            for (npy_intp j = 0; j < nsys; ++j) {
                complex128 value = -decay * ddos[ddos_index(nsys, iado, i, j)];

                complex128 left_h(0.0, 0.0);
                complex128 right_h(0.0, 0.0);
                for (npy_intp k = 0; k < nsys; ++k) {
                    left_h += h[op_index(nsys, i, k)] * ddos[ddos_index(nsys, iado, k, j)];
                    right_h += ddos[ddos_index(nsys, iado, i, k)] * h[op_index(nsys, k, j)];
                }
                value -= I * (left_h - right_h);
                out[ddos_index(nsys, iado, i, j)] = value;
            }
        }

        for (npy_intp mp = 0; mp < nind; ++mp) {
            const std::int64_t occupation = keys[iado * nind + mp];
            const std::int64_t m = mode[mp];
            if (m < 0 || m >= nmode) {
                Py_DECREF(out_obj);
                PyErr_SetString(PyExc_ValueError, "mode index out of range for Q");
                return nullptr;
            }

            const std::int64_t minus_pos = minus_index[iado * nind + mp];
            if (occupation > 0 && minus_pos >= 0) {
                const complex128 scale = std::sqrt(static_cast<double>(occupation)) / std::sqrt(etaa[mp]);
                const complex128 left_coef = -I * scale * etal[mp];
                const complex128 right_coef = I * scale * etar[mp];
                for (npy_intp i = 0; i < nsys; ++i) {
                    for (npy_intp j = 0; j < nsys; ++j) {
                        complex128 left_q(0.0, 0.0);
                        complex128 right_q(0.0, 0.0);
                        for (npy_intp k = 0; k < nsys; ++k) {
                            left_q += q[q_index(nsys, m, i, k)] *
                                      ddos[ddos_index(nsys, minus_pos, k, j)];
                            right_q += ddos[ddos_index(nsys, minus_pos, i, k)] *
                                       q[q_index(nsys, m, k, j)];
                        }
                        out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                    }
                }
            }

            const std::int64_t plus_pos = plus_index[iado * nind + mp];
            if (plus_pos >= 0) {
                const complex128 scale = std::sqrt(static_cast<double>(occupation + 1)) * std::sqrt(etaa[mp]);
                const complex128 left_coef = -I * scale;
                const complex128 right_coef = I * scale;
                for (npy_intp i = 0; i < nsys; ++i) {
                    for (npy_intp j = 0; j < nsys; ++j) {
                        complex128 left_q(0.0, 0.0);
                        complex128 right_q(0.0, 0.0);
                        for (npy_intp k = 0; k < nsys; ++k) {
                            left_q += q[q_index(nsys, m, i, k)] *
                                      ddos[ddos_index(nsys, plus_pos, k, j)];
                            right_q += ddos[ddos_index(nsys, plus_pos, i, k)] *
                                       q[q_index(nsys, m, k, j)];
                        }
                        out[ddos_index(nsys, iado, i, j)] += left_coef * left_q + right_coef * right_q;
                    }
                }
            }
        }
    }

    return out_obj;
}

PyObject* dop853_by_index_impl(PyObject* args, bool accepted_step_output) {
    PyObject* ddos_obj = nullptr;
    PyObject* keys_obj = nullptr;
    PyObject* minus_obj = nullptr;
    PyObject* plus_obj = nullptr;
    PyObject* expn_obj = nullptr;
    PyObject* etal_obj = nullptr;
    PyObject* etar_obj = nullptr;
    PyObject* etaa_obj = nullptr;
    PyObject* mode_obj = nullptr;
    PyObject* h_obj = nullptr;
    PyObject* h_dipole_obj = nullptr;
    PyObject* q_obj = nullptr;
    PyObject* q_dipole_obj = nullptr;
    PyObject* pulse_system = nullptr;
    PyObject* pulse_coupling = nullptr;
    PyObject* t_eval_obj = nullptr;
    double t0 = 0.0;
    double t_bound_input = 0.0;
    double rtol = 0.0;
    double atol = 0.0;
    int threads = 1;
    PyObject* lower_offset_obj = nullptr;
    PyObject* lower_src_obj = nullptr;
    PyObject* lower_mode_obj = nullptr;
    PyObject* lower_left_obj = nullptr;
    PyObject* lower_right_obj = nullptr;
    PyObject* upper_offset_obj = nullptr;
    PyObject* upper_src_obj = nullptr;
    PyObject* upper_mode_obj = nullptr;
    PyObject* upper_left_obj = nullptr;
    PyObject* upper_right_obj = nullptr;

    if (accepted_step_output) {
        if (!PyArg_ParseTuple(
                args,
                "OOOOOOOOOOOOOOOdddd|iOOOOOOOOOO",
                &ddos_obj,
                &keys_obj,
                &minus_obj,
                &plus_obj,
                &expn_obj,
                &etal_obj,
                &etar_obj,
                &etaa_obj,
                &mode_obj,
                &h_obj,
                &h_dipole_obj,
                &q_obj,
                &q_dipole_obj,
                &pulse_system,
                &pulse_coupling,
                &t0,
                &t_bound_input,
                &rtol,
                &atol,
                &threads,
                &lower_offset_obj,
                &lower_src_obj,
                &lower_mode_obj,
                &lower_left_obj,
                &lower_right_obj,
                &upper_offset_obj,
                &upper_src_obj,
                &upper_mode_obj,
                &upper_left_obj,
                &upper_right_obj)) {
            return nullptr;
        }
    } else {
        if (!PyArg_ParseTuple(
                args,
                "OOOOOOOOOOOOOOOOdd|iOOOOOOOOOO",
                &ddos_obj,
                &keys_obj,
                &minus_obj,
                &plus_obj,
                &expn_obj,
                &etal_obj,
                &etar_obj,
                &etaa_obj,
                &mode_obj,
                &h_obj,
                &h_dipole_obj,
                &q_obj,
                &q_dipole_obj,
                &pulse_system,
                &pulse_coupling,
                &t_eval_obj,
                &rtol,
                &atol,
                &threads,
                &lower_offset_obj,
                &lower_src_obj,
                &lower_mode_obj,
                &lower_left_obj,
                &lower_right_obj,
                &upper_offset_obj,
                &upper_src_obj,
                &upper_mode_obj,
                &upper_left_obj,
                &upper_right_obj)) {
            return nullptr;
        }
    }

    const int write_flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
    ArrayRef ddos_ref(ddos_obj, NPY_COMPLEX128, write_flags);
    ArrayRef keys_ref(keys_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef minus_ref(minus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef plus_ref(plus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef expn_ref(expn_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etal_ref(etal_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etar_ref(etar_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etaa_ref(etaa_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef mode_ref(mode_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_ref(h_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_dipole_ref(h_dipole_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_ref(q_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_dipole_ref(q_dipole_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef t_eval_ref;
    if (!accepted_step_output) {
        t_eval_ref.reset(t_eval_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }
    ArrayRef lower_offset_ref;
    ArrayRef lower_src_ref;
    ArrayRef lower_mode_ref;
    ArrayRef lower_left_ref;
    ArrayRef lower_right_ref;
    ArrayRef upper_offset_ref;
    ArrayRef upper_src_ref;
    ArrayRef upper_mode_ref;
    ArrayRef upper_left_ref;
    ArrayRef upper_right_ref;
    if (!ddos_ref || !keys_ref || !minus_ref || !plus_ref || !expn_ref || !etal_ref ||
        !etar_ref || !etaa_ref || !mode_ref || !h_ref || !h_dipole_ref || !q_ref ||
        !q_dipole_ref || (!accepted_step_output && !t_eval_ref)) {
        return nullptr;
    }
    if (reinterpret_cast<PyObject*>(ddos_ref.obj) != ddos_obj) {
        PyErr_SetString(PyExc_ValueError, "ddos must be a writable C-contiguous complex128 array");
        return nullptr;
    }
    if ((pulse_system != Py_None && !PyCallable_Check(pulse_system)) ||
        (pulse_coupling != Py_None && !PyCallable_Check(pulse_coupling))) {
        PyErr_SetString(PyExc_TypeError, "pulse arguments must be callable or None");
        return nullptr;
    }
    if (!std::isfinite(rtol) || !std::isfinite(atol) || rtol <= 0.0 || atol <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "rtol and atol must be finite and positive");
        return nullptr;
    }
    if (threads < 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be non-negative");
        return nullptr;
    }

    if (!require_ndim(ddos_ref.obj, 3, "ddos") ||
        !require_ndim(keys_ref.obj, 2, "keys") ||
        !require_ndim(minus_ref.obj, 2, "minus_index") ||
        !require_ndim(plus_ref.obj, 2, "plus_index") ||
        !require_ndim(expn_ref.obj, 1, "expn") ||
        !require_ndim(etal_ref.obj, 1, "etal") ||
        !require_ndim(etar_ref.obj, 1, "etar") ||
        !require_ndim(etaa_ref.obj, 1, "etaa") ||
        !require_ndim(mode_ref.obj, 1, "mode") ||
        !require_ndim(h_ref.obj, 2, "H") ||
        !require_ndim(h_dipole_ref.obj, 2, "H_dipole") ||
        !require_ndim(q_ref.obj, 3, "Q") ||
        !require_ndim(q_dipole_ref.obj, 3, "Q_dipole") ||
        (!accepted_step_output && !require_ndim(t_eval_ref.obj, 1, "t_eval"))) {
        return nullptr;
    }

    const npy_intp nmax = PyArray_DIM(ddos_ref.obj, 0);
    const npy_intp nsys = PyArray_DIM(ddos_ref.obj, 1);
    const npy_intp nind = PyArray_DIM(keys_ref.obj, 1);
    const npy_intp nmode = PyArray_DIM(q_ref.obj, 0);
    const npy_intp n_times = accepted_step_output ? 2 : PyArray_DIM(t_eval_ref.obj, 0);
    if (nmax <= 0 || nsys <= 0 || nind <= 0 || nmode <= 0 || n_times <= 0 ||
        PyArray_DIM(ddos_ref.obj, 2) != nsys ||
        PyArray_DIM(keys_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 1) != nind ||
        PyArray_DIM(plus_ref.obj, 0) != nmax ||
        PyArray_DIM(plus_ref.obj, 1) != nind ||
        PyArray_DIM(expn_ref.obj, 0) != nind ||
        PyArray_DIM(etal_ref.obj, 0) != nind ||
        PyArray_DIM(etar_ref.obj, 0) != nind ||
        PyArray_DIM(etaa_ref.obj, 0) != nind ||
        PyArray_DIM(mode_ref.obj, 0) != nind ||
        PyArray_DIM(h_ref.obj, 0) != nsys ||
        PyArray_DIM(h_ref.obj, 1) != nsys ||
        PyArray_DIM(h_dipole_ref.obj, 0) != nsys ||
        PyArray_DIM(h_dipole_ref.obj, 1) != nsys ||
        PyArray_DIM(q_ref.obj, 1) != nsys ||
        PyArray_DIM(q_ref.obj, 2) != nsys ||
        PyArray_DIM(q_dipole_ref.obj, 0) != nmode ||
        PyArray_DIM(q_dipole_ref.obj, 1) != nsys ||
        PyArray_DIM(q_dipole_ref.obj, 2) != nsys) {
        PyErr_SetString(PyExc_ValueError, "HEOM native DOP853 input shapes are inconsistent");
        return nullptr;
    }

    const auto* mode = reinterpret_cast<const std::int64_t*>(PyArray_DATA(mode_ref.obj));
    for (npy_intp i = 0; i < nind; ++i) {
        if (mode[i] < 0 || mode[i] >= nmode) {
            PyErr_SetString(PyExc_ValueError, "mode index out of range for Q");
            return nullptr;
        }
    }
    double adaptive_t_eval[2] = {t0, t_bound_input};
    const auto* t_eval = accepted_step_output
        ? adaptive_t_eval
        : reinterpret_cast<const double*>(PyArray_DATA(t_eval_ref.obj));
    for (npy_intp i = 0; i < n_times; ++i) {
        if (!std::isfinite(t_eval[i]) || (i > 0 && t_eval[i] <= t_eval[i - 1])) {
            PyErr_SetString(
                PyExc_ValueError,
                accepted_step_output
                    ? "t_span must be finite and strictly increasing"
                    : "t_eval must be finite and strictly increasing");
            return nullptr;
        }
    }

    const bool any_edge_arg =
        lower_offset_obj != nullptr || lower_src_obj != nullptr || lower_mode_obj != nullptr ||
        lower_left_obj != nullptr || lower_right_obj != nullptr || upper_offset_obj != nullptr ||
        upper_src_obj != nullptr || upper_mode_obj != nullptr || upper_left_obj != nullptr ||
        upper_right_obj != nullptr;
    const bool all_edge_args =
        lower_offset_obj != nullptr && lower_src_obj != nullptr && lower_mode_obj != nullptr &&
        lower_left_obj != nullptr && lower_right_obj != nullptr && upper_offset_obj != nullptr &&
        upper_src_obj != nullptr && upper_mode_obj != nullptr && upper_left_obj != nullptr &&
        upper_right_obj != nullptr;
    if (any_edge_arg && !all_edge_args) {
        PyErr_SetString(PyExc_ValueError, "all ten HEOM edge-table arrays must be provided together");
        return nullptr;
    }

    const std::int64_t* lower_offset = nullptr;
    const std::int64_t* lower_src = nullptr;
    const std::int64_t* lower_mode = nullptr;
    const complex128* lower_left = nullptr;
    const complex128* lower_right = nullptr;
    const std::int64_t* upper_offset = nullptr;
    const std::int64_t* upper_src = nullptr;
    const std::int64_t* upper_mode = nullptr;
    const complex128* upper_left = nullptr;
    const complex128* upper_right = nullptr;

    if (all_edge_args) {
        lower_offset_ref.reset(lower_offset_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        lower_src_ref.reset(lower_src_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        lower_mode_ref.reset(lower_mode_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        lower_left_ref.reset(lower_left_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        lower_right_ref.reset(lower_right_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        upper_offset_ref.reset(upper_offset_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        upper_src_ref.reset(upper_src_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        upper_mode_ref.reset(upper_mode_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
        upper_left_ref.reset(upper_left_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        upper_right_ref.reset(upper_right_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (!lower_offset_ref || !lower_src_ref || !lower_mode_ref || !lower_left_ref ||
            !lower_right_ref || !upper_offset_ref || !upper_src_ref || !upper_mode_ref ||
            !upper_left_ref || !upper_right_ref) {
            return nullptr;
        }
        if (!require_ndim(lower_offset_ref.obj, 1, "lower_offset") ||
            !require_ndim(lower_src_ref.obj, 1, "lower_src") ||
            !require_ndim(lower_mode_ref.obj, 1, "lower_mode") ||
            !require_ndim(lower_left_ref.obj, 1, "lower_left") ||
            !require_ndim(lower_right_ref.obj, 1, "lower_right") ||
            !require_ndim(upper_offset_ref.obj, 1, "upper_offset") ||
            !require_ndim(upper_src_ref.obj, 1, "upper_src") ||
            !require_ndim(upper_mode_ref.obj, 1, "upper_mode") ||
            !require_ndim(upper_left_ref.obj, 1, "upper_left") ||
            !require_ndim(upper_right_ref.obj, 1, "upper_right")) {
            return nullptr;
        }
        if (PyArray_DIM(lower_offset_ref.obj, 0) != nmax + 1 ||
            PyArray_DIM(upper_offset_ref.obj, 0) != nmax + 1) {
            PyErr_SetString(PyExc_ValueError, "edge offsets must have shape (nado + 1,)");
            return nullptr;
        }
        lower_offset = reinterpret_cast<const std::int64_t*>(PyArray_DATA(lower_offset_ref.obj));
        upper_offset = reinterpret_cast<const std::int64_t*>(PyArray_DATA(upper_offset_ref.obj));
        const std::int64_t n_lower = lower_offset[nmax];
        const std::int64_t n_upper = upper_offset[nmax];
        if (lower_offset[0] != 0 || upper_offset[0] != 0 || n_lower < 0 || n_upper < 0) {
            PyErr_SetString(PyExc_ValueError, "edge offsets are invalid");
            return nullptr;
        }
        for (npy_intp i = 0; i < nmax; ++i) {
            if (lower_offset[i] > lower_offset[i + 1] || upper_offset[i] > upper_offset[i + 1]) {
                PyErr_SetString(PyExc_ValueError, "edge offsets must be nondecreasing");
                return nullptr;
            }
        }
        if (PyArray_DIM(lower_src_ref.obj, 0) != n_lower ||
            PyArray_DIM(lower_mode_ref.obj, 0) != n_lower ||
            PyArray_DIM(lower_left_ref.obj, 0) != n_lower ||
            PyArray_DIM(lower_right_ref.obj, 0) != n_lower ||
            PyArray_DIM(upper_src_ref.obj, 0) != n_upper ||
            PyArray_DIM(upper_mode_ref.obj, 0) != n_upper ||
            PyArray_DIM(upper_left_ref.obj, 0) != n_upper ||
            PyArray_DIM(upper_right_ref.obj, 0) != n_upper) {
            PyErr_SetString(PyExc_ValueError, "edge table shapes are inconsistent");
            return nullptr;
        }
        lower_src = reinterpret_cast<const std::int64_t*>(PyArray_DATA(lower_src_ref.obj));
        lower_mode = reinterpret_cast<const std::int64_t*>(PyArray_DATA(lower_mode_ref.obj));
        lower_left = reinterpret_cast<const complex128*>(PyArray_DATA(lower_left_ref.obj));
        lower_right = reinterpret_cast<const complex128*>(PyArray_DATA(lower_right_ref.obj));
        upper_src = reinterpret_cast<const std::int64_t*>(PyArray_DATA(upper_src_ref.obj));
        upper_mode = reinterpret_cast<const std::int64_t*>(PyArray_DATA(upper_mode_ref.obj));
        upper_left = reinterpret_cast<const complex128*>(PyArray_DATA(upper_left_ref.obj));
        upper_right = reinterpret_cast<const complex128*>(PyArray_DATA(upper_right_ref.obj));
        for (std::int64_t i = 0; i < n_lower; ++i) {
            if (lower_src[i] < 0 || lower_src[i] >= nmax || lower_mode[i] < 0 || lower_mode[i] >= nmode) {
                PyErr_SetString(PyExc_ValueError, "lower edge table contains an out-of-range index");
                return nullptr;
            }
        }
        for (std::int64_t i = 0; i < n_upper; ++i) {
            if (upper_src[i] < 0 || upper_src[i] >= nmax || upper_mode[i] < 0 || upper_mode[i] >= nmode) {
                PyErr_SetString(PyExc_ValueError, "upper edge table contains an out-of-range index");
                return nullptr;
            }
        }
    }

    PyObject* output_obj = nullptr;
    PyArrayObject* output_array = nullptr;
    if (!accepted_step_output) {
        npy_intp output_dims[3] = {n_times, nsys, nsys};
        output_obj = PyArray_SimpleNew(3, output_dims, NPY_COMPLEX128);
        if (output_obj == nullptr) {
            return nullptr;
        }
        output_array = reinterpret_cast<PyArrayObject*>(output_obj);
    }

    HeomRhsEvaluator evaluator{
        reinterpret_cast<const std::int64_t*>(PyArray_DATA(keys_ref.obj)),
        reinterpret_cast<const std::int64_t*>(PyArray_DATA(minus_ref.obj)),
        reinterpret_cast<const std::int64_t*>(PyArray_DATA(plus_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(expn_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(etal_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(etar_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(etaa_ref.obj)),
        mode,
        reinterpret_cast<const complex128*>(PyArray_DATA(h_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(h_dipole_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(q_ref.obj)),
        reinterpret_cast<const complex128*>(PyArray_DATA(q_dipole_ref.obj)),
        pulse_system,
        pulse_coupling,
        nmax,
        nsys,
        nind,
        nmode,
        normalize_thread_count(threads),
        nullptr,
        lower_offset,
        lower_src,
        lower_mode,
        lower_left,
        lower_right,
        upper_offset,
        upper_src,
        upper_mode,
        upper_left,
        upper_right,
        {},
        {},
        {},
    };
    try {
        evaluator.decay_rates.resize(static_cast<std::size_t>(nmax));
        for (npy_intp iado = 0; iado < nmax; ++iado) {
            complex128 decay(0.0, 0.0);
            for (npy_intp mp = 0; mp < nind; ++mp) {
                decay += static_cast<double>(evaluator.keys[iado * nind + mp]) * evaluator.expn[mp];
            }
            evaluator.decay_rates[static_cast<std::size_t>(iado)] = decay;
        }
        if (pulse_system != Py_None) {
            evaluator.h_work.resize(static_cast<std::size_t>(nsys * nsys));
        }
        if (pulse_coupling != Py_None) {
            evaluator.q_work.resize(static_cast<std::size_t>(nmode * nsys * nsys));
        }
    } catch (const std::bad_alloc&) {
        Py_XDECREF(output_obj);
        PyErr_NoMemory();
        return nullptr;
    }

    auto* state = reinterpret_cast<complex128*>(PyArray_DATA(ddos_ref.obj));
    auto* reduced_states = accepted_step_output
        ? nullptr
        : reinterpret_cast<complex128*>(PyArray_DATA(output_array));
    const std::size_t state_size = static_cast<std::size_t>(nmax * nsys * nsys);
    const std::size_t reduced_size = static_cast<std::size_t>(nsys * nsys);
    const bool use_pool = evaluator.threads > 1 && (
        nmax >= kParallelRhsMinAdos ||
        state_size >= pyqed::dop853::parallel_vector_min_size);
    pyqed::dop853::ThreadPool pool(use_pool ? evaluator.threads : 1);
    evaluator.pool = &pool;
    std::vector<double> adaptive_times;
    std::vector<complex128> adaptive_reduced_states;
    if (accepted_step_output) {
        try {
            adaptive_times.reserve(1024);
            adaptive_reduced_states.reserve(1024 * reduced_size);
        } catch (const std::bad_alloc&) {
            PyErr_NoMemory();
            return nullptr;
        }
    }
    auto observe = [&](
                       std::size_t output_index,
                       double time,
                       const pyqed::dop853::OutputView& view) {
        complex128* destination = nullptr;
        if (accepted_step_output) {
            adaptive_times.push_back(time);
            const std::size_t offset = adaptive_reduced_states.size();
            adaptive_reduced_states.resize(offset + reduced_size);
            destination = adaptive_reduced_states.data() + offset;
        } else {
            destination = reduced_states + output_index * reduced_size;
        }
        for (std::size_t i = 0; i < reduced_size; ++i) {
            destination[i] = view[i];
        }
    };
    pyqed::dop853::Stats stats;
    try {
        if (accepted_step_output || evaluator.has_python_callbacks()) {
            stats = pyqed::dop853::integrate(
                evaluator,
                state,
                state_size,
                t_eval,
                static_cast<std::size_t>(n_times),
                observe,
                rtol,
                atol,
                evaluator.threads,
                &pool,
                accepted_step_output);
        } else {
            Py_BEGIN_ALLOW_THREADS
            stats = pyqed::dop853::integrate(
                evaluator,
                state,
                state_size,
                t_eval,
                static_cast<std::size_t>(n_times),
                observe,
                rtol,
                atol,
                evaluator.threads,
                &pool,
                false);
            Py_END_ALLOW_THREADS
        }
    } catch (const std::bad_alloc&) {
        Py_XDECREF(output_obj);
        PyErr_NoMemory();
        return nullptr;
    }

    if (!stats.success) {
        Py_XDECREF(output_obj);
        if (PyErr_Occurred()) {
            return nullptr;
        }
        PyErr_SetString(PyExc_RuntimeError, stats.message.c_str());
        return nullptr;
    }

    if (accepted_step_output) {
        const npy_intp adaptive_n_times = static_cast<npy_intp>(adaptive_times.size());
        npy_intp time_dims[1] = {adaptive_n_times};
        PyObject* times_obj = PyArray_SimpleNew(1, time_dims, NPY_DOUBLE);
        if (times_obj == nullptr) {
            return nullptr;
        }
        npy_intp output_dims[3] = {adaptive_n_times, nsys, nsys};
        output_obj = PyArray_SimpleNew(3, output_dims, NPY_COMPLEX128);
        if (output_obj == nullptr) {
            Py_DECREF(times_obj);
            return nullptr;
        }
        auto* times_array = reinterpret_cast<PyArrayObject*>(times_obj);
        output_array = reinterpret_cast<PyArrayObject*>(output_obj);
        std::copy(
            adaptive_times.begin(),
            adaptive_times.end(),
            reinterpret_cast<double*>(PyArray_DATA(times_array)));
        std::copy(
            adaptive_reduced_states.begin(),
            adaptive_reduced_states.end(),
            reinterpret_cast<complex128*>(PyArray_DATA(output_array)));

        PyObject* result = PyTuple_New(5);
        if (result == nullptr) {
            Py_DECREF(times_obj);
            Py_DECREF(output_obj);
            return nullptr;
        }
        PyTuple_SET_ITEM(result, 0, times_obj);
        PyTuple_SET_ITEM(result, 1, output_obj);
        PyTuple_SET_ITEM(result, 2, PyLong_FromLongLong(stats.nfev));
        PyTuple_SET_ITEM(result, 3, PyLong_FromLongLong(stats.n_steps));
        PyTuple_SET_ITEM(result, 4, PyLong_FromLongLong(stats.n_rejected));
        return result;
    }

    PyObject* result = PyTuple_New(4);
    if (result == nullptr) {
        Py_DECREF(output_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, output_obj);
    PyTuple_SET_ITEM(result, 1, PyLong_FromLongLong(stats.nfev));
    PyTuple_SET_ITEM(result, 2, PyLong_FromLongLong(stats.n_steps));
    PyTuple_SET_ITEM(result, 3, PyLong_FromLongLong(stats.n_rejected));
    return result;
}

PyObject* dop853_by_index(PyObject*, PyObject* args) {
    return dop853_by_index_impl(args, false);
}

PyObject* dop853_adaptive_by_index(PyObject*, PyObject* args) {
    return dop853_by_index_impl(args, true);
}

PyObject* rk4_step_by_index(PyObject*, PyObject* args) {
    PyObject* ddos_obj = nullptr;
    PyObject* keys_obj = nullptr;
    PyObject* minus_obj = nullptr;
    PyObject* plus_obj = nullptr;
    PyObject* expn_obj = nullptr;
    PyObject* etal_obj = nullptr;
    PyObject* etar_obj = nullptr;
    PyObject* etaa_obj = nullptr;
    PyObject* mode_obj = nullptr;
    PyObject* h0_obj = nullptr;
    PyObject* q0_obj = nullptr;
    PyObject* hhalf_obj = nullptr;
    PyObject* qhalf_obj = nullptr;
    PyObject* h1_obj = nullptr;
    PyObject* q1_obj = nullptr;
    PyObject* k1_obj = nullptr;
    PyObject* k2_obj = nullptr;
    PyObject* k3_obj = nullptr;
    PyObject* k4_obj = nullptr;
    PyObject* work_obj = nullptr;
    PyObject* decay_obj = nullptr;
    double dt = 0.0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOOOOd|OOOOOO",
            &ddos_obj,
            &keys_obj,
            &minus_obj,
            &plus_obj,
            &expn_obj,
            &etal_obj,
            &etar_obj,
            &etaa_obj,
            &mode_obj,
            &h0_obj,
            &q0_obj,
            &hhalf_obj,
            &qhalf_obj,
            &h1_obj,
            &q1_obj,
            &dt,
            &k1_obj,
            &k2_obj,
            &k3_obj,
            &k4_obj,
            &work_obj,
            &decay_obj)) {
        return nullptr;
    }

    ArrayRef ddos_ref(
        ddos_obj,
        NPY_COMPLEX128,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    ArrayRef keys_ref(keys_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef minus_ref(minus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef plus_ref(plus_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef expn_ref(expn_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etal_ref(etal_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etar_ref(etar_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef etaa_ref(etaa_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef mode_ref(mode_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef h0_ref(h0_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef q0_ref(q0_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef hhalf_ref(hhalf_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef qhalf_ref(qhalf_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef h1_ref(h1_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef q1_ref(q1_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!ddos_ref || !keys_ref || !minus_ref || !plus_ref || !expn_ref || !etal_ref ||
        !etar_ref || !etaa_ref || !mode_ref || !h0_ref || !q0_ref || !hhalf_ref ||
        !qhalf_ref || !h1_ref || !q1_ref) {
        return nullptr;
    }
    if (reinterpret_cast<PyObject*>(ddos_ref.obj) != ddos_obj) {
        PyErr_SetString(PyExc_ValueError, "ddos must be a writable C-contiguous complex128 array");
        return nullptr;
    }

    if (!require_ndim(ddos_ref.obj, 3, "ddos") ||
        !require_ndim(keys_ref.obj, 2, "keys") ||
        !require_ndim(minus_ref.obj, 2, "minus_index") ||
        !require_ndim(plus_ref.obj, 2, "plus_index") ||
        !require_ndim(expn_ref.obj, 1, "expn") ||
        !require_ndim(etal_ref.obj, 1, "etal") ||
        !require_ndim(etar_ref.obj, 1, "etar") ||
        !require_ndim(etaa_ref.obj, 1, "etaa") ||
        !require_ndim(mode_ref.obj, 1, "mode") ||
        !require_ndim(h0_ref.obj, 2, "H0") ||
        !require_ndim(q0_ref.obj, 3, "Q0") ||
        !require_ndim(hhalf_ref.obj, 2, "Hhalf") ||
        !require_ndim(qhalf_ref.obj, 3, "Qhalf") ||
        !require_ndim(h1_ref.obj, 2, "H1") ||
        !require_ndim(q1_ref.obj, 3, "Q1")) {
        return nullptr;
    }

    const npy_intp nmax = PyArray_DIM(ddos_ref.obj, 0);
    const npy_intp nsys = PyArray_DIM(ddos_ref.obj, 1);
    const npy_intp nind = PyArray_DIM(keys_ref.obj, 1);
    const npy_intp nmode = PyArray_DIM(q0_ref.obj, 0);
    const npy_intp total = nmax * nsys * nsys;

    if (PyArray_DIM(ddos_ref.obj, 2) != nsys ||
        PyArray_DIM(keys_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 0) != nmax ||
        PyArray_DIM(minus_ref.obj, 1) != nind ||
        PyArray_DIM(plus_ref.obj, 0) != nmax ||
        PyArray_DIM(plus_ref.obj, 1) != nind ||
        PyArray_DIM(expn_ref.obj, 0) != nind ||
        PyArray_DIM(etal_ref.obj, 0) != nind ||
        PyArray_DIM(etar_ref.obj, 0) != nind ||
        PyArray_DIM(etaa_ref.obj, 0) != nind ||
        PyArray_DIM(mode_ref.obj, 0) != nind ||
        PyArray_DIM(h0_ref.obj, 0) != nsys ||
        PyArray_DIM(h0_ref.obj, 1) != nsys ||
        PyArray_DIM(hhalf_ref.obj, 0) != nsys ||
        PyArray_DIM(hhalf_ref.obj, 1) != nsys ||
        PyArray_DIM(h1_ref.obj, 0) != nsys ||
        PyArray_DIM(h1_ref.obj, 1) != nsys ||
        PyArray_DIM(q0_ref.obj, 1) != nsys ||
        PyArray_DIM(q0_ref.obj, 2) != nsys ||
        PyArray_DIM(qhalf_ref.obj, 0) != nmode ||
        PyArray_DIM(qhalf_ref.obj, 1) != nsys ||
        PyArray_DIM(qhalf_ref.obj, 2) != nsys ||
        PyArray_DIM(q1_ref.obj, 0) != nmode ||
        PyArray_DIM(q1_ref.obj, 1) != nsys ||
        PyArray_DIM(q1_ref.obj, 2) != nsys) {
        PyErr_SetString(PyExc_ValueError, "HEOM native RK4 input shapes are inconsistent");
        return nullptr;
    }

    const auto* mode = reinterpret_cast<const std::int64_t*>(PyArray_DATA(mode_ref.obj));
    for (npy_intp mp = 0; mp < nind; ++mp) {
        if (mode[mp] < 0 || mode[mp] >= nmode) {
            PyErr_SetString(PyExc_ValueError, "mode index out of range for Q");
            return nullptr;
        }
    }

    auto check_workspace = [&](PyArrayObject* array, PyObject* original, const char* name) -> bool {
        if (reinterpret_cast<PyObject*>(array) != original) {
            PyErr_Format(PyExc_ValueError, "%s must be a writable C-contiguous complex128 array", name);
            return false;
        }
        if (!require_ndim(array, 3, name)) {
            return false;
        }
        if (PyArray_DIM(array, 0) != nmax ||
            PyArray_DIM(array, 1) != nsys ||
            PyArray_DIM(array, 2) != nsys) {
            PyErr_Format(PyExc_ValueError, "%s has the wrong shape", name);
            return false;
        }
        return true;
    };

    std::vector<complex128> k1;
    std::vector<complex128> k2;
    std::vector<complex128> k3;
    std::vector<complex128> k4;
    std::vector<complex128> work;
    ArrayRef k1_ref;
    ArrayRef k2_ref;
    ArrayRef k3_ref;
    ArrayRef k4_ref;
    ArrayRef work_ref;
    ArrayRef decay_ref;
    complex128* k1_data = nullptr;
    complex128* k2_data = nullptr;
    complex128* k3_data = nullptr;
    complex128* k4_data = nullptr;
    complex128* work_data = nullptr;
    const complex128* decay_rates = nullptr;

    const bool has_workspace = k1_obj != nullptr || k2_obj != nullptr || k3_obj != nullptr ||
                               k4_obj != nullptr || work_obj != nullptr;
    if (has_workspace) {
        if (k1_obj == nullptr || k2_obj == nullptr || k3_obj == nullptr ||
            k4_obj == nullptr || work_obj == nullptr) {
            PyErr_SetString(PyExc_ValueError, "all five RK4 workspace arrays must be provided together");
            return nullptr;
        }
        const int work_flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
        k1_ref.reset(k1_obj, NPY_COMPLEX128, work_flags);
        k2_ref.reset(k2_obj, NPY_COMPLEX128, work_flags);
        k3_ref.reset(k3_obj, NPY_COMPLEX128, work_flags);
        k4_ref.reset(k4_obj, NPY_COMPLEX128, work_flags);
        work_ref.reset(work_obj, NPY_COMPLEX128, work_flags);
        if (!k1_ref || !k2_ref || !k3_ref || !k4_ref || !work_ref) {
            return nullptr;
        }
        if (!check_workspace(k1_ref.obj, k1_obj, "k1") ||
            !check_workspace(k2_ref.obj, k2_obj, "k2") ||
            !check_workspace(k3_ref.obj, k3_obj, "k3") ||
            !check_workspace(k4_ref.obj, k4_obj, "k4") ||
            !check_workspace(work_ref.obj, work_obj, "work")) {
            return nullptr;
        }
        k1_data = reinterpret_cast<complex128*>(PyArray_DATA(k1_ref.obj));
        k2_data = reinterpret_cast<complex128*>(PyArray_DATA(k2_ref.obj));
        k3_data = reinterpret_cast<complex128*>(PyArray_DATA(k3_ref.obj));
        k4_data = reinterpret_cast<complex128*>(PyArray_DATA(k4_ref.obj));
        work_data = reinterpret_cast<complex128*>(PyArray_DATA(work_ref.obj));
    } else {
        try {
            k1.resize(static_cast<std::size_t>(total));
            k2.resize(static_cast<std::size_t>(total));
            k3.resize(static_cast<std::size_t>(total));
            k4.resize(static_cast<std::size_t>(total));
            work.resize(static_cast<std::size_t>(total));
        } catch (const std::bad_alloc&) {
            PyErr_NoMemory();
            return nullptr;
        }
        k1_data = k1.data();
        k2_data = k2.data();
        k3_data = k3.data();
        k4_data = k4.data();
        work_data = work.data();
    }

    if (decay_obj != nullptr) {
        decay_ref.reset(decay_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (!decay_ref) {
            return nullptr;
        }
        if (!require_ndim(decay_ref.obj, 1, "decay_rates")) {
            return nullptr;
        }
        if (PyArray_DIM(decay_ref.obj, 0) != nmax) {
            PyErr_SetString(PyExc_ValueError, "decay_rates has the wrong shape");
            return nullptr;
        }
        decay_rates = reinterpret_cast<const complex128*>(PyArray_DATA(decay_ref.obj));
    }

    auto* ddos = reinterpret_cast<complex128*>(PyArray_DATA(ddos_ref.obj));
    const auto* keys = reinterpret_cast<const std::int64_t*>(PyArray_DATA(keys_ref.obj));
    const auto* minus_index = reinterpret_cast<const std::int64_t*>(PyArray_DATA(minus_ref.obj));
    const auto* plus_index = reinterpret_cast<const std::int64_t*>(PyArray_DATA(plus_ref.obj));
    const auto* expn = reinterpret_cast<const complex128*>(PyArray_DATA(expn_ref.obj));
    const auto* etal = reinterpret_cast<const complex128*>(PyArray_DATA(etal_ref.obj));
    const auto* etar = reinterpret_cast<const complex128*>(PyArray_DATA(etar_ref.obj));
    const auto* etaa = reinterpret_cast<const complex128*>(PyArray_DATA(etaa_ref.obj));
    const auto* h0 = reinterpret_cast<const complex128*>(PyArray_DATA(h0_ref.obj));
    const auto* q0 = reinterpret_cast<const complex128*>(PyArray_DATA(q0_ref.obj));
    const auto* hhalf = reinterpret_cast<const complex128*>(PyArray_DATA(hhalf_ref.obj));
    const auto* qhalf = reinterpret_cast<const complex128*>(PyArray_DATA(qhalf_ref.obj));
    const auto* h1 = reinterpret_cast<const complex128*>(PyArray_DATA(h1_ref.obj));
    const auto* q1 = reinterpret_cast<const complex128*>(PyArray_DATA(q1_ref.obj));

    rhs_into_raw(ddos, k1_data, keys, minus_index, plus_index, expn, etal, etar, etaa,
                 mode, h0, q0, nmax, nsys, nind, decay_rates);

    for (npy_intp idx = 0; idx < total; ++idx) {
        work_data[idx] = ddos[idx] + k1_data[idx] * (dt / 2.0);
    }
    rhs_into_raw(work_data, k2_data, keys, minus_index, plus_index, expn, etal, etar, etaa,
                 mode, hhalf, qhalf, nmax, nsys, nind, decay_rates);

    for (npy_intp idx = 0; idx < total; ++idx) {
        work_data[idx] = ddos[idx] + k2_data[idx] * (dt / 2.0);
    }
    rhs_into_raw(work_data, k3_data, keys, minus_index, plus_index, expn, etal, etar, etaa,
                 mode, hhalf, qhalf, nmax, nsys, nind, decay_rates);

    for (npy_intp idx = 0; idx < total; ++idx) {
        work_data[idx] = ddos[idx] + k3_data[idx] * dt;
    }
    rhs_into_raw(work_data, k4_data, keys, minus_index, plus_index, expn, etal, etar, etaa,
                 mode, h1, q1, nmax, nsys, nind, decay_rates);

    for (npy_intp idx = 0; idx < total; ++idx) {
        ddos[idx] += (k1_data[idx] + 2.0 * k2_data[idx] + 2.0 * k3_data[idx] + k4_data[idx]) * (dt / 6.0);
    }

    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"available", available, METH_NOARGS, "Return True when the C++ HEOM extension is loaded."},
    {"rhs_by_index", rhs_by_index, METH_VARARGS, "Evaluate the dense arbitrary-key HEOM RHS."},
    {"dop853_by_index", dop853_by_index, METH_VARARGS,
     "Integrate HEOM ADOs with native adaptive DOP853 and reduced dense output."},
    {"dop853_adaptive_by_index", dop853_adaptive_by_index, METH_VARARGS,
     "Integrate HEOM ADOs with native adaptive DOP853 and accepted-step reduced output."},
    {"rk4_step_by_index", rk4_step_by_index, METH_VARARGS, "Advance dense HEOM ADOs by one RK4 step in place."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_heom_cpp",
    "C++ HEOM hierarchy kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__heom_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
