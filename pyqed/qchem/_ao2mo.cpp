#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <climits>
#include <new>
#include <vector>

namespace {
#ifdef __APPLE__
extern "C" void cblas_dgemm(int, int, int, int, int, int, double,
                            const double*, int, const double*, int,
                            double, double*, int);
#endif

void multiply(const double* a, const double* b, double* out,
              int m, int n, int k, bool transpose_a = false) {
#ifdef __APPLE__
    cblas_dgemm(101, transpose_a ? 112 : 111, 111, m, n, k,
                1.0, a, transpose_a ? m : k, b, n, 0.0, out, n);
#else
    std::fill(out, out + static_cast<size_t>(m) * n, 0.0);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) {
            const double value = transpose_a ? a[static_cast<size_t>(p) * m + i]
                                             : a[static_cast<size_t>(i) * k + p];
            for (int j = 0; j < n; ++j)
                out[static_cast<size_t>(i) * n + j] += value * b[static_cast<size_t>(p) * n + j];
        }
#endif
}

void transform(const double* eri, const double* coeff, double* out, int nao, int nmo) {
    const size_t ao2 = static_cast<size_t>(nao) * nao;
    const size_t mo2 = static_cast<size_t>(nmo) * nmo;
    const size_t apairs = static_cast<size_t>(nao) * (nao + 1) / 2;
    const size_t mpairs = static_cast<size_t>(nmo) * (nmo + 1) / 2;
    std::vector<double> half(apairs * mpairs), work(ao2);
    std::vector<double> tmp(static_cast<size_t>(nao) * nmo), block(mo2);
    size_t ij = 0;
    for (int i = 0; i < nao; ++i)
        for (int j = 0; j <= i; ++j, ++ij) {
            multiply(eri + (static_cast<size_t>(i) * nao + j) * ao2,
                     coeff, tmp.data(), nao, nmo, nao);
            multiply(coeff, tmp.data(), block.data(), nmo, nmo, nao, true);
            size_t pq = 0;
            for (int p = 0; p < nmo; ++p)
                for (int q = 0; q <= p; ++q, ++pq)
                    half[ij * mpairs + pq] = block[static_cast<size_t>(p) * nmo + q];
        }
    size_t pq = 0;
    for (int p = 0; p < nmo; ++p)
        for (int q = 0; q <= p; ++q, ++pq) {
            ij = 0;
            for (int i = 0; i < nao; ++i)
                for (int j = 0; j <= i; ++j, ++ij) {
                    const double value = half[ij * mpairs + pq];
                    work[static_cast<size_t>(i) * nao + j] = value;
                    work[static_cast<size_t>(j) * nao + i] = value;
                }
            multiply(work.data(), coeff, tmp.data(), nao, nmo, nao);
            multiply(coeff, tmp.data(), block.data(), nmo, nmo, nao, true);
            // Pair exchange makes each output block contiguous.
            std::copy(block.begin(), block.end(), out + (static_cast<size_t>(p) * nmo + q) * mo2);
            if (p != q)
                std::copy(block.begin(), block.end(), out + (static_cast<size_t>(q) * nmo + p) * mo2);
        }
}

PyObject* dense_pair_transform(PyObject*, PyObject* args) {
    PyObject *source, *coefficients;
    if (!PyArg_ParseTuple(args, "OO", &source, &coefficients)) return nullptr;
    auto* eri = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(source, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coeff = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coefficients, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!eri || !coeff) { Py_XDECREF(eri); Py_XDECREF(coeff); return nullptr; }
    bool valid = PyArray_NDIM(eri) == 4 && PyArray_NDIM(coeff) == 2;
    npy_intp nao = valid ? PyArray_DIM(coeff, 0) : 0;
    npy_intp nmo = valid ? PyArray_DIM(coeff, 1) : 0;
    valid = valid && nao > 0 && nao < INT_MAX && nmo < INT_MAX;
    for (int axis = 0; valid && axis < 4; ++axis) valid = PyArray_DIM(eri, axis) == nao;
    if (!valid) {
        Py_DECREF(eri); Py_DECREF(coeff);
        PyErr_SetString(PyExc_ValueError, "Expected dense (nao,nao,nao,nao) ERIs and (nao,nmo) coefficients.");
        return nullptr;
    }
    npy_intp shape[4] = {nmo, nmo, nmo, nmo};
    auto* result = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(4, shape, NPY_DOUBLE));
    bool allocation_failed = false;
    if (result && nmo) {
        Py_BEGIN_ALLOW_THREADS
        try {
            transform(static_cast<double*>(PyArray_DATA(eri)), static_cast<double*>(PyArray_DATA(coeff)),
                      static_cast<double*>(PyArray_DATA(result)), static_cast<int>(nao), static_cast<int>(nmo));
        } catch (const std::bad_alloc&) { allocation_failed = true; }
        Py_END_ALLOW_THREADS
    }
    Py_DECREF(eri); Py_DECREF(coeff);
    if (allocation_failed) { Py_DECREF(result); return PyErr_NoMemory(); }
    return reinterpret_cast<PyObject*>(result);
}

PyObject* packed_half_transform(PyObject*, PyObject* args) {
    PyObject *source, *coefficients;
    Py_ssize_t start = 0, stop = -1;
    if (!PyArg_ParseTuple(args, "OO|nn", &source, &coefficients, &start, &stop)) return nullptr;
    auto* eri = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(source, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* coeff = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(coefficients, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!eri || !coeff) { Py_XDECREF(eri); Py_XDECREF(coeff); return nullptr; }
    const int rank = PyArray_NDIM(eri);
    bool valid = (rank == 1 || rank == 2) && PyArray_NDIM(coeff) == 2;
    const npy_intp nao = valid ? PyArray_DIM(coeff, 0) : 0;
    const npy_intp nmo = valid ? PyArray_DIM(coeff, 1) : 0;
    const npy_intp ap = nao * (nao + 1) / 2, mp = nmo * (nmo + 1) / 2;
    const npy_intp rows = rank == 1 ? ap : rank == 2 ? PyArray_DIM(eri, 0) : 0;
    if (stop < 0) stop = rows;
    valid = valid && nao > 0 && nao < INT_MAX && nmo < INT_MAX &&
            start >= 0 && stop >= start && stop <= rows;
    if (valid) valid = rank == 1 ? PyArray_DIM(eri, 0) == ap * (ap + 1) / 2
                                : PyArray_DIM(eri, 1) == ap;
    if (!valid) {
        Py_DECREF(eri); Py_DECREF(coeff);
        PyErr_SetString(PyExc_ValueError, "Expected s8 ERIs or packed AO-pair rows, coefficients, and a valid row interval.");
        return nullptr;
    }
    npy_intp shape[2] = {stop - start, mp};
    auto* result = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, shape, NPY_DOUBLE));
    bool failed = false;
    if (result && nmo && stop > start) {
        const auto* input = static_cast<const double*>(PyArray_DATA(eri));
        const auto* c = static_cast<const double*>(PyArray_DATA(coeff));
        auto* output = static_cast<double*>(PyArray_DATA(result));
        Py_BEGIN_ALLOW_THREADS
        try {
            std::vector<double> work(nao * nao), tmp(nao * nmo), block(nmo * nmo);
            for (npy_intp row = start; row < stop; ++row) {
                npy_intp pair = 0;
                for (npy_intp i = 0; i < nao; ++i)
                    for (npy_intp j = 0; j <= i; ++j, ++pair) {
                        const npy_intp hi = std::max(row, pair), lo = std::min(row, pair);
                        const double value = input[rank == 1 ? hi * (hi + 1) / 2 + lo : row * ap + pair];
                        work[i * nao + j] = value;
                        work[j * nao + i] = value;
                    }
                multiply(work.data(), c, tmp.data(), nao, nmo, nao);
                multiply(c, tmp.data(), block.data(), nmo, nmo, nao, true);
                npy_intp pair_mo = 0;
                for (npy_intp i = 0; i < nmo; ++i)
                    for (npy_intp j = 0; j <= i; ++j, ++pair_mo)
                        output[(row - start) * mp + pair_mo] = block[i * nmo + j];
            }
        } catch (const std::bad_alloc&) { failed = true; }
        Py_END_ALLOW_THREADS
    }
    Py_DECREF(eri); Py_DECREF(coeff);
    if (failed) { Py_DECREF(result); return PyErr_NoMemory(); }
    return reinterpret_cast<PyObject*>(result);
}

PyMethodDef methods[] = {{"dense_pair_transform", dense_pair_transform, METH_VARARGS,
                         "Transform real eightfold-symmetric dense AO ERIs using independent orbital pairs."},
                        {"packed_half_transform", packed_half_transform, METH_VARARGS,
                         "Transform a bounded set of packed AO-pair rows using two matrix products."},
                        {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ao2mo", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__ao2mo() {
    import_array();
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
#ifdef __APPLE__
    const int has_blas = 1;
#else
    const int has_blas = 0;
#endif
    if (PyModule_AddIntConstant(result, "has_blas", has_blas) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
