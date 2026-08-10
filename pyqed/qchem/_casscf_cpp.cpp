#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace {

#if defined(__APPLE__) || defined(PYQED_USE_CBLAS)
#define PYQED_HAVE_CBLAS 1
#endif

#ifdef __APPLE__
extern "C" void dsyev_(
    char* jobz,
    char* uplo,
    int* n,
    double* a,
    int* lda,
    double* w,
    double* work,
    int* lwork,
    int* info
);
#endif

#ifdef PYQED_HAVE_CBLAS
extern "C" void cblas_dgemv(
    const int order,
    const int trans,
    const int m,
    const int n,
    const double alpha,
    const double* a,
    const int lda,
    const double* x,
    const int inc_x,
    const double beta,
    double* y,
    const int inc_y
);
extern "C" void cblas_dgemm(
    const int order,
    const int trans_a,
    const int trans_b,
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* a,
    const int lda,
    const double* b,
    const int ldb,
    const double beta,
    double* c,
    const int ldc
);
#endif

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

    explicit operator bool() const {
        return obj != nullptr;
    }

    void reset(PyObject* value, int typenum, int flags) {
        Py_XDECREF(obj);
        obj = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(value, typenum, flags));
    }
};

struct SequenceRef {
    PyObject* obj = nullptr;

    SequenceRef() = default;
    SequenceRef(PyObject* value, const char* message)
        : obj(PySequence_Fast(value, message)) {}

    ~SequenceRef() {
        Py_XDECREF(obj);
    }

    SequenceRef(const SequenceRef&) = delete;
    SequenceRef& operator=(const SequenceRef&) = delete;

    explicit operator bool() const {
        return obj != nullptr;
    }

    Py_ssize_t size() const {
        return PySequence_Fast_GET_SIZE(obj);
    }

    PyObject* item(Py_ssize_t index) const {
        return PySequence_Fast_GET_ITEM(obj, index);
    }
};

inline npy_intp idx2(npy_intp n, npy_intp i, npy_intp j) {
    return i * n + j;
}

inline npy_intp idx3(npy_intp n, npy_intp s, npy_intp i, npy_intp j) {
    return (s * n + i) * n + j;
}

inline npy_intp idx_binary(npy_intp n, npy_intp det, npy_intp spin, npy_intp orbital) {
    return (det * 2 + spin) * n + orbital;
}

inline npy_intp idx4(npy_intp n, npy_intp i, npy_intp j, npy_intp k, npy_intp l) {
    return ((i * n + j) * n + k) * n + l;
}

inline npy_intp idx_occ(npy_intp n, npy_intp row, npy_intp orbital) {
    return row * n + orbital;
}

inline npy_intp idx6(npy_intp n, npy_intp s, npy_intp t, npy_intp i, npy_intp j, npy_intp k, npy_intp l) {
    return (((((s * 2 + t) * n + i) * n + j) * n + k) * n + l);
}

inline npy_intp idx6_plain(
    npy_intp n,
    npy_intp i,
    npy_intp j,
    npy_intp k,
    npy_intp l,
    npy_intp m,
    npy_intp o
) {
    return (((((i * n + j) * n + k) * n + l) * n + m) * n + o);
}

PyObject* vector_to_int32_array(const std::vector<npy_int32>& values) {
    npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
    PyObject* obj = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (obj == nullptr) {
        return nullptr;
    }
    if (!values.empty()) {
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj)),
            values.data(),
            values.size() * sizeof(npy_int32)
        );
    }
    return obj;
}

PyObject* vector_to_int8_array(const std::vector<npy_int8>& values) {
    npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
    PyObject* obj = PyArray_SimpleNew(1, dims, NPY_INT8);
    if (obj == nullptr) {
        return nullptr;
    }
    if (!values.empty()) {
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj)),
            values.data(),
            values.size() * sizeof(npy_int8)
        );
    }
    return obj;
}

PyObject* build_link_tuple(
    const std::vector<npy_int32>& i,
    const std::vector<npy_int32>& j,
    const std::vector<npy_int32>& p,
    const std::vector<npy_int32>& q,
    const std::vector<npy_int8>& phase
) {
    PyObject* result = PyTuple_New(5);
    if (result == nullptr) {
        return nullptr;
    }
    PyObject* arrays[5] = {
        vector_to_int32_array(i),
        vector_to_int32_array(j),
        vector_to_int32_array(p),
        vector_to_int32_array(q),
        vector_to_int8_array(phase),
    };
    for (int idx = 0; idx < 5; ++idx) {
        if (arrays[idx] == nullptr) {
            for (int jdx = 0; jdx < idx; ++jdx) {
                Py_DECREF(arrays[jdx]);
            }
            Py_DECREF(result);
            return nullptr;
        }
        PyTuple_SET_ITEM(result, idx, arrays[idx]);
    }
    return result;
}

PyObject* build_double_link_tuple(
    const std::vector<npy_int32>& i,
    const std::vector<npy_int32>& j,
    const std::vector<npy_int32>& p,
    const std::vector<npy_int32>& q,
    const std::vector<npy_int32>& r,
    const std::vector<npy_int32>& s,
    const std::vector<npy_int8>& phase
) {
    PyObject* result = PyTuple_New(7);
    if (result == nullptr) {
        return nullptr;
    }
    PyObject* arrays[7] = {
        vector_to_int32_array(i),
        vector_to_int32_array(j),
        vector_to_int32_array(p),
        vector_to_int32_array(q),
        vector_to_int32_array(r),
        vector_to_int32_array(s),
        vector_to_int8_array(phase),
    };
    for (int idx = 0; idx < 7; ++idx) {
        if (arrays[idx] == nullptr) {
            for (int jdx = 0; jdx < idx; ++jdx) {
                Py_DECREF(arrays[jdx]);
            }
            Py_DECREF(result);
            return nullptr;
        }
        PyTuple_SET_ITEM(result, idx, arrays[idx]);
    }
    return result;
}

inline std::uint64_t orbital_mask(npy_intp orbital) {
    return std::uint64_t{1} << static_cast<unsigned>(orbital);
}

inline npy_int8 orbital_phase(std::uint64_t bits, npy_intp orbital) {
    if (orbital <= 0) {
        return 1;
    }
    const std::uint64_t mask = orbital_mask(orbital) - 1;
    return (__builtin_popcountll(bits & mask) & 1) ? -1 : 1;
}

std::uint64_t row_to_bits(const npy_int8* strings, npy_intp n_mo, npy_intp row) {
    std::uint64_t bits = 0;
    const npy_int8* data = strings + row * n_mo;
    for (npy_intp p = 0; p < n_mo; ++p) {
        if (data[p]) {
            bits |= orbital_mask(p);
        }
    }
    return bits;
}

bool validate_orbital_hessian_inputs(
    PyArrayObject* h1,
    PyArrayObject* eri,
    PyArrayObject* dm1,
    PyArrayObject* dm2,
    PyArrayObject* kappa
) {
    if (
        PyArray_NDIM(h1) != 2 ||
        PyArray_NDIM(eri) != 4 ||
        PyArray_NDIM(dm1) != 2 ||
        PyArray_NDIM(dm2) != 4 ||
        PyArray_NDIM(kappa) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "orbital_hessian_action_from_integrals expects h1/dm1/kappa as 2D arrays and eri/dm2 as 4D arrays.");
        return false;
    }
    const npy_intp n = PyArray_DIM(h1, 0);
    if (
        PyArray_DIM(h1, 1) != n ||
        PyArray_DIM(dm1, 0) != n ||
        PyArray_DIM(dm1, 1) != n ||
        PyArray_DIM(kappa, 0) != n ||
        PyArray_DIM(kappa, 1) != n
    ) {
        PyErr_SetString(PyExc_ValueError, "h1, dm1, and kappa must be square arrays with matching dimensions.");
        return false;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (PyArray_DIM(eri, axis) != n || PyArray_DIM(dm2, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "eri and dm2 must have shape (n, n, n, n) matching h1.");
            return false;
        }
    }
    return true;
}

PyObject* orbital_hessian_action_from_integrals(PyObject*, PyObject* args) {
    PyObject* h1_obj = nullptr;
    PyObject* eri_obj = nullptr;
    PyObject* dm1_obj = nullptr;
    PyObject* dm2_obj = nullptr;
    PyObject* kappa_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOOO", &h1_obj, &eri_obj, &dm1_obj, &dm2_obj, &kappa_obj)) {
        return nullptr;
    }

    ArrayRef h1(h1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm1(dm1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm2(dm2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef kappa(kappa_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!h1 || !eri || !dm1 || !dm2 || !kappa) {
        return nullptr;
    }
    if (!validate_orbital_hessian_inputs(h1.obj, eri.obj, dm1.obj, dm2.obj, kappa.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h1.obj, 0);
    npy_intp dims[2] = {n, n};
    PyObject* grad_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    PyObject* dfock_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (grad_obj == nullptr || dfock_obj == nullptr) {
        Py_XDECREF(grad_obj);
        Py_XDECREF(dfock_obj);
        return nullptr;
    }

    const double* h = static_cast<const double*>(PyArray_DATA(h1.obj));
    const double* g = static_cast<const double*>(PyArray_DATA(eri.obj));
    const double* d1 = static_cast<const double*>(PyArray_DATA(dm1.obj));
    const double* d2 = static_cast<const double*>(PyArray_DATA(dm2.obj));
    const double* k = static_cast<const double*>(PyArray_DATA(kappa.obj));
    double* df = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dfock_obj)));
    double* grad = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(grad_obj)));

    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            double value = 0.0;

            for (npy_intp r = 0; r < n; ++r) {
                double dh1_pr = 0.0;
                for (npy_intp a = 0; a < n; ++a) {
                    dh1_pr += h[idx2(n, p, a)] * k[idx2(n, a, r)];
                    dh1_pr -= k[idx2(n, p, a)] * h[idx2(n, a, r)];
                }
                value += dh1_pr * d1[idx2(n, r, q)];
            }

            for (npy_intp r = 0; r < n; ++r) {
                for (npy_intp s = 0; s < n; ++s) {
                    for (npy_intp t = 0; t < n; ++t) {
                        const double rho = d2[idx4(n, r, q, s, t)];
                        if (rho == 0.0) {
                            continue;
                        }
                        double deri = 0.0;
                        for (npy_intp a = 0; a < n; ++a) {
                            deri += k[idx2(n, a, p)] * g[idx4(n, a, r, s, t)];
                            deri += k[idx2(n, a, r)] * g[idx4(n, p, a, s, t)];
                            deri += k[idx2(n, a, s)] * g[idx4(n, p, r, a, t)];
                            deri += k[idx2(n, a, t)] * g[idx4(n, p, r, s, a)];
                        }
                        value += deri * rho;
                    }
                }
            }

            df[idx2(n, p, q)] = value;
        }
    }

    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            grad[idx2(n, p, q)] = 2.0 * (df[idx2(n, p, q)] - df[idx2(n, q, p)]);
        }
    }

    Py_DECREF(dfock_obj);
    return grad_obj;
}

bool validate_ci_inputs(PyArrayObject* binary, PyArrayObject* h1, PyArrayObject* h2) {
    if (PyArray_NDIM(binary) != 3 || PyArray_NDIM(h1) != 3 || PyArray_NDIM(h2) != 6) {
        PyErr_SetString(PyExc_ValueError, "ci_hamiltonian expects Binary(ndef,2,n), H1(2,n,n), and H2(2,2,n,n,n,n).");
        return false;
    }
    const npy_intp n = PyArray_DIM(binary, 2);
    if (
        PyArray_DIM(binary, 1) != 2 ||
        PyArray_DIM(h1, 0) != 2 ||
        PyArray_DIM(h1, 1) != n ||
        PyArray_DIM(h1, 2) != n ||
        PyArray_DIM(h2, 0) != 2 ||
        PyArray_DIM(h2, 1) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "ci_hamiltonian spin/orbital dimensions do not match.");
        return false;
    }
    for (int axis = 2; axis < 6; ++axis) {
        if (PyArray_DIM(h2, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "H2 orbital dimensions must all match Binary.shape[2].");
            return false;
        }
    }
    return true;
}

bool validate_1d_index_array(PyArrayObject* arr, const char* name) {
    if (PyArray_NDIM(arr) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be a 1D index array.", name);
        return false;
    }
    return true;
}

bool validate_2d_mask_array(PyArrayObject* arr, npy_intp n, const char* name) {
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != n) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (k, norb).", name);
        return false;
    }
    return true;
}

bool validate_3d_mask_array(PyArrayObject* arr, npy_intp n, const char* name) {
    if (PyArray_NDIM(arr) != 3 || PyArray_DIM(arr, 0) != 2 || PyArray_DIM(arr, 2) != n) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (2, k, norb).", name);
        return false;
    }
    return true;
}

PyObject* ci_hamiltonian(PyObject*, PyObject* args) {
    PyObject* binary_obj = nullptr;
    PyObject* h1_obj = nullptr;
    PyObject* h2_obj = nullptr;
    PyObject* sc1_obj = nullptr;
    PyObject* sc2_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOOOO", &binary_obj, &h1_obj, &h2_obj, &sc1_obj, &sc2_obj)) {
        return nullptr;
    }

    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef h1(h1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h2(h2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!binary || !h1 || !h2) {
        return nullptr;
    }
    if (!validate_ci_inputs(binary.obj, h1.obj, h2.obj)) {
        return nullptr;
    }

    SequenceRef sc1(sc1_obj, "SC1 must be a sequence.");
    SequenceRef sc2(sc2_obj, "SC2 must be a sequence.");
    if (!sc1 || !sc2) {
        return nullptr;
    }
    if (sc1.size() != 10 || sc2.size() != 14) {
        PyErr_SetString(PyExc_ValueError, "SC1 and SC2 have unexpected lengths.");
        return nullptr;
    }

    ArrayRef I_A(sc1.item(0), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_A(sc1.item(1), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef a_t(sc1.item(2), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef a(sc1.item(3), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef I_B(sc1.item(4), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_B(sc1.item(5), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef b_t(sc1.item(6), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef b(sc1.item(7), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef ca(sc1.item(8), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef cb(sc1.item(9), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_AA(sc2.item(0), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_AA(sc2.item(1), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef aa_t(sc2.item(2), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef aa(sc2.item(3), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef I_BB(sc2.item(4), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_BB(sc2.item(5), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef bb_t(sc2.item(6), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef bb(sc2.item(7), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef I_AB(sc2.item(8), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_AB(sc2.item(9), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef ab_t(sc2.item(10), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef ab(sc2.item(11), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef ba_t(sc2.item(12), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef ba(sc2.item(13), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    if (!I_A || !J_A || !a_t || !a || !I_B || !J_B || !b_t || !b || !ca || !cb ||
        !I_AA || !J_AA || !aa_t || !aa || !I_BB || !J_BB || !bb_t || !bb ||
        !I_AB || !J_AB || !ab_t || !ab || !ba_t || !ba) {
        return nullptr;
    }

    const npy_intp ndet = PyArray_DIM(binary.obj, 0);
    const npy_intp n = PyArray_DIM(binary.obj, 2);

    if (
        !validate_1d_index_array(I_A.obj, "I_A") ||
        !validate_1d_index_array(J_A.obj, "J_A") ||
        !validate_2d_mask_array(a_t.obj, n, "a_t") ||
        !validate_2d_mask_array(a.obj, n, "a") ||
        !validate_1d_index_array(I_B.obj, "I_B") ||
        !validate_1d_index_array(J_B.obj, "J_B") ||
        !validate_2d_mask_array(b_t.obj, n, "b_t") ||
        !validate_2d_mask_array(b.obj, n, "b") ||
        !validate_2d_mask_array(ca.obj, n, "ca") ||
        !validate_2d_mask_array(cb.obj, n, "cb") ||
        !validate_1d_index_array(I_AA.obj, "I_AA") ||
        !validate_1d_index_array(J_AA.obj, "J_AA") ||
        !validate_3d_mask_array(aa_t.obj, n, "aa_t") ||
        !validate_3d_mask_array(aa.obj, n, "aa") ||
        !validate_1d_index_array(I_BB.obj, "I_BB") ||
        !validate_1d_index_array(J_BB.obj, "J_BB") ||
        !validate_3d_mask_array(bb_t.obj, n, "bb_t") ||
        !validate_3d_mask_array(bb.obj, n, "bb") ||
        !validate_1d_index_array(I_AB.obj, "I_AB") ||
        !validate_1d_index_array(J_AB.obj, "J_AB") ||
        !validate_2d_mask_array(ab_t.obj, n, "ab_t") ||
        !validate_2d_mask_array(ab.obj, n, "ab") ||
        !validate_2d_mask_array(ba_t.obj, n, "ba_t") ||
        !validate_2d_mask_array(ba.obj, n, "ba")
    ) {
        return nullptr;
    }

    npy_intp dims[2] = {ndet, ndet};
    PyObject* hci_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (hci_obj == nullptr) {
        return nullptr;
    }

    const npy_int8* occ = static_cast<const npy_int8*>(PyArray_DATA(binary.obj));
    const double* h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
    const double* h2_data = static_cast<const double*>(PyArray_DATA(h2.obj));
    double* hci = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(hci_obj)));

    for (npy_intp det = 0; det < ndet; ++det) {
        double value = 0.0;
        for (npy_intp spin = 0; spin < 2; ++spin) {
            for (npy_intp p = 0; p < n; ++p) {
                value += h1_data[idx3(n, spin, p, p)] * static_cast<double>(occ[idx_binary(n, det, spin, p)]);
            }
        }
        for (npy_intp s = 0; s < 2; ++s) {
            for (npy_intp t = 0; t < 2; ++t) {
                for (npy_intp p = 0; p < n; ++p) {
                    const double occ_p = static_cast<double>(occ[idx_binary(n, det, s, p)]);
                    if (occ_p == 0.0) {
                        continue;
                    }
                    for (npy_intp q = 0; q < n; ++q) {
                        value += 0.5 * h2_data[idx6(n, s, t, p, p, q, q)] * occ_p * static_cast<double>(occ[idx_binary(n, det, t, q)]);
                    }
                }
            }
        }
        hci[det * ndet + det] = value;
    }

    const npy_intp* i_a = static_cast<const npy_intp*>(PyArray_DATA(I_A.obj));
    const npy_intp* j_a = static_cast<const npy_intp*>(PyArray_DATA(J_A.obj));
    const npy_int8* at = static_cast<const npy_int8*>(PyArray_DATA(a_t.obj));
    const npy_int8* am = static_cast<const npy_int8*>(PyArray_DATA(a.obj));
    const npy_int8* cam = static_cast<const npy_int8*>(PyArray_DATA(ca.obj));
    const npy_intp n_single_a = PyArray_DIM(I_A.obj, 0);
    for (npy_intp k = 0; k < n_single_a; ++k) {
        double value = 0.0;
        for (npy_intp p = 0; p < n; ++p) {
            const double at_p = static_cast<double>(at[k * n + p]);
            if (at_p == 0.0) {
                continue;
            }
            for (npy_intp q = 0; q < n; ++q) {
                const double aq = static_cast<double>(am[k * n + q]);
                if (aq == 0.0) {
                    continue;
                }
                const double pair = at_p * aq;
                value += h1_data[idx3(n, 0, p, q)] * pair;
                for (npy_intp r = 0; r < n; ++r) {
                    value += h2_data[idx6(n, 0, 0, p, q, r, r)] * pair * static_cast<double>(cam[k * n + r]);
                    value += h2_data[idx6(n, 0, 1, p, q, r, r)] * pair * static_cast<double>(occ[idx_binary(n, i_a[k], 1, r)]);
                }
            }
        }
        hci[i_a[k] * ndet + j_a[k]] -= value;
    }

    const npy_intp* i_b = static_cast<const npy_intp*>(PyArray_DATA(I_B.obj));
    const npy_intp* j_b = static_cast<const npy_intp*>(PyArray_DATA(J_B.obj));
    const npy_int8* bt = static_cast<const npy_int8*>(PyArray_DATA(b_t.obj));
    const npy_int8* bm = static_cast<const npy_int8*>(PyArray_DATA(b.obj));
    const npy_int8* cbm = static_cast<const npy_int8*>(PyArray_DATA(cb.obj));
    const npy_intp n_single_b = PyArray_DIM(I_B.obj, 0);
    for (npy_intp k = 0; k < n_single_b; ++k) {
        double value = 0.0;
        for (npy_intp p = 0; p < n; ++p) {
            const double bt_p = static_cast<double>(bt[k * n + p]);
            if (bt_p == 0.0) {
                continue;
            }
            for (npy_intp q = 0; q < n; ++q) {
                const double bq = static_cast<double>(bm[k * n + q]);
                if (bq == 0.0) {
                    continue;
                }
                const double pair = bt_p * bq;
                value += h1_data[idx3(n, 1, p, q)] * pair;
                for (npy_intp r = 0; r < n; ++r) {
                    value += h2_data[idx6(n, 1, 1, p, q, r, r)] * pair * static_cast<double>(cbm[k * n + r]);
                    value += h2_data[idx6(n, 1, 0, p, q, r, r)] * pair * static_cast<double>(occ[idx_binary(n, i_b[k], 0, r)]);
                }
            }
        }
        hci[i_b[k] * ndet + j_b[k]] -= value;
    }

    const npy_intp* i_aa = static_cast<const npy_intp*>(PyArray_DATA(I_AA.obj));
    const npy_intp* j_aa = static_cast<const npy_intp*>(PyArray_DATA(J_AA.obj));
    const npy_int8* aat = static_cast<const npy_int8*>(PyArray_DATA(aa_t.obj));
    const npy_int8* aam = static_cast<const npy_int8*>(PyArray_DATA(aa.obj));
    const npy_intp n_aa = PyArray_DIM(I_AA.obj, 0);
    for (npy_intp k = 0; k < n_aa; ++k) {
        double value = 0.0;
        for (npy_intp p = 0; p < n; ++p) {
            const double at0 = static_cast<double>(aat[(0 * n_aa + k) * n + p]);
            if (at0 == 0.0) {
                continue;
            }
            for (npy_intp q = 0; q < n; ++q) {
                const double a0 = static_cast<double>(aam[(0 * n_aa + k) * n + q]);
                if (a0 == 0.0) {
                    continue;
                }
                for (npy_intp r = 0; r < n; ++r) {
                    const double at1 = static_cast<double>(aat[(1 * n_aa + k) * n + r]);
                    if (at1 == 0.0) {
                        continue;
                    }
                    for (npy_intp s = 0; s < n; ++s) {
                        value += h2_data[idx6(n, 0, 0, p, q, r, s)] * at0 * a0 * at1 * static_cast<double>(aam[(1 * n_aa + k) * n + s]);
                    }
                }
            }
        }
        hci[i_aa[k] * ndet + j_aa[k]] = value;
    }

    const npy_intp* i_bb = static_cast<const npy_intp*>(PyArray_DATA(I_BB.obj));
    const npy_intp* j_bb = static_cast<const npy_intp*>(PyArray_DATA(J_BB.obj));
    const npy_int8* bbt = static_cast<const npy_int8*>(PyArray_DATA(bb_t.obj));
    const npy_int8* bbm = static_cast<const npy_int8*>(PyArray_DATA(bb.obj));
    const npy_intp n_bb = PyArray_DIM(I_BB.obj, 0);
    for (npy_intp k = 0; k < n_bb; ++k) {
        double value = 0.0;
        for (npy_intp p = 0; p < n; ++p) {
            const double bt0 = static_cast<double>(bbt[(0 * n_bb + k) * n + p]);
            if (bt0 == 0.0) {
                continue;
            }
            for (npy_intp q = 0; q < n; ++q) {
                const double b0 = static_cast<double>(bbm[(0 * n_bb + k) * n + q]);
                if (b0 == 0.0) {
                    continue;
                }
                for (npy_intp r = 0; r < n; ++r) {
                    const double bt1 = static_cast<double>(bbt[(1 * n_bb + k) * n + r]);
                    if (bt1 == 0.0) {
                        continue;
                    }
                    for (npy_intp s = 0; s < n; ++s) {
                        value += h2_data[idx6(n, 1, 1, p, q, r, s)] * bt0 * b0 * bt1 * static_cast<double>(bbm[(1 * n_bb + k) * n + s]);
                    }
                }
            }
        }
        hci[i_bb[k] * ndet + j_bb[k]] = value;
    }

    const npy_intp* i_ab = static_cast<const npy_intp*>(PyArray_DATA(I_AB.obj));
    const npy_intp* j_ab = static_cast<const npy_intp*>(PyArray_DATA(J_AB.obj));
    const npy_int8* abt = static_cast<const npy_int8*>(PyArray_DATA(ab_t.obj));
    const npy_int8* abm = static_cast<const npy_int8*>(PyArray_DATA(ab.obj));
    const npy_int8* bat = static_cast<const npy_int8*>(PyArray_DATA(ba_t.obj));
    const npy_int8* bam = static_cast<const npy_int8*>(PyArray_DATA(ba.obj));
    const npy_intp n_ab = PyArray_DIM(I_AB.obj, 0);
    for (npy_intp k = 0; k < n_ab; ++k) {
        double value = 0.0;
        for (npy_intp p = 0; p < n; ++p) {
            const double abt_p = static_cast<double>(abt[k * n + p]);
            if (abt_p == 0.0) {
                continue;
            }
            for (npy_intp q = 0; q < n; ++q) {
                const double ab_q = static_cast<double>(abm[k * n + q]);
                if (ab_q == 0.0) {
                    continue;
                }
                for (npy_intp r = 0; r < n; ++r) {
                    const double bat_r = static_cast<double>(bat[k * n + r]);
                    if (bat_r == 0.0) {
                        continue;
                    }
                    for (npy_intp s = 0; s < n; ++s) {
                        value += h2_data[idx6(n, 0, 1, p, q, r, s)] * abt_p * ab_q * bat_r * static_cast<double>(bam[k * n + s]);
                    }
                }
            }
        }
        hci[i_ab[k] * ndet + j_ab[k]] = value;
    }

    return hci_obj;
}

bool validate_spin_string_sigma_inputs(
    PyArrayObject* h1,
    PyArrayObject* eri_same,
    PyArrayObject* eri_cross,
    PyArrayObject* hdiag,
    PyArrayObject* c,
    PyArrayObject* alpha_occ,
    PyArrayObject* beta_occ
) {
    if (
        PyArray_NDIM(h1) != 2 ||
        PyArray_NDIM(eri_same) != 4 ||
        PyArray_NDIM(eri_cross) != 4 ||
        PyArray_NDIM(hdiag) != 1 ||
        PyArray_NDIM(c) != 1 ||
        PyArray_NDIM(alpha_occ) != 2 ||
        PyArray_NDIM(beta_occ) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "sigma_compact_spin_string expects h1(nao,nao), ERIs(nao,nao,nao,nao), vectors(ndet), and occupations(nstring,nao).");
        return false;
    }
    const npy_intp n = PyArray_DIM(h1, 0);
    const npy_intp n_alpha = PyArray_DIM(alpha_occ, 0);
    const npy_intp n_beta = PyArray_DIM(beta_occ, 0);
    const npy_intp n_det = n_alpha * n_beta;
    if (
        PyArray_DIM(h1, 1) != n ||
        PyArray_DIM(alpha_occ, 1) != n ||
        PyArray_DIM(beta_occ, 1) != n ||
        PyArray_DIM(hdiag, 0) != n_det ||
        PyArray_DIM(c, 0) != n_det
    ) {
        PyErr_SetString(PyExc_ValueError, "sigma_compact_spin_string dimensions do not match.");
        return false;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (PyArray_DIM(eri_same, axis) != n || PyArray_DIM(eri_cross, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "ERI tensors must have shape (norb, norb, norb, norb).");
            return false;
        }
    }
    return true;
}

bool validate_link_group(
    PyArrayObject* I,
    PyArrayObject* J,
    PyArrayObject* p,
    PyArrayObject* q,
    PyArrayObject* phase,
    const char* name
) {
    if (
        PyArray_NDIM(I) != 1 ||
        PyArray_NDIM(J) != 1 ||
        PyArray_NDIM(p) != 1 ||
        PyArray_NDIM(q) != 1 ||
        PyArray_NDIM(phase) != 1
    ) {
        PyErr_Format(PyExc_ValueError, "%s spin-string single-link arrays must be 1D.", name);
        return false;
    }
    const npy_intp n_link = PyArray_DIM(I, 0);
    if (
        PyArray_DIM(J, 0) != n_link ||
        PyArray_DIM(p, 0) != n_link ||
        PyArray_DIM(q, 0) != n_link ||
        PyArray_DIM(phase, 0) != n_link
    ) {
        PyErr_Format(PyExc_ValueError, "%s spin-string single-link arrays have inconsistent lengths.", name);
        return false;
    }
    return true;
}

bool validate_double_link_group(
    PyArrayObject* I,
    PyArrayObject* J,
    PyArrayObject* p,
    PyArrayObject* q,
    PyArrayObject* r,
    PyArrayObject* s,
    PyArrayObject* phase,
    const char* name
) {
    if (
        PyArray_NDIM(I) != 1 ||
        PyArray_NDIM(J) != 1 ||
        PyArray_NDIM(p) != 1 ||
        PyArray_NDIM(q) != 1 ||
        PyArray_NDIM(r) != 1 ||
        PyArray_NDIM(s) != 1 ||
        PyArray_NDIM(phase) != 1
    ) {
        PyErr_Format(PyExc_ValueError, "%s spin-string double-link arrays must be 1D.", name);
        return false;
    }
    const npy_intp n_link = PyArray_DIM(I, 0);
    if (
        PyArray_DIM(J, 0) != n_link ||
        PyArray_DIM(p, 0) != n_link ||
        PyArray_DIM(q, 0) != n_link ||
        PyArray_DIM(r, 0) != n_link ||
        PyArray_DIM(s, 0) != n_link ||
        PyArray_DIM(phase, 0) != n_link
    ) {
        PyErr_Format(PyExc_ValueError, "%s spin-string double-link arrays have inconsistent lengths.", name);
        return false;
    }
    return true;
}

PyObject* single_string_links(PyObject*, PyObject* args) {
    PyObject* strings_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &strings_obj)) {
        return nullptr;
    }
    ArrayRef strings_ref(strings_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!strings_ref) {
        return nullptr;
    }
    PyArrayObject* strings = strings_ref.obj;
    if (PyArray_NDIM(strings) != 2) {
        PyErr_SetString(PyExc_ValueError, "single_string_links expects a 2D int8 occupation array.");
        return nullptr;
    }
    const npy_intp n_string = PyArray_DIM(strings, 0);
    const npy_intp n_mo = PyArray_DIM(strings, 1);
    if (n_mo > 62) {
        PyErr_SetString(PyExc_NotImplementedError, "native string link generation supports at most 62 orbitals.");
        return nullptr;
    }
    const npy_int8* data = static_cast<const npy_int8*>(PyArray_DATA(strings));

    std::vector<std::uint64_t> bits(static_cast<std::size_t>(n_string), 0);
    std::unordered_map<std::uint64_t, npy_int32> lookup;
    lookup.reserve(static_cast<std::size_t>(n_string) * 2 + 1);
    for (npy_intp row = 0; row < n_string; ++row) {
        bits[static_cast<std::size_t>(row)] = row_to_bits(data, n_mo, row);
        lookup.emplace(bits[static_cast<std::size_t>(row)], static_cast<npy_int32>(row));
    }

    std::vector<npy_int32> I;
    std::vector<npy_int32> J;
    std::vector<npy_int32> p_idx;
    std::vector<npy_int32> q_idx;
    std::vector<npy_int8> phases;

    try {
        for (npy_intp ket = 0; ket < n_string; ++ket) {
            const std::uint64_t ket_bits = bits[static_cast<std::size_t>(ket)];
            for (npy_intp q = 0; q < n_mo; ++q) {
                if ((ket_bits & orbital_mask(q)) == 0) {
                    continue;
                }
                const std::uint64_t removed = ket_bits ^ orbital_mask(q);
                for (npy_intp p = 0; p < n_mo; ++p) {
                    if (ket_bits & orbital_mask(p)) {
                        continue;
                    }
                    const std::uint64_t bra_bits = removed | orbital_mask(p);
                    const auto found = lookup.find(bra_bits);
                    if (found == lookup.end()) {
                        PyErr_SetString(PyExc_ValueError, "Occupation strings are not closed under single excitations.");
                        return nullptr;
                    }
                    const npy_int8 phase = orbital_phase(ket_bits, p) * orbital_phase(bra_bits, q);
                    I.push_back(found->second);
                    J.push_back(static_cast<npy_int32>(ket));
                    p_idx.push_back(static_cast<npy_int32>(p));
                    q_idx.push_back(static_cast<npy_int32>(q));
                    phases.push_back(phase);
                }
            }
        }
    } catch (...) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate native single string links.");
        return nullptr;
    }

    return build_link_tuple(I, J, p_idx, q_idx, phases);
}

PyObject* double_string_links(PyObject*, PyObject* args) {
    PyObject* strings_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &strings_obj)) {
        return nullptr;
    }
    ArrayRef strings_ref(strings_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!strings_ref) {
        return nullptr;
    }
    PyArrayObject* strings = strings_ref.obj;
    if (PyArray_NDIM(strings) != 2) {
        PyErr_SetString(PyExc_ValueError, "double_string_links expects a 2D int8 occupation array.");
        return nullptr;
    }
    const npy_intp n_string = PyArray_DIM(strings, 0);
    const npy_intp n_mo = PyArray_DIM(strings, 1);
    if (n_mo > 62) {
        PyErr_SetString(PyExc_NotImplementedError, "native string link generation supports at most 62 orbitals.");
        return nullptr;
    }
    const npy_int8* data = static_cast<const npy_int8*>(PyArray_DATA(strings));

    std::vector<std::uint64_t> bits(static_cast<std::size_t>(n_string), 0);
    std::unordered_map<std::uint64_t, npy_int32> lookup;
    lookup.reserve(static_cast<std::size_t>(n_string) * 2 + 1);
    for (npy_intp row = 0; row < n_string; ++row) {
        bits[static_cast<std::size_t>(row)] = row_to_bits(data, n_mo, row);
        lookup.emplace(bits[static_cast<std::size_t>(row)], static_cast<npy_int32>(row));
    }

    std::vector<npy_int32> I;
    std::vector<npy_int32> J;
    std::vector<npy_int32> p_idx;
    std::vector<npy_int32> q_idx;
    std::vector<npy_int32> r_idx;
    std::vector<npy_int32> s_idx;
    std::vector<npy_int8> phases;
    std::vector<npy_intp> occ;
    std::vector<npy_intp> vir;

    try {
        for (npy_intp ket = 0; ket < n_string; ++ket) {
            const std::uint64_t ket_bits = bits[static_cast<std::size_t>(ket)];
            occ.clear();
            vir.clear();
            for (npy_intp orb = 0; orb < n_mo; ++orb) {
                if (ket_bits & orbital_mask(orb)) {
                    occ.push_back(orb);
                } else {
                    vir.push_back(orb);
                }
            }
            for (std::size_t iq = 0; iq < occ.size(); ++iq) {
                const npy_intp q = occ[iq];
                for (std::size_t is = iq + 1; is < occ.size(); ++is) {
                    const npy_intp s = occ[is];
                    const std::uint64_t removed = ket_bits ^ orbital_mask(q) ^ orbital_mask(s);
                    for (std::size_t ip = 0; ip < vir.size(); ++ip) {
                        const npy_intp p = vir[ip];
                        for (std::size_t ir = ip + 1; ir < vir.size(); ++ir) {
                            const npy_intp r = vir[ir];
                            const std::uint64_t bra_bits = removed | orbital_mask(p) | orbital_mask(r);
                            const auto found = lookup.find(bra_bits);
                            if (found == lookup.end()) {
                                PyErr_SetString(PyExc_ValueError, "Occupation strings are not closed under double excitations.");
                                return nullptr;
                            }
                            const npy_int8 phase =
                                orbital_phase(ket_bits, r) *
                                orbital_phase(bra_bits, s) *
                                orbital_phase(ket_bits, p) *
                                orbital_phase(bra_bits, q);
                            I.push_back(found->second);
                            J.push_back(static_cast<npy_int32>(ket));
                            p_idx.push_back(static_cast<npy_int32>(r));
                            q_idx.push_back(static_cast<npy_int32>(s));
                            r_idx.push_back(static_cast<npy_int32>(p));
                            s_idx.push_back(static_cast<npy_int32>(q));
                            phases.push_back(phase);
                        }
                    }
                }
            }
        }
    } catch (...) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate native double string links.");
        return nullptr;
    }

    return build_double_link_tuple(I, J, p_idx, q_idx, r_idx, s_idx, phases);
}

PyObject* sigma_compact_spin_string(PyObject*, PyObject* args) {
    const Py_ssize_t nargs = PyTuple_Size(args);
    const bool has_prepared_work = (nargs == 37 || nargs == 43 || nargs == 44);
    const bool has_ordered_links = (nargs == 43 || nargs == 44);
    if (nargs != 31 && nargs != 37 && nargs != 43 && nargs != 44) {
        PyErr_SetString(PyExc_TypeError, "sigma_compact_spin_string expects 31, 37, 43, or 44 positional arguments.");
        return nullptr;
    }
    int requested_workers = 1;
    if (nargs == 44) {
        const long parsed_workers = PyLong_AsLong(PyTuple_GET_ITEM(args, 43));
        if (PyErr_Occurred()) {
            return nullptr;
        }
        requested_workers = static_cast<int>(std::max<long>(1, parsed_workers));
    }

    ArrayRef h1(PyTuple_GET_ITEM(args, 0), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri_same(PyTuple_GET_ITEM(args, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri_cross(PyTuple_GET_ITEM(args, 2), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef hdiag(PyTuple_GET_ITEM(args, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef c(PyTuple_GET_ITEM(args, 4), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef alpha_occ(PyTuple_GET_ITEM(args, 5), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef beta_occ(PyTuple_GET_ITEM(args, 6), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_A(PyTuple_GET_ITEM(args, 7), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_A(PyTuple_GET_ITEM(args, 8), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_A(PyTuple_GET_ITEM(args, 9), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_A(PyTuple_GET_ITEM(args, 10), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_A(PyTuple_GET_ITEM(args, 11), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_B(PyTuple_GET_ITEM(args, 12), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_B(PyTuple_GET_ITEM(args, 13), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_B(PyTuple_GET_ITEM(args, 14), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_B(PyTuple_GET_ITEM(args, 15), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_B(PyTuple_GET_ITEM(args, 16), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_AA(PyTuple_GET_ITEM(args, 17), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_AA(PyTuple_GET_ITEM(args, 18), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_AA(PyTuple_GET_ITEM(args, 19), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_AA(PyTuple_GET_ITEM(args, 20), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef r_AA(PyTuple_GET_ITEM(args, 21), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef s_AA(PyTuple_GET_ITEM(args, 22), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_AA(PyTuple_GET_ITEM(args, 23), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_BB(PyTuple_GET_ITEM(args, 24), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_BB(PyTuple_GET_ITEM(args, 25), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_BB(PyTuple_GET_ITEM(args, 26), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_BB(PyTuple_GET_ITEM(args, 27), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef r_BB(PyTuple_GET_ITEM(args, 28), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef s_BB(PyTuple_GET_ITEM(args, 29), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_BB(PyTuple_GET_ITEM(args, 30), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef prepared_alpha_offsets;
    ArrayRef prepared_beta_offsets;
    ArrayRef prepared_alpha_order;
    ArrayRef prepared_beta_order;
    ArrayRef prepared_alpha_cross_diag;
    ArrayRef prepared_beta_cross_diag;
    ArrayRef prepared_alpha_i;
    ArrayRef prepared_alpha_j;
    ArrayRef prepared_alpha_phase;
    ArrayRef prepared_beta_i;
    ArrayRef prepared_beta_j;
    ArrayRef prepared_beta_phase;
    if (has_prepared_work) {
        prepared_alpha_offsets.reset(PyTuple_GET_ITEM(args, 31), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        prepared_beta_offsets.reset(PyTuple_GET_ITEM(args, 32), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        prepared_alpha_order.reset(PyTuple_GET_ITEM(args, 33), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        prepared_beta_order.reset(PyTuple_GET_ITEM(args, 34), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        prepared_alpha_cross_diag.reset(PyTuple_GET_ITEM(args, 35), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        prepared_beta_cross_diag.reset(PyTuple_GET_ITEM(args, 36), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    }
    if (has_ordered_links) {
        prepared_alpha_i.reset(PyTuple_GET_ITEM(args, 37), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        prepared_alpha_j.reset(PyTuple_GET_ITEM(args, 38), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        prepared_alpha_phase.reset(PyTuple_GET_ITEM(args, 39), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        prepared_beta_i.reset(PyTuple_GET_ITEM(args, 40), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        prepared_beta_j.reset(PyTuple_GET_ITEM(args, 41), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        prepared_beta_phase.reset(PyTuple_GET_ITEM(args, 42), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    }

    if (!h1 || !eri_same || !eri_cross || !hdiag || !c || !alpha_occ || !beta_occ ||
        !I_A || !J_A || !p_A || !q_A || !phase_A ||
        !I_B || !J_B || !p_B || !q_B || !phase_B ||
        !I_AA || !J_AA || !p_AA || !q_AA || !r_AA || !s_AA || !phase_AA ||
        !I_BB || !J_BB || !p_BB || !q_BB || !r_BB || !s_BB || !phase_BB ||
        (has_prepared_work && (
            !prepared_alpha_offsets || !prepared_beta_offsets ||
            !prepared_alpha_order || !prepared_beta_order ||
            !prepared_alpha_cross_diag || !prepared_beta_cross_diag
        )) ||
        (has_ordered_links && (
            !prepared_alpha_i || !prepared_alpha_j || !prepared_alpha_phase ||
            !prepared_beta_i || !prepared_beta_j || !prepared_beta_phase
        ))) {
        return nullptr;
    }
    if (
        !validate_spin_string_sigma_inputs(h1.obj, eri_same.obj, eri_cross.obj, hdiag.obj, c.obj, alpha_occ.obj, beta_occ.obj) ||
        !validate_link_group(I_A.obj, J_A.obj, p_A.obj, q_A.obj, phase_A.obj, "alpha") ||
        !validate_link_group(I_B.obj, J_B.obj, p_B.obj, q_B.obj, phase_B.obj, "beta") ||
        !validate_double_link_group(I_AA.obj, J_AA.obj, p_AA.obj, q_AA.obj, r_AA.obj, s_AA.obj, phase_AA.obj, "alpha-alpha") ||
        !validate_double_link_group(I_BB.obj, J_BB.obj, p_BB.obj, q_BB.obj, r_BB.obj, s_BB.obj, phase_BB.obj, "beta-beta")
    ) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h1.obj, 0);
    const npy_intp n_alpha = PyArray_DIM(alpha_occ.obj, 0);
    const npy_intp n_beta = PyArray_DIM(beta_occ.obj, 0);
    const npy_intp n_det = n_alpha * n_beta;
    npy_intp dims[1] = {n_det};
    PyObject* sigma_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (sigma_obj == nullptr) {
        return nullptr;
    }

    const double* h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
    const double* same = static_cast<const double*>(PyArray_DATA(eri_same.obj));
    const double* cross = static_cast<const double*>(PyArray_DATA(eri_cross.obj));
    const double* diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));
    const double* cvec = static_cast<const double*>(PyArray_DATA(c.obj));
    const npy_int8* alpha = static_cast<const npy_int8*>(PyArray_DATA(alpha_occ.obj));
    const npy_int8* beta = static_cast<const npy_int8*>(PyArray_DATA(beta_occ.obj));
    double* sigma = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(sigma_obj)));

    const npy_int32* i_a = static_cast<const npy_int32*>(PyArray_DATA(I_A.obj));
    const npy_int32* j_a = static_cast<const npy_int32*>(PyArray_DATA(J_A.obj));
    const npy_int32* pa = static_cast<const npy_int32*>(PyArray_DATA(p_A.obj));
    const npy_int32* qa = static_cast<const npy_int32*>(PyArray_DATA(q_A.obj));
    const npy_int8* phase_a = static_cast<const npy_int8*>(PyArray_DATA(phase_A.obj));

    const npy_int32* i_b = static_cast<const npy_int32*>(PyArray_DATA(I_B.obj));
    const npy_int32* j_b = static_cast<const npy_int32*>(PyArray_DATA(J_B.obj));
    const npy_int32* pb = static_cast<const npy_int32*>(PyArray_DATA(p_B.obj));
    const npy_int32* qb = static_cast<const npy_int32*>(PyArray_DATA(q_B.obj));
    const npy_int8* phase_b = static_cast<const npy_int8*>(PyArray_DATA(phase_B.obj));

    const npy_int32* i_aa = static_cast<const npy_int32*>(PyArray_DATA(I_AA.obj));
    const npy_int32* j_aa = static_cast<const npy_int32*>(PyArray_DATA(J_AA.obj));
    const npy_int32* p_aa = static_cast<const npy_int32*>(PyArray_DATA(p_AA.obj));
    const npy_int32* q_aa = static_cast<const npy_int32*>(PyArray_DATA(q_AA.obj));
    const npy_int32* r_aa = static_cast<const npy_int32*>(PyArray_DATA(r_AA.obj));
    const npy_int32* s_aa = static_cast<const npy_int32*>(PyArray_DATA(s_AA.obj));
    const npy_int8* phase_aa = static_cast<const npy_int8*>(PyArray_DATA(phase_AA.obj));

    const npy_int32* i_bb = static_cast<const npy_int32*>(PyArray_DATA(I_BB.obj));
    const npy_int32* j_bb = static_cast<const npy_int32*>(PyArray_DATA(J_BB.obj));
    const npy_int32* p_bb = static_cast<const npy_int32*>(PyArray_DATA(p_BB.obj));
    const npy_int32* q_bb = static_cast<const npy_int32*>(PyArray_DATA(q_BB.obj));
    const npy_int32* r_bb = static_cast<const npy_int32*>(PyArray_DATA(r_BB.obj));
    const npy_int32* s_bb = static_cast<const npy_int32*>(PyArray_DATA(s_BB.obj));
    const npy_int8* phase_bb = static_cast<const npy_int8*>(PyArray_DATA(phase_BB.obj));

    const npy_intp n_link_a = PyArray_DIM(I_A.obj, 0);
    const npy_intp n_link_b = PyArray_DIM(I_B.obj, 0);
    const npy_intp n_link_aa = PyArray_DIM(I_AA.obj, 0);
    const npy_intp n_link_bb = PyArray_DIM(I_BB.obj, 0);
    const npy_intp n_transition = n * n;
    if (has_prepared_work) {
        if (
            PyArray_NDIM(prepared_alpha_offsets.obj) != 1 ||
            PyArray_NDIM(prepared_beta_offsets.obj) != 1 ||
            PyArray_NDIM(prepared_alpha_order.obj) != 1 ||
            PyArray_NDIM(prepared_beta_order.obj) != 1 ||
            PyArray_NDIM(prepared_alpha_cross_diag.obj) != 2 ||
            PyArray_NDIM(prepared_beta_cross_diag.obj) != 2 ||
            PyArray_DIM(prepared_alpha_offsets.obj, 0) != n_transition + 1 ||
            PyArray_DIM(prepared_beta_offsets.obj, 0) != n_transition + 1 ||
            PyArray_DIM(prepared_alpha_order.obj, 0) != n_link_a ||
            PyArray_DIM(prepared_beta_order.obj, 0) != n_link_b ||
            PyArray_DIM(prepared_alpha_cross_diag.obj, 0) != n_transition ||
            PyArray_DIM(prepared_alpha_cross_diag.obj, 1) != n_alpha ||
            PyArray_DIM(prepared_beta_cross_diag.obj, 0) != n_transition ||
            PyArray_DIM(prepared_beta_cross_diag.obj, 1) != n_beta
        ) {
            Py_DECREF(sigma_obj);
            PyErr_SetString(PyExc_ValueError, "Prepared spin-string sigma work arrays have inconsistent shapes.");
            return nullptr;
        }
    }
    if (has_ordered_links) {
        if (
            PyArray_NDIM(prepared_alpha_i.obj) != 1 ||
            PyArray_NDIM(prepared_alpha_j.obj) != 1 ||
            PyArray_NDIM(prepared_alpha_phase.obj) != 1 ||
            PyArray_NDIM(prepared_beta_i.obj) != 1 ||
            PyArray_NDIM(prepared_beta_j.obj) != 1 ||
            PyArray_NDIM(prepared_beta_phase.obj) != 1 ||
            PyArray_DIM(prepared_alpha_i.obj, 0) != n_link_a ||
            PyArray_DIM(prepared_alpha_j.obj, 0) != n_link_a ||
            PyArray_DIM(prepared_alpha_phase.obj, 0) != n_link_a ||
            PyArray_DIM(prepared_beta_i.obj, 0) != n_link_b ||
            PyArray_DIM(prepared_beta_j.obj, 0) != n_link_b ||
            PyArray_DIM(prepared_beta_phase.obj, 0) != n_link_b
        ) {
            Py_DECREF(sigma_obj);
            PyErr_SetString(PyExc_ValueError, "Prepared ordered spin-string link arrays have inconsistent shapes.");
            return nullptr;
        }
    }
    const npy_int32* ordered_i_a = nullptr;
    const npy_int32* ordered_j_a = nullptr;
    const npy_int8* ordered_phase_a = nullptr;
    const npy_int32* ordered_i_b = nullptr;
    const npy_int32* ordered_j_b = nullptr;
    const npy_int8* ordered_phase_b = nullptr;
    if (has_ordered_links) {
        ordered_i_a = static_cast<const npy_int32*>(PyArray_DATA(prepared_alpha_i.obj));
        ordered_j_a = static_cast<const npy_int32*>(PyArray_DATA(prepared_alpha_j.obj));
        ordered_phase_a = static_cast<const npy_int8*>(PyArray_DATA(prepared_alpha_phase.obj));
        ordered_i_b = static_cast<const npy_int32*>(PyArray_DATA(prepared_beta_i.obj));
        ordered_j_b = static_cast<const npy_int32*>(PyArray_DATA(prepared_beta_j.obj));
        ordered_phase_b = static_cast<const npy_int8*>(PyArray_DATA(prepared_beta_phase.obj));
    }

    std::vector<double> alpha_cross_diag;
    std::vector<double> beta_cross_diag;
    std::vector<npy_intp> alpha_offsets;
    std::vector<npy_intp> beta_offsets;
    std::vector<npy_intp> alpha_order;
    std::vector<npy_intp> beta_order;
    const double* alpha_cross_diag_data = nullptr;
    const double* beta_cross_diag_data = nullptr;
    const npy_intp* alpha_offsets_data = nullptr;
    const npy_intp* beta_offsets_data = nullptr;
    const npy_intp* alpha_order_data = nullptr;
    const npy_intp* beta_order_data = nullptr;
    try {
        if (has_prepared_work) {
            alpha_cross_diag_data = static_cast<const double*>(PyArray_DATA(prepared_alpha_cross_diag.obj));
            beta_cross_diag_data = static_cast<const double*>(PyArray_DATA(prepared_beta_cross_diag.obj));
            alpha_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(prepared_alpha_offsets.obj));
            beta_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(prepared_beta_offsets.obj));
            alpha_order_data = static_cast<const npy_intp*>(PyArray_DATA(prepared_alpha_order.obj));
            beta_order_data = static_cast<const npy_intp*>(PyArray_DATA(prepared_beta_order.obj));
        } else {
            alpha_cross_diag.assign(
                static_cast<std::size_t>(n_transition) * static_cast<std::size_t>(n_alpha),
                0.0
            );
            beta_cross_diag.assign(
                static_cast<std::size_t>(n_transition) * static_cast<std::size_t>(n_beta),
                0.0
            );
            for (npy_intp p = 0; p < n; ++p) {
                for (npy_intp q = 0; q < n; ++q) {
                    const npy_intp transition = p * n + q;
                    double* alpha_diag = alpha_cross_diag.data() + transition * n_alpha;
                    double* beta_diag = beta_cross_diag.data() + transition * n_beta;
                    for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                        double value = 0.0;
                        for (npy_intp r = 0; r < n; ++r) {
                            if (alpha[idx_occ(n, ia, r)]) {
                                value += cross[idx4(n, p, q, r, r)];
                            }
                        }
                        alpha_diag[ia] = value;
                    }
                    for (npy_intp ib = 0; ib < n_beta; ++ib) {
                        double value = 0.0;
                        for (npy_intp r = 0; r < n; ++r) {
                            if (beta[idx_occ(n, ib, r)]) {
                                value += cross[idx4(n, p, q, r, r)];
                            }
                        }
                        beta_diag[ib] = value;
                    }
                }
            }

            std::vector<npy_intp> alpha_counts(static_cast<std::size_t>(n_transition), 0);
            std::vector<npy_intp> beta_counts(static_cast<std::size_t>(n_transition), 0);
            for (npy_intp link = 0; link < n_link_a; ++link) {
                ++alpha_counts[static_cast<std::size_t>(pa[link] * n + qa[link])];
            }
            for (npy_intp link = 0; link < n_link_b; ++link) {
                ++beta_counts[static_cast<std::size_t>(pb[link] * n + qb[link])];
            }
            alpha_offsets.assign(static_cast<std::size_t>(n_transition) + 1, 0);
            beta_offsets.assign(static_cast<std::size_t>(n_transition) + 1, 0);
            for (npy_intp transition = 0; transition < n_transition; ++transition) {
                alpha_offsets[static_cast<std::size_t>(transition) + 1] =
                    alpha_offsets[static_cast<std::size_t>(transition)] +
                    alpha_counts[static_cast<std::size_t>(transition)];
                beta_offsets[static_cast<std::size_t>(transition) + 1] =
                    beta_offsets[static_cast<std::size_t>(transition)] +
                    beta_counts[static_cast<std::size_t>(transition)];
            }
            alpha_order.assign(static_cast<std::size_t>(n_link_a), 0);
            beta_order.assign(static_cast<std::size_t>(n_link_b), 0);
            std::vector<npy_intp> alpha_cursor = alpha_offsets;
            std::vector<npy_intp> beta_cursor = beta_offsets;
            for (npy_intp link = 0; link < n_link_a; ++link) {
                const npy_intp transition = pa[link] * n + qa[link];
                alpha_order[static_cast<std::size_t>(alpha_cursor[static_cast<std::size_t>(transition)]++)] = link;
            }
            for (npy_intp link = 0; link < n_link_b; ++link) {
                const npy_intp transition = pb[link] * n + qb[link];
                beta_order[static_cast<std::size_t>(beta_cursor[static_cast<std::size_t>(transition)]++)] = link;
            }
            alpha_cross_diag_data = alpha_cross_diag.data();
            beta_cross_diag_data = beta_cross_diag.data();
            alpha_offsets_data = alpha_offsets.data();
            beta_offsets_data = beta_offsets.data();
            alpha_order_data = alpha_order.data();
            beta_order_data = beta_order.data();
        }
    } catch (...) {
        Py_DECREF(sigma_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate spin-string sigma work arrays.");
        return nullptr;
    }

    int worker_count = 1;
    if (has_ordered_links && requested_workers > 1 && n_transition > 1) {
        worker_count = std::min<int>(requested_workers, static_cast<int>(n_transition));
    }
    std::vector<double> cross_sigma_buffers;
    try {
        if (worker_count > 1) {
            cross_sigma_buffers.assign(
                static_cast<std::size_t>(worker_count) * static_cast<std::size_t>(n_det),
                0.0
            );
        }
    } catch (...) {
        Py_DECREF(sigma_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate parallel spin-string sigma buffers.");
        return nullptr;
    }
    std::vector<std::thread> cross_threads;
    if (worker_count > 1) {
        try {
            cross_threads.reserve(static_cast<std::size_t>(worker_count - 1));
        } catch (...) {
            Py_DECREF(sigma_obj);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate parallel spin-string worker handles.");
            return nullptr;
        }
    }

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp det = 0; det < n_det; ++det) {
        sigma[det] = diag[det] * cvec[det];
    }

    for (npy_intp link = 0; link < n_link_a; ++link) {
        const npy_intp ia = i_a[link];
        const npy_intp ja = j_a[link];
        const npy_intp p = pa[link];
        const npy_intp q = qa[link];
        const double sign = static_cast<double>(phase_a[link]);
        double same_part = -sign * h1_data[idx2(n, p, q)];
        for (npy_intp r = 0; r < n; ++r) {
            if (alpha[idx_occ(n, ja, r)] && r != q) {
                same_part -= sign * same[idx4(n, p, q, r, r)];
            }
        }
        const double* beta_diag = beta_cross_diag_data + (p * n + q) * n_beta;
        double* sigma_row = sigma + ia * n_beta;
        const double* c_row = cvec + ja * n_beta;
        for (npy_intp ib = 0; ib < n_beta; ++ib) {
            sigma_row[ib] += (same_part - sign * beta_diag[ib]) * c_row[ib];
        }
    }

    for (npy_intp link = 0; link < n_link_b; ++link) {
        const npy_intp ib = i_b[link];
        const npy_intp jb = j_b[link];
        const npy_intp p = pb[link];
        const npy_intp q = qb[link];
        const double sign = static_cast<double>(phase_b[link]);
        double same_part = -sign * h1_data[idx2(n, p, q)];
        for (npy_intp r = 0; r < n; ++r) {
            if (beta[idx_occ(n, jb, r)] && r != q) {
                same_part -= sign * same[idx4(n, p, q, r, r)];
            }
        }
        const double* alpha_diag = alpha_cross_diag_data + (p * n + q) * n_alpha;
        for (npy_intp ia = 0; ia < n_alpha; ++ia) {
            sigma[ia * n_beta + ib] +=
                (same_part - sign * alpha_diag[ia]) * cvec[ia * n_beta + jb];
        }
    }

    for (npy_intp link = 0; link < n_link_aa; ++link) {
        const npy_intp ia = i_aa[link];
        const npy_intp ja = j_aa[link];
        const double val = (
            static_cast<double>(phase_aa[link]) *
            same[idx4(n, p_aa[link], q_aa[link], r_aa[link], s_aa[link])]
        );
        for (npy_intp ib = 0; ib < n_beta; ++ib) {
            sigma[ia * n_beta + ib] += val * cvec[ja * n_beta + ib];
        }
    }

    for (npy_intp link = 0; link < n_link_bb; ++link) {
        const npy_intp ib = i_bb[link];
        const npy_intp jb = j_bb[link];
        const double val = (
            static_cast<double>(phase_bb[link]) *
            same[idx4(n, p_bb[link], q_bb[link], r_bb[link], s_bb[link])]
        );
        for (npy_intp ia = 0; ia < n_alpha; ++ia) {
            sigma[ia * n_beta + ib] += val * cvec[ia * n_beta + jb];
        }
    }

    if (has_ordered_links && worker_count > 1) {
        auto cross_worker = [&](int worker_id) {
            double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (
                npy_intp transition_a = worker_id;
                transition_a < n_transition;
                transition_a += worker_count
            ) {
                const npy_intp alpha_begin = alpha_offsets_data[transition_a];
                const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
                if (alpha_begin == alpha_end) {
                    continue;
                }
                const npy_intp p = transition_a / n;
                const npy_intp q = transition_a - p * n;
                for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                    const npy_intp beta_begin = beta_offsets_data[transition_b];
                    const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                    if (beta_begin == beta_end) {
                        continue;
                    }
                    const npy_intp r = transition_b / n;
                    const npy_intp s = transition_b - r * n;
                    const double eri_value = cross[idx4(n, p, q, r, s)];
                    if (eri_value == 0.0) {
                        continue;
                    }
                    for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                        const npy_intp ia = ordered_i_a[alpha_pos];
                        const npy_intp ja = ordered_j_a[alpha_pos];
                        const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                        double* sigma_row = local_sigma + ia * n_beta;
                        const double* c_row = cvec + ja * n_beta;
                        for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                            sigma_row[ordered_i_b[beta_pos]] +=
                                alpha_value *
                                static_cast<double>(ordered_phase_b[beta_pos]) *
                                c_row[ordered_j_b[beta_pos]];
                        }
                    }
                }
            }
        };

        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            cross_threads.emplace_back(cross_worker, worker_id);
        }
        cross_worker(0);
        for (std::thread& thread : cross_threads) {
            thread.join();
        }
        for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
            const double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (npy_intp det = 0; det < n_det; ++det) {
                sigma[det] += local_sigma[det];
            }
        }
    } else if (has_ordered_links) {
        for (npy_intp transition_a = 0; transition_a < n_transition; ++transition_a) {
            const npy_intp alpha_begin = alpha_offsets_data[transition_a];
            const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
            if (alpha_begin == alpha_end) {
                continue;
            }
            const npy_intp p = transition_a / n;
            const npy_intp q = transition_a - p * n;
            for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                const npy_intp beta_begin = beta_offsets_data[transition_b];
                const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                if (beta_begin == beta_end) {
                    continue;
                }
                const npy_intp r = transition_b / n;
                const npy_intp s = transition_b - r * n;
                const double eri_value = cross[idx4(n, p, q, r, s)];
                if (eri_value == 0.0) {
                    continue;
                }
                for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                    const npy_intp ia = ordered_i_a[alpha_pos];
                    const npy_intp ja = ordered_j_a[alpha_pos];
                    const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                    double* sigma_row = sigma + ia * n_beta;
                    const double* c_row = cvec + ja * n_beta;
                    for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                        sigma_row[ordered_i_b[beta_pos]] +=
                            alpha_value *
                            static_cast<double>(ordered_phase_b[beta_pos]) *
                            c_row[ordered_j_b[beta_pos]];
                    }
                }
            }
        }
    } else {
        for (npy_intp transition_a = 0; transition_a < n_transition; ++transition_a) {
            const npy_intp alpha_begin = alpha_offsets_data[transition_a];
            const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
            if (alpha_begin == alpha_end) {
                continue;
            }
            const npy_intp p = transition_a / n;
            const npy_intp q = transition_a - p * n;
            for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                const npy_intp beta_begin = beta_offsets_data[transition_b];
                const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                if (beta_begin == beta_end) {
                    continue;
                }
                const npy_intp r = transition_b / n;
                const npy_intp s = transition_b - r * n;
                const double eri_value = cross[idx4(n, p, q, r, s)];
                if (eri_value == 0.0) {
                    continue;
                }
                for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                    const npy_intp la = alpha_order_data[alpha_pos];
                    const npy_intp ia = i_a[la];
                    const npy_intp ja = j_a[la];
                    const double alpha_value = eri_value * static_cast<double>(phase_a[la]);
                    double* sigma_row = sigma + ia * n_beta;
                    const double* c_row = cvec + ja * n_beta;
                    for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                        const npy_intp lb = beta_order_data[beta_pos];
                        sigma_row[i_b[lb]] +=
                            alpha_value *
                            static_cast<double>(phase_b[lb]) *
                            c_row[j_b[lb]];
                    }
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    return sigma_obj;
}

PyObject* sigma_compact_spin0_pair(PyObject*, PyObject* args) {
    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != 45 && nargs != 46) {
        PyErr_SetString(PyExc_TypeError, "sigma_compact_spin0_pair expects 45 or 46 positional arguments.");
        return nullptr;
    }
    int requested_workers = 1;
    if (nargs == 46) {
        const long parsed_workers = PyLong_AsLong(PyTuple_GET_ITEM(args, 45));
        if (PyErr_Occurred()) {
            return nullptr;
        }
        requested_workers = static_cast<int>(std::max<long>(1, parsed_workers));
    }

    ArrayRef h1(PyTuple_GET_ITEM(args, 0), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri_same(PyTuple_GET_ITEM(args, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri_cross(PyTuple_GET_ITEM(args, 2), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef hdiag(PyTuple_GET_ITEM(args, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef c_pair(PyTuple_GET_ITEM(args, 4), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_left(PyTuple_GET_ITEM(args, 5), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_right(PyTuple_GET_ITEM(args, 6), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef alpha_occ(PyTuple_GET_ITEM(args, 7), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef beta_occ(PyTuple_GET_ITEM(args, 8), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_A(PyTuple_GET_ITEM(args, 9), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_A(PyTuple_GET_ITEM(args, 10), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_A(PyTuple_GET_ITEM(args, 11), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_A(PyTuple_GET_ITEM(args, 12), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_A(PyTuple_GET_ITEM(args, 13), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_B(PyTuple_GET_ITEM(args, 14), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_B(PyTuple_GET_ITEM(args, 15), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_B(PyTuple_GET_ITEM(args, 16), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_B(PyTuple_GET_ITEM(args, 17), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_B(PyTuple_GET_ITEM(args, 18), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_AA(PyTuple_GET_ITEM(args, 19), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_AA(PyTuple_GET_ITEM(args, 20), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_AA(PyTuple_GET_ITEM(args, 21), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_AA(PyTuple_GET_ITEM(args, 22), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef r_AA(PyTuple_GET_ITEM(args, 23), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef s_AA(PyTuple_GET_ITEM(args, 24), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_AA(PyTuple_GET_ITEM(args, 25), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef I_BB(PyTuple_GET_ITEM(args, 26), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef J_BB(PyTuple_GET_ITEM(args, 27), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef p_BB(PyTuple_GET_ITEM(args, 28), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef q_BB(PyTuple_GET_ITEM(args, 29), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef r_BB(PyTuple_GET_ITEM(args, 30), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef s_BB(PyTuple_GET_ITEM(args, 31), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_BB(PyTuple_GET_ITEM(args, 32), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    ArrayRef alpha_offsets(PyTuple_GET_ITEM(args, 33), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef beta_offsets(PyTuple_GET_ITEM(args, 34), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef alpha_order(PyTuple_GET_ITEM(args, 35), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef beta_order(PyTuple_GET_ITEM(args, 36), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef alpha_cross_diag(PyTuple_GET_ITEM(args, 37), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef beta_cross_diag(PyTuple_GET_ITEM(args, 38), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_alpha_i(PyTuple_GET_ITEM(args, 39), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_alpha_j(PyTuple_GET_ITEM(args, 40), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_alpha_phase(PyTuple_GET_ITEM(args, 41), NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_beta_i(PyTuple_GET_ITEM(args, 42), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_beta_j(PyTuple_GET_ITEM(args, 43), NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef ordered_beta_phase(PyTuple_GET_ITEM(args, 44), NPY_INT8, NPY_ARRAY_IN_ARRAY);

    if (!h1 || !eri_same || !eri_cross || !hdiag || !c_pair || !pair_left || !pair_right ||
        !alpha_occ || !beta_occ ||
        !I_A || !J_A || !p_A || !q_A || !phase_A ||
        !I_B || !J_B || !p_B || !q_B || !phase_B ||
        !I_AA || !J_AA || !p_AA || !q_AA || !r_AA || !s_AA || !phase_AA ||
        !I_BB || !J_BB || !p_BB || !q_BB || !r_BB || !s_BB || !phase_BB ||
        !alpha_offsets || !beta_offsets || !alpha_order || !beta_order ||
        !alpha_cross_diag || !beta_cross_diag ||
        !ordered_alpha_i || !ordered_alpha_j || !ordered_alpha_phase ||
        !ordered_beta_i || !ordered_beta_j || !ordered_beta_phase) {
        return nullptr;
    }
    if (
        !validate_link_group(I_A.obj, J_A.obj, p_A.obj, q_A.obj, phase_A.obj, "alpha") ||
        !validate_link_group(I_B.obj, J_B.obj, p_B.obj, q_B.obj, phase_B.obj, "beta") ||
        !validate_double_link_group(I_AA.obj, J_AA.obj, p_AA.obj, q_AA.obj, r_AA.obj, s_AA.obj, phase_AA.obj, "alpha-alpha") ||
        !validate_double_link_group(I_BB.obj, J_BB.obj, p_BB.obj, q_BB.obj, r_BB.obj, s_BB.obj, phase_BB.obj, "beta-beta")
    ) {
        return nullptr;
    }
    if (
        PyArray_NDIM(h1.obj) != 2 ||
        PyArray_NDIM(eri_same.obj) != 4 ||
        PyArray_NDIM(eri_cross.obj) != 4 ||
        PyArray_NDIM(hdiag.obj) != 1 ||
        PyArray_NDIM(c_pair.obj) != 1 ||
        PyArray_NDIM(pair_left.obj) != 1 ||
        PyArray_NDIM(pair_right.obj) != 1 ||
        PyArray_NDIM(alpha_occ.obj) != 2 ||
        PyArray_NDIM(beta_occ.obj) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "sigma_compact_spin0_pair received arrays with invalid ranks.");
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h1.obj, 0);
    const npy_intp n_alpha = PyArray_DIM(alpha_occ.obj, 0);
    const npy_intp n_beta = PyArray_DIM(beta_occ.obj, 0);
    const npy_intp n_det = n_alpha * n_beta;
    const npy_intp n_pair = PyArray_DIM(c_pair.obj, 0);
    const npy_intp n_transition = n * n;
    const npy_intp n_link_a = PyArray_DIM(I_A.obj, 0);
    const npy_intp n_link_b = PyArray_DIM(I_B.obj, 0);
    const npy_intp n_link_aa = PyArray_DIM(I_AA.obj, 0);
    const npy_intp n_link_bb = PyArray_DIM(I_BB.obj, 0);
    if (
        PyArray_DIM(h1.obj, 1) != n ||
        PyArray_DIM(alpha_occ.obj, 1) != n ||
        PyArray_DIM(beta_occ.obj, 1) != n ||
        PyArray_DIM(hdiag.obj, 0) != n_det ||
        PyArray_DIM(pair_left.obj, 0) != n_pair ||
        PyArray_DIM(pair_right.obj, 0) != n_pair ||
        PyArray_NDIM(alpha_offsets.obj) != 1 ||
        PyArray_NDIM(beta_offsets.obj) != 1 ||
        PyArray_NDIM(alpha_order.obj) != 1 ||
        PyArray_NDIM(beta_order.obj) != 1 ||
        PyArray_NDIM(alpha_cross_diag.obj) != 2 ||
        PyArray_NDIM(beta_cross_diag.obj) != 2 ||
        PyArray_DIM(alpha_offsets.obj, 0) != n_transition + 1 ||
        PyArray_DIM(beta_offsets.obj, 0) != n_transition + 1 ||
        PyArray_DIM(alpha_order.obj, 0) != n_link_a ||
        PyArray_DIM(beta_order.obj, 0) != n_link_b ||
        PyArray_DIM(alpha_cross_diag.obj, 0) != n_transition ||
        PyArray_DIM(alpha_cross_diag.obj, 1) != n_alpha ||
        PyArray_DIM(beta_cross_diag.obj, 0) != n_transition ||
        PyArray_DIM(beta_cross_diag.obj, 1) != n_beta ||
        PyArray_NDIM(ordered_alpha_i.obj) != 1 ||
        PyArray_NDIM(ordered_alpha_j.obj) != 1 ||
        PyArray_NDIM(ordered_alpha_phase.obj) != 1 ||
        PyArray_NDIM(ordered_beta_i.obj) != 1 ||
        PyArray_NDIM(ordered_beta_j.obj) != 1 ||
        PyArray_NDIM(ordered_beta_phase.obj) != 1 ||
        PyArray_DIM(ordered_alpha_i.obj, 0) != n_link_a ||
        PyArray_DIM(ordered_alpha_j.obj, 0) != n_link_a ||
        PyArray_DIM(ordered_alpha_phase.obj, 0) != n_link_a ||
        PyArray_DIM(ordered_beta_i.obj, 0) != n_link_b ||
        PyArray_DIM(ordered_beta_j.obj, 0) != n_link_b ||
        PyArray_DIM(ordered_beta_phase.obj, 0) != n_link_b
    ) {
        PyErr_SetString(PyExc_ValueError, "sigma_compact_spin0_pair array shapes are inconsistent.");
        return nullptr;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (PyArray_DIM(eri_same.obj, axis) != n || PyArray_DIM(eri_cross.obj, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "ERI tensors must have shape (norb, norb, norb, norb).");
            return nullptr;
        }
    }

    npy_intp dims[1] = {n_pair};
    PyObject* sigma_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (sigma_obj == nullptr) {
        return nullptr;
    }

    const npy_intp* left = static_cast<const npy_intp*>(PyArray_DATA(pair_left.obj));
    const npy_intp* right = static_cast<const npy_intp*>(PyArray_DATA(pair_right.obj));
    constexpr double inv_sqrt2 = 0.70710678118654752440084436210484903928;
    for (npy_intp pair = 0; pair < n_pair; ++pair) {
        const npy_intp ldet = left[pair];
        const npy_intp rdet = right[pair];
        if (ldet < 0 || ldet >= n_det || rdet < 0 || rdet >= n_det) {
            Py_DECREF(sigma_obj);
            PyErr_SetString(PyExc_ValueError, "Spin0 pair indices are out of range.");
            return nullptr;
        }
    }

    const double* h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
    const double* same = static_cast<const double*>(PyArray_DATA(eri_same.obj));
    const double* cross = static_cast<const double*>(PyArray_DATA(eri_cross.obj));
    const double* diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));
    const double* cdata = static_cast<const double*>(PyArray_DATA(c_pair.obj));
    const npy_int8* alpha = static_cast<const npy_int8*>(PyArray_DATA(alpha_occ.obj));
    const npy_int8* beta = static_cast<const npy_int8*>(PyArray_DATA(beta_occ.obj));
    double* sigma_pair = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(sigma_obj)));

    const npy_int32* i_a = static_cast<const npy_int32*>(PyArray_DATA(I_A.obj));
    const npy_int32* j_a = static_cast<const npy_int32*>(PyArray_DATA(J_A.obj));
    const npy_int32* pa = static_cast<const npy_int32*>(PyArray_DATA(p_A.obj));
    const npy_int32* qa = static_cast<const npy_int32*>(PyArray_DATA(q_A.obj));
    const npy_int8* phase_a = static_cast<const npy_int8*>(PyArray_DATA(phase_A.obj));
    const npy_int32* i_b = static_cast<const npy_int32*>(PyArray_DATA(I_B.obj));
    const npy_int32* j_b = static_cast<const npy_int32*>(PyArray_DATA(J_B.obj));
    const npy_int32* pb = static_cast<const npy_int32*>(PyArray_DATA(p_B.obj));
    const npy_int32* qb = static_cast<const npy_int32*>(PyArray_DATA(q_B.obj));
    const npy_int8* phase_b = static_cast<const npy_int8*>(PyArray_DATA(phase_B.obj));
    const npy_int32* i_aa = static_cast<const npy_int32*>(PyArray_DATA(I_AA.obj));
    const npy_int32* j_aa = static_cast<const npy_int32*>(PyArray_DATA(J_AA.obj));
    const npy_int32* p_aa = static_cast<const npy_int32*>(PyArray_DATA(p_AA.obj));
    const npy_int32* q_aa = static_cast<const npy_int32*>(PyArray_DATA(q_AA.obj));
    const npy_int32* r_aa = static_cast<const npy_int32*>(PyArray_DATA(r_AA.obj));
    const npy_int32* s_aa = static_cast<const npy_int32*>(PyArray_DATA(s_AA.obj));
    const npy_int8* phase_aa = static_cast<const npy_int8*>(PyArray_DATA(phase_AA.obj));
    const npy_int32* i_bb = static_cast<const npy_int32*>(PyArray_DATA(I_BB.obj));
    const npy_int32* j_bb = static_cast<const npy_int32*>(PyArray_DATA(J_BB.obj));
    const npy_int32* p_bb = static_cast<const npy_int32*>(PyArray_DATA(p_BB.obj));
    const npy_int32* q_bb = static_cast<const npy_int32*>(PyArray_DATA(q_BB.obj));
    const npy_int32* r_bb = static_cast<const npy_int32*>(PyArray_DATA(r_BB.obj));
    const npy_int32* s_bb = static_cast<const npy_int32*>(PyArray_DATA(s_BB.obj));
    const npy_int8* phase_bb = static_cast<const npy_int8*>(PyArray_DATA(phase_BB.obj));
    const npy_intp* alpha_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(alpha_offsets.obj));
    const npy_intp* beta_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(beta_offsets.obj));
    const double* alpha_cross_diag_data = static_cast<const double*>(PyArray_DATA(alpha_cross_diag.obj));
    const double* beta_cross_diag_data = static_cast<const double*>(PyArray_DATA(beta_cross_diag.obj));
    const npy_int32* ordered_i_a = static_cast<const npy_int32*>(PyArray_DATA(ordered_alpha_i.obj));
    const npy_int32* ordered_j_a = static_cast<const npy_int32*>(PyArray_DATA(ordered_alpha_j.obj));
    const npy_int8* ordered_phase_a = static_cast<const npy_int8*>(PyArray_DATA(ordered_alpha_phase.obj));
    const npy_int32* ordered_i_b = static_cast<const npy_int32*>(PyArray_DATA(ordered_beta_i.obj));
    const npy_int32* ordered_j_b = static_cast<const npy_int32*>(PyArray_DATA(ordered_beta_j.obj));
    const npy_int8* ordered_phase_b = static_cast<const npy_int8*>(PyArray_DATA(ordered_beta_phase.obj));

    auto equal_bytes = [](const void* lhs, const void* rhs, std::size_t nbytes) {
        return nbytes == 0 || std::memcmp(lhs, rhs, nbytes) == 0;
    };
    bool use_reduced_cross = (
        n_alpha == n_beta &&
        n_pair == (n_alpha * (n_alpha + 1)) / 2 &&
        n_link_a == n_link_b &&
        equal_bytes(alpha, beta, static_cast<std::size_t>(n_alpha) * static_cast<std::size_t>(n) * sizeof(npy_int8)) &&
        equal_bytes(
            alpha_offsets_data,
            beta_offsets_data,
            static_cast<std::size_t>(n_transition + 1) * sizeof(npy_intp)
        ) &&
        equal_bytes(
            ordered_i_a,
            ordered_i_b,
            static_cast<std::size_t>(n_link_a) * sizeof(npy_int32)
        ) &&
        equal_bytes(
            ordered_j_a,
            ordered_j_b,
            static_cast<std::size_t>(n_link_a) * sizeof(npy_int32)
        ) &&
        equal_bytes(
            ordered_phase_a,
            ordered_phase_b,
            static_cast<std::size_t>(n_link_a) * sizeof(npy_int8)
        )
    );

    int worker_count = 1;
    if (requested_workers > 1 && n_transition > 1) {
        worker_count = std::min<int>(requested_workers, static_cast<int>(n_transition));
    }
    static thread_local std::vector<double> c_det;
    static thread_local std::vector<double> sigma_det;
    static thread_local std::vector<double> cross_sigma_buffers;
    static thread_local std::vector<npy_intp> pair_row_start;
    static thread_local std::vector<uint32_t> pair_index_lookup;
    static thread_local std::vector<double> pair_scale_lookup;
    static thread_local std::vector<double> pair_coeff_lookup;
    std::vector<std::thread> worker_threads;
    std::vector<std::thread> cross_threads;
    try {
        c_det.assign(static_cast<std::size_t>(n_det), 0.0);
        sigma_det.resize(static_cast<std::size_t>(n_det));
        pair_row_start.clear();
        pair_index_lookup.clear();
        pair_scale_lookup.clear();
        pair_coeff_lookup.clear();
        cross_sigma_buffers.clear();
        if (use_reduced_cross) {
            for (npy_intp ia = 0; ia < n_alpha && use_reduced_cross; ++ia) {
                const npy_intp row_start = ia * n_alpha - (ia * (ia - 1)) / 2;
                for (npy_intp ib = ia; ib < n_alpha; ++ib) {
                    const npy_intp pair = row_start + (ib - ia);
                    if (
                        left[pair] != ia * n_beta + ib ||
                        right[pair] != ib * n_beta + ia
                    ) {
                        use_reduced_cross = false;
                        break;
                    }
                }
            }
        }
        if (use_reduced_cross) {
            pair_row_start.assign(static_cast<std::size_t>(n_alpha), 0);
            for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                pair_row_start[static_cast<std::size_t>(ia)] =
                    ia * n_alpha - (ia * (ia - 1)) / 2;
            }
        }
        constexpr std::size_t pair_lookup_limit = 128'000;
        const std::size_t pair_lookup_size =
            static_cast<std::size_t>(n_alpha) * static_cast<std::size_t>(n_alpha);
        if (use_reduced_cross && pair_lookup_size <= pair_lookup_limit) {
            pair_index_lookup.assign(pair_lookup_size, 0);
            pair_scale_lookup.assign(pair_lookup_size, 0.0);
            pair_coeff_lookup.assign(pair_lookup_size, 0.0);
        }
        if (worker_count > 1) {
            cross_sigma_buffers.assign(
                static_cast<std::size_t>(worker_count) *
                    static_cast<std::size_t>(use_reduced_cross ? n_pair : n_det),
                0.0
            );
            worker_threads.reserve(static_cast<std::size_t>(worker_count - 1));
            cross_threads.reserve(static_cast<std::size_t>(worker_count - 1));
        }
    } catch (...) {
        Py_DECREF(sigma_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate spin0 sigma work arrays.");
        return nullptr;
    }

    double* cvec = c_det.data();
    double* sigma = sigma_det.data();

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp pair = 0; pair < n_pair; ++pair) {
        const npy_intp ldet = left[pair];
        const npy_intp rdet = right[pair];
        if (ldet == rdet) {
            cvec[ldet] = cdata[pair];
        } else {
            const double scaled = cdata[pair] * inv_sqrt2;
            cvec[ldet] = scaled;
            cvec[rdet] = scaled;
        }
    }

    if (use_reduced_cross && !pair_index_lookup.empty()) {
        for (npy_intp ia = 0; ia < n_alpha; ++ia) {
            const npy_intp row_start = pair_row_start[static_cast<std::size_t>(ia)];
            for (npy_intp ib = ia; ib < n_alpha; ++ib) {
                const npy_intp pair = row_start + (ib - ia);
                const double scale = (ia == ib) ? 1.0 : inv_sqrt2;
                const double coeff = scale * cdata[pair];
                const std::size_t key_ab =
                    static_cast<std::size_t>(ia) * static_cast<std::size_t>(n_alpha) +
                    static_cast<std::size_t>(ib);
                const std::size_t key_ba =
                    static_cast<std::size_t>(ib) * static_cast<std::size_t>(n_alpha) +
                    static_cast<std::size_t>(ia);
                pair_index_lookup[key_ab] = static_cast<uint32_t>(pair);
                pair_index_lookup[key_ba] = static_cast<uint32_t>(pair);
                pair_scale_lookup[key_ab] = scale;
                pair_scale_lookup[key_ba] = scale;
                pair_coeff_lookup[key_ab] = coeff;
                pair_coeff_lookup[key_ba] = coeff;
            }
        }
    }

    for (npy_intp det = 0; det < n_det; ++det) {
        sigma[det] = use_reduced_cross ? 0.0 : diag[det] * cvec[det];
    }

    auto triangular_pair_index = [&](npy_intp left_index, npy_intp right_index) {
        const npy_intp lo = std::min(left_index, right_index);
        const npy_intp hi = std::max(left_index, right_index);
        if (!pair_row_start.empty()) {
            return pair_row_start[static_cast<std::size_t>(lo)] + (hi - lo);
        }
        if (left_index <= right_index) {
            return left_index * n_alpha - (left_index * (left_index - 1)) / 2 +
                (right_index - left_index);
        }
        return right_index * n_alpha - (right_index * (right_index - 1)) / 2 +
            (left_index - right_index);
    };

    auto add_reduced_cross_cached = [&](
        double* out,
        npy_intp bra_alpha,
        npy_intp bra_beta,
        npy_intp ket_alpha,
        npy_intp ket_beta,
        double value
    ) {
        const std::size_t bra_key =
            static_cast<std::size_t>(bra_alpha) * static_cast<std::size_t>(n_alpha) +
            static_cast<std::size_t>(bra_beta);
        const std::size_t ket_key =
            static_cast<std::size_t>(ket_alpha) * static_cast<std::size_t>(n_alpha) +
            static_cast<std::size_t>(ket_beta);
        out[static_cast<npy_intp>(pair_index_lookup[bra_key])] += (
            pair_scale_lookup[bra_key] *
            value *
            pair_coeff_lookup[ket_key]
        );
    };

    auto add_reduced_cross_formula = [&](
        double* out,
        npy_intp bra_alpha,
        npy_intp bra_beta,
        npy_intp ket_alpha,
        npy_intp ket_beta,
        double value
    ) {
        const npy_intp bra_pair = triangular_pair_index(bra_alpha, bra_beta);
        const npy_intp ket_pair = triangular_pair_index(ket_alpha, ket_beta);
        const double bra_scale = (bra_alpha == bra_beta) ? 1.0 : inv_sqrt2;
        const double ket_scale = (ket_alpha == ket_beta) ? 1.0 : inv_sqrt2;
        out[bra_pair] += (
            bra_scale *
            value *
            ket_scale *
            cdata[ket_pair]
        );
    };

    auto reduced_cross_worker = [&](int worker_id, double* out) {
        auto run = [&](auto&& add_cross) {
            for (
                npy_intp transition_a = worker_id;
                transition_a < n_transition;
                transition_a += worker_count
            ) {
                const npy_intp alpha_begin = alpha_offsets_data[transition_a];
                const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
                if (alpha_begin == alpha_end) {
                    continue;
                }
                const npy_intp p = transition_a / n;
                const npy_intp q = transition_a - p * n;
                for (npy_intp transition_b = transition_a; transition_b < n_transition; ++transition_b) {
                    const npy_intp beta_begin = alpha_offsets_data[transition_b];
                    const npy_intp beta_end = alpha_offsets_data[transition_b + 1];
                    if (beta_begin == beta_end) {
                        continue;
                    }
                    const npy_intp r = transition_b / n;
                    const npy_intp s = transition_b - r * n;
                    const double eri_value = cross[idx4(n, p, q, r, s)];
                    if (eri_value == 0.0) {
                        continue;
                    }
                    if (transition_b == transition_a) {
                        for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                            const npy_intp ia = ordered_i_a[alpha_pos];
                            const npy_intp ja = ordered_j_a[alpha_pos];
                            const double alpha_value =
                                eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                            add_cross(
                                out,
                                ia,
                                ia,
                                ja,
                                ja,
                                alpha_value * static_cast<double>(ordered_phase_a[alpha_pos])
                            );
                            for (npy_intp beta_pos = alpha_pos + 1; beta_pos < beta_end; ++beta_pos) {
                                add_cross(
                                    out,
                                    ia,
                                    ordered_i_a[beta_pos],
                                    ja,
                                    ordered_j_a[beta_pos],
                                    2.0 * alpha_value * static_cast<double>(ordered_phase_a[beta_pos])
                                );
                            }
                        }
                    } else {
                        for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                            const npy_intp ia = ordered_i_a[alpha_pos];
                            const npy_intp ja = ordered_j_a[alpha_pos];
                            const double alpha_value =
                                2.0 * eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                            for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                                add_cross(
                                    out,
                                    ia,
                                    ordered_i_a[beta_pos],
                                    ja,
                                    ordered_j_a[beta_pos],
                                    alpha_value * static_cast<double>(ordered_phase_a[beta_pos])
                                );
                            }
                        }
                    }
                }
            }
        };
        if (!pair_index_lookup.empty()) {
            run(add_reduced_cross_cached);
        } else {
            run(add_reduced_cross_formula);
        }
    };

    if (worker_count > 1) {
        auto alpha_same_worker = [&](int worker_id) {
            const npy_intp row_begin = (n_alpha * worker_id) / worker_count;
            const npy_intp row_end = (n_alpha * (worker_id + 1)) / worker_count;
            for (npy_intp link = 0; link < n_link_a; ++link) {
                const npy_intp ia = i_a[link];
                if (ia < row_begin || ia >= row_end) {
                    continue;
                }
                const npy_intp ja = j_a[link];
                const npy_intp p = pa[link];
                const npy_intp q = qa[link];
                const double sign = static_cast<double>(phase_a[link]);
                double same_part = -sign * h1_data[idx2(n, p, q)];
                for (npy_intp r = 0; r < n; ++r) {
                    if (alpha[idx_occ(n, ja, r)] && r != q) {
                        same_part -= sign * same[idx4(n, p, q, r, r)];
                    }
                }
                const double* beta_diag = beta_cross_diag_data + (p * n + q) * n_beta;
                double* sigma_row = sigma + ia * n_beta;
                const double* c_row = cvec + ja * n_beta;
                for (npy_intp ib = 0; ib < n_beta; ++ib) {
                    sigma_row[ib] += (same_part - sign * beta_diag[ib]) * c_row[ib];
                }
            }

            for (npy_intp link = 0; link < n_link_aa; ++link) {
                const npy_intp ia = i_aa[link];
                if (ia < row_begin || ia >= row_end) {
                    continue;
                }
                const npy_intp ja = j_aa[link];
                const double val = (
                    static_cast<double>(phase_aa[link]) *
                    same[idx4(n, p_aa[link], q_aa[link], r_aa[link], s_aa[link])]
                );
                for (npy_intp ib = 0; ib < n_beta; ++ib) {
                    sigma[ia * n_beta + ib] += val * cvec[ja * n_beta + ib];
                }
            }
        };

        auto beta_same_worker = [&](int worker_id) {
            const npy_intp col_begin = (n_beta * worker_id) / worker_count;
            const npy_intp col_end = (n_beta * (worker_id + 1)) / worker_count;
            if (!use_reduced_cross) {
                for (npy_intp link = 0; link < n_link_b; ++link) {
                    const npy_intp ib = i_b[link];
                    if (ib < col_begin || ib >= col_end) {
                        continue;
                    }
                    const npy_intp jb = j_b[link];
                    const npy_intp p = pb[link];
                    const npy_intp q = qb[link];
                    const double sign = static_cast<double>(phase_b[link]);
                    double same_part = -sign * h1_data[idx2(n, p, q)];
                    for (npy_intp r = 0; r < n; ++r) {
                        if (beta[idx_occ(n, jb, r)] && r != q) {
                            same_part -= sign * same[idx4(n, p, q, r, r)];
                        }
                    }
                    const double* alpha_diag = alpha_cross_diag_data + (p * n + q) * n_alpha;
                    for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                        sigma[ia * n_beta + ib] +=
                            (same_part - sign * alpha_diag[ia]) * cvec[ia * n_beta + jb];
                    }
                }

                for (npy_intp link = 0; link < n_link_bb; ++link) {
                    const npy_intp ib = i_bb[link];
                    if (ib < col_begin || ib >= col_end) {
                        continue;
                    }
                    const npy_intp jb = j_bb[link];
                    const double val = (
                        static_cast<double>(phase_bb[link]) *
                        same[idx4(n, p_bb[link], q_bb[link], r_bb[link], s_bb[link])]
                    );
                    for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                        sigma[ia * n_beta + ib] += val * cvec[ia * n_beta + jb];
                    }
                }
            }

            if (use_reduced_cross) {
                double* local_cross_sigma = cross_sigma_buffers.data() +
                    static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_pair);
                reduced_cross_worker(worker_id, local_cross_sigma);
            }
        };

        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            worker_threads.emplace_back(alpha_same_worker, worker_id);
        }
        alpha_same_worker(0);
        for (std::thread& thread : worker_threads) {
            thread.join();
        }
        worker_threads.clear();

        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            worker_threads.emplace_back(beta_same_worker, worker_id);
        }
        beta_same_worker(0);
        for (std::thread& thread : worker_threads) {
            thread.join();
        }
    } else {
        for (npy_intp link = 0; link < n_link_a; ++link) {
            const npy_intp ia = i_a[link];
            const npy_intp ja = j_a[link];
            const npy_intp p = pa[link];
            const npy_intp q = qa[link];
            const double sign = static_cast<double>(phase_a[link]);
            double same_part = -sign * h1_data[idx2(n, p, q)];
            for (npy_intp r = 0; r < n; ++r) {
                if (alpha[idx_occ(n, ja, r)] && r != q) {
                    same_part -= sign * same[idx4(n, p, q, r, r)];
                }
            }
            const double* beta_diag = beta_cross_diag_data + (p * n + q) * n_beta;
            double* sigma_row = sigma + ia * n_beta;
            const double* c_row = cvec + ja * n_beta;
            for (npy_intp ib = 0; ib < n_beta; ++ib) {
                sigma_row[ib] += (same_part - sign * beta_diag[ib]) * c_row[ib];
            }
        }

        if (!use_reduced_cross) {
            for (npy_intp link = 0; link < n_link_b; ++link) {
                const npy_intp ib = i_b[link];
                const npy_intp jb = j_b[link];
                const npy_intp p = pb[link];
                const npy_intp q = qb[link];
                const double sign = static_cast<double>(phase_b[link]);
                double same_part = -sign * h1_data[idx2(n, p, q)];
                for (npy_intp r = 0; r < n; ++r) {
                    if (beta[idx_occ(n, jb, r)] && r != q) {
                        same_part -= sign * same[idx4(n, p, q, r, r)];
                    }
                }
                const double* alpha_diag = alpha_cross_diag_data + (p * n + q) * n_alpha;
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    sigma[ia * n_beta + ib] +=
                        (same_part - sign * alpha_diag[ia]) * cvec[ia * n_beta + jb];
                }
            }
        }

        for (npy_intp link = 0; link < n_link_aa; ++link) {
            const npy_intp ia = i_aa[link];
            const npy_intp ja = j_aa[link];
            const double val = (
                static_cast<double>(phase_aa[link]) *
                same[idx4(n, p_aa[link], q_aa[link], r_aa[link], s_aa[link])]
            );
            for (npy_intp ib = 0; ib < n_beta; ++ib) {
                sigma[ia * n_beta + ib] += val * cvec[ja * n_beta + ib];
            }
        }

        if (!use_reduced_cross) {
            for (npy_intp link = 0; link < n_link_bb; ++link) {
                const npy_intp ib = i_bb[link];
                const npy_intp jb = j_bb[link];
                const double val = (
                    static_cast<double>(phase_bb[link]) *
                    same[idx4(n, p_bb[link], q_bb[link], r_bb[link], s_bb[link])]
                );
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    sigma[ia * n_beta + ib] += val * cvec[ia * n_beta + jb];
                }
            }
        }
    }

    if (use_reduced_cross) {
        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            const double diag_part = (
                ldet == rdet
                    ? diag[ldet]
                    : 0.5 * (diag[ldet] + diag[rdet])
            ) * cdata[pair];
            if (ldet == rdet) {
                sigma_pair[pair] = diag_part + 2.0 * sigma[ldet];
            } else {
                sigma_pair[pair] = diag_part + 2.0 * (sigma[ldet] + sigma[rdet]) * inv_sqrt2;
            }
        }

        if (worker_count > 1) {
            for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
                const double* local_sigma = cross_sigma_buffers.data() +
                    static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_pair);
                for (npy_intp pair = 0; pair < n_pair; ++pair) {
                    sigma_pair[pair] += local_sigma[pair];
                }
            }
        } else {
            reduced_cross_worker(0, sigma_pair);
        }
    } else if (worker_count > 1) {
        auto cross_worker = [&](int worker_id) {
            double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (
                npy_intp transition_a = worker_id;
                transition_a < n_transition;
                transition_a += worker_count
            ) {
                const npy_intp alpha_begin = alpha_offsets_data[transition_a];
                const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
                if (alpha_begin == alpha_end) {
                    continue;
                }
                const npy_intp p = transition_a / n;
                const npy_intp q = transition_a - p * n;
                for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                    const npy_intp beta_begin = beta_offsets_data[transition_b];
                    const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                    if (beta_begin == beta_end) {
                        continue;
                    }
                    const npy_intp r = transition_b / n;
                    const npy_intp s = transition_b - r * n;
                    const double eri_value = cross[idx4(n, p, q, r, s)];
                    if (eri_value == 0.0) {
                        continue;
                    }
                    for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                        const npy_intp ia = ordered_i_a[alpha_pos];
                        const npy_intp ja = ordered_j_a[alpha_pos];
                        const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                        double* sigma_row = local_sigma + ia * n_beta;
                        const double* c_row = cvec + ja * n_beta;
                        for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                            sigma_row[ordered_i_b[beta_pos]] +=
                                alpha_value *
                                static_cast<double>(ordered_phase_b[beta_pos]) *
                                c_row[ordered_j_b[beta_pos]];
                        }
                    }
                }
            }
        };
        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            cross_threads.emplace_back(cross_worker, worker_id);
        }
        cross_worker(0);
        for (std::thread& thread : cross_threads) {
            thread.join();
        }
        for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
            const double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (npy_intp det = 0; det < n_det; ++det) {
                sigma[det] += local_sigma[det];
            }
        }
    } else {
        for (npy_intp transition_a = 0; transition_a < n_transition; ++transition_a) {
            const npy_intp alpha_begin = alpha_offsets_data[transition_a];
            const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
            if (alpha_begin == alpha_end) {
                continue;
            }
            const npy_intp p = transition_a / n;
            const npy_intp q = transition_a - p * n;
            for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                const npy_intp beta_begin = beta_offsets_data[transition_b];
                const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                if (beta_begin == beta_end) {
                    continue;
                }
                const npy_intp r = transition_b / n;
                const npy_intp s = transition_b - r * n;
                const double eri_value = cross[idx4(n, p, q, r, s)];
                if (eri_value == 0.0) {
                    continue;
                }
                for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                    const npy_intp ia = ordered_i_a[alpha_pos];
                    const npy_intp ja = ordered_j_a[alpha_pos];
                    const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                    double* sigma_row = sigma + ia * n_beta;
                    const double* c_row = cvec + ja * n_beta;
                    for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                        sigma_row[ordered_i_b[beta_pos]] +=
                            alpha_value *
                            static_cast<double>(ordered_phase_b[beta_pos]) *
                            c_row[ordered_j_b[beta_pos]];
                    }
                }
            }
        }

        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            if (ldet == rdet) {
                sigma_pair[pair] = sigma[ldet];
            } else {
                sigma_pair[pair] = (sigma[ldet] + sigma[rdet]) * inv_sqrt2;
            }
        }
    }

    Py_END_ALLOW_THREADS

    return sigma_obj;
}

struct Spin0PairSigmaWorkspace {
    ArrayRef h1;
    ArrayRef eri_same;
    ArrayRef eri_cross;
    ArrayRef hdiag;
    ArrayRef c_pair;
    ArrayRef pair_left;
    ArrayRef pair_right;
    ArrayRef alpha_occ;
    ArrayRef beta_occ;
    ArrayRef I_A;
    ArrayRef J_A;
    ArrayRef p_A;
    ArrayRef q_A;
    ArrayRef phase_A;
    ArrayRef I_B;
    ArrayRef J_B;
    ArrayRef p_B;
    ArrayRef q_B;
    ArrayRef phase_B;
    ArrayRef I_AA;
    ArrayRef J_AA;
    ArrayRef p_AA;
    ArrayRef q_AA;
    ArrayRef r_AA;
    ArrayRef s_AA;
    ArrayRef phase_AA;
    ArrayRef I_BB;
    ArrayRef J_BB;
    ArrayRef p_BB;
    ArrayRef q_BB;
    ArrayRef r_BB;
    ArrayRef s_BB;
    ArrayRef phase_BB;
    ArrayRef alpha_offsets;
    ArrayRef beta_offsets;
    ArrayRef alpha_order;
    ArrayRef beta_order;
    ArrayRef alpha_cross_diag;
    ArrayRef beta_cross_diag;
    ArrayRef ordered_alpha_i;
    ArrayRef ordered_alpha_j;
    ArrayRef ordered_alpha_phase;
    ArrayRef ordered_beta_i;
    ArrayRef ordered_beta_j;
    ArrayRef ordered_beta_phase;

    npy_intp n = 0;
    npy_intp n_alpha = 0;
    npy_intp n_beta = 0;
    npy_intp n_det = 0;
    npy_intp n_pair = 0;
    npy_intp n_transition = 0;
    npy_intp n_link_a = 0;
    npy_intp n_link_b = 0;
    npy_intp n_link_aa = 0;
    npy_intp n_link_bb = 0;
    int worker_count = 1;
    bool use_reduced_cross = false;

    const double* h1_data = nullptr;
    const double* same = nullptr;
    const double* cross = nullptr;
    const double* diag = nullptr;
    const npy_intp* left = nullptr;
    const npy_intp* right = nullptr;
    const npy_int8* alpha = nullptr;
    const npy_int8* beta = nullptr;
    const npy_int32* i_a = nullptr;
    const npy_int32* j_a = nullptr;
    const npy_int32* pa = nullptr;
    const npy_int32* qa = nullptr;
    const npy_int8* phase_a = nullptr;
    const npy_int32* i_b = nullptr;
    const npy_int32* j_b = nullptr;
    const npy_int32* pb = nullptr;
    const npy_int32* qb = nullptr;
    const npy_int8* phase_b = nullptr;
    const npy_int32* i_aa = nullptr;
    const npy_int32* j_aa = nullptr;
    const npy_int32* p_aa = nullptr;
    const npy_int32* q_aa = nullptr;
    const npy_int32* r_aa = nullptr;
    const npy_int32* s_aa = nullptr;
    const npy_int8* phase_aa = nullptr;
    const npy_int32* i_bb = nullptr;
    const npy_int32* j_bb = nullptr;
    const npy_int32* p_bb = nullptr;
    const npy_int32* q_bb = nullptr;
    const npy_int32* r_bb = nullptr;
    const npy_int32* s_bb = nullptr;
    const npy_int8* phase_bb = nullptr;
    const npy_intp* alpha_offsets_data = nullptr;
    const npy_intp* beta_offsets_data = nullptr;
    const double* alpha_cross_diag_data = nullptr;
    const double* beta_cross_diag_data = nullptr;
    const npy_int32* ordered_i_a = nullptr;
    const npy_int32* ordered_j_a = nullptr;
    const npy_int8* ordered_phase_a = nullptr;
    const npy_int32* ordered_i_b = nullptr;
    const npy_int32* ordered_j_b = nullptr;
    const npy_int8* ordered_phase_b = nullptr;

    std::vector<double> c_det;
    std::vector<double> sigma_det;
    std::vector<double> cross_sigma_buffers;
    std::vector<npy_intp> pair_row_start;
    std::vector<uint32_t> pair_index_lookup;
    std::vector<double> pair_scale_lookup;
    std::vector<double> pair_coeff_lookup;
#ifdef PYQED_HAVE_CBLAS
    bool use_blas_pair = false;
    npy_intp n_orbital_pair = 0;
    std::vector<npy_intp> blas_link_offsets;
    std::vector<npy_int32> blas_link_pair;
    std::vector<npy_int32> blas_link_ket;
    std::vector<npy_int8> blas_link_sign;
    std::vector<double> effective_pair_eri;
    std::vector<double> blas_sigma_buffers;
    std::vector<double> blas_t1_buffers;
    std::vector<double> blas_vt1_buffers;
#endif

    static constexpr double inv_sqrt2 = 0.70710678118654752440084436210484903928;

    bool initialize(PyObject* args, int requested_workers) {
        h1.reset(PyTuple_GET_ITEM(args, 0), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        eri_same.reset(PyTuple_GET_ITEM(args, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        eri_cross.reset(PyTuple_GET_ITEM(args, 2), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        hdiag.reset(PyTuple_GET_ITEM(args, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        c_pair.reset(PyTuple_GET_ITEM(args, 4), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        pair_left.reset(PyTuple_GET_ITEM(args, 5), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        pair_right.reset(PyTuple_GET_ITEM(args, 6), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        alpha_occ.reset(PyTuple_GET_ITEM(args, 7), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        beta_occ.reset(PyTuple_GET_ITEM(args, 8), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        I_A.reset(PyTuple_GET_ITEM(args, 9), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        J_A.reset(PyTuple_GET_ITEM(args, 10), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        p_A.reset(PyTuple_GET_ITEM(args, 11), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        q_A.reset(PyTuple_GET_ITEM(args, 12), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        phase_A.reset(PyTuple_GET_ITEM(args, 13), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        I_B.reset(PyTuple_GET_ITEM(args, 14), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        J_B.reset(PyTuple_GET_ITEM(args, 15), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        p_B.reset(PyTuple_GET_ITEM(args, 16), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        q_B.reset(PyTuple_GET_ITEM(args, 17), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        phase_B.reset(PyTuple_GET_ITEM(args, 18), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        I_AA.reset(PyTuple_GET_ITEM(args, 19), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        J_AA.reset(PyTuple_GET_ITEM(args, 20), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        p_AA.reset(PyTuple_GET_ITEM(args, 21), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        q_AA.reset(PyTuple_GET_ITEM(args, 22), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        r_AA.reset(PyTuple_GET_ITEM(args, 23), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        s_AA.reset(PyTuple_GET_ITEM(args, 24), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        phase_AA.reset(PyTuple_GET_ITEM(args, 25), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        I_BB.reset(PyTuple_GET_ITEM(args, 26), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        J_BB.reset(PyTuple_GET_ITEM(args, 27), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        p_BB.reset(PyTuple_GET_ITEM(args, 28), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        q_BB.reset(PyTuple_GET_ITEM(args, 29), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        r_BB.reset(PyTuple_GET_ITEM(args, 30), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        s_BB.reset(PyTuple_GET_ITEM(args, 31), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        phase_BB.reset(PyTuple_GET_ITEM(args, 32), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        alpha_offsets.reset(PyTuple_GET_ITEM(args, 33), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        beta_offsets.reset(PyTuple_GET_ITEM(args, 34), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        alpha_order.reset(PyTuple_GET_ITEM(args, 35), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        beta_order.reset(PyTuple_GET_ITEM(args, 36), NPY_INTP, NPY_ARRAY_IN_ARRAY);
        alpha_cross_diag.reset(PyTuple_GET_ITEM(args, 37), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        beta_cross_diag.reset(PyTuple_GET_ITEM(args, 38), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        ordered_alpha_i.reset(PyTuple_GET_ITEM(args, 39), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        ordered_alpha_j.reset(PyTuple_GET_ITEM(args, 40), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        ordered_alpha_phase.reset(PyTuple_GET_ITEM(args, 41), NPY_INT8, NPY_ARRAY_IN_ARRAY);
        ordered_beta_i.reset(PyTuple_GET_ITEM(args, 42), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        ordered_beta_j.reset(PyTuple_GET_ITEM(args, 43), NPY_INT32, NPY_ARRAY_IN_ARRAY);
        ordered_beta_phase.reset(PyTuple_GET_ITEM(args, 44), NPY_INT8, NPY_ARRAY_IN_ARRAY);

        if (!h1 || !eri_same || !eri_cross || !hdiag || !c_pair || !pair_left || !pair_right ||
            !alpha_occ || !beta_occ ||
            !I_A || !J_A || !p_A || !q_A || !phase_A ||
            !I_B || !J_B || !p_B || !q_B || !phase_B ||
            !I_AA || !J_AA || !p_AA || !q_AA || !r_AA || !s_AA || !phase_AA ||
            !I_BB || !J_BB || !p_BB || !q_BB || !r_BB || !s_BB || !phase_BB ||
            !alpha_offsets || !beta_offsets || !alpha_order || !beta_order ||
            !alpha_cross_diag || !beta_cross_diag ||
            !ordered_alpha_i || !ordered_alpha_j || !ordered_alpha_phase ||
            !ordered_beta_i || !ordered_beta_j || !ordered_beta_phase) {
            return false;
        }
        if (
            !validate_link_group(I_A.obj, J_A.obj, p_A.obj, q_A.obj, phase_A.obj, "alpha") ||
            !validate_link_group(I_B.obj, J_B.obj, p_B.obj, q_B.obj, phase_B.obj, "beta") ||
            !validate_double_link_group(I_AA.obj, J_AA.obj, p_AA.obj, q_AA.obj, r_AA.obj, s_AA.obj, phase_AA.obj, "alpha-alpha") ||
            !validate_double_link_group(I_BB.obj, J_BB.obj, p_BB.obj, q_BB.obj, r_BB.obj, s_BB.obj, phase_BB.obj, "beta-beta")
        ) {
            return false;
        }
        if (
            PyArray_NDIM(h1.obj) != 2 ||
            PyArray_NDIM(eri_same.obj) != 4 ||
            PyArray_NDIM(eri_cross.obj) != 4 ||
            PyArray_NDIM(hdiag.obj) != 1 ||
            PyArray_NDIM(c_pair.obj) != 1 ||
            PyArray_NDIM(pair_left.obj) != 1 ||
            PyArray_NDIM(pair_right.obj) != 1 ||
            PyArray_NDIM(alpha_occ.obj) != 2 ||
            PyArray_NDIM(beta_occ.obj) != 2
        ) {
            PyErr_SetString(PyExc_ValueError, "spin0 pair sigma workspace received arrays with invalid ranks.");
            return false;
        }

        n = PyArray_DIM(h1.obj, 0);
        n_alpha = PyArray_DIM(alpha_occ.obj, 0);
        n_beta = PyArray_DIM(beta_occ.obj, 0);
        n_det = n_alpha * n_beta;
        n_pair = PyArray_DIM(c_pair.obj, 0);
        n_transition = n * n;
        n_link_a = PyArray_DIM(I_A.obj, 0);
        n_link_b = PyArray_DIM(I_B.obj, 0);
        n_link_aa = PyArray_DIM(I_AA.obj, 0);
        n_link_bb = PyArray_DIM(I_BB.obj, 0);
        if (
            PyArray_DIM(h1.obj, 1) != n ||
            PyArray_DIM(alpha_occ.obj, 1) != n ||
            PyArray_DIM(beta_occ.obj, 1) != n ||
            PyArray_DIM(hdiag.obj, 0) != n_det ||
            PyArray_DIM(pair_left.obj, 0) != n_pair ||
            PyArray_DIM(pair_right.obj, 0) != n_pair ||
            PyArray_NDIM(alpha_offsets.obj) != 1 ||
            PyArray_NDIM(beta_offsets.obj) != 1 ||
            PyArray_NDIM(alpha_order.obj) != 1 ||
            PyArray_NDIM(beta_order.obj) != 1 ||
            PyArray_NDIM(alpha_cross_diag.obj) != 2 ||
            PyArray_NDIM(beta_cross_diag.obj) != 2 ||
            PyArray_DIM(alpha_offsets.obj, 0) != n_transition + 1 ||
            PyArray_DIM(beta_offsets.obj, 0) != n_transition + 1 ||
            PyArray_DIM(alpha_order.obj, 0) != n_link_a ||
            PyArray_DIM(beta_order.obj, 0) != n_link_b ||
            PyArray_DIM(alpha_cross_diag.obj, 0) != n_transition ||
            PyArray_DIM(alpha_cross_diag.obj, 1) != n_alpha ||
            PyArray_DIM(beta_cross_diag.obj, 0) != n_transition ||
            PyArray_DIM(beta_cross_diag.obj, 1) != n_beta ||
            PyArray_NDIM(ordered_alpha_i.obj) != 1 ||
            PyArray_NDIM(ordered_alpha_j.obj) != 1 ||
            PyArray_NDIM(ordered_alpha_phase.obj) != 1 ||
            PyArray_NDIM(ordered_beta_i.obj) != 1 ||
            PyArray_NDIM(ordered_beta_j.obj) != 1 ||
            PyArray_NDIM(ordered_beta_phase.obj) != 1 ||
            PyArray_DIM(ordered_alpha_i.obj, 0) != n_link_a ||
            PyArray_DIM(ordered_alpha_j.obj, 0) != n_link_a ||
            PyArray_DIM(ordered_alpha_phase.obj, 0) != n_link_a ||
            PyArray_DIM(ordered_beta_i.obj, 0) != n_link_b ||
            PyArray_DIM(ordered_beta_j.obj, 0) != n_link_b ||
            PyArray_DIM(ordered_beta_phase.obj, 0) != n_link_b
        ) {
            PyErr_SetString(PyExc_ValueError, "spin0 pair sigma workspace array shapes are inconsistent.");
            return false;
        }
        for (int axis = 0; axis < 4; ++axis) {
            if (PyArray_DIM(eri_same.obj, axis) != n || PyArray_DIM(eri_cross.obj, axis) != n) {
                PyErr_SetString(PyExc_ValueError, "ERI tensors must have shape (norb, norb, norb, norb).");
                return false;
            }
        }

        h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
        same = static_cast<const double*>(PyArray_DATA(eri_same.obj));
        cross = static_cast<const double*>(PyArray_DATA(eri_cross.obj));
        diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));
        left = static_cast<const npy_intp*>(PyArray_DATA(pair_left.obj));
        right = static_cast<const npy_intp*>(PyArray_DATA(pair_right.obj));
        alpha = static_cast<const npy_int8*>(PyArray_DATA(alpha_occ.obj));
        beta = static_cast<const npy_int8*>(PyArray_DATA(beta_occ.obj));
        i_a = static_cast<const npy_int32*>(PyArray_DATA(I_A.obj));
        j_a = static_cast<const npy_int32*>(PyArray_DATA(J_A.obj));
        pa = static_cast<const npy_int32*>(PyArray_DATA(p_A.obj));
        qa = static_cast<const npy_int32*>(PyArray_DATA(q_A.obj));
        phase_a = static_cast<const npy_int8*>(PyArray_DATA(phase_A.obj));
        i_b = static_cast<const npy_int32*>(PyArray_DATA(I_B.obj));
        j_b = static_cast<const npy_int32*>(PyArray_DATA(J_B.obj));
        pb = static_cast<const npy_int32*>(PyArray_DATA(p_B.obj));
        qb = static_cast<const npy_int32*>(PyArray_DATA(q_B.obj));
        phase_b = static_cast<const npy_int8*>(PyArray_DATA(phase_B.obj));
        i_aa = static_cast<const npy_int32*>(PyArray_DATA(I_AA.obj));
        j_aa = static_cast<const npy_int32*>(PyArray_DATA(J_AA.obj));
        p_aa = static_cast<const npy_int32*>(PyArray_DATA(p_AA.obj));
        q_aa = static_cast<const npy_int32*>(PyArray_DATA(q_AA.obj));
        r_aa = static_cast<const npy_int32*>(PyArray_DATA(r_AA.obj));
        s_aa = static_cast<const npy_int32*>(PyArray_DATA(s_AA.obj));
        phase_aa = static_cast<const npy_int8*>(PyArray_DATA(phase_AA.obj));
        i_bb = static_cast<const npy_int32*>(PyArray_DATA(I_BB.obj));
        j_bb = static_cast<const npy_int32*>(PyArray_DATA(J_BB.obj));
        p_bb = static_cast<const npy_int32*>(PyArray_DATA(p_BB.obj));
        q_bb = static_cast<const npy_int32*>(PyArray_DATA(q_BB.obj));
        r_bb = static_cast<const npy_int32*>(PyArray_DATA(r_BB.obj));
        s_bb = static_cast<const npy_int32*>(PyArray_DATA(s_BB.obj));
        phase_bb = static_cast<const npy_int8*>(PyArray_DATA(phase_BB.obj));
        alpha_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(alpha_offsets.obj));
        beta_offsets_data = static_cast<const npy_intp*>(PyArray_DATA(beta_offsets.obj));
        alpha_cross_diag_data = static_cast<const double*>(PyArray_DATA(alpha_cross_diag.obj));
        beta_cross_diag_data = static_cast<const double*>(PyArray_DATA(beta_cross_diag.obj));
        ordered_i_a = static_cast<const npy_int32*>(PyArray_DATA(ordered_alpha_i.obj));
        ordered_j_a = static_cast<const npy_int32*>(PyArray_DATA(ordered_alpha_j.obj));
        ordered_phase_a = static_cast<const npy_int8*>(PyArray_DATA(ordered_alpha_phase.obj));
        ordered_i_b = static_cast<const npy_int32*>(PyArray_DATA(ordered_beta_i.obj));
        ordered_j_b = static_cast<const npy_int32*>(PyArray_DATA(ordered_beta_j.obj));
        ordered_phase_b = static_cast<const npy_int8*>(PyArray_DATA(ordered_beta_phase.obj));

        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            if (ldet < 0 || ldet >= n_det || rdet < 0 || rdet >= n_det) {
                PyErr_SetString(PyExc_ValueError, "Spin0 pair indices are out of range.");
                return false;
            }
        }

        auto equal_bytes = [](const void* lhs, const void* rhs, std::size_t nbytes) {
            return nbytes == 0 || std::memcmp(lhs, rhs, nbytes) == 0;
        };
        use_reduced_cross = (
            n_alpha == n_beta &&
            n_pair == (n_alpha * (n_alpha + 1)) / 2 &&
            n_link_a == n_link_b &&
            equal_bytes(alpha, beta, static_cast<std::size_t>(n_alpha) * static_cast<std::size_t>(n) * sizeof(npy_int8)) &&
            equal_bytes(
                alpha_offsets_data,
                beta_offsets_data,
                static_cast<std::size_t>(n_transition + 1) * sizeof(npy_intp)
            ) &&
            equal_bytes(
                ordered_i_a,
                ordered_i_b,
                static_cast<std::size_t>(n_link_a) * sizeof(npy_int32)
            ) &&
            equal_bytes(
                ordered_j_a,
                ordered_j_b,
                static_cast<std::size_t>(n_link_a) * sizeof(npy_int32)
            ) &&
            equal_bytes(
                ordered_phase_a,
                ordered_phase_b,
                static_cast<std::size_t>(n_link_a) * sizeof(npy_int8)
            )
        );
        if (use_reduced_cross) {
            for (npy_intp ia = 0; ia < n_alpha && use_reduced_cross; ++ia) {
                const npy_intp row_start = ia * n_alpha - (ia * (ia - 1)) / 2;
                for (npy_intp ib = ia; ib < n_alpha; ++ib) {
                    const npy_intp pair = row_start + (ib - ia);
                    if (
                        left[pair] != ia * n_beta + ib ||
                        right[pair] != ib * n_beta + ia
                    ) {
                        use_reduced_cross = false;
                        break;
                    }
                }
            }
        }

        worker_count = 1;
        if (requested_workers > 1 && n_transition > 1) {
            worker_count = std::min<int>(requested_workers, static_cast<int>(n_transition));
        }
        try {
            c_det.resize(static_cast<std::size_t>(n_det));
            sigma_det.resize(static_cast<std::size_t>(n_det));
            pair_row_start.clear();
            pair_index_lookup.clear();
            pair_scale_lookup.clear();
            pair_coeff_lookup.clear();
            if (use_reduced_cross) {
                pair_row_start.assign(static_cast<std::size_t>(n_alpha), 0);
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    pair_row_start[static_cast<std::size_t>(ia)] =
                        ia * n_alpha - (ia * (ia - 1)) / 2;
                }
            }
            constexpr std::size_t pair_lookup_limit = 128'000;
            const std::size_t pair_lookup_size =
                static_cast<std::size_t>(n_alpha) * static_cast<std::size_t>(n_alpha);
            if (use_reduced_cross && pair_lookup_size <= pair_lookup_limit) {
                pair_index_lookup.assign(pair_lookup_size, 0);
                pair_scale_lookup.assign(pair_lookup_size, 0.0);
                pair_coeff_lookup.assign(pair_lookup_size, 0.0);
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    const npy_intp row_start = pair_row_start[static_cast<std::size_t>(ia)];
                    for (npy_intp ib = ia; ib < n_alpha; ++ib) {
                        const npy_intp pair = row_start + (ib - ia);
                        const double scale = (ia == ib) ? 1.0 : inv_sqrt2;
                        const std::size_t key_ab =
                            static_cast<std::size_t>(ia) * static_cast<std::size_t>(n_alpha) +
                            static_cast<std::size_t>(ib);
                        const std::size_t key_ba =
                            static_cast<std::size_t>(ib) * static_cast<std::size_t>(n_alpha) +
                            static_cast<std::size_t>(ia);
                        pair_index_lookup[key_ab] = static_cast<uint32_t>(pair);
                        pair_index_lookup[key_ba] = static_cast<uint32_t>(pair);
                        pair_scale_lookup[key_ab] = scale;
                        pair_scale_lookup[key_ba] = scale;
                    }
                }
            }
            if (worker_count > 1) {
                cross_sigma_buffers.resize(
                    static_cast<std::size_t>(worker_count) *
                        static_cast<std::size_t>(use_reduced_cross ? n_pair : n_det)
                );
            } else {
                cross_sigma_buffers.clear();
            }
#ifdef PYQED_HAVE_CBLAS
            if (!initialize_blas_pair()) {
                return false;
            }
#endif
        } catch (...) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate spin0 sigma workspace.");
            return false;
        }
        return true;
    }

    bool prepare_scratch() {
        try {
            std::fill(c_det.begin(), c_det.end(), 0.0);
            if (worker_count > 1) {
                std::fill(cross_sigma_buffers.begin(), cross_sigma_buffers.end(), 0.0);
            }
        } catch (...) {
            PyErr_SetString(PyExc_MemoryError, "Failed to prepare spin0 sigma workspace.");
            return false;
        }
        return true;
    }

    bool apply(const double* cdata, double* sigma_pair);
#ifdef PYQED_HAVE_CBLAS
    bool initialize_blas_pair();
    bool apply_blas_det(const double* cdata, double* sigma_data);
    bool apply_blas_pair(const double* cdata, double* sigma_pair);
#endif
};

#ifdef PYQED_HAVE_CBLAS
bool Spin0PairSigmaWorkspace::initialize_blas_pair() {
    use_blas_pair = false;
    if (!use_reduced_cross || n_alpha != n_beta || n <= 0) {
        return true;
    }

    n_orbital_pair = n * (n + 1) / 2;
    const auto orbital_pair = [](npy_intp p, npy_intp q) -> npy_intp {
        const npy_intp hi = std::max(p, q);
        const npy_intp lo = std::min(p, q);
        return hi * (hi + 1) / 2 + lo;
    };

    std::vector<npy_intp> link_counts(static_cast<std::size_t>(n_alpha), 0);
    for (npy_intp bra = 0; bra < n_alpha; ++bra) {
        for (npy_intp orb = 0; orb < n; ++orb) {
            if (alpha[idx_occ(n, bra, orb)]) {
                ++link_counts[static_cast<std::size_t>(bra)];
            }
        }
    }
    for (npy_intp link = 0; link < n_link_a; ++link) {
        ++link_counts[static_cast<std::size_t>(i_a[link])];
    }

    blas_link_offsets.assign(static_cast<std::size_t>(n_alpha + 1), 0);
    for (npy_intp bra = 0; bra < n_alpha; ++bra) {
        blas_link_offsets[static_cast<std::size_t>(bra + 1)] =
            blas_link_offsets[static_cast<std::size_t>(bra)] +
            link_counts[static_cast<std::size_t>(bra)];
    }
    const std::size_t n_blas_link =
        static_cast<std::size_t>(blas_link_offsets.back());
    blas_link_pair.assign(n_blas_link, 0);
    blas_link_ket.assign(n_blas_link, 0);
    blas_link_sign.assign(n_blas_link, 0);
    std::vector<npy_intp> cursor = blas_link_offsets;

    for (npy_intp bra = 0; bra < n_alpha; ++bra) {
        for (npy_intp orb = 0; orb < n; ++orb) {
            if (!alpha[idx_occ(n, bra, orb)]) {
                continue;
            }
            const npy_intp slot = cursor[static_cast<std::size_t>(bra)]++;
            blas_link_pair[static_cast<std::size_t>(slot)] =
                static_cast<npy_int32>(orbital_pair(orb, orb));
            blas_link_ket[static_cast<std::size_t>(slot)] =
                static_cast<npy_int32>(bra);
            blas_link_sign[static_cast<std::size_t>(slot)] = 1;
        }
    }
    for (npy_intp link = 0; link < n_link_a; ++link) {
        const npy_intp bra = i_a[link];
        const npy_intp slot = cursor[static_cast<std::size_t>(bra)]++;
        blas_link_pair[static_cast<std::size_t>(slot)] =
            static_cast<npy_int32>(orbital_pair(pa[link], qa[link]));
        blas_link_ket[static_cast<std::size_t>(slot)] = j_a[link];
        // The compact link phase is the negative of the spin-free E_pq
        // matrix element used by the gather/scatter factorization.
        blas_link_sign[static_cast<std::size_t>(slot)] = -phase_a[link];
    }

    npy_intp nocc = 0;
    for (npy_intp orb = 0; orb < n; ++orb) {
        nocc += alpha[idx_occ(n, 0, orb)] != 0;
    }
    const double nelec = static_cast<double>(2 * nocc);
    if (nelec <= 0.0) {
        return true;
    }
    std::vector<double> f1e(static_cast<std::size_t>(n * n), 0.0);
    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            double value = h1_data[idx2(n, p, q)];
            for (npy_intp i = 0; i < n; ++i) {
                value -= 0.5 * cross[idx4(n, p, i, i, q)];
            }
            f1e[static_cast<std::size_t>(p * n + q)] = value / nelec;
        }
    }

    effective_pair_eri.assign(
        static_cast<std::size_t>(n_orbital_pair * n_orbital_pair),
        0.0
    );
    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q <= p; ++q) {
            const npy_intp pq = orbital_pair(p, q);
            for (npy_intp r = 0; r < n; ++r) {
                for (npy_intp s = 0; s <= r; ++s) {
                    const npy_intp rs = orbital_pair(r, s);
                    double value = cross[idx4(n, p, q, r, s)];
                    if (p == q) {
                        value += f1e[static_cast<std::size_t>(r * n + s)];
                    }
                    if (r == s) {
                        value += f1e[static_cast<std::size_t>(p * n + q)];
                    }
                    effective_pair_eri[
                        static_cast<std::size_t>(pq * n_orbital_pair + rs)
                    ] = 0.5 * value;
                }
            }
        }
    }

    const std::size_t workers = static_cast<std::size_t>(worker_count);
    const std::size_t det_size = static_cast<std::size_t>(n_det);
    const std::size_t intermediate_size =
        static_cast<std::size_t>(n_orbital_pair * n_beta);
    blas_sigma_buffers.assign(workers * det_size, 0.0);
    blas_t1_buffers.assign(workers * intermediate_size, 0.0);
    blas_vt1_buffers.assign(workers * intermediate_size, 0.0);
    use_blas_pair = true;
    return true;
}

bool Spin0PairSigmaWorkspace::apply_blas_pair(
    const double* cdata,
    double* sigma_pair
) {
    std::fill(c_det.begin(), c_det.end(), 0.0);
    for (npy_intp pair = 0; pair < n_pair; ++pair) {
        const npy_intp ldet = left[pair];
        const npy_intp rdet = right[pair];
        if (ldet == rdet) {
            c_det[static_cast<std::size_t>(ldet)] = cdata[pair];
        } else {
            const double scaled = cdata[pair] * inv_sqrt2;
            c_det[static_cast<std::size_t>(ldet)] = scaled;
            c_det[static_cast<std::size_t>(rdet)] = scaled;
        }
    }

    if (!apply_blas_det(c_det.data(), sigma_det.data())) {
        return false;
    }
    for (npy_intp pair = 0; pair < n_pair; ++pair) {
        const npy_intp ldet = left[pair];
        const npy_intp rdet = right[pair];
        sigma_pair[pair] = (
            ldet == rdet
                ? sigma_det[static_cast<std::size_t>(ldet)]
                : (
                    sigma_det[static_cast<std::size_t>(ldet)] +
                    sigma_det[static_cast<std::size_t>(rdet)]
                ) * inv_sqrt2
        );
    }
    return true;
}

bool Spin0PairSigmaWorkspace::apply_blas_det(
    const double* cdata,
    double* sigma_data
) {
    std::fill(blas_sigma_buffers.begin(), blas_sigma_buffers.end(), 0.0);

    constexpr int col_major = 102;
    constexpr int no_trans = 111;
    const std::size_t det_size = static_cast<std::size_t>(n_det);
    const std::size_t intermediate_size =
        static_cast<std::size_t>(n_orbital_pair * n_beta);
    auto worker = [&](int worker_id) {
        double* local_sigma = blas_sigma_buffers.data() +
            static_cast<std::size_t>(worker_id) * det_size;
        double* t1 = blas_t1_buffers.data() +
            static_cast<std::size_t>(worker_id) * intermediate_size;
        double* vt1 = blas_vt1_buffers.data() +
            static_cast<std::size_t>(worker_id) * intermediate_size;

        for (
            npy_intp bra_a = worker_id;
            bra_a < n_alpha;
            bra_a += worker_count
        ) {
            std::fill(t1, t1 + intermediate_size, 0.0);
            const npy_intp alpha_begin =
                blas_link_offsets[static_cast<std::size_t>(bra_a)];
            const npy_intp alpha_end =
                blas_link_offsets[static_cast<std::size_t>(bra_a + 1)];

            for (npy_intp pos = alpha_begin; pos < alpha_end; ++pos) {
                const npy_intp orbital = blas_link_pair[static_cast<std::size_t>(pos)];
                const npy_intp ket_a = blas_link_ket[static_cast<std::size_t>(pos)];
                const double sign = static_cast<double>(
                    blas_link_sign[static_cast<std::size_t>(pos)]
                );
                double* target = t1 + orbital * n_beta;
                const double* source = cdata + ket_a * n_beta;
                for (npy_intp beta = 0; beta < n_beta; ++beta) {
                    target[beta] += sign * source[beta];
                }
            }

            const double* c_row = cdata + bra_a * n_beta;
            for (npy_intp bra_b = 0; bra_b < n_beta; ++bra_b) {
                const npy_intp beta_begin =
                    blas_link_offsets[static_cast<std::size_t>(bra_b)];
                const npy_intp beta_end =
                    blas_link_offsets[static_cast<std::size_t>(bra_b + 1)];
                for (npy_intp pos = beta_begin; pos < beta_end; ++pos) {
                    const npy_intp orbital = blas_link_pair[static_cast<std::size_t>(pos)];
                    const npy_intp ket_b = blas_link_ket[static_cast<std::size_t>(pos)];
                    const double sign = static_cast<double>(
                        blas_link_sign[static_cast<std::size_t>(pos)]
                    );
                    t1[orbital * n_beta + bra_b] += sign * c_row[ket_b];
                }
            }

            cblas_dgemm(
                col_major,
                no_trans,
                no_trans,
                static_cast<int>(n_beta),
                static_cast<int>(n_orbital_pair),
                static_cast<int>(n_orbital_pair),
                1.0,
                t1,
                static_cast<int>(n_beta),
                effective_pair_eri.data(),
                static_cast<int>(n_orbital_pair),
                0.0,
                vt1,
                static_cast<int>(n_beta)
            );

            double* sigma_row = local_sigma + bra_a * n_beta;
            for (npy_intp bra_b = 0; bra_b < n_beta; ++bra_b) {
                const npy_intp beta_begin =
                    blas_link_offsets[static_cast<std::size_t>(bra_b)];
                const npy_intp beta_end =
                    blas_link_offsets[static_cast<std::size_t>(bra_b + 1)];
                for (npy_intp pos = beta_begin; pos < beta_end; ++pos) {
                    const npy_intp orbital = blas_link_pair[static_cast<std::size_t>(pos)];
                    const npy_intp ket_b = blas_link_ket[static_cast<std::size_t>(pos)];
                    const double sign = static_cast<double>(
                        blas_link_sign[static_cast<std::size_t>(pos)]
                    );
                    sigma_row[ket_b] += sign * vt1[orbital * n_beta + bra_b];
                }
            }

            for (npy_intp pos = alpha_begin; pos < alpha_end; ++pos) {
                const npy_intp orbital = blas_link_pair[static_cast<std::size_t>(pos)];
                const npy_intp ket_a = blas_link_ket[static_cast<std::size_t>(pos)];
                const double sign = static_cast<double>(
                    blas_link_sign[static_cast<std::size_t>(pos)]
                );
                double* target = local_sigma + ket_a * n_beta;
                const double* source = vt1 + orbital * n_beta;
                for (npy_intp beta = 0; beta < n_beta; ++beta) {
                    target[beta] += sign * source[beta];
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));
    Py_BEGIN_ALLOW_THREADS
    for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
        threads.emplace_back(worker, worker_id);
    }
    worker(0);
    for (std::thread& thread : threads) {
        thread.join();
    }
    Py_END_ALLOW_THREADS

    std::fill(sigma_data, sigma_data + n_det, 0.0);
    for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
        const double* source = blas_sigma_buffers.data() +
            static_cast<std::size_t>(worker_id) * det_size;
        for (npy_intp det = 0; det < n_det; ++det) {
            sigma_data[det] += source[det];
        }
    }
    return true;
}
#endif

bool Spin0PairSigmaWorkspace::apply(const double* cdata, double* sigma_pair) {
#ifdef PYQED_HAVE_CBLAS
    if (use_blas_pair) {
        return apply_blas_pair(cdata, sigma_pair);
    }
#endif
    if (!prepare_scratch()) {
        return false;
    }

    double* cvec = c_det.data();
    double* sigma = sigma_det.data();
    std::vector<std::thread> worker_threads;
    std::vector<std::thread> cross_threads;
    worker_threads.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));
    cross_threads.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp pair = 0; pair < n_pair; ++pair) {
        const npy_intp ldet = left[pair];
        const npy_intp rdet = right[pair];
        if (ldet == rdet) {
            cvec[ldet] = cdata[pair];
        } else {
            const double scaled = cdata[pair] * inv_sqrt2;
            cvec[ldet] = scaled;
            cvec[rdet] = scaled;
        }
    }

    if (use_reduced_cross && !pair_coeff_lookup.empty()) {
        for (npy_intp ia = 0; ia < n_alpha; ++ia) {
            const npy_intp row_start = pair_row_start[static_cast<std::size_t>(ia)];
            for (npy_intp ib = ia; ib < n_alpha; ++ib) {
                const npy_intp pair = row_start + (ib - ia);
                const double scale = (ia == ib) ? 1.0 : inv_sqrt2;
                const double coeff = scale * cdata[pair];
                const std::size_t key_ab =
                    static_cast<std::size_t>(ia) * static_cast<std::size_t>(n_alpha) +
                    static_cast<std::size_t>(ib);
                const std::size_t key_ba =
                    static_cast<std::size_t>(ib) * static_cast<std::size_t>(n_alpha) +
                    static_cast<std::size_t>(ia);
                pair_coeff_lookup[key_ab] = coeff;
                pair_coeff_lookup[key_ba] = coeff;
            }
        }
    }

    for (npy_intp det = 0; det < n_det; ++det) {
        sigma[det] = use_reduced_cross ? 0.0 : diag[det] * cvec[det];
    }

    auto triangular_pair_index = [&](npy_intp left_index, npy_intp right_index) {
        const npy_intp lo = std::min(left_index, right_index);
        const npy_intp hi = std::max(left_index, right_index);
        if (!pair_row_start.empty()) {
            return pair_row_start[static_cast<std::size_t>(lo)] + (hi - lo);
        }
        if (left_index <= right_index) {
            return left_index * n_alpha - (left_index * (left_index - 1)) / 2 +
                (right_index - left_index);
        }
        return right_index * n_alpha - (right_index * (right_index - 1)) / 2 +
            (left_index - right_index);
    };

    auto add_reduced_cross_cached = [&](
        double* out,
        npy_intp bra_alpha,
        npy_intp bra_beta,
        npy_intp ket_alpha,
        npy_intp ket_beta,
        double value
    ) {
        const std::size_t bra_key =
            static_cast<std::size_t>(bra_alpha) * static_cast<std::size_t>(n_alpha) +
            static_cast<std::size_t>(bra_beta);
        const std::size_t ket_key =
            static_cast<std::size_t>(ket_alpha) * static_cast<std::size_t>(n_alpha) +
            static_cast<std::size_t>(ket_beta);
        out[static_cast<npy_intp>(pair_index_lookup[bra_key])] += (
            pair_scale_lookup[bra_key] *
            value *
            pair_coeff_lookup[ket_key]
        );
    };

    auto add_reduced_cross_formula = [&](
        double* out,
        npy_intp bra_alpha,
        npy_intp bra_beta,
        npy_intp ket_alpha,
        npy_intp ket_beta,
        double value
    ) {
        const npy_intp bra_pair = triangular_pair_index(bra_alpha, bra_beta);
        const npy_intp ket_pair = triangular_pair_index(ket_alpha, ket_beta);
        const double bra_scale = (bra_alpha == bra_beta) ? 1.0 : inv_sqrt2;
        const double ket_scale = (ket_alpha == ket_beta) ? 1.0 : inv_sqrt2;
        out[bra_pair] += (
            bra_scale *
            value *
            ket_scale *
            cdata[ket_pair]
        );
    };

    auto reduced_cross_worker = [&](int worker_id, double* out) {
        auto run = [&](auto&& add_cross) {
            for (
                npy_intp transition_a = worker_id;
                transition_a < n_transition;
                transition_a += worker_count
            ) {
                const npy_intp alpha_begin = alpha_offsets_data[transition_a];
                const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
                if (alpha_begin == alpha_end) {
                    continue;
                }
                const npy_intp p = transition_a / n;
                const npy_intp q = transition_a - p * n;
                for (npy_intp transition_b = transition_a; transition_b < n_transition; ++transition_b) {
                    const npy_intp beta_begin = alpha_offsets_data[transition_b];
                    const npy_intp beta_end = alpha_offsets_data[transition_b + 1];
                    if (beta_begin == beta_end) {
                        continue;
                    }
                    const npy_intp r = transition_b / n;
                    const npy_intp s = transition_b - r * n;
                    const double eri_value = cross[idx4(n, p, q, r, s)];
                    if (eri_value == 0.0) {
                        continue;
                    }
                    if (transition_b == transition_a) {
                        for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                            const npy_intp ia = ordered_i_a[alpha_pos];
                            const npy_intp ja = ordered_j_a[alpha_pos];
                            const double alpha_value =
                                eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                            add_cross(
                                out,
                                ia,
                                ia,
                                ja,
                                ja,
                                alpha_value * static_cast<double>(ordered_phase_a[alpha_pos])
                            );
                            for (npy_intp beta_pos = alpha_pos + 1; beta_pos < beta_end; ++beta_pos) {
                                add_cross(
                                    out,
                                    ia,
                                    ordered_i_a[beta_pos],
                                    ja,
                                    ordered_j_a[beta_pos],
                                    2.0 * alpha_value * static_cast<double>(ordered_phase_a[beta_pos])
                                );
                            }
                        }
                    } else {
                        for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                            const npy_intp ia = ordered_i_a[alpha_pos];
                            const npy_intp ja = ordered_j_a[alpha_pos];
                            const double alpha_value =
                                2.0 * eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                            for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                                add_cross(
                                    out,
                                    ia,
                                    ordered_i_a[beta_pos],
                                    ja,
                                    ordered_j_a[beta_pos],
                                    alpha_value * static_cast<double>(ordered_phase_a[beta_pos])
                                );
                            }
                        }
                    }
                }
            }
        };
        if (!pair_index_lookup.empty()) {
            run(add_reduced_cross_cached);
        } else {
            run(add_reduced_cross_formula);
        }
    };

    if (worker_count > 1) {
        auto alpha_same_worker = [&](int worker_id) {
            const npy_intp row_begin = (n_alpha * worker_id) / worker_count;
            const npy_intp row_end = (n_alpha * (worker_id + 1)) / worker_count;
            for (npy_intp link = 0; link < n_link_a; ++link) {
                const npy_intp ia = i_a[link];
                if (ia < row_begin || ia >= row_end) {
                    continue;
                }
                const npy_intp ja = j_a[link];
                const npy_intp p = pa[link];
                const npy_intp q = qa[link];
                const double sign = static_cast<double>(phase_a[link]);
                double same_part = -sign * h1_data[idx2(n, p, q)];
                for (npy_intp r = 0; r < n; ++r) {
                    if (alpha[idx_occ(n, ja, r)] && r != q) {
                        same_part -= sign * same[idx4(n, p, q, r, r)];
                    }
                }
                const double* beta_diag = beta_cross_diag_data + (p * n + q) * n_beta;
                double* sigma_row = sigma + ia * n_beta;
                const double* c_row = cvec + ja * n_beta;
                for (npy_intp ib = 0; ib < n_beta; ++ib) {
                    sigma_row[ib] += (same_part - sign * beta_diag[ib]) * c_row[ib];
                }
            }

            for (npy_intp link = 0; link < n_link_aa; ++link) {
                const npy_intp ia = i_aa[link];
                if (ia < row_begin || ia >= row_end) {
                    continue;
                }
                const npy_intp ja = j_aa[link];
                const double val = (
                    static_cast<double>(phase_aa[link]) *
                    same[idx4(n, p_aa[link], q_aa[link], r_aa[link], s_aa[link])]
                );
                for (npy_intp ib = 0; ib < n_beta; ++ib) {
                    sigma[ia * n_beta + ib] += val * cvec[ja * n_beta + ib];
                }
            }
        };

        auto beta_same_worker = [&](int worker_id) {
            const npy_intp col_begin = (n_beta * worker_id) / worker_count;
            const npy_intp col_end = (n_beta * (worker_id + 1)) / worker_count;
            if (!use_reduced_cross) {
                for (npy_intp link = 0; link < n_link_b; ++link) {
                    const npy_intp ib = i_b[link];
                    if (ib < col_begin || ib >= col_end) {
                        continue;
                    }
                    const npy_intp jb = j_b[link];
                    const npy_intp p = pb[link];
                    const npy_intp q = qb[link];
                    const double sign = static_cast<double>(phase_b[link]);
                    double same_part = -sign * h1_data[idx2(n, p, q)];
                    for (npy_intp r = 0; r < n; ++r) {
                        if (beta[idx_occ(n, jb, r)] && r != q) {
                            same_part -= sign * same[idx4(n, p, q, r, r)];
                        }
                    }
                    const double* alpha_diag = alpha_cross_diag_data + (p * n + q) * n_alpha;
                    for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                        sigma[ia * n_beta + ib] +=
                            (same_part - sign * alpha_diag[ia]) * cvec[ia * n_beta + jb];
                    }
                }

                for (npy_intp link = 0; link < n_link_bb; ++link) {
                    const npy_intp ib = i_bb[link];
                    if (ib < col_begin || ib >= col_end) {
                        continue;
                    }
                    const npy_intp jb = j_bb[link];
                    const double val = (
                        static_cast<double>(phase_bb[link]) *
                        same[idx4(n, p_bb[link], q_bb[link], r_bb[link], s_bb[link])]
                    );
                    for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                        sigma[ia * n_beta + ib] += val * cvec[ia * n_beta + jb];
                    }
                }
            }

            if (use_reduced_cross) {
                double* local_cross_sigma = cross_sigma_buffers.data() +
                    static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_pair);
                reduced_cross_worker(worker_id, local_cross_sigma);
            }
        };

        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            worker_threads.emplace_back(alpha_same_worker, worker_id);
        }
        alpha_same_worker(0);
        for (std::thread& thread : worker_threads) {
            thread.join();
        }
        worker_threads.clear();

        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            worker_threads.emplace_back(beta_same_worker, worker_id);
        }
        beta_same_worker(0);
        for (std::thread& thread : worker_threads) {
            thread.join();
        }
    } else {
        for (npy_intp link = 0; link < n_link_a; ++link) {
            const npy_intp ia = i_a[link];
            const npy_intp ja = j_a[link];
            const npy_intp p = pa[link];
            const npy_intp q = qa[link];
            const double sign = static_cast<double>(phase_a[link]);
            double same_part = -sign * h1_data[idx2(n, p, q)];
            for (npy_intp r = 0; r < n; ++r) {
                if (alpha[idx_occ(n, ja, r)] && r != q) {
                    same_part -= sign * same[idx4(n, p, q, r, r)];
                }
            }
            const double* beta_diag = beta_cross_diag_data + (p * n + q) * n_beta;
            double* sigma_row = sigma + ia * n_beta;
            const double* c_row = cvec + ja * n_beta;
            for (npy_intp ib = 0; ib < n_beta; ++ib) {
                sigma_row[ib] += (same_part - sign * beta_diag[ib]) * c_row[ib];
            }
        }

        if (!use_reduced_cross) {
            for (npy_intp link = 0; link < n_link_b; ++link) {
                const npy_intp ib = i_b[link];
                const npy_intp jb = j_b[link];
                const npy_intp p = pb[link];
                const npy_intp q = qb[link];
                const double sign = static_cast<double>(phase_b[link]);
                double same_part = -sign * h1_data[idx2(n, p, q)];
                for (npy_intp r = 0; r < n; ++r) {
                    if (beta[idx_occ(n, jb, r)] && r != q) {
                        same_part -= sign * same[idx4(n, p, q, r, r)];
                    }
                }
                const double* alpha_diag = alpha_cross_diag_data + (p * n + q) * n_alpha;
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    sigma[ia * n_beta + ib] +=
                        (same_part - sign * alpha_diag[ia]) * cvec[ia * n_beta + jb];
                }
            }
        }

        for (npy_intp link = 0; link < n_link_aa; ++link) {
            const npy_intp ia = i_aa[link];
            const npy_intp ja = j_aa[link];
            const double val = (
                static_cast<double>(phase_aa[link]) *
                same[idx4(n, p_aa[link], q_aa[link], r_aa[link], s_aa[link])]
            );
            for (npy_intp ib = 0; ib < n_beta; ++ib) {
                sigma[ia * n_beta + ib] += val * cvec[ja * n_beta + ib];
            }
        }

        if (!use_reduced_cross) {
            for (npy_intp link = 0; link < n_link_bb; ++link) {
                const npy_intp ib = i_bb[link];
                const npy_intp jb = j_bb[link];
                const double val = (
                    static_cast<double>(phase_bb[link]) *
                    same[idx4(n, p_bb[link], q_bb[link], r_bb[link], s_bb[link])]
                );
                for (npy_intp ia = 0; ia < n_alpha; ++ia) {
                    sigma[ia * n_beta + ib] += val * cvec[ia * n_beta + jb];
                }
            }
        }
    }

    if (use_reduced_cross) {
        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            const double diag_part = (
                ldet == rdet
                    ? diag[ldet]
                    : 0.5 * (diag[ldet] + diag[rdet])
            ) * cdata[pair];
            if (ldet == rdet) {
                sigma_pair[pair] = diag_part + 2.0 * sigma[ldet];
            } else {
                sigma_pair[pair] = diag_part + 2.0 * (sigma[ldet] + sigma[rdet]) * inv_sqrt2;
            }
        }

        if (worker_count > 1) {
            for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
                const double* local_sigma = cross_sigma_buffers.data() +
                    static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_pair);
                for (npy_intp pair = 0; pair < n_pair; ++pair) {
                    sigma_pair[pair] += local_sigma[pair];
                }
            }
        } else {
            reduced_cross_worker(0, sigma_pair);
        }
    } else if (worker_count > 1) {
        auto cross_worker = [&](int worker_id) {
            double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (
                npy_intp transition_a = worker_id;
                transition_a < n_transition;
                transition_a += worker_count
            ) {
                const npy_intp alpha_begin = alpha_offsets_data[transition_a];
                const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
                if (alpha_begin == alpha_end) {
                    continue;
                }
                const npy_intp p = transition_a / n;
                const npy_intp q = transition_a - p * n;
                for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                    const npy_intp beta_begin = beta_offsets_data[transition_b];
                    const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                    if (beta_begin == beta_end) {
                        continue;
                    }
                    const npy_intp r = transition_b / n;
                    const npy_intp s = transition_b - r * n;
                    const double eri_value = cross[idx4(n, p, q, r, s)];
                    if (eri_value == 0.0) {
                        continue;
                    }
                    for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                        const npy_intp ia = ordered_i_a[alpha_pos];
                        const npy_intp ja = ordered_j_a[alpha_pos];
                        const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                        double* sigma_row = local_sigma + ia * n_beta;
                        const double* c_row = cvec + ja * n_beta;
                        for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                            sigma_row[ordered_i_b[beta_pos]] +=
                                alpha_value *
                                static_cast<double>(ordered_phase_b[beta_pos]) *
                                c_row[ordered_j_b[beta_pos]];
                        }
                    }
                }
            }
        };
        for (int worker_id = 1; worker_id < worker_count; ++worker_id) {
            cross_threads.emplace_back(cross_worker, worker_id);
        }
        cross_worker(0);
        for (std::thread& thread : cross_threads) {
            thread.join();
        }
        for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
            const double* local_sigma = cross_sigma_buffers.data() +
                static_cast<std::size_t>(worker_id) * static_cast<std::size_t>(n_det);
            for (npy_intp det = 0; det < n_det; ++det) {
                sigma[det] += local_sigma[det];
            }
        }
    } else {
        for (npy_intp transition_a = 0; transition_a < n_transition; ++transition_a) {
            const npy_intp alpha_begin = alpha_offsets_data[transition_a];
            const npy_intp alpha_end = alpha_offsets_data[transition_a + 1];
            if (alpha_begin == alpha_end) {
                continue;
            }
            const npy_intp p = transition_a / n;
            const npy_intp q = transition_a - p * n;
            for (npy_intp transition_b = 0; transition_b < n_transition; ++transition_b) {
                const npy_intp beta_begin = beta_offsets_data[transition_b];
                const npy_intp beta_end = beta_offsets_data[transition_b + 1];
                if (beta_begin == beta_end) {
                    continue;
                }
                const npy_intp r = transition_b / n;
                const npy_intp s = transition_b - r * n;
                const double eri_value = cross[idx4(n, p, q, r, s)];
                if (eri_value == 0.0) {
                    continue;
                }
                for (npy_intp alpha_pos = alpha_begin; alpha_pos < alpha_end; ++alpha_pos) {
                    const npy_intp ia = ordered_i_a[alpha_pos];
                    const npy_intp ja = ordered_j_a[alpha_pos];
                    const double alpha_value = eri_value * static_cast<double>(ordered_phase_a[alpha_pos]);
                    double* sigma_row = sigma + ia * n_beta;
                    const double* c_row = cvec + ja * n_beta;
                    for (npy_intp beta_pos = beta_begin; beta_pos < beta_end; ++beta_pos) {
                        sigma_row[ordered_i_b[beta_pos]] +=
                            alpha_value *
                            static_cast<double>(ordered_phase_b[beta_pos]) *
                            c_row[ordered_j_b[beta_pos]];
                    }
                }
            }
        }

        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            if (ldet == rdet) {
                sigma_pair[pair] = sigma[ldet];
            } else {
                sigma_pair[pair] = (sigma[ldet] + sigma[rdet]) * inv_sqrt2;
            }
        }
    }

    Py_END_ALLOW_THREADS
    return true;
}

constexpr const char* spin0_workspace_capsule_name =
    "pyqed.qchem._casscf_cpp.Spin0PairSigmaWorkspace";

void destroy_spin0_workspace_capsule(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, spin0_workspace_capsule_name);
    if (pointer == nullptr) {
        PyErr_Clear();
        return;
    }
    delete static_cast<Spin0PairSigmaWorkspace*>(pointer);
}

PyObject* create_spin0_pair_workspace(PyObject*, PyObject* args) {
    if (PyTuple_Size(args) != 46) {
        PyErr_SetString(
            PyExc_TypeError,
            "create_spin0_pair_workspace expects 46 positional arguments."
        );
        return nullptr;
    }
    const long parsed_workers = PyLong_AsLong(PyTuple_GET_ITEM(args, 45));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    auto* workspace = new (std::nothrow) Spin0PairSigmaWorkspace();
    if (workspace == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    if (!workspace->initialize(
            args,
            static_cast<int>(std::max<long>(1, parsed_workers))
        )) {
        delete workspace;
        return nullptr;
    }
#ifdef PYQED_HAVE_CBLAS
    if (!workspace->use_blas_pair) {
        delete workspace;
        PyErr_SetString(
            PyExc_NotImplementedError,
            "Packed BLAS direct-CI workspace requires balanced restricted spin strings."
        );
        return nullptr;
    }
#else
    delete workspace;
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Packed BLAS direct-CI workspace is not available on this platform."
    );
    return nullptr;
#endif
    return PyCapsule_New(
        workspace,
        spin0_workspace_capsule_name,
        destroy_spin0_workspace_capsule
    );
}

PyObject* apply_spin0_pair_workspace_det(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* c_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &c_object)) {
        return nullptr;
    }
    auto* workspace = static_cast<Spin0PairSigmaWorkspace*>(
        PyCapsule_GetPointer(capsule, spin0_workspace_capsule_name)
    );
    if (workspace == nullptr) {
        return nullptr;
    }
    ArrayRef c(c_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!c) {
        return nullptr;
    }
    if (PyArray_NDIM(c.obj) != 1 || PyArray_DIM(c.obj, 0) != workspace->n_det) {
        PyErr_SetString(
            PyExc_ValueError,
            "Direct-CI workspace input has an incompatible determinant dimension."
        );
        return nullptr;
    }
    npy_intp output_dims[1] = {workspace->n_det};
    PyObject* output_object = PyArray_SimpleNew(1, output_dims, NPY_DOUBLE);
    if (output_object == nullptr) {
        return nullptr;
    }
#ifdef PYQED_HAVE_CBLAS
    auto* output = reinterpret_cast<PyArrayObject*>(output_object);
    if (!workspace->apply_blas_det(
            static_cast<const double*>(PyArray_DATA(c.obj)),
            static_cast<double*>(PyArray_DATA(output))
        )) {
        Py_DECREF(output_object);
        return nullptr;
    }
    return output_object;
#else
    Py_DECREF(output_object);
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Packed BLAS direct-CI workspace is not available on this platform."
    );
    return nullptr;
#endif
}

bool jacobi_eigh_small(
    std::vector<double> a,
    int n,
    std::vector<double>& evals,
    std::vector<double>& evecs
) {
    evals.assign(static_cast<std::size_t>(n), 0.0);
    evecs.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        evecs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] = 1.0;
    }
    if (n <= 0) {
        return false;
    }
    if (n == 1) {
        evals[0] = a[0];
        return true;
    }

#ifdef __APPLE__
    {
        const int ni = n;
        const int lda = std::max(1, ni);
        std::vector<double> a_col(static_cast<std::size_t>(lda) * static_cast<std::size_t>(ni));
        for (int row = 0; row < ni; ++row) {
            for (int col = 0; col < ni; ++col) {
                a_col[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * static_cast<std::size_t>(lda)] =
                    a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ni) + static_cast<std::size_t>(col)];
            }
        }
        std::vector<double> w(static_cast<std::size_t>(ni), 0.0);
        char jobz = 'V';
        char uplo = 'U';
        int n_arg = ni;
        int lda_arg = lda;
        int lwork = -1;
        int info = 0;
        double work_query = 0.0;
        dsyev_(
            &jobz,
            &uplo,
            &n_arg,
            a_col.data(),
            &lda_arg,
            w.data(),
            &work_query,
            &lwork,
            &info
        );
        if (info == 0) {
            lwork = std::max(1, static_cast<int>(work_query));
            std::vector<double> work(static_cast<std::size_t>(lwork), 0.0);
            n_arg = ni;
            lda_arg = lda;
            info = 0;
            dsyev_(
                &jobz,
                &uplo,
                &n_arg,
                a_col.data(),
                &lda_arg,
                w.data(),
                work.data(),
                &lwork,
                &info
            );
            if (info == 0) {
                evals = std::move(w);
                for (int col = 0; col < ni; ++col) {
                    for (int row = 0; row < ni; ++row) {
                        evecs[
                            static_cast<std::size_t>(row) * static_cast<std::size_t>(ni) +
                            static_cast<std::size_t>(col)
                        ] = a_col[
                            static_cast<std::size_t>(row) +
                            static_cast<std::size_t>(col) * static_cast<std::size_t>(lda)
                        ];
                    }
                }
                return true;
            }
        }
    }
#endif

    const int max_iter = std::max(64, 64 * n * n);
    for (int iter = 0; iter < max_iter; ++iter) {
        int p = 0;
        int q = 1;
        double max_offdiag = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double value = std::fabs(
                    a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]
                );
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag < 1e-14) {
            break;
        }

        const std::size_t pp = static_cast<std::size_t>(p) * static_cast<std::size_t>(n) + static_cast<std::size_t>(p);
        const std::size_t qq = static_cast<std::size_t>(q) * static_cast<std::size_t>(n) + static_cast<std::size_t>(q);
        const std::size_t pq = static_cast<std::size_t>(p) * static_cast<std::size_t>(n) + static_cast<std::size_t>(q);
        const double app = a[pp];
        const double aqq = a[qq];
        const double apq = a[pq];
        if (apq == 0.0) {
            continue;
        }
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
            (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const std::size_t kp = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(p);
            const std::size_t kq = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(q);
            const double akp = a[kp];
            const double akq = a[kq];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            a[kp] = new_kp;
            a[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) + static_cast<std::size_t>(k)] = new_kp;
            a[kq] = new_kq;
            a[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) + static_cast<std::size_t>(k)] = new_kq;
        }
        a[pp] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[qq] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[pq] = 0.0;
        a[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) + static_cast<std::size_t>(p)] = 0.0;

        for (int k = 0; k < n; ++k) {
            const std::size_t kp = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(p);
            const std::size_t kq = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(q);
            const double vkp = evecs[kp];
            const double vkq = evecs[kq];
            evecs[kp] = c * vkp - s * vkq;
            evecs[kq] = s * vkp + c * vkq;
        }
    }

    for (int i = 0; i < n; ++i) {
        evals[static_cast<std::size_t>(i)] =
            a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
    }
    return true;
}

bool native_blas_dims(std::size_t rows, int cols) {
    return (
        cols >= 0 &&
        rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        cols <= std::numeric_limits<int>::max()
    );
}

void projected_matrix_vt_av(
    const std::vector<double>& V,
    const std::vector<double>& AV,
    std::size_t n,
    int m,
    std::vector<double>& T,
    std::vector<double>& T_col
) {
#ifdef PYQED_HAVE_CBLAS
    constexpr int col_major = 102;
    constexpr int no_trans = 111;
    constexpr int trans = 112;
    if (native_blas_dims(n, m)) {
        T_col.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
        cblas_dgemm(
            col_major,
            trans,
            no_trans,
            m,
            m,
            static_cast<int>(n),
            1.0,
            V.data(),
            static_cast<int>(n),
            AV.data(),
            static_cast<int>(n),
            0.0,
            T_col.data(),
            m
        );
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < m; ++col) {
                T[static_cast<std::size_t>(row) * static_cast<std::size_t>(m) + static_cast<std::size_t>(col)] =
                    T_col[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * static_cast<std::size_t>(m)];
            }
        }
        return;
    }
#endif
    std::fill(T.begin(), T.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        const double* vi = V.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < m; ++j) {
            const double* avj = AV.data() + static_cast<std::size_t>(j) * n;
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                dot += vi[k] * avj[k];
            }
            T[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = dot;
        }
    }
}

void column_major_linear_combination(
    const std::vector<double>& M,
    std::size_t n,
    int m,
    const std::vector<double>& coeff,
    double* out
) {
#ifdef PYQED_HAVE_CBLAS
    constexpr int col_major = 102;
    constexpr int no_trans = 111;
    if (native_blas_dims(n, m)) {
        cblas_dgemv(
            col_major,
            no_trans,
            static_cast<int>(n),
            m,
            1.0,
            M.data(),
            static_cast<int>(n),
            coeff.data(),
            1,
            0.0,
            out,
            1
        );
        return;
    }
#endif
    std::fill(out, out + n, 0.0);
    for (int col = 0; col < m; ++col) {
        const double c = coeff[static_cast<std::size_t>(col)];
        const double* mcol = M.data() + static_cast<std::size_t>(col) * n;
        for (std::size_t k = 0; k < n; ++k) {
            out[k] += c * mcol[k];
        }
    }
}

void project_out_subspace(
    const std::vector<double>& V,
    std::size_t n,
    int m,
    std::vector<double>& corr,
    std::vector<double>& overlap
) {
    overlap.assign(static_cast<std::size_t>(m), 0.0);
#ifdef PYQED_HAVE_CBLAS
    constexpr int col_major = 102;
    constexpr int no_trans = 111;
    constexpr int trans = 112;
    if (native_blas_dims(n, m)) {
        cblas_dgemv(
            col_major,
            trans,
            static_cast<int>(n),
            m,
            1.0,
            V.data(),
            static_cast<int>(n),
            corr.data(),
            1,
            0.0,
            overlap.data(),
            1
        );
        cblas_dgemv(
            col_major,
            no_trans,
            static_cast<int>(n),
            m,
            -1.0,
            V.data(),
            static_cast<int>(n),
            overlap.data(),
            1,
            1.0,
            corr.data(),
            1
        );
        return;
    }
#endif
    for (int col = 0; col < m; ++col) {
        const double* vcol = V.data() + static_cast<std::size_t>(col) * n;
        double dot = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            dot += vcol[k] * corr[k];
        }
        overlap[static_cast<std::size_t>(col)] = dot;
    }
    for (int col = 0; col < m; ++col) {
        const double* vcol = V.data() + static_cast<std::size_t>(col) * n;
        const double dot = overlap[static_cast<std::size_t>(col)];
        for (std::size_t k = 0; k < n; ++k) {
            corr[k] -= dot * vcol[k];
        }
    }
}

void build_restart_block(
    const std::vector<double>& M,
    std::size_t n,
    int m,
    int keep,
    const std::vector<double>& evecs,
    const std::vector<int>& root_order,
    std::vector<double>& coeff,
    std::vector<double>& out
) {
    coeff.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(keep), 0.0);
    for (int kept = 0; kept < keep; ++kept) {
        const int eig_col = root_order[static_cast<std::size_t>(kept)];
        for (int row = 0; row < m; ++row) {
            coeff[static_cast<std::size_t>(row) + static_cast<std::size_t>(kept) * static_cast<std::size_t>(m)] =
                evecs[static_cast<std::size_t>(row) * static_cast<std::size_t>(m) + static_cast<std::size_t>(eig_col)];
        }
    }
#ifdef PYQED_HAVE_CBLAS
    constexpr int col_major = 102;
    constexpr int no_trans = 111;
    if (native_blas_dims(n, std::max(m, keep))) {
        cblas_dgemm(
            col_major,
            no_trans,
            no_trans,
            static_cast<int>(n),
            keep,
            m,
            1.0,
            M.data(),
            static_cast<int>(n),
            coeff.data(),
            m,
            0.0,
            out.data(),
            static_cast<int>(n)
        );
        return;
    }
#endif
    for (int kept = 0; kept < keep; ++kept) {
        double* dst = out.data() + static_cast<std::size_t>(kept) * n;
        std::fill(dst, dst + n, 0.0);
        for (int col = 0; col < m; ++col) {
            const double c = coeff[static_cast<std::size_t>(col) + static_cast<std::size_t>(kept) * static_cast<std::size_t>(m)];
            const double* src = M.data() + static_cast<std::size_t>(col) * n;
            for (std::size_t k = 0; k < n; ++k) {
                dst[k] += c * src[k];
            }
        }
    }
}

PyObject* davidson_spin0_pair(PyObject*, PyObject* args) {
    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != 51) {
        PyErr_SetString(PyExc_TypeError, "davidson_spin0_pair expects 51 positional arguments.");
        return nullptr;
    }

    const long parsed_workers = PyLong_AsLong(PyTuple_GET_ITEM(args, 45));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const int workers = static_cast<int>(std::max<long>(1, parsed_workers));
    const long parsed_nroots = PyLong_AsLong(PyTuple_GET_ITEM(args, 46));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const int nroots = static_cast<int>(parsed_nroots);
    if (nroots != 1) {
        PyErr_SetString(PyExc_NotImplementedError, "native spin0 Davidson currently supports one root.");
        return nullptr;
    }
    const double energy_tol = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 47));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const double residual_tol = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 48));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const long parsed_max_cycle = PyLong_AsLong(PyTuple_GET_ITEM(args, 49));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const long parsed_max_subspace = PyLong_AsLong(PyTuple_GET_ITEM(args, 50));
    if (PyErr_Occurred()) {
        return nullptr;
    }
    const int max_cycle = static_cast<int>(std::max<long>(1, parsed_max_cycle));
    const int max_subspace = static_cast<int>(std::max<long>(2, parsed_max_subspace));

    ArrayRef hdiag(PyTuple_GET_ITEM(args, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef c_dummy(PyTuple_GET_ITEM(args, 4), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_left(PyTuple_GET_ITEM(args, 5), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_right(PyTuple_GET_ITEM(args, 6), NPY_INTP, NPY_ARRAY_IN_ARRAY);
    if (!hdiag || !c_dummy || !pair_left || !pair_right) {
        return nullptr;
    }
    if (
        PyArray_NDIM(hdiag.obj) != 1 ||
        PyArray_NDIM(c_dummy.obj) != 1 ||
        PyArray_NDIM(pair_left.obj) != 1 ||
        PyArray_NDIM(pair_right.obj) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "native spin0 Davidson received invalid diagonal or pair arrays.");
        return nullptr;
    }
    const npy_intp n_pair = PyArray_DIM(pair_left.obj, 0);
    const npy_intp n_det = PyArray_DIM(hdiag.obj, 0);
    if (
        n_pair < 1 ||
        PyArray_DIM(c_dummy.obj, 0) != n_pair ||
        PyArray_DIM(pair_right.obj, 0) != n_pair
    ) {
        PyErr_SetString(PyExc_ValueError, "native spin0 Davidson pair dimensions are inconsistent.");
        return nullptr;
    }
    const double* diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));
    const npy_intp* left = static_cast<const npy_intp*>(PyArray_DATA(pair_left.obj));
    const npy_intp* right = static_cast<const npy_intp*>(PyArray_DATA(pair_right.obj));

    std::vector<double> spin0_diag;
    std::vector<npy_intp> guess_order;
    try {
        spin0_diag.assign(static_cast<std::size_t>(n_pair), 0.0);
        guess_order.resize(static_cast<std::size_t>(n_pair));
        std::iota(guess_order.begin(), guess_order.end(), 0);
        for (npy_intp pair = 0; pair < n_pair; ++pair) {
            const npy_intp ldet = left[pair];
            const npy_intp rdet = right[pair];
            if (ldet < 0 || ldet >= n_det || rdet < 0 || rdet >= n_det) {
                PyErr_SetString(PyExc_ValueError, "native spin0 Davidson pair indices are out of range.");
                return nullptr;
            }
            spin0_diag[static_cast<std::size_t>(pair)] =
                (ldet == rdet) ? diag[ldet] : 0.5 * (diag[ldet] + diag[rdet]);
        }
        const int initial_cols = static_cast<int>(std::min<npy_intp>(n_pair, 2));
        std::partial_sort(
            guess_order.begin(),
            guess_order.begin() + initial_cols,
            guess_order.end(),
            [&](npy_intp lhs, npy_intp rhs) {
                return spin0_diag[static_cast<std::size_t>(lhs)] <
                    spin0_diag[static_cast<std::size_t>(rhs)];
            }
        );
    } catch (...) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate native spin0 Davidson work arrays.");
        return nullptr;
    }

    const std::size_t n = static_cast<std::size_t>(n_pair);
    const int capacity = static_cast<int>(std::min<npy_intp>(n_pair, max_subspace));
    std::vector<double> V;
    std::vector<double> AV;
    std::vector<double> T;
    std::vector<double> evals;
    std::vector<double> evecs;
    std::vector<double> T_col;
    std::vector<double> ritz;
    std::vector<double> aritz;
    std::vector<double> resid;
    std::vector<double> corr;
    std::vector<double> root_coeff;
    std::vector<double> overlap;
    std::vector<double> restart_V;
    std::vector<double> restart_AV;
    std::vector<double> restart_coeff;
    std::vector<int> root_order;
    try {
        V.assign(n * static_cast<std::size_t>(capacity), 0.0);
        AV.assign(n * static_cast<std::size_t>(capacity), 0.0);
        T.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(capacity), 0.0);
        T_col.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(capacity), 0.0);
        ritz.assign(n, 0.0);
        aritz.assign(n, 0.0);
        resid.assign(n, 0.0);
        corr.assign(n, 0.0);
        root_coeff.assign(static_cast<std::size_t>(capacity), 0.0);
        overlap.assign(static_cast<std::size_t>(capacity), 0.0);
        restart_V.assign(n * static_cast<std::size_t>(capacity), 0.0);
        restart_AV.assign(n * static_cast<std::size_t>(capacity), 0.0);
        restart_coeff.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(capacity), 0.0);
        root_order.resize(static_cast<std::size_t>(capacity));
    } catch (...) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate native spin0 Davidson subspace.");
        return nullptr;
    }

    Spin0PairSigmaWorkspace sigma_workspace;
    if (!sigma_workspace.initialize(args, workers)) {
        return nullptr;
    }
    auto sigma_matvec = [&](const double* input, double* output) -> bool {
        return sigma_workspace.apply(input, output);
    };

    int m = static_cast<int>(std::min<npy_intp>(n_pair, 2));
    for (int col = 0; col < m; ++col) {
        V[static_cast<std::size_t>(col) * n + static_cast<std::size_t>(guess_order[static_cast<std::size_t>(col)])] = 1.0;
        if (!sigma_matvec(V.data() + static_cast<std::size_t>(col) * n, AV.data() + static_cast<std::size_t>(col) * n)) {
            return nullptr;
        }
    }

    double theta = 0.0;
    double previous_theta = std::numeric_limits<double>::infinity();
    double resid_norm = std::numeric_limits<double>::infinity();
    for (int cycle = 0; cycle < max_cycle; ++cycle) {
        projected_matrix_vt_av(V, AV, n, m, T, T_col);
        if (!jacobi_eigh_small(
                std::vector<double>(T.begin(), T.begin() + static_cast<std::ptrdiff_t>(m) * m),
                m,
                evals,
                evecs
            )) {
            PyErr_SetString(PyExc_RuntimeError, "native spin0 Davidson failed to diagonalize subspace.");
            return nullptr;
        }
        for (int i = 0; i < m; ++i) {
            root_order[static_cast<std::size_t>(i)] = i;
        }
        std::sort(
            root_order.begin(),
            root_order.begin() + m,
            [&](int lhs, int rhs) {
                return evals[static_cast<std::size_t>(lhs)] <
                    evals[static_cast<std::size_t>(rhs)];
            }
        );
        const int root_col = root_order[0];
        theta = evals[static_cast<std::size_t>(root_col)];

        for (int col = 0; col < m; ++col) {
            root_coeff[static_cast<std::size_t>(col)] =
                evecs[static_cast<std::size_t>(col) * static_cast<std::size_t>(m) + static_cast<std::size_t>(root_col)];
        }
        column_major_linear_combination(V, n, m, root_coeff, ritz.data());
        column_major_linear_combination(AV, n, m, root_coeff, aritz.data());

        resid_norm = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            resid[k] = aritz[k] - theta * ritz[k];
            resid_norm += resid[k] * resid[k];
        }
        resid_norm = std::sqrt(resid_norm);
        const bool energy_converged =
            std::isfinite(previous_theta) && std::fabs(theta - previous_theta) < energy_tol;
        if (energy_converged && resid_norm < residual_tol) {
            break;
        }
        previous_theta = theta;

        for (std::size_t k = 0; k < n; ++k) {
            double denom = theta - spin0_diag[k];
            if (std::fabs(denom) <= 1e-12) {
                denom = denom >= 0.0 ? 1e-12 : -1e-12;
            }
            corr[k] = resid[k] / denom;
        }
        project_out_subspace(V, n, m, corr, overlap);
        double corr_norm = 0.0;
        for (double value : corr) {
            corr_norm += value * value;
        }
        corr_norm = std::sqrt(corr_norm);
        if (corr_norm <= 1e-12) {
            break;
        }
        for (double& value : corr) {
            value /= corr_norm;
        }

        if (m + 1 > capacity) {
            int keep = std::min(m, 4);
            keep = std::min(keep, std::max(1, capacity - 1));
            build_restart_block(V, n, m, keep, evecs, root_order, restart_coeff, restart_V);
            build_restart_block(AV, n, m, keep, evecs, root_order, restart_coeff, restart_AV);
            std::copy(corr.begin(), corr.end(), restart_V.begin() + static_cast<std::size_t>(keep) * n);
            if (!sigma_matvec(
                    restart_V.data() + static_cast<std::size_t>(keep) * n,
                    restart_AV.data() + static_cast<std::size_t>(keep) * n
                )) {
                return nullptr;
            }
            const std::size_t restart_size = static_cast<std::size_t>(keep + 1) * n;
            std::copy(restart_V.begin(), restart_V.begin() + static_cast<std::ptrdiff_t>(restart_size), V.begin());
            std::copy(restart_AV.begin(), restart_AV.begin() + static_cast<std::ptrdiff_t>(restart_size), AV.begin());
            m = keep + 1;
        } else {
            std::copy(corr.begin(), corr.end(), V.begin() + static_cast<std::size_t>(m) * n);
            if (!sigma_matvec(V.data() + static_cast<std::size_t>(m) * n, AV.data() + static_cast<std::size_t>(m) * n)) {
                return nullptr;
            }
            ++m;
        }
    }

    npy_intp energy_dims[1] = {1};
    PyObject* energies_obj = PyArray_SimpleNew(1, energy_dims, NPY_DOUBLE);
    if (energies_obj == nullptr) {
        return nullptr;
    }
    *static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(energies_obj))) = theta;

    npy_intp vec_dims[2] = {n_pair, 1};
    PyObject* vecs_obj = PyArray_SimpleNew(2, vec_dims, NPY_DOUBLE);
    if (vecs_obj == nullptr) {
        Py_DECREF(energies_obj);
        return nullptr;
    }
    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vecs_obj)), ritz.data(), n * sizeof(double));
    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(energies_obj);
        Py_DECREF(vecs_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, energies_obj);
    PyTuple_SET_ITEM(result, 1, vecs_obj);
    return result;
}

PyObject* davidson_rhf_workspace(PyObject*, PyObject* args) {
    if (PyTuple_Size(args) != 8) {
        PyErr_SetString(PyExc_TypeError, "davidson_rhf_workspace expects 8 positional arguments.");
        return nullptr;
    }
    auto* workspace = static_cast<Spin0PairSigmaWorkspace*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(args, 0), spin0_workspace_capsule_name)
    );
    if (workspace == nullptr) {
        return nullptr;
    }
    ArrayRef hdiag(PyTuple_GET_ITEM(args, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!hdiag || PyArray_NDIM(hdiag.obj) != 1 ||
        PyArray_DIM(hdiag.obj, 0) != workspace->n_det) {
        PyErr_SetString(PyExc_ValueError, "Native RHF Davidson received an incompatible diagonal.");
        return nullptr;
    }
    const int nroots = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 2)));
    const double energy_tol = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 3));
    const double residual_tol = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 4));
    const int max_cycle = static_cast<int>(std::max<long>(
        1, PyLong_AsLong(PyTuple_GET_ITEM(args, 5))
    ));
    const int requested_subspace = static_cast<int>(std::max<long>(
        2, PyLong_AsLong(PyTuple_GET_ITEM(args, 6))
    ));
    ArrayRef guess(PyTuple_GET_ITEM(args, 7), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (!guess || PyArray_NDIM(guess.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "Native RHF Davidson guess must be a 2D array.");
        return nullptr;
    }
    if (nroots < 1) {
        PyErr_SetString(PyExc_ValueError, "Native RHF Davidson requires at least one root.");
        return nullptr;
    }
#ifndef PYQED_HAVE_CBLAS
    PyErr_SetString(PyExc_NotImplementedError, "Native packed-BLAS RHF Davidson is unavailable on this platform.");
    return nullptr;
#else
    if (!workspace->use_blas_pair) {
        PyErr_SetString(PyExc_NotImplementedError, "Native RHF Davidson requires a packed-BLAS workspace.");
        return nullptr;
    }
    const npy_intp n_dim = workspace->n_det;
    if (n_dim < 2 || nroots > n_dim) {
        PyErr_SetString(PyExc_NotImplementedError, "Native RHF Davidson received an unsupported determinant/root count.");
        return nullptr;
    }
    if (PyArray_DIM(guess.obj, 0) != n_dim) {
        PyErr_SetString(PyExc_ValueError, "Native RHF Davidson guess has an incompatible determinant dimension.");
        return nullptr;
    }
    const std::size_t n = static_cast<std::size_t>(n_dim);
    const int minimum_capacity = std::min<int>(
        static_cast<int>(n_dim),
        std::max(2, 2 * nroots)
    );
    const int capacity = static_cast<int>(std::min<npy_intp>(
        n_dim,
        std::max(requested_subspace, minimum_capacity)
    ));
    const int supplied_guess_cols = static_cast<int>(PyArray_DIM(guess.obj, 1));
    if (supplied_guess_cols > 0 && supplied_guess_cols < nroots) {
        PyErr_SetString(PyExc_ValueError, "Native RHF Davidson guess has fewer columns than requested roots.");
        return nullptr;
    }
    const int initial_cols = supplied_guess_cols > 0
        ? std::min(capacity, supplied_guess_cols)
        : std::min<int>(static_cast<int>(n_dim), std::max(2, nroots));
    const int thick_keep_target = std::min(
        capacity,
        std::max(nroots + 1, 2 * nroots + 2)
    );
    const int restart_capacity = std::min(
        capacity,
        thick_keep_target + nroots
    );
    const double* diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));

    std::vector<npy_intp> guess_order;
    if (supplied_guess_cols == 0) {
        guess_order.assign(static_cast<std::size_t>(initial_cols), -1);
        std::vector<double> guess_values(
            static_cast<std::size_t>(initial_cols),
            std::numeric_limits<double>::infinity()
        );
        for (npy_intp index = 0; index < n_dim; ++index) {
            int position = initial_cols;
            for (int col = 0; col < initial_cols; ++col) {
                if (diag[index] < guess_values[static_cast<std::size_t>(col)]) {
                    position = col;
                    break;
                }
            }
            if (position < initial_cols) {
                for (int col = initial_cols - 1; col > position; --col) {
                    guess_values[static_cast<std::size_t>(col)] =
                        guess_values[static_cast<std::size_t>(col - 1)];
                    guess_order[static_cast<std::size_t>(col)] =
                        guess_order[static_cast<std::size_t>(col - 1)];
                }
                guess_values[static_cast<std::size_t>(position)] = diag[index];
                guess_order[static_cast<std::size_t>(position)] = index;
            }
        }
    }

    std::vector<double> V;
    std::vector<double> AV;
    std::vector<double> T;
    std::vector<double> evals;
    std::vector<double> evecs;
    std::vector<double> T_col;
    std::vector<double> ritz;
    std::vector<double> aritz;
    std::vector<double> resid;
    std::vector<double> corr;
    std::vector<double> corrections;
    std::vector<double> root_coeff;
    std::vector<double> overlap;
    std::vector<double> restart_V;
    std::vector<double> restart_AV;
    std::vector<double> restart_coeff;
    std::vector<int> root_order;
    try {
        V.assign(n * static_cast<std::size_t>(capacity), 0.0);
        AV.assign(n * static_cast<std::size_t>(capacity), 0.0);
        T.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(capacity), 0.0);
        T_col.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(capacity), 0.0);
        ritz.assign(n * static_cast<std::size_t>(nroots), 0.0);
        aritz.assign(n * static_cast<std::size_t>(nroots), 0.0);
        resid.assign(n * static_cast<std::size_t>(nroots), 0.0);
        corr.assign(n, 0.0);
        corrections.assign(n * static_cast<std::size_t>(nroots), 0.0);
        root_coeff.assign(
            static_cast<std::size_t>(capacity) * static_cast<std::size_t>(restart_capacity),
            0.0
        );
        overlap.assign(static_cast<std::size_t>(capacity), 0.0);
        restart_V.assign(n * static_cast<std::size_t>(restart_capacity), 0.0);
        restart_AV.assign(n * static_cast<std::size_t>(restart_capacity), 0.0);
        restart_coeff.assign(static_cast<std::size_t>(capacity) * static_cast<std::size_t>(restart_capacity), 0.0);
        root_order.resize(static_cast<std::size_t>(capacity));
    } catch (...) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate native RHF Davidson subspace.");
        return nullptr;
    }

    auto sigma_matvec = [&](const double* input, double* output) -> bool {
        return workspace->apply_blas_det(input, output);
    };
    int m = initial_cols;
    const double* guess_data = static_cast<const double*>(PyArray_DATA(guess.obj));
    for (int col = 0; col < m; ++col) {
        if (supplied_guess_cols > 0) {
            for (std::size_t row = 0; row < n; ++row) {
                V[static_cast<std::size_t>(col) * n + row] =
                    guess_data[row * static_cast<std::size_t>(supplied_guess_cols) +
                               static_cast<std::size_t>(col)];
            }
        } else {
            V[static_cast<std::size_t>(col) * n +
              static_cast<std::size_t>(guess_order[static_cast<std::size_t>(col)])] = 1.0;
        }
        if (!sigma_matvec(
                V.data() + static_cast<std::size_t>(col) * n,
                AV.data() + static_cast<std::size_t>(col) * n
            )) {
            return nullptr;
        }
    }

    std::vector<double> theta(
        static_cast<std::size_t>(nroots),
        std::numeric_limits<double>::infinity()
    );
    std::vector<double> previous_theta = theta;
    std::vector<double> resid_norm(static_cast<std::size_t>(nroots), 0.0);
    std::vector<double> energy_change(
        static_cast<std::size_t>(nroots),
        std::numeric_limits<double>::infinity()
    );
    bool converged = false;
    int iterations = 0;
    for (int cycle = 0; cycle < max_cycle; ++cycle) {
        iterations = cycle + 1;
        projected_matrix_vt_av(V, AV, n, m, T, T_col);
        if (!jacobi_eigh_small(
                std::vector<double>(T.begin(), T.begin() + static_cast<std::ptrdiff_t>(m) * m),
                m,
                evals,
                evecs
            )) {
            PyErr_SetString(PyExc_RuntimeError, "Native RHF Davidson failed to diagonalize its subspace.");
            return nullptr;
        }
        for (int index = 0; index < m; ++index) {
            root_order[static_cast<std::size_t>(index)] = index;
        }
        std::sort(
            root_order.begin(),
            root_order.begin() + m,
            [&](int lhs, int rhs) { return evals[lhs] < evals[rhs]; }
        );
        for (int root = 0; root < nroots; ++root) {
            theta[static_cast<std::size_t>(root)] =
                evals[static_cast<std::size_t>(root_order[static_cast<std::size_t>(root)])];
        }
        build_restart_block(V, n, m, nroots, evecs, root_order, root_coeff, ritz);
        build_restart_block(AV, n, m, nroots, evecs, root_order, root_coeff, aritz);

        bool all_converged = cycle > 0;
        bool residuals_converged = true;
        for (int root = 0; root < nroots; ++root) {
            const std::size_t offset = static_cast<std::size_t>(root) * n;
            double norm_sq = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                resid[offset + k] = aritz[offset + k] -
                    theta[static_cast<std::size_t>(root)] * ritz[offset + k];
                norm_sq += resid[offset + k] * resid[offset + k];
            }
            resid_norm[static_cast<std::size_t>(root)] = std::sqrt(norm_sq);
            energy_change[static_cast<std::size_t>(root)] = std::fabs(
                theta[static_cast<std::size_t>(root)] -
                previous_theta[static_cast<std::size_t>(root)]
            );
            residuals_converged = residuals_converged &&
                resid_norm[static_cast<std::size_t>(root)] < residual_tol;
            all_converged = all_converged &&
                energy_change[static_cast<std::size_t>(root)] < energy_tol &&
                resid_norm[static_cast<std::size_t>(root)] < residual_tol;
        }
        if (all_converged || residuals_converged) {
            converged = true;
            break;
        }
        previous_theta = theta;

        int accepted = 0;
        for (int root = 0; root < nroots; ++root) {
            if (accepted >= capacity - nroots) {
                break;
            }
            if (resid_norm[static_cast<std::size_t>(root)] < residual_tol) {
                continue;
            }
            const std::size_t offset = static_cast<std::size_t>(root) * n;
            for (std::size_t k = 0; k < n; ++k) {
                double denominator = theta[static_cast<std::size_t>(root)] - diag[k];
                if (std::fabs(denominator) <= 1e-12) {
                    denominator = denominator >= 0.0 ? 1e-12 : -1e-12;
                }
                corr[k] = resid[offset + k] / denominator;
            }
            for (int pass = 0; pass < 2; ++pass) {
                project_out_subspace(V, n, m, corr, overlap);
                for (int previous = 0; previous < accepted; ++previous) {
                    const double* prior = corrections.data() + static_cast<std::size_t>(previous) * n;
                    double dot = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        dot += prior[k] * corr[k];
                    }
                    for (std::size_t k = 0; k < n; ++k) {
                        corr[k] -= dot * prior[k];
                    }
                }
            }
            double corr_norm = 0.0;
            for (double value : corr) {
                corr_norm += value * value;
            }
            corr_norm = std::sqrt(corr_norm);
            if (corr_norm <= 1e-12) {
                continue;
            }
            double* target = corrections.data() + static_cast<std::size_t>(accepted) * n;
            for (std::size_t k = 0; k < n; ++k) {
                target[k] = corr[k] / corr_norm;
            }
            ++accepted;
        }

        if (accepted == 0) {
            break;
        }
        if (m + accepted > capacity) {
            int keep = std::min(m, thick_keep_target);
            keep = std::min(keep, capacity - accepted);
            build_restart_block(V, n, m, keep, evecs, root_order, restart_coeff, restart_V);
            build_restart_block(AV, n, m, keep, evecs, root_order, restart_coeff, restart_AV);
            for (int col = 0; col < accepted; ++col) {
                std::copy(
                    corrections.begin() + static_cast<std::ptrdiff_t>(col) * static_cast<std::ptrdiff_t>(n),
                    corrections.begin() + static_cast<std::ptrdiff_t>(col + 1) * static_cast<std::ptrdiff_t>(n),
                    restart_V.begin() + static_cast<std::ptrdiff_t>(keep + col) * static_cast<std::ptrdiff_t>(n)
                );
                if (!sigma_matvec(
                        restart_V.data() + static_cast<std::size_t>(keep + col) * n,
                        restart_AV.data() + static_cast<std::size_t>(keep + col) * n
                    )) {
                    return nullptr;
                }
            }
            const std::size_t restart_size = static_cast<std::size_t>(keep + accepted) * n;
            std::copy(restart_V.begin(), restart_V.begin() + static_cast<std::ptrdiff_t>(restart_size), V.begin());
            std::copy(restart_AV.begin(), restart_AV.begin() + static_cast<std::ptrdiff_t>(restart_size), AV.begin());
            m = keep + accepted;
        } else {
            for (int col = 0; col < accepted; ++col) {
                std::copy(
                    corrections.begin() + static_cast<std::ptrdiff_t>(col) * static_cast<std::ptrdiff_t>(n),
                    corrections.begin() + static_cast<std::ptrdiff_t>(col + 1) * static_cast<std::ptrdiff_t>(n),
                    V.begin() + static_cast<std::ptrdiff_t>(m + col) * static_cast<std::ptrdiff_t>(n)
                );
                if (!sigma_matvec(
                        V.data() + static_cast<std::size_t>(m + col) * n,
                        AV.data() + static_cast<std::size_t>(m + col) * n
                    )) {
                    return nullptr;
                }
            }
            m += accepted;
        }
    }

    if (!converged) {
        PyErr_SetString(PyExc_RuntimeError, "Native RHF Davidson did not converge within max_cycle iterations.");
        return nullptr;
    }

    npy_intp energy_dims[1] = {nroots};
    PyObject* energies_obj = PyArray_SimpleNew(1, energy_dims, NPY_DOUBLE);
    if (energies_obj == nullptr) {
        return nullptr;
    }
    std::memcpy(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(energies_obj)),
        theta.data(),
        static_cast<std::size_t>(nroots) * sizeof(double)
    );
    npy_intp vector_dims[2] = {n_dim, nroots};
    PyObject* vectors_obj = PyArray_EMPTY(2, vector_dims, NPY_DOUBLE, 1);
    if (vectors_obj == nullptr) {
        Py_DECREF(energies_obj);
        return nullptr;
    }
    std::memcpy(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(vectors_obj)),
        ritz.data(),
        n * static_cast<std::size_t>(nroots) * sizeof(double)
    );
    PyObject* residuals_obj = PyArray_SimpleNew(1, energy_dims, NPY_DOUBLE);
    PyObject* changes_obj = PyArray_SimpleNew(1, energy_dims, NPY_DOUBLE);
    if (residuals_obj == nullptr || changes_obj == nullptr) {
        Py_DECREF(energies_obj);
        Py_DECREF(vectors_obj);
        Py_XDECREF(residuals_obj);
        Py_XDECREF(changes_obj);
        return nullptr;
    }
    std::memcpy(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(residuals_obj)),
        resid_norm.data(),
        static_cast<std::size_t>(nroots) * sizeof(double)
    );
    std::memcpy(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(changes_obj)),
        energy_change.data(),
        static_cast<std::size_t>(nroots) * sizeof(double)
    );
    PyObject* diagnostics_obj = Py_BuildValue(
        "{s:O,s:i,s:i,s:N,s:N}",
        "converged", Py_True,
        "iterations", iterations,
        "subspace_dimension", m,
        "residual_norms", residuals_obj,
        "energy_changes", changes_obj
    );
    if (diagnostics_obj == nullptr) {
        Py_DECREF(energies_obj);
        Py_DECREF(vectors_obj);
        return nullptr;
    }
    PyObject* result = PyTuple_New(3);
    if (result == nullptr) {
        Py_DECREF(energies_obj);
        Py_DECREF(vectors_obj);
        Py_DECREF(diagnostics_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, energies_obj);
    PyTuple_SET_ITEM(result, 1, vectors_obj);
    PyTuple_SET_ITEM(result, 2, diagnostics_obj);
    return result;
#endif
}

bool validate_1d_same_length(
    PyArrayObject* a,
    PyArrayObject* b,
    PyArrayObject* c,
    PyArrayObject* d,
    PyArrayObject* e,
    const char* name
) {
    if (
        PyArray_NDIM(a) != 1 ||
        PyArray_NDIM(b) != 1 ||
        PyArray_NDIM(c) != 1 ||
        PyArray_NDIM(d) != 1 ||
        PyArray_NDIM(e) != 1
    ) {
        PyErr_Format(PyExc_ValueError, "%s link arrays must be 1D.", name);
        return false;
    }
    const npy_intp n = PyArray_DIM(a, 0);
    if (
        PyArray_DIM(b, 0) != n ||
        PyArray_DIM(c, 0) != n ||
        PyArray_DIM(d, 0) != n ||
        PyArray_DIM(e, 0) != n
    ) {
        PyErr_Format(PyExc_ValueError, "%s link arrays must have matching lengths.", name);
        return false;
    }
    return true;
}

bool validate_conn_value_block(
    PyArrayObject* values,
    PyArrayObject* i_idx,
    PyArrayObject* j_idx,
    const char* name
) {
    if (
        PyArray_NDIM(values) != 1 ||
        PyArray_NDIM(i_idx) != 1 ||
        PyArray_NDIM(j_idx) != 1
    ) {
        PyErr_Format(PyExc_ValueError, "%s connection arrays must be 1D.", name);
        return false;
    }
    const npy_intp n = PyArray_DIM(values, 0);
    if (PyArray_DIM(i_idx, 0) != n || PyArray_DIM(j_idx, 0) != n) {
        PyErr_Format(PyExc_ValueError, "%s connection arrays must have matching lengths.", name);
        return false;
    }
    return true;
}

inline bool conn_index_in_range(npy_int32 idx, npy_intp n) {
    return idx >= 0 && static_cast<npy_intp>(idx) < n;
}

void add_conn_values_block(
    double* sigma,
    const double* c,
    npy_intp n_det,
    npy_intp n_vec,
    const double* values,
    const npy_int32* i_idx,
    const npy_int32* j_idx,
    npy_intp n_link
) {
    for (npy_intp link = 0; link < n_link; ++link) {
        const npy_intp i = static_cast<npy_intp>(i_idx[link]);
        const npy_intp j = static_cast<npy_intp>(j_idx[link]);
        const double value = values[link];
        double* sigma_row = sigma + i * n_vec;
        const double* c_row = c + j * n_vec;
        for (npy_intp root = 0; root < n_vec; ++root) {
            sigma_row[root] += value * c_row[root];
        }
    }
}

PyObject* sigma_values_conn(PyObject*, PyObject* args) {
    PyObject* hdiag_obj = nullptr;
    PyObject* h_a_obj = nullptr;
    PyObject* h_b_obj = nullptr;
    PyObject* h_aa_obj = nullptr;
    PyObject* h_bb_obj = nullptr;
    PyObject* h_ab_obj = nullptr;
    PyObject* c_obj = nullptr;
    PyObject* i_a_obj = nullptr;
    PyObject* j_a_obj = nullptr;
    PyObject* i_b_obj = nullptr;
    PyObject* j_b_obj = nullptr;
    PyObject* i_aa_obj = nullptr;
    PyObject* j_aa_obj = nullptr;
    PyObject* i_bb_obj = nullptr;
    PyObject* j_bb_obj = nullptr;
    PyObject* i_ab_obj = nullptr;
    PyObject* j_ab_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOOOOOO",
            &hdiag_obj,
            &h_a_obj,
            &h_b_obj,
            &h_aa_obj,
            &h_bb_obj,
            &h_ab_obj,
            &c_obj,
            &i_a_obj,
            &j_a_obj,
            &i_b_obj,
            &j_b_obj,
            &i_aa_obj,
            &j_aa_obj,
            &i_bb_obj,
            &j_bb_obj,
            &i_ab_obj,
            &j_ab_obj
        )) {
        return nullptr;
    }

    ArrayRef hdiag(hdiag_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_a(h_a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_b(h_b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_aa(h_aa_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_bb(h_bb_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h_ab(h_ab_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef c(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef i_a(i_a_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef j_a(j_a_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef i_b(i_b_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef j_b(j_b_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef i_aa(i_aa_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef j_aa(j_aa_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef i_bb(i_bb_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef j_bb(j_bb_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef i_ab(i_ab_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    ArrayRef j_ab(j_ab_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!hdiag || !h_a || !h_b || !h_aa || !h_bb || !h_ab || !c ||
        !i_a || !j_a || !i_b || !j_b || !i_aa || !j_aa || !i_bb || !j_bb ||
        !i_ab || !j_ab) {
        return nullptr;
    }
    if (PyArray_NDIM(hdiag.obj) != 1 || (PyArray_NDIM(c.obj) != 1 && PyArray_NDIM(c.obj) != 2)) {
        PyErr_SetString(PyExc_ValueError, "sigma_values_conn expects hdiag as 1D and c as 1D or 2D.");
        return nullptr;
    }
    if (
        !validate_conn_value_block(h_a.obj, i_a.obj, j_a.obj, "A") ||
        !validate_conn_value_block(h_b.obj, i_b.obj, j_b.obj, "B") ||
        !validate_conn_value_block(h_aa.obj, i_aa.obj, j_aa.obj, "AA") ||
        !validate_conn_value_block(h_bb.obj, i_bb.obj, j_bb.obj, "BB") ||
        !validate_conn_value_block(h_ab.obj, i_ab.obj, j_ab.obj, "AB")
    ) {
        return nullptr;
    }

    const npy_intp n_det = PyArray_DIM(hdiag.obj, 0);
    const bool input_is_matrix = PyArray_NDIM(c.obj) == 2;
    const npy_intp n_vec = input_is_matrix ? PyArray_DIM(c.obj, 1) : 1;
    if (PyArray_DIM(c.obj, 0) != n_det || n_vec < 1) {
        PyErr_SetString(PyExc_ValueError, "sigma_values_conn received a CI vector shape inconsistent with hdiag.");
        return nullptr;
    }

    auto check_range = [&](PyArrayObject* i_obj, PyArrayObject* j_obj, const char* name) -> bool {
        const auto* i_data = static_cast<const npy_int32*>(PyArray_DATA(i_obj));
        const auto* j_data = static_cast<const npy_int32*>(PyArray_DATA(j_obj));
        const npy_intp n_link = PyArray_DIM(i_obj, 0);
        for (npy_intp link = 0; link < n_link; ++link) {
            if (!conn_index_in_range(i_data[link], n_det) || !conn_index_in_range(j_data[link], n_det)) {
                PyErr_Format(PyExc_ValueError, "%s connection index out of range.", name);
                return false;
            }
        }
        return true;
    };
    if (
        !check_range(i_a.obj, j_a.obj, "A") ||
        !check_range(i_b.obj, j_b.obj, "B") ||
        !check_range(i_aa.obj, j_aa.obj, "AA") ||
        !check_range(i_bb.obj, j_bb.obj, "BB") ||
        !check_range(i_ab.obj, j_ab.obj, "AB")
    ) {
        return nullptr;
    }

    npy_intp dims[2] = {n_det, n_vec};
    PyObject* sigma_obj = input_is_matrix
        ? PyArray_SimpleNew(2, dims, NPY_DOUBLE)
        : PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (sigma_obj == nullptr) {
        return nullptr;
    }

    const double* diag = static_cast<const double*>(PyArray_DATA(hdiag.obj));
    const double* c_data = static_cast<const double*>(PyArray_DATA(c.obj));
    double* sigma = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(sigma_obj)));
    const double* h_a_data = static_cast<const double*>(PyArray_DATA(h_a.obj));
    const double* h_b_data = static_cast<const double*>(PyArray_DATA(h_b.obj));
    const double* h_aa_data = static_cast<const double*>(PyArray_DATA(h_aa.obj));
    const double* h_bb_data = static_cast<const double*>(PyArray_DATA(h_bb.obj));
    const double* h_ab_data = static_cast<const double*>(PyArray_DATA(h_ab.obj));
    const npy_int32* i_a_data = static_cast<const npy_int32*>(PyArray_DATA(i_a.obj));
    const npy_int32* j_a_data = static_cast<const npy_int32*>(PyArray_DATA(j_a.obj));
    const npy_int32* i_b_data = static_cast<const npy_int32*>(PyArray_DATA(i_b.obj));
    const npy_int32* j_b_data = static_cast<const npy_int32*>(PyArray_DATA(j_b.obj));
    const npy_int32* i_aa_data = static_cast<const npy_int32*>(PyArray_DATA(i_aa.obj));
    const npy_int32* j_aa_data = static_cast<const npy_int32*>(PyArray_DATA(j_aa.obj));
    const npy_int32* i_bb_data = static_cast<const npy_int32*>(PyArray_DATA(i_bb.obj));
    const npy_int32* j_bb_data = static_cast<const npy_int32*>(PyArray_DATA(j_bb.obj));
    const npy_int32* i_ab_data = static_cast<const npy_int32*>(PyArray_DATA(i_ab.obj));
    const npy_int32* j_ab_data = static_cast<const npy_int32*>(PyArray_DATA(j_ab.obj));
    const npy_intp n_a = PyArray_DIM(i_a.obj, 0);
    const npy_intp n_b = PyArray_DIM(i_b.obj, 0);
    const npy_intp n_aa = PyArray_DIM(i_aa.obj, 0);
    const npy_intp n_bb = PyArray_DIM(i_bb.obj, 0);
    const npy_intp n_ab = PyArray_DIM(i_ab.obj, 0);

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp det = 0; det < n_det; ++det) {
        const double diag_value = diag[det];
        const double* c_row = c_data + det * n_vec;
        double* sigma_row = sigma + det * n_vec;
        for (npy_intp root = 0; root < n_vec; ++root) {
            sigma_row[root] = diag_value * c_row[root];
        }
    }

    add_conn_values_block(
        sigma,
        c_data,
        n_det,
        n_vec,
        h_a_data,
        i_a_data,
        j_a_data,
        n_a
    );
    add_conn_values_block(
        sigma,
        c_data,
        n_det,
        n_vec,
        h_b_data,
        i_b_data,
        j_b_data,
        n_b
    );
    add_conn_values_block(
        sigma,
        c_data,
        n_det,
        n_vec,
        h_aa_data,
        i_aa_data,
        j_aa_data,
        n_aa
    );
    add_conn_values_block(
        sigma,
        c_data,
        n_det,
        n_vec,
        h_bb_data,
        i_bb_data,
        j_bb_data,
        n_bb
    );
    add_conn_values_block(
        sigma,
        c_data,
        n_det,
        n_vec,
        h_ab_data,
        i_ab_data,
        j_ab_data,
        n_ab
    );

    Py_END_ALLOW_THREADS

    return sigma_obj;
}

PyObject* scatter_opposite_spin_rdm2(PyObject*, PyObject* args) {
    PyObject* dm2_obj = nullptr;
    PyObject* pa_obj = nullptr;
    PyObject* qa_obj = nullptr;
    PyObject* bra_a_obj = nullptr;
    PyObject* ket_a_obj = nullptr;
    PyObject* phase_a_obj = nullptr;
    PyObject* rb_obj = nullptr;
    PyObject* sb_obj = nullptr;
    PyObject* bra_b_obj = nullptr;
    PyObject* ket_b_obj = nullptr;
    PyObject* phase_b_obj = nullptr;
    PyObject* cbra_obj = nullptr;
    PyObject* cket_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOO",
            &dm2_obj,
            &pa_obj,
            &qa_obj,
            &bra_a_obj,
            &ket_a_obj,
            &phase_a_obj,
            &rb_obj,
            &sb_obj,
            &bra_b_obj,
            &ket_b_obj,
            &phase_b_obj,
            &cbra_obj,
            &cket_obj
        )) {
        return nullptr;
    }

    ArrayRef dm2(dm2_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    ArrayRef pa(pa_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef qa(qa_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef bra_a(bra_a_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef ket_a(ket_a_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_a(phase_a_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef rb(rb_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef sb(sb_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef bra_b(bra_b_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef ket_b(ket_b_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef phase_b(phase_b_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    ArrayRef cbra(cbra_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef cket(cket_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!dm2 || !pa || !qa || !bra_a || !ket_a || !phase_a ||
        !rb || !sb || !bra_b || !ket_b || !phase_b || !cbra || !cket) {
        PyArray_ResolveWritebackIfCopy(dm2.obj);
        return nullptr;
    }

    if (
        PyArray_NDIM(dm2.obj) != 4 ||
        PyArray_NDIM(cbra.obj) != 2 ||
        PyArray_NDIM(cket.obj) != 2 ||
        !validate_1d_same_length(pa.obj, qa.obj, bra_a.obj, ket_a.obj, phase_a.obj, "alpha") ||
        !validate_1d_same_length(rb.obj, sb.obj, bra_b.obj, ket_b.obj, phase_b.obj, "beta")
    ) {
        PyArray_ResolveWritebackIfCopy(dm2.obj);
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(dm2.obj, 0);
    for (int axis = 1; axis < 4; ++axis) {
        if (PyArray_DIM(dm2.obj, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "dm2 must have shape (n, n, n, n).");
            PyArray_ResolveWritebackIfCopy(dm2.obj);
            return nullptr;
        }
    }
    if (
        PyArray_DIM(cbra.obj, 0) != PyArray_DIM(cket.obj, 0) ||
        PyArray_DIM(cbra.obj, 1) != PyArray_DIM(cket.obj, 1)
    ) {
        PyErr_SetString(PyExc_ValueError, "cbra and cket coefficient matrices must have matching shapes.");
        PyArray_ResolveWritebackIfCopy(dm2.obj);
        return nullptr;
    }

    const npy_intp n_beta = PyArray_DIM(cbra.obj, 1);
    const npy_intp n_link_a = PyArray_DIM(pa.obj, 0);
    const npy_intp n_link_b = PyArray_DIM(rb.obj, 0);

    double* d2 = static_cast<double*>(PyArray_DATA(dm2.obj));
    const double* cb = static_cast<const double*>(PyArray_DATA(cbra.obj));
    const double* ck = static_cast<const double*>(PyArray_DATA(cket.obj));
    const npy_intp* pa_data = static_cast<const npy_intp*>(PyArray_DATA(pa.obj));
    const npy_intp* qa_data = static_cast<const npy_intp*>(PyArray_DATA(qa.obj));
    const npy_intp* bra_a_data = static_cast<const npy_intp*>(PyArray_DATA(bra_a.obj));
    const npy_intp* ket_a_data = static_cast<const npy_intp*>(PyArray_DATA(ket_a.obj));
    const npy_intp* phase_a_data = static_cast<const npy_intp*>(PyArray_DATA(phase_a.obj));
    const npy_intp* rb_data = static_cast<const npy_intp*>(PyArray_DATA(rb.obj));
    const npy_intp* sb_data = static_cast<const npy_intp*>(PyArray_DATA(sb.obj));
    const npy_intp* bra_b_data = static_cast<const npy_intp*>(PyArray_DATA(bra_b.obj));
    const npy_intp* ket_b_data = static_cast<const npy_intp*>(PyArray_DATA(ket_b.obj));
    const npy_intp* phase_b_data = static_cast<const npy_intp*>(PyArray_DATA(phase_b.obj));

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp la = 0; la < n_link_a; ++la) {
        const npy_intp p = pa_data[la];
        const npy_intp q = qa_data[la];
        const npy_intp bra_a_idx = bra_a_data[la];
        const npy_intp ket_a_idx = ket_a_data[la];
        const double phase_alpha = static_cast<double>(phase_a_data[la]);
        for (npy_intp lb = 0; lb < n_link_b; ++lb) {
            const double value = (
                phase_alpha *
                static_cast<double>(phase_b_data[lb]) *
                cb[bra_a_idx * n_beta + bra_b_data[lb]] *
                ck[ket_a_idx * n_beta + ket_b_data[lb]]
            );
            d2[idx4(n, p, q, rb_data[lb], sb_data[lb])] += value;
        }
    }

    Py_END_ALLOW_THREADS

    PyArray_ResolveWritebackIfCopy(dm2.obj);
    Py_RETURN_NONE;
}

struct SpinFreeLink {
    npy_intp bra;
    npy_intp ket;
    double phase;
};

inline bool bit_is_set(std::uint64_t bits, npy_intp idx) {
    return ((bits >> static_cast<unsigned>(idx)) & 1ULL) != 0ULL;
}

inline int popcount_u64(std::uint64_t bits) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(bits));
#else
    return __builtin_popcountll(bits);
#endif
}

inline int fermion_phase(std::uint64_t bits, npy_intp idx) {
    const std::uint64_t mask = (idx <= 0) ? 0ULL : ((1ULL << static_cast<unsigned>(idx)) - 1ULL);
    return (popcount_u64(bits & mask) & 1) ? -1 : 1;
}

bool annihilate_bit(std::uint64_t bits, npy_intp idx, std::uint64_t& out_bits, int& phase) {
    if (!bit_is_set(bits, idx)) {
        phase = 0;
        return false;
    }
    phase = fermion_phase(bits, idx);
    out_bits = bits ^ (1ULL << static_cast<unsigned>(idx));
    return true;
}

bool create_bit(std::uint64_t bits, npy_intp idx, std::uint64_t& out_bits, int& phase) {
    if (bit_is_set(bits, idx)) {
        phase = 0;
        return false;
    }
    phase = fermion_phase(bits, idx);
    out_bits = bits | (1ULL << static_cast<unsigned>(idx));
    return true;
}

inline npy_intp single_bit_index(std::uint64_t bits) {
#ifdef _MSC_VER
    unsigned long idx = 0;
    _BitScanForward64(&idx, bits);
    return static_cast<npy_intp>(idx);
#else
    return static_cast<npy_intp>(__builtin_ctzll(bits));
#endif
}

inline double spin_orbital_energy(const double* mo_energy, npy_intp nmo, npy_intp spin_orbital) {
    return mo_energy[spin_orbital % nmo];
}

int caspt2_external_class_id(
    std::uint64_t bits,
    npy_intp ncore,
    npy_intp ncas,
    npy_intp nmo
) {
    const npy_intp nocc = ncore + ncas;
    int core_holes = 0;
    int virt_particles = 0;
    for (npy_intp spin = 0; spin < 2; ++spin) {
        const npy_intp offset = spin * nmo;
        for (npy_intp i = 0; i < ncore; ++i) {
            if (!bit_is_set(bits, offset + i)) {
                ++core_holes;
            }
        }
        for (npy_intp r = nocc; r < nmo; ++r) {
            if (bit_is_set(bits, offset + r)) {
                ++virt_particles;
            }
        }
    }

    if (core_holes == 2 && virt_particles == 2) {
        return 0;  // Sijrs
    }
    if (core_holes == 2 && virt_particles == 1) {
        return 1;  // Sijr
    }
    if (core_holes == 1 && virt_particles == 2) {
        return 2;  // Srsi
    }
    if (core_holes == 2 && virt_particles == 0) {
        return 3;  // Sij
    }
    if (core_holes == 0 && virt_particles == 2) {
        return 4;  // Srs
    }
    if (core_holes == 1 && virt_particles == 1) {
        return 5;  // Sir
    }
    if (core_holes == 1 && virt_particles == 0) {
        return 6;  // Si
    }
    if (core_holes == 0 && virt_particles == 1) {
        return 7;  // Sr
    }
    return -1;
}

PyObject* caspt2_external_space(PyObject*, PyObject* args) {
    PyObject* ref_obj = nullptr;
    Py_ssize_t ncore_arg = 0;
    Py_ssize_t ncas_arg = 0;
    Py_ssize_t nmo_arg = 0;
    if (!PyArg_ParseTuple(args, "Onnn", &ref_obj, &ncore_arg, &ncas_arg, &nmo_arg)) {
        return nullptr;
    }

    const npy_intp ncore = static_cast<npy_intp>(ncore_arg);
    const npy_intp ncas = static_cast<npy_intp>(ncas_arg);
    const npy_intp nmo = static_cast<npy_intp>(nmo_arg);
    if (ncore < 0 || ncas < 0 || nmo < 0 || ncore + ncas > nmo) {
        PyErr_SetString(PyExc_ValueError, "caspt2_external_space received inconsistent ncore/ncas/nmo.");
        return nullptr;
    }
    const npy_intp nspinorb = 2 * nmo;
    if (nspinorb >= 63) {
        PyErr_SetString(PyExc_NotImplementedError, "native CASPT2 external-space builder requires 2*nmo < 63 for bit encoding.");
        return nullptr;
    }

    ArrayRef ref_bits(ref_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!ref_bits) {
        return nullptr;
    }
    if (PyArray_NDIM(ref_bits.obj) != 1) {
        PyErr_SetString(PyExc_ValueError, "caspt2_external_space expects a one-dimensional uint64 reference determinant array.");
        return nullptr;
    }

    const npy_intp nref = PyArray_DIM(ref_bits.obj, 0);
    const auto* ref_data = static_cast<const std::uint64_t*>(PyArray_DATA(ref_bits.obj));
    std::unordered_map<std::uint64_t, std::int8_t> cas_set;
    cas_set.reserve(static_cast<std::size_t>(nref) * 2 + 1);
    for (npy_intp i = 0; i < nref; ++i) {
        cas_set.emplace(ref_data[i], static_cast<std::int8_t>(0));
    }

    std::unordered_map<std::uint64_t, std::int8_t> external;
    external.reserve(static_cast<std::size_t>(nref) * static_cast<std::size_t>(std::max<npy_intp>(nspinorb, 1)));
    std::vector<npy_intp> occ;
    std::vector<npy_intp> unocc;
    occ.reserve(static_cast<std::size_t>(nspinorb));
    unocc.reserve(static_cast<std::size_t>(nspinorb));

    for (npy_intp ref = 0; ref < nref; ++ref) {
        const std::uint64_t bits = ref_data[ref];
        occ.clear();
        unocc.clear();
        for (npy_intp idx = 0; idx < nspinorb; ++idx) {
            if (bit_is_set(bits, idx)) {
                occ.push_back(idx);
            } else {
                unocc.push_back(idx);
            }
        }

        for (const npy_intp q : occ) {
            const std::uint64_t cleared = bits ^ (1ULL << static_cast<unsigned>(q));
            for (const npy_intp p : unocc) {
                if (p / nmo != q / nmo) {
                    continue;
                }
                const std::uint64_t det = cleared | (1ULL << static_cast<unsigned>(p));
                if (cas_set.find(det) != cas_set.end()) {
                    continue;
                }
                auto it = external.find(det);
                if (it == external.end() || it->second > 1) {
                    external[det] = 1;
                }
            }
        }

        for (std::size_t iq = 0; iq < occ.size(); ++iq) {
            for (std::size_t is = iq + 1; is < occ.size(); ++is) {
                const npy_intp q = occ[iq];
                const npy_intp s = occ[is];
                const std::uint64_t cleared =
                    bits ^
                    (1ULL << static_cast<unsigned>(q)) ^
                    (1ULL << static_cast<unsigned>(s));
                for (std::size_t ip = 0; ip < unocc.size(); ++ip) {
                    for (std::size_t ir = ip + 1; ir < unocc.size(); ++ir) {
                        const npy_intp p = unocc[ip];
                        const npy_intp r = unocc[ir];
                        const int hole_alpha = (q < nmo ? 1 : 0) + (s < nmo ? 1 : 0);
                        const int particle_alpha = (p < nmo ? 1 : 0) + (r < nmo ? 1 : 0);
                        if (hole_alpha != particle_alpha) {
                            continue;
                        }
                        const std::uint64_t det =
                            cleared |
                            (1ULL << static_cast<unsigned>(p)) |
                            (1ULL << static_cast<unsigned>(r));
                        if (cas_set.find(det) != cas_set.end()) {
                            continue;
                        }
                        auto it = external.find(det);
                        if (it == external.end()) {
                            external[det] = 2;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::uint64_t> keys;
    keys.reserve(external.size());
    for (const auto& item : external) {
        keys.push_back(item.first);
    }
    std::sort(keys.begin(), keys.end());

    const npy_intp next = static_cast<npy_intp>(keys.size());
    npy_intp dims[1] = {next};
    PyObject* det_obj = PyArray_SimpleNew(1, dims, NPY_UINT64);
    PyObject* rank_obj = PyArray_SimpleNew(1, dims, NPY_INT8);
    PyObject* class_obj = PyArray_SimpleNew(1, dims, NPY_INT8);
    if (det_obj == nullptr || rank_obj == nullptr || class_obj == nullptr) {
        Py_XDECREF(det_obj);
        Py_XDECREF(rank_obj);
        Py_XDECREF(class_obj);
        return nullptr;
    }

    auto* det_data = static_cast<std::uint64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(det_obj)));
    auto* rank_data = static_cast<std::int8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rank_obj)));
    auto* class_data = static_cast<std::int8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(class_obj)));
    for (npy_intp idx = 0; idx < next; ++idx) {
        const std::uint64_t det = keys[static_cast<std::size_t>(idx)];
        det_data[idx] = det;
        rank_data[idx] = external[det];
        class_data[idx] = static_cast<std::int8_t>(caspt2_external_class_id(det, ncore, ncas, nmo));
    }

    PyObject* result = PyTuple_New(3);
    if (result == nullptr) {
        Py_DECREF(det_obj);
        Py_DECREF(rank_obj);
        Py_DECREF(class_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, det_obj);
    PyTuple_SET_ITEM(result, 1, rank_obj);
    PyTuple_SET_ITEM(result, 2, class_obj);
    return result;
}

double caspt2_hamiltonian_element_bits(
    std::uint64_t bra_bits,
    std::uint64_t ket_bits,
    const double* h1,
    const double* eri,
    npy_intp nmo
) {
    const std::uint64_t holes = ket_bits & ~bra_bits;
    const std::uint64_t particles = bra_bits & ~ket_bits;
    const int rank = popcount_u64(holes);
    if (rank != popcount_u64(particles) || rank > 2) {
        return 0.0;
    }

    double value = 0.0;

    if (rank == 0) {
        std::uint64_t occ_i = ket_bits;
        while (occ_i != 0ULL) {
            const npy_intp i = single_bit_index(occ_i);
            occ_i &= occ_i - 1ULL;
            value += h1[idx2(nmo, i % nmo, i % nmo)];
        }
        occ_i = ket_bits;
        while (occ_i != 0ULL) {
            const npy_intp i = single_bit_index(occ_i);
            occ_i &= occ_i - 1ULL;
            std::uint64_t occ_j = ket_bits;
            while (occ_j != 0ULL) {
                const npy_intp j = single_bit_index(occ_j);
                occ_j &= occ_j - 1ULL;
                double term = eri[idx4(nmo, i % nmo, i % nmo, j % nmo, j % nmo)];
                if (i / nmo == j / nmo) {
                    term -= eri[idx4(nmo, i % nmo, j % nmo, j % nmo, i % nmo)];
                }
                value += 0.5 * term;
            }
        }
        return value;
    }

    if (rank == 1) {
        const npy_intp q = single_bit_index(holes);
        const npy_intp p = single_bit_index(particles);
        if (p / nmo != q / nmo) {
            return 0.0;
        }
        std::uint64_t bits1 = 0ULL;
        std::uint64_t bits2 = 0ULL;
        int phase1 = 0;
        int phase2 = 0;
        if (!annihilate_bit(ket_bits, q, bits1, phase1)) {
            return 0.0;
        }
        if (!create_bit(bits1, p, bits2, phase2) || bits2 != bra_bits) {
            return 0.0;
        }

        value = h1[idx2(nmo, p % nmo, q % nmo)];
        std::uint64_t common = ket_bits & bra_bits;
        while (common != 0ULL) {
            const npy_intp j = single_bit_index(common);
            common &= common - 1ULL;
            value += eri[idx4(nmo, p % nmo, q % nmo, j % nmo, j % nmo)];
            if (p / nmo == j / nmo) {
                value -= eri[idx4(nmo, p % nmo, j % nmo, j % nmo, q % nmo)];
            }
        }
        return static_cast<double>(phase1 * phase2) * value;
    }

    std::uint64_t q_iter = holes;
    while (q_iter != 0ULL) {
        const npy_intp q = single_bit_index(q_iter);
        q_iter &= q_iter - 1ULL;

        std::uint64_t bits1 = 0ULL;
        int phase1 = 0;
        if (!annihilate_bit(ket_bits, q, bits1, phase1)) {
            continue;
        }

        std::uint64_t s_iter = holes & ~(1ULL << static_cast<unsigned>(q));
        while (s_iter != 0ULL) {
            const npy_intp s = single_bit_index(s_iter);
            s_iter &= s_iter - 1ULL;

            std::uint64_t bits2 = 0ULL;
            int phase2 = 0;
            if (!annihilate_bit(bits1, s, bits2, phase2)) {
                continue;
            }

            std::uint64_t p_iter = particles;
            while (p_iter != 0ULL) {
                const npy_intp p = single_bit_index(p_iter);
                p_iter &= p_iter - 1ULL;

                std::uint64_t r_iter = particles & ~(1ULL << static_cast<unsigned>(p));
                while (r_iter != 0ULL) {
                    const npy_intp r = single_bit_index(r_iter);
                    r_iter &= r_iter - 1ULL;

                    if (p / nmo != q / nmo || r / nmo != s / nmo) {
                        continue;
                    }
                    std::uint64_t bits3 = 0ULL;
                    std::uint64_t bits4 = 0ULL;
                    int phase3 = 0;
                    int phase4 = 0;
                    if (!create_bit(bits2, r, bits3, phase3)) {
                        continue;
                    }
                    if (!create_bit(bits3, p, bits4, phase4)) {
                        continue;
                    }
                    if (bits4 != bra_bits) {
                        continue;
                    }
                    value += (
                        0.5 *
                        static_cast<double>(phase1 * phase2 * phase3 * phase4) *
                        eri[idx4(nmo, p % nmo, q % nmo, r % nmo, s % nmo)]
                    );
                }
            }
        }
    }
    return value;
}

double caspt2_fock_denominator_bits(
    std::uint64_t det_bits,
    const double* occ_average,
    const double* mo_energy,
    npy_intp nmo
) {
    double value = 0.0;
    const npy_intp nspinorb = 2 * nmo;
    for (npy_intp idx = 0; idx < nspinorb; ++idx) {
        const double occ = bit_is_set(det_bits, idx) ? 1.0 : 0.0;
        value += (occ_average[idx] - occ) * spin_orbital_energy(mo_energy, nmo, idx);
    }
    return value;
}

bool validate_caspt2_external_inputs(
    PyArrayObject* external,
    PyArrayObject* ref_bits,
    PyArrayObject* ci,
    PyArrayObject* h1,
    PyArrayObject* eri,
    PyArrayObject* mo_energy,
    PyArrayObject* occ_average,
    int zeroth_order
) {
    if (
        PyArray_NDIM(external) != 1 ||
        PyArray_NDIM(ref_bits) != 1 ||
        PyArray_NDIM(ci) != 1 ||
        PyArray_NDIM(h1) != 2 ||
        PyArray_NDIM(eri) != 4 ||
        PyArray_NDIM(mo_energy) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_external_kernel expects external/ref/ci/mo_energy as 1D, h1 as 2D, and eri as 4D arrays.");
        return false;
    }
    const npy_intp nref = PyArray_DIM(ref_bits, 0);
    const npy_intp nmo = PyArray_DIM(h1, 0);
    if (
        PyArray_DIM(ci, 0) != nref ||
        PyArray_DIM(h1, 1) != nmo ||
        PyArray_DIM(mo_energy, 0) != nmo
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_external_kernel CI/reference or h1/mo_energy dimensions do not match.");
        return false;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (PyArray_DIM(eri, axis) != nmo) {
            PyErr_SetString(PyExc_ValueError, "caspt2_external_kernel eri must have shape (nmo, nmo, nmo, nmo).");
            return false;
        }
    }
    if (2 * nmo >= 63) {
        PyErr_SetString(PyExc_NotImplementedError, "native CASPT2 external kernel requires 2*nmo < 63 for bit encoding.");
        return false;
    }
    if (zeroth_order == 0) {
        if (occ_average == nullptr || PyArray_NDIM(occ_average) != 1 || PyArray_DIM(occ_average, 0) != 2 * nmo) {
            PyErr_SetString(PyExc_ValueError, "fock zeroth-order CASPT2 requires occ_average with shape (2*nmo,).");
            return false;
        }
    } else if (zeroth_order != 1) {
        PyErr_SetString(PyExc_ValueError, "caspt2_external_kernel zeroth_order must be 0 (fock) or 1 (EN).");
        return false;
    }
    return true;
}

PyObject* caspt2_external_kernel(PyObject*, PyObject* args) {
    PyObject* external_obj = nullptr;
    PyObject* ref_obj = nullptr;
    PyObject* ci_obj = nullptr;
    PyObject* h1_obj = nullptr;
    PyObject* eri_obj = nullptr;
    PyObject* mo_energy_obj = nullptr;
    PyObject* occ_average_obj = nullptr;
    double e_ref = 0.0;
    double e_nuc = 0.0;
    int zeroth_order = 0;
    double real_shift = 0.0;
    double imaginary_shift = 0.0;
    double denominator_tol = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOddiddd",
            &external_obj,
            &ref_obj,
            &ci_obj,
            &h1_obj,
            &eri_obj,
            &mo_energy_obj,
            &occ_average_obj,
            &e_ref,
            &e_nuc,
            &zeroth_order,
            &real_shift,
            &imaginary_shift,
            &denominator_tol
        )) {
        return nullptr;
    }

    ArrayRef external(external_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef ref_bits(ref_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h1(h1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_energy(mo_energy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef occ_average;
    PyArrayObject* occ_average_arr = nullptr;
    if (zeroth_order == 0) {
        occ_average.reset(occ_average_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!occ_average) {
            return nullptr;
        }
        occ_average_arr = occ_average.obj;
    }
    if (!external || !ref_bits || !ci || !h1 || !eri || !mo_energy) {
        return nullptr;
    }
    if (!validate_caspt2_external_inputs(
            external.obj,
            ref_bits.obj,
            ci.obj,
            h1.obj,
            eri.obj,
            mo_energy.obj,
            occ_average_arr,
            zeroth_order
        )) {
        return nullptr;
    }

    const npy_intp next = PyArray_DIM(external.obj, 0);
    const npy_intp nref = PyArray_DIM(ref_bits.obj, 0);
    const npy_intp nmo = PyArray_DIM(h1.obj, 0);
    npy_intp dims[1] = {next};
    PyObject* couplings_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyObject* denominators_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyObject* energies_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyObject* amplitudes_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (
        couplings_obj == nullptr ||
        denominators_obj == nullptr ||
        energies_obj == nullptr ||
        amplitudes_obj == nullptr
    ) {
        Py_XDECREF(couplings_obj);
        Py_XDECREF(denominators_obj);
        Py_XDECREF(energies_obj);
        Py_XDECREF(amplitudes_obj);
        return nullptr;
    }

    const auto* external_data = static_cast<const std::uint64_t*>(PyArray_DATA(external.obj));
    const auto* ref_data = static_cast<const std::uint64_t*>(PyArray_DATA(ref_bits.obj));
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    const auto* h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
    const auto* eri_data = static_cast<const double*>(PyArray_DATA(eri.obj));
    const auto* mo_energy_data = static_cast<const double*>(PyArray_DATA(mo_energy.obj));
    const auto* occ_average_data = zeroth_order == 0
        ? static_cast<const double*>(PyArray_DATA(occ_average.obj))
        : nullptr;
    auto* couplings = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(couplings_obj)));
    auto* denominators = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(denominators_obj)));
    auto* energies = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(energies_obj)));
    auto* amplitudes = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(amplitudes_obj)));

    bool near_zero = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp mu = 0; mu < next; ++mu) {
        const std::uint64_t det = external_data[mu];
        double coupling = 0.0;
        for (npy_intp ket = 0; ket < nref; ++ket) {
            const double coeff = ci_data[ket];
            if (coeff == 0.0) {
                continue;
            }
            coupling += coeff * caspt2_hamiltonian_element_bits(
                det,
                ref_data[ket],
                h1_data,
                eri_data,
                nmo
            );
        }
        couplings[mu] = coupling;

        double denominator = 0.0;
        if (zeroth_order == 0) {
            denominator = caspt2_fock_denominator_bits(
                det,
                occ_average_data,
                mo_energy_data,
                nmo
            );
        } else {
            const double ext_diag =
                e_nuc +
                caspt2_hamiltonian_element_bits(det, det, h1_data, eri_data, nmo);
            denominator = e_ref - ext_diag;
        }
        denominators[mu] = denominator;

        if (coupling == 0.0) {
            energies[mu] = 0.0;
            amplitudes[mu] = 0.0;
            continue;
        }
        const double shifted = denominator - real_shift;
        double weight = 0.0;
        if (imaginary_shift != 0.0) {
            weight = shifted / (shifted * shifted + imaginary_shift * imaginary_shift);
        } else {
            if (std::abs(shifted) < denominator_tol) {
                near_zero = true;
                weight = 0.0;
            } else {
                weight = 1.0 / shifted;
            }
        }
        energies[mu] = coupling * coupling * weight;
        amplitudes[mu] = coupling * weight;
    }
    Py_END_ALLOW_THREADS

    if (near_zero) {
        Py_DECREF(couplings_obj);
        Py_DECREF(denominators_obj);
        Py_DECREF(energies_obj);
        Py_DECREF(amplitudes_obj);
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "Encountered a near-zero CASPT2 denominator. Use real_shift or imaginary_shift to regularize intruder states."
        );
        return nullptr;
    }

    PyObject* result = PyTuple_New(4);
    if (result == nullptr) {
        Py_DECREF(couplings_obj);
        Py_DECREF(denominators_obj);
        Py_DECREF(energies_obj);
        Py_DECREF(amplitudes_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, couplings_obj);
    PyTuple_SET_ITEM(result, 1, denominators_obj);
    PyTuple_SET_ITEM(result, 2, energies_obj);
    PyTuple_SET_ITEM(result, 3, amplitudes_obj);
    return result;
}

bool validate_caspt2_strong_contract_inputs(
    PyArrayObject* couplings,
    PyArrayObject* denominators,
    PyArrayObject* classes
) {
    if (
        PyArray_NDIM(couplings) != 1 ||
        PyArray_NDIM(denominators) != 1 ||
        PyArray_NDIM(classes) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_strong_contract expects one-dimensional coupling, denominator, and class arrays.");
        return false;
    }
    if (
        PyArray_DIM(denominators, 0) != PyArray_DIM(couplings, 0) ||
        PyArray_DIM(classes, 0) != PyArray_DIM(couplings, 0)
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_strong_contract input arrays must have matching lengths.");
        return false;
    }
    return true;
}

PyObject* caspt2_strong_contract(PyObject*, PyObject* args) {
    PyObject* couplings_obj = nullptr;
    PyObject* denominators_obj = nullptr;
    PyObject* classes_obj = nullptr;
    double real_shift = 0.0;
    double imaginary_shift = 0.0;
    double denominator_tol = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOddd",
            &couplings_obj,
            &denominators_obj,
            &classes_obj,
            &real_shift,
            &imaginary_shift,
            &denominator_tol
        )) {
        return nullptr;
    }

    ArrayRef couplings(couplings_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef denominators(denominators_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef classes(classes_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!couplings || !denominators || !classes) {
        return nullptr;
    }
    if (!validate_caspt2_strong_contract_inputs(couplings.obj, denominators.obj, classes.obj)) {
        return nullptr;
    }

    const npy_intp next = PyArray_DIM(couplings.obj, 0);
    npy_intp ext_dims[1] = {next};
    npy_intp class_dims[1] = {8};
    PyObject* energies_obj = PyArray_ZEROS(1, ext_dims, NPY_DOUBLE, 0);
    PyObject* amplitudes_obj = PyArray_ZEROS(1, ext_dims, NPY_DOUBLE, 0);
    PyObject* counts_obj = PyArray_ZEROS(1, class_dims, NPY_INTP, 0);
    PyObject* norms_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    PyObject* denominator_moments_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    PyObject* class_denominators_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    PyObject* class_amplitudes_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    PyObject* component_energies_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    if (
        energies_obj == nullptr ||
        amplitudes_obj == nullptr ||
        counts_obj == nullptr ||
        norms_obj == nullptr ||
        denominator_moments_obj == nullptr ||
        class_denominators_obj == nullptr ||
        class_amplitudes_obj == nullptr ||
        component_energies_obj == nullptr
    ) {
        Py_XDECREF(energies_obj);
        Py_XDECREF(amplitudes_obj);
        Py_XDECREF(counts_obj);
        Py_XDECREF(norms_obj);
        Py_XDECREF(denominator_moments_obj);
        Py_XDECREF(class_denominators_obj);
        Py_XDECREF(class_amplitudes_obj);
        Py_XDECREF(component_energies_obj);
        return nullptr;
    }

    const auto* coupling_data = static_cast<const double*>(PyArray_DATA(couplings.obj));
    const auto* denominator_data = static_cast<const double*>(PyArray_DATA(denominators.obj));
    const auto* class_data = static_cast<const std::int8_t*>(PyArray_DATA(classes.obj));
    auto* energies = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(energies_obj)));
    auto* amplitudes = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(amplitudes_obj)));
    auto* counts = static_cast<npy_intp*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_obj)));
    auto* norms = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(norms_obj)));
    auto* denominator_moments = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(denominator_moments_obj)));
    auto* class_denominators = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(class_denominators_obj)));
    auto* class_amplitudes = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(class_amplitudes_obj)));
    auto* component_energies = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(component_energies_obj)));

    bool near_zero = false;
    bool unsupported_class = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp idx = 0; idx < next; ++idx) {
        const int class_id = static_cast<int>(class_data[idx]);
        const double coupling = coupling_data[idx];
        if (class_id < 0 || class_id >= 8) {
            if (std::abs(coupling) > denominator_tol) {
                unsupported_class = true;
            }
            continue;
        }
        const double coupling2 = coupling * coupling;
        counts[class_id] += 1;
        norms[class_id] += coupling2;
        denominator_moments[class_id] += coupling2 * denominator_data[idx];
    }

    for (int class_id = 0; class_id < 8; ++class_id) {
        if (norms[class_id] <= 0.0) {
            denominator_moments[class_id] = 0.0;
            class_denominators[class_id] = 0.0;
            class_amplitudes[class_id] = 0.0;
            component_energies[class_id] = 0.0;
            continue;
        }
        class_denominators[class_id] = denominator_moments[class_id] / norms[class_id];
        const double shifted = class_denominators[class_id] - real_shift;
        double weight = 0.0;
        if (imaginary_shift != 0.0) {
            weight = shifted / (shifted * shifted + imaginary_shift * imaginary_shift);
        } else if (std::abs(shifted) < denominator_tol) {
            near_zero = true;
        } else {
            weight = 1.0 / shifted;
        }
        class_amplitudes[class_id] = weight;
        component_energies[class_id] = norms[class_id] * weight;
    }

    if (!near_zero && !unsupported_class) {
        for (npy_intp idx = 0; idx < next; ++idx) {
            const int class_id = static_cast<int>(class_data[idx]);
            if (class_id < 0 || class_id >= 8 || norms[class_id] <= 0.0) {
                continue;
            }
            const double weight = class_amplitudes[class_id];
            const double coupling = coupling_data[idx];
            energies[idx] = coupling * coupling * weight;
            amplitudes[idx] = coupling * weight;
        }
    }
    Py_END_ALLOW_THREADS

    if (unsupported_class) {
        Py_DECREF(energies_obj);
        Py_DECREF(amplitudes_obj);
        Py_DECREF(counts_obj);
        Py_DECREF(norms_obj);
        Py_DECREF(denominator_moments_obj);
        Py_DECREF(class_denominators_obj);
        Py_DECREF(class_amplitudes_obj);
        Py_DECREF(component_energies_obj);
        PyErr_SetString(
            PyExc_NotImplementedError,
            "Encountered CASPT2 external determinants outside the eight standard internally contracted perturber classes."
        );
        return nullptr;
    }
    if (near_zero) {
        Py_DECREF(energies_obj);
        Py_DECREF(amplitudes_obj);
        Py_DECREF(counts_obj);
        Py_DECREF(norms_obj);
        Py_DECREF(denominator_moments_obj);
        Py_DECREF(class_denominators_obj);
        Py_DECREF(class_amplitudes_obj);
        Py_DECREF(component_energies_obj);
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "Encountered a near-zero CASPT2 denominator. Use real_shift or imaginary_shift to regularize intruder states."
        );
        return nullptr;
    }

    PyObject* result = PyTuple_New(8);
    if (result == nullptr) {
        Py_DECREF(energies_obj);
        Py_DECREF(amplitudes_obj);
        Py_DECREF(counts_obj);
        Py_DECREF(norms_obj);
        Py_DECREF(denominator_moments_obj);
        Py_DECREF(class_denominators_obj);
        Py_DECREF(class_amplitudes_obj);
        Py_DECREF(component_energies_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, energies_obj);
    PyTuple_SET_ITEM(result, 1, amplitudes_obj);
    PyTuple_SET_ITEM(result, 2, counts_obj);
    PyTuple_SET_ITEM(result, 3, norms_obj);
    PyTuple_SET_ITEM(result, 4, denominator_moments_obj);
    PyTuple_SET_ITEM(result, 5, class_denominators_obj);
    PyTuple_SET_ITEM(result, 6, class_amplitudes_obj);
    PyTuple_SET_ITEM(result, 7, component_energies_obj);
    return result;
}

bool validate_caspt2_en_coupled_contract_inputs(
    PyArrayObject* external,
    PyArrayObject* couplings,
    PyArrayObject* classes,
    PyArrayObject* h1,
    PyArrayObject* eri
) {
    if (
        PyArray_NDIM(external) != 1 ||
        PyArray_NDIM(couplings) != 1 ||
        PyArray_NDIM(classes) != 1 ||
        PyArray_NDIM(h1) != 2 ||
        PyArray_NDIM(eri) != 4
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_en_coupled_contract expects external/couplings/classes as 1D, h1 as 2D, and eri as 4D arrays.");
        return false;
    }
    if (
        PyArray_DIM(couplings, 0) != PyArray_DIM(external, 0) ||
        PyArray_DIM(classes, 0) != PyArray_DIM(external, 0)
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_en_coupled_contract external, coupling, and class arrays must have matching lengths.");
        return false;
    }
    const npy_intp nmo = PyArray_DIM(h1, 0);
    if (PyArray_DIM(h1, 1) != nmo) {
        PyErr_SetString(PyExc_ValueError, "caspt2_en_coupled_contract h1 must be square.");
        return false;
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (PyArray_DIM(eri, axis) != nmo) {
            PyErr_SetString(PyExc_ValueError, "caspt2_en_coupled_contract eri must have shape (nmo, nmo, nmo, nmo).");
            return false;
        }
    }
    if (2 * nmo >= 63) {
        PyErr_SetString(PyExc_NotImplementedError, "native CASPT2 coupled EN contraction requires 2*nmo < 63 for bit encoding.");
        return false;
    }
    return true;
}

PyObject* caspt2_en_coupled_contract(PyObject*, PyObject* args) {
    PyObject* external_obj = nullptr;
    PyObject* couplings_obj = nullptr;
    PyObject* classes_obj = nullptr;
    PyObject* h1_obj = nullptr;
    PyObject* eri_obj = nullptr;
    double e_ref = 0.0;
    double e_nuc = 0.0;
    double denominator_tol = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOddd",
            &external_obj,
            &couplings_obj,
            &classes_obj,
            &h1_obj,
            &eri_obj,
            &e_ref,
            &e_nuc,
            &denominator_tol
        )) {
        return nullptr;
    }

    ArrayRef external(external_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef couplings(couplings_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef classes(classes_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    ArrayRef h1(h1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef eri(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!external || !couplings || !classes || !h1 || !eri) {
        return nullptr;
    }
    if (!validate_caspt2_en_coupled_contract_inputs(
            external.obj,
            couplings.obj,
            classes.obj,
            h1.obj,
            eri.obj
        )) {
        return nullptr;
    }

    npy_intp matrix_dims[2] = {8, 8};
    npy_intp class_dims[1] = {8};
    PyObject* metric_obj = PyArray_ZEROS(2, matrix_dims, NPY_DOUBLE, 0);
    PyObject* denominator_matrix_obj = PyArray_ZEROS(2, matrix_dims, NPY_DOUBLE, 0);
    PyObject* rhs_obj = PyArray_ZEROS(1, class_dims, NPY_DOUBLE, 0);
    PyObject* counts_obj = PyArray_ZEROS(1, class_dims, NPY_INTP, 0);
    if (
        metric_obj == nullptr ||
        denominator_matrix_obj == nullptr ||
        rhs_obj == nullptr ||
        counts_obj == nullptr
    ) {
        Py_XDECREF(metric_obj);
        Py_XDECREF(denominator_matrix_obj);
        Py_XDECREF(rhs_obj);
        Py_XDECREF(counts_obj);
        return nullptr;
    }

    const npy_intp next = PyArray_DIM(external.obj, 0);
    const npy_intp nmo = PyArray_DIM(h1.obj, 0);
    const auto* external_data = static_cast<const std::uint64_t*>(PyArray_DATA(external.obj));
    const auto* coupling_data = static_cast<const double*>(PyArray_DATA(couplings.obj));
    const auto* class_data = static_cast<const std::int8_t*>(PyArray_DATA(classes.obj));
    const auto* h1_data = static_cast<const double*>(PyArray_DATA(h1.obj));
    const auto* eri_data = static_cast<const double*>(PyArray_DATA(eri.obj));
    auto* metric = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(metric_obj)));
    auto* denominator_matrix = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(denominator_matrix_obj)));
    auto* rhs = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rhs_obj)));
    auto* counts = static_cast<npy_intp*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_obj)));

    bool unsupported_class = false;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp mu = 0; mu < next; ++mu) {
        const int class_mu = static_cast<int>(class_data[mu]);
        const double coupling_mu = coupling_data[mu];
        if (class_mu < 0 || class_mu >= 8) {
            if (std::abs(coupling_mu) > denominator_tol) {
                unsupported_class = true;
            }
            continue;
        }
        const double coupling2 = coupling_mu * coupling_mu;
        counts[class_mu] += 1;
        metric[class_mu * 8 + class_mu] += coupling2;
        rhs[class_mu] += coupling2;
    }

    if (!unsupported_class) {
        for (npy_intp mu = 0; mu < next; ++mu) {
            const int class_mu = static_cast<int>(class_data[mu]);
            const double coupling_mu = coupling_data[mu];
            if (class_mu < 0 || class_mu >= 8 || coupling_mu == 0.0) {
                continue;
            }
            const std::uint64_t det_mu = external_data[mu];
            for (npy_intp nu = 0; nu < next; ++nu) {
                const int class_nu = static_cast<int>(class_data[nu]);
                const double coupling_nu = coupling_data[nu];
                if (class_nu < 0 || class_nu >= 8 || coupling_nu == 0.0) {
                    continue;
                }
                const double h_mu_nu = caspt2_hamiltonian_element_bits(
                    det_mu,
                    external_data[nu],
                    h1_data,
                    eri_data,
                    nmo
                );
                const double projected = (mu == nu)
                    ? (e_ref - e_nuc - h_mu_nu)
                    : (-h_mu_nu);
                denominator_matrix[class_mu * 8 + class_nu] +=
                    coupling_mu * coupling_nu * projected;
            }
        }

        for (int row = 0; row < 8; ++row) {
            for (int col = row + 1; col < 8; ++col) {
                const double averaged = 0.5 * (
                    denominator_matrix[row * 8 + col] +
                    denominator_matrix[col * 8 + row]
                );
                denominator_matrix[row * 8 + col] = averaged;
                denominator_matrix[col * 8 + row] = averaged;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (unsupported_class) {
        Py_DECREF(metric_obj);
        Py_DECREF(denominator_matrix_obj);
        Py_DECREF(rhs_obj);
        Py_DECREF(counts_obj);
        PyErr_SetString(
            PyExc_NotImplementedError,
            "Encountered CASPT2 external determinants outside the eight standard internally contracted perturber classes."
        );
        return nullptr;
    }

    PyObject* result = PyTuple_New(4);
    if (result == nullptr) {
        Py_DECREF(metric_obj);
        Py_DECREF(denominator_matrix_obj);
        Py_DECREF(rhs_obj);
        Py_DECREF(counts_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, metric_obj);
    PyTuple_SET_ITEM(result, 1, denominator_matrix_obj);
    PyTuple_SET_ITEM(result, 2, rhs_obj);
    PyTuple_SET_ITEM(result, 3, counts_obj);
    return result;
}

bool validate_caspt2_contracted_solve_inputs(
    PyArrayObject* metric,
    PyArrayObject* denominator_matrix,
    PyArrayObject* rhs
) {
    if (
        PyArray_NDIM(metric) != 2 ||
        PyArray_NDIM(denominator_matrix) != 2 ||
        PyArray_NDIM(rhs) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_solve_contracted expects metric(n,n), denominator_matrix(n,n), and rhs(n).");
        return false;
    }
    const npy_intp n = PyArray_DIM(rhs, 0);
    if (
        PyArray_DIM(metric, 0) != n ||
        PyArray_DIM(metric, 1) != n ||
        PyArray_DIM(denominator_matrix, 0) != n ||
        PyArray_DIM(denominator_matrix, 1) != n
    ) {
        PyErr_SetString(PyExc_ValueError, "caspt2_solve_contracted input dimensions are inconsistent.");
        return false;
    }
    return true;
}

bool solve_dense_linear_system(
    std::vector<double>& matrix,
    std::vector<double>& rhs,
    npy_intp n,
    double tolerance,
    npy_intp& failed_pivot
) {
    failed_pivot = -1;
    for (npy_intp col = 0; col < n; ++col) {
        npy_intp pivot = col;
        double pivot_abs = std::abs(matrix[static_cast<std::size_t>(col * n + col)]);
        for (npy_intp row = col + 1; row < n; ++row) {
            const double value_abs = std::abs(matrix[static_cast<std::size_t>(row * n + col)]);
            if (value_abs > pivot_abs) {
                pivot_abs = value_abs;
                pivot = row;
            }
        }
        if (pivot_abs < tolerance) {
            failed_pivot = col;
            return false;
        }
        if (pivot != col) {
            for (npy_intp j = col; j < n; ++j) {
                std::swap(
                    matrix[static_cast<std::size_t>(col * n + j)],
                    matrix[static_cast<std::size_t>(pivot * n + j)]
                );
            }
            std::swap(rhs[static_cast<std::size_t>(col)], rhs[static_cast<std::size_t>(pivot)]);
        }

        const double pivot_value = matrix[static_cast<std::size_t>(col * n + col)];
        for (npy_intp row = col + 1; row < n; ++row) {
            const double factor = matrix[static_cast<std::size_t>(row * n + col)] / pivot_value;
            matrix[static_cast<std::size_t>(row * n + col)] = 0.0;
            for (npy_intp j = col + 1; j < n; ++j) {
                matrix[static_cast<std::size_t>(row * n + j)] -=
                    factor * matrix[static_cast<std::size_t>(col * n + j)];
            }
            rhs[static_cast<std::size_t>(row)] -= factor * rhs[static_cast<std::size_t>(col)];
        }
    }

    for (npy_intp row = n - 1; row >= 0; --row) {
        double value = rhs[static_cast<std::size_t>(row)];
        for (npy_intp col = row + 1; col < n; ++col) {
            value -= matrix[static_cast<std::size_t>(row * n + col)] * rhs[static_cast<std::size_t>(col)];
        }
        const double pivot_value = matrix[static_cast<std::size_t>(row * n + row)];
        if (std::abs(pivot_value) < tolerance) {
            failed_pivot = row;
            return false;
        }
        rhs[static_cast<std::size_t>(row)] = value / pivot_value;
        if (row == 0) {
            break;
        }
    }
    return true;
}

PyObject* caspt2_solve_contracted(PyObject*, PyObject* args) {
    PyObject* metric_obj = nullptr;
    PyObject* denominator_matrix_obj = nullptr;
    PyObject* rhs_obj = nullptr;
    double real_shift = 0.0;
    double tolerance = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "OOOdd",
            &metric_obj,
            &denominator_matrix_obj,
            &rhs_obj,
            &real_shift,
            &tolerance
        )) {
        return nullptr;
    }

    ArrayRef metric(metric_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef denominator_matrix(denominator_matrix_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef rhs_arr(rhs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!metric || !denominator_matrix || !rhs_arr) {
        return nullptr;
    }
    if (!validate_caspt2_contracted_solve_inputs(metric.obj, denominator_matrix.obj, rhs_arr.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(rhs_arr.obj, 0);
    npy_intp dims[1] = {n};
    PyObject* amplitudes_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (amplitudes_obj == nullptr) {
        return nullptr;
    }
    if (n == 0) {
        return amplitudes_obj;
    }

    const auto* metric_data = static_cast<const double*>(PyArray_DATA(metric.obj));
    const auto* denominator_data = static_cast<const double*>(PyArray_DATA(denominator_matrix.obj));
    const auto* rhs_data = static_cast<const double*>(PyArray_DATA(rhs_arr.obj));
    auto* amplitudes_data = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(amplitudes_obj)));

    std::vector<double> matrix(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
    try {
        for (npy_intp i = 0; i < n; ++i) {
            rhs[static_cast<std::size_t>(i)] = rhs_data[i];
            for (npy_intp j = 0; j < n; ++j) {
                matrix[static_cast<std::size_t>(i * n + j)] =
                    denominator_data[i * n + j] - real_shift * metric_data[i * n + j];
            }
        }
    } catch (const std::bad_alloc&) {
        Py_DECREF(amplitudes_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate contracted CASPT2 solve workspace.");
        return nullptr;
    }

    npy_intp failed_pivot = -1;
    bool success = false;
    Py_BEGIN_ALLOW_THREADS
    success = solve_dense_linear_system(matrix, rhs, n, tolerance, failed_pivot);
    Py_END_ALLOW_THREADS
    if (!success) {
        Py_DECREF(amplitudes_obj);
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "Encountered a near-singular contracted CASPT2 linear system."
        );
        return nullptr;
    }

    for (npy_intp i = 0; i < n; ++i) {
        amplitudes_data[i] = rhs[static_cast<std::size_t>(i)];
    }
    return amplitudes_obj;
}

bool validate_nevpt_contract_inputs(PyArrayObject* h2e, PyArrayObject* ci, PyArrayObject* binary) {
    if (PyArray_NDIM(h2e) != 4 || PyArray_NDIM(ci) != 1 || PyArray_NDIM(binary) != 3) {
        PyErr_SetString(PyExc_ValueError, "NEVPT contracted terms expect h2e(n,n,n,n), ci(ndet), binary(ndet,2,n).");
        return false;
    }
    const npy_intp n = PyArray_DIM(h2e, 0);
    for (int axis = 1; axis < 4; ++axis) {
        if (PyArray_DIM(h2e, axis) != n) {
            PyErr_SetString(PyExc_ValueError, "h2e must have shape (n, n, n, n).");
            return false;
        }
    }
    if (PyArray_DIM(binary, 1) != 2 || PyArray_DIM(binary, 2) != n || PyArray_DIM(binary, 0) != PyArray_DIM(ci, 0)) {
        PyErr_SetString(PyExc_ValueError, "ci and binary dimensions are inconsistent.");
        return false;
    }
    if (2 * n >= 63) {
        PyErr_SetString(PyExc_ValueError, "Native contracted NEVPT2 currently requires 2*ncas < 63 for bit encoding.");
        return false;
    }
    return true;
}

bool validate_nevpt_ci_binary_inputs(PyArrayObject* ci, PyArrayObject* binary) {
    if (PyArray_NDIM(ci) != 1 || PyArray_NDIM(binary) != 3) {
        PyErr_SetString(PyExc_ValueError, "NEVPT RDM builder expects ci(ndet), binary(ndet,2,n).");
        return false;
    }
    const npy_intp n = PyArray_DIM(binary, 2);
    if (PyArray_DIM(binary, 1) != 2 || PyArray_DIM(binary, 0) != PyArray_DIM(ci, 0)) {
        PyErr_SetString(PyExc_ValueError, "ci and binary dimensions are inconsistent.");
        return false;
    }
    if (2 * n >= 63) {
        PyErr_SetString(PyExc_ValueError, "Native NEVPT2 RDM builder currently requires 2*ncas < 63 for bit encoding.");
        return false;
    }
    return true;
}

bool build_spin_free_links(
    PyArrayObject* binary,
    std::vector<std::vector<SpinFreeLink>>& links,
    std::vector<std::uint64_t>& det_bits
) {
    const npy_intp ndet = PyArray_DIM(binary, 0);
    const npy_intp n = PyArray_DIM(binary, 2);
    const auto* occ = static_cast<const npy_int8*>(PyArray_DATA(binary));

    det_bits.assign(static_cast<std::size_t>(ndet), 0ULL);
    std::unordered_map<std::uint64_t, npy_intp> det_index;
    det_index.reserve(static_cast<std::size_t>(ndet) * 2);
    for (npy_intp det = 0; det < ndet; ++det) {
        std::uint64_t bits = 0ULL;
        for (npy_intp spin = 0; spin < 2; ++spin) {
            const npy_intp offset = spin * n;
            for (npy_intp orb = 0; orb < n; ++orb) {
                if (occ[idx_binary(n, det, spin, orb)] != 0) {
                    bits |= (1ULL << static_cast<unsigned>(offset + orb));
                }
            }
        }
        det_bits[static_cast<std::size_t>(det)] = bits;
        det_index.emplace(bits, det);
    }

    links.assign(static_cast<std::size_t>(n * n), {});
    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            auto& pair_links = links[static_cast<std::size_t>(p * n + q)];
            pair_links.reserve(static_cast<std::size_t>(2 * ndet));
            for (npy_intp ket = 0; ket < ndet; ++ket) {
                const std::uint64_t bits0 = det_bits[static_cast<std::size_t>(ket)];
                for (npy_intp spin = 0; spin < 2; ++spin) {
                    const npy_intp offset = spin * n;
                    std::uint64_t bits1 = 0ULL;
                    std::uint64_t bits2 = 0ULL;
                    int phase1 = 0;
                    int phase2 = 0;
                    if (!annihilate_bit(bits0, offset + q, bits1, phase1)) {
                        continue;
                    }
                    if (!create_bit(bits1, offset + p, bits2, phase2)) {
                        continue;
                    }
                    auto found = det_index.find(bits2);
                    if (found != det_index.end()) {
                        pair_links.push_back({found->second, ket, static_cast<double>(phase1 * phase2)});
                    }
                }
            }
        }
    }
    return true;
}

void apply_spin_free_pair(
    const std::vector<SpinFreeLink>& pair_links,
    const double* input,
    double* output,
    npy_intp ndet
) {
    std::fill(output, output + ndet, 0.0);
    for (const auto& link : pair_links) {
        output[link.bra] += link.phase * input[link.ket];
    }
}

void build_evecs_and_t2(
    const double* ci,
    const std::vector<std::vector<SpinFreeLink>>& links,
    npy_intp n,
    npy_intp ndet,
    std::vector<double>& evecs,
    std::vector<double>& t2
) {
    const npy_intp n2 = n * n;
    const npy_intp n4 = n2 * n2;
    evecs.assign(static_cast<std::size_t>(n2 * ndet), 0.0);
    t2.assign(static_cast<std::size_t>(n4 * ndet), 0.0);

    for (npy_intp pair = 0; pair < n2; ++pair) {
        apply_spin_free_pair(
            links[static_cast<std::size_t>(pair)],
            ci,
            evecs.data() + pair * ndet,
            ndet
        );
    }

    for (npy_intp second = 0; second < n2; ++second) {
        const double* base = evecs.data() + second * ndet;
        for (npy_intp first = 0; first < n2; ++first) {
            apply_spin_free_pair(
                links[static_cast<std::size_t>(first)],
                base,
                t2.data() + (first * n2 + second) * ndet,
                ndet
            );
        }
    }
}

double dot_product(const double* lhs, const double* rhs, npy_intp n) {
    double value = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        value += lhs[i] * rhs[i];
    }
    return value;
}

void fill_spin_free_rdms123(
    const double* ci,
    const std::vector<double>& evecs,
    const std::vector<double>& t2,
    npy_intp n,
    npy_intp ndet,
    double* rdm1,
    double* rdm2,
    double* rdm3
) {
    const npy_intp n2 = n * n;
    const npy_intp n4 = n2 * n2;
    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            rdm1[p * n + q] = dot_product(ci, evecs.data() + (p * n + q) * ndet, ndet);
        }
    }

    std::vector<double> left(static_cast<std::size_t>(n2 * ndet), 0.0);
    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            const double* src = evecs.data() + (q * n + p) * ndet;
            double* dst = left.data() + (p * n + q) * ndet;
            std::copy(src, src + ndet, dst);
        }
    }

#ifdef PYQED_HAVE_CBLAS
    constexpr int row_major = 101;
    constexpr int no_trans = 111;
    constexpr int trans = 112;
    if (
        n2 <= static_cast<npy_intp>(std::numeric_limits<int>::max()) &&
        n4 <= static_cast<npy_intp>(std::numeric_limits<int>::max()) &&
        ndet <= static_cast<npy_intp>(std::numeric_limits<int>::max())
    ) {
        cblas_dgemm(
            row_major,
            no_trans,
            trans,
            static_cast<int>(n2),
            static_cast<int>(n2),
            static_cast<int>(ndet),
            1.0,
            left.data(),
            static_cast<int>(ndet),
            evecs.data(),
            static_cast<int>(ndet),
            0.0,
            rdm2,
            static_cast<int>(n2)
        );
        cblas_dgemm(
            row_major,
            no_trans,
            trans,
            static_cast<int>(n2),
            static_cast<int>(n4),
            static_cast<int>(ndet),
            1.0,
            left.data(),
            static_cast<int>(ndet),
            t2.data(),
            static_cast<int>(ndet),
            0.0,
            rdm3,
            static_cast<int>(n4)
        );
        return;
    }
#endif

    for (npy_intp p = 0; p < n; ++p) {
        for (npy_intp q = 0; q < n; ++q) {
            const double* lhs = left.data() + (p * n + q) * ndet;
            for (npy_intp r = 0; r < n; ++r) {
                for (npy_intp s = 0; s < n; ++s) {
                    const npy_intp rs = r * n + s;
                    rdm2[idx4(n, p, q, r, s)] = dot_product(
                        lhs,
                        evecs.data() + rs * ndet,
                        ndet
                    );
                    for (npy_intp t = 0; t < n; ++t) {
                        for (npy_intp u = 0; u < n; ++u) {
                            rdm3[idx6_plain(n, p, q, r, s, t, u)] = dot_product(
                                lhs,
                                t2.data() + (rs * n2 + t * n + u) * ndet,
                                ndet
                            );
                        }
                    }
                }
            }
        }
    }
}

void compute_pair_overlaps(
    const std::vector<double>& evecs,
    const double* vec,
    npy_intp n,
    npy_intp ndet,
    double* overlaps
) {
    const npy_intp n2 = n * n;
#ifdef PYQED_HAVE_CBLAS
    constexpr int row_major = 101;
    constexpr int no_trans = 111;
    if (
        n2 <= static_cast<npy_intp>(std::numeric_limits<int>::max()) &&
        ndet <= static_cast<npy_intp>(std::numeric_limits<int>::max()) &&
        static_cast<std::size_t>(n2) * static_cast<std::size_t>(ndet) >= 512
    ) {
        cblas_dgemv(
            row_major,
            no_trans,
            static_cast<int>(n2),
            static_cast<int>(ndet),
            1.0,
            evecs.data(),
            static_cast<int>(ndet),
            vec,
            1,
            0.0,
            overlaps,
            1
        );
        return;
    }
#endif
    for (npy_intp pair = 0; pair < n2; ++pair) {
        const double* left = evecs.data() + pair * ndet;
        double value = 0.0;
        for (npy_intp det = 0; det < ndet; ++det) {
            value += left[det] * vec[det];
        }
        overlaps[pair] = value;
    }
}

PyObject* make_nevpt_terms_result(npy_intp n, PyObject** out0, PyObject** out1, PyObject** out2) {
    npy_intp dims[6] = {n, n, n, n, n, n};
    *out0 = PyArray_ZEROS(6, dims, NPY_DOUBLE, 0);
    *out1 = PyArray_ZEROS(6, dims, NPY_DOUBLE, 0);
    *out2 = PyArray_ZEROS(6, dims, NPY_DOUBLE, 0);
    if (*out0 == nullptr || *out1 == nullptr || *out2 == nullptr) {
        Py_XDECREF(*out0);
        Py_XDECREF(*out1);
        Py_XDECREF(*out2);
        return nullptr;
    }
    PyObject* tuple = PyTuple_New(3);
    if (tuple == nullptr) {
        Py_DECREF(*out0);
        Py_DECREF(*out1);
        Py_DECREF(*out2);
        return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, *out0);
    PyTuple_SET_ITEM(tuple, 1, *out1);
    PyTuple_SET_ITEM(tuple, 2, *out2);
    return tuple;
}

PyObject* nevpt_spin_free_rdms123(PyObject*, PyObject* args) {
    PyObject* ci_obj = nullptr;
    PyObject* binary_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &ci_obj, &binary_obj)) {
        return nullptr;
    }

    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!ci || !binary) {
        return nullptr;
    }
    if (!validate_nevpt_ci_binary_inputs(ci.obj, binary.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(binary.obj, 2);
    const npy_intp ndet = PyArray_DIM(ci.obj, 0);
    npy_intp dims1[2] = {n, n};
    npy_intp dims2[4] = {n, n, n, n};
    npy_intp dims3[6] = {n, n, n, n, n, n};

    PyObject* rdm1_obj = PyArray_ZEROS(2, dims1, NPY_DOUBLE, 0);
    PyObject* rdm2_obj = PyArray_ZEROS(4, dims2, NPY_DOUBLE, 0);
    PyObject* rdm3_obj = PyArray_ZEROS(6, dims3, NPY_DOUBLE, 0);
    if (rdm1_obj == nullptr || rdm2_obj == nullptr || rdm3_obj == nullptr) {
        Py_XDECREF(rdm1_obj);
        Py_XDECREF(rdm2_obj);
        Py_XDECREF(rdm3_obj);
        return nullptr;
    }

    std::vector<std::vector<SpinFreeLink>> links;
    std::vector<std::uint64_t> det_bits;
    if (!build_spin_free_links(binary.obj, links, det_bits)) {
        Py_DECREF(rdm1_obj);
        Py_DECREF(rdm2_obj);
        Py_DECREF(rdm3_obj);
        return nullptr;
    }

    std::vector<double> evecs;
    std::vector<double> t2;
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    auto* rdm1 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rdm1_obj)));
    auto* rdm2 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rdm2_obj)));
    auto* rdm3 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(rdm3_obj)));

    Py_BEGIN_ALLOW_THREADS
    build_evecs_and_t2(ci_data, links, n, ndet, evecs, t2);
    fill_spin_free_rdms123(ci_data, evecs, t2, n, ndet, rdm1, rdm2, rdm3);
    Py_END_ALLOW_THREADS

    PyObject* result = PyTuple_New(3);
    if (result == nullptr) {
        Py_DECREF(rdm1_obj);
        Py_DECREF(rdm2_obj);
        Py_DECREF(rdm3_obj);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, rdm1_obj);
    PyTuple_SET_ITEM(result, 1, rdm2_obj);
    PyTuple_SET_ITEM(result, 2, rdm3_obj);
    return result;
}

void fill_nevpt_a16_terms(
    const double* h,
    const std::vector<std::vector<SpinFreeLink>>& links,
    const std::vector<double>& evecs,
    const std::vector<double>& t2,
    npy_intp n,
    npy_intp ndet,
    double* term0,
    double* term1,
    double* term2
) {
    const npy_intp n2 = n * n;
    std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
    std::vector<double> ov(static_cast<std::size_t>(n2), 0.0);

    for (npy_intp a = 0; a < n; ++a) {
        for (npy_intp c = 0; c < n; ++c) {
            const npy_intp ac = a * n + c;
            for (npy_intp k = 0; k < n; ++k) {
                for (npy_intp i = 0; i < n; ++i) {
                    const npy_intp ki = k * n + i;
                    const double* base = t2.data() + (ki * n2 + ac) * ndet;
                    for (npy_intp q = 0; q < n; ++q) {
                        for (npy_intp j = 0; j < n; ++j) {
                            apply_spin_free_pair(links[static_cast<std::size_t>(q * n + j)], base, temp.data(), ndet);
                            compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                            for (npy_intp b = 0; b < n; ++b) {
                                const double coeff = h[idx4(n, k, b, i, j)];
                                if (coeff == 0.0) {
                                    continue;
                                }
                                for (npy_intp p = 0; p < n; ++p) {
                                    for (npy_intp r = 0; r < n; ++r) {
                                        term0[idx6_plain(n, p, q, r, a, b, c)] += coeff * ov[p * n + r];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (npy_intp i = 0; i < n; ++i) {
        for (npy_intp k = 0; k < n; ++k) {
            const npy_intp ik = i * n + k;
            for (npy_intp j = 0; j < n; ++j) {
                for (npy_intp c = 0; c < n; ++c) {
                    const npy_intp jc = j * n + c;
                    const double* base = t2.data() + (jc * n2 + ik) * ndet;
                    for (npy_intp q = 0; q < n; ++q) {
                        for (npy_intp b = 0; b < n; ++b) {
                            apply_spin_free_pair(links[static_cast<std::size_t>(q * n + b)], base, temp.data(), ndet);
                            compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                            for (npy_intp a = 0; a < n; ++a) {
                                const double coeff = h[idx4(n, i, j, k, a)];
                                if (coeff == 0.0) {
                                    continue;
                                }
                                for (npy_intp p = 0; p < n; ++p) {
                                    for (npy_intp r = 0; r < n; ++r) {
                                        term1[idx6_plain(n, p, q, r, a, b, c)] += coeff * ov[p * n + r];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (npy_intp k = 0; k < n; ++k) {
        for (npy_intp i = 0; i < n; ++i) {
            const npy_intp ki = k * n + i;
            for (npy_intp a = 0; a < n; ++a) {
                for (npy_intp j = 0; j < n; ++j) {
                    const npy_intp aj = a * n + j;
                    const double* base = t2.data() + (aj * n2 + ki) * ndet;
                    for (npy_intp q = 0; q < n; ++q) {
                        for (npy_intp b = 0; b < n; ++b) {
                            apply_spin_free_pair(links[static_cast<std::size_t>(q * n + b)], base, temp.data(), ndet);
                            compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                            for (npy_intp c = 0; c < n; ++c) {
                                const double coeff = h[idx4(n, k, c, i, j)];
                                if (coeff == 0.0) {
                                    continue;
                                }
                                for (npy_intp p = 0; p < n; ++p) {
                                    for (npy_intp r = 0; r < n; ++r) {
                                        term2[idx6_plain(n, p, q, r, a, b, c)] += coeff * ov[p * n + r];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void fill_nevpt_a22_terms(
    const double* h,
    const std::vector<std::vector<SpinFreeLink>>& links,
    const std::vector<double>& evecs,
    const std::vector<double>& t2,
    npy_intp n,
    npy_intp ndet,
    double* term0,
    double* term1,
    double* term2
) {
    const npy_intp n2 = n * n;

    auto fill_term0 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        std::vector<double> ov(static_cast<std::size_t>(n2), 0.0);
        for (npy_intp a = 0; a < n; ++a) {
            for (npy_intp c = 0; c < n; ++c) {
                const npy_intp ac = a * n + c;
                for (npy_intp p = 0; p < n; ++p) {
                    for (npy_intp r = 0; r < n; ++r) {
                        const npy_intp pr = p * n + r;
                        const double* base = t2.data() + (pr * n2 + ac) * ndet;
                        for (npy_intp q = 0; q < n; ++q) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(q * n + j)], base, temp.data(), ndet);
                                compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                                for (npy_intp b = 0; b < n; ++b) {
                                    const double coeff = h[idx4(n, p, q, r, b)];
                                    if (coeff == 0.0) {
                                        continue;
                                    }
                                    for (npy_intp i = 0; i < n; ++i) {
                                        for (npy_intp k = 0; k < n; ++k) {
                                            term0[idx6_plain(n, i, j, k, a, b, c)] += coeff * ov[i * n + k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto fill_term1 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        std::vector<double> ov(static_cast<std::size_t>(n2), 0.0);
        for (npy_intp p = 0; p < n; ++p) {
            for (npy_intp r = 0; r < n; ++r) {
                const npy_intp pr = p * n + r;
                for (npy_intp q = 0; q < n; ++q) {
                    for (npy_intp c = 0; c < n; ++c) {
                        const npy_intp qc = q * n + c;
                        const double* base = t2.data() + (qc * n2 + pr) * ndet;
                        for (npy_intp b = 0; b < n; ++b) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(b * n + j)], base, temp.data(), ndet);
                                compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                                for (npy_intp a = 0; a < n; ++a) {
                                    const double coeff = h[idx4(n, p, q, r, a)];
                                    if (coeff == 0.0) {
                                        continue;
                                    }
                                    for (npy_intp i = 0; i < n; ++i) {
                                        for (npy_intp k = 0; k < n; ++k) {
                                            term1[idx6_plain(n, i, j, k, a, b, c)] += coeff * ov[i * n + k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto fill_term2 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        std::vector<double> ov(static_cast<std::size_t>(n2), 0.0);
        for (npy_intp r = 0; r < n; ++r) {
            for (npy_intp p = 0; p < n; ++p) {
                const npy_intp rp = r * n + p;
                for (npy_intp a = 0; a < n; ++a) {
                    for (npy_intp q = 0; q < n; ++q) {
                        const npy_intp aq = a * n + q;
                        const double* base = t2.data() + (aq * n2 + rp) * ndet;
                        for (npy_intp b = 0; b < n; ++b) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(b * n + j)], base, temp.data(), ndet);
                                compute_pair_overlaps(evecs, temp.data(), n, ndet, ov.data());
                                for (npy_intp c = 0; c < n; ++c) {
                                    const double coeff = h[idx4(n, r, c, p, q)];
                                    if (coeff == 0.0) {
                                        continue;
                                    }
                                    for (npy_intp i = 0; i < n; ++i) {
                                        for (npy_intp k = 0; k < n; ++k) {
                                            term2[idx6_plain(n, i, j, k, a, b, c)] += coeff * ov[i * n + k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    if (n >= 5 && ndet >= 20) {
        std::thread worker0(fill_term0);
        std::thread worker1(fill_term1);
        fill_term2();
        worker0.join();
        worker1.join();
    } else {
        fill_term0();
        fill_term1();
        fill_term2();
    }
}

inline npy_intp idx_h2ev(npy_intp n, npy_intp ncore, npy_intp p, npy_intp q, npy_intp i, npy_intp r) {
    return (((p * n + q) * ncore + i) * n + r);
}

void accumulate_nevpt_a22_4rdm_energy(
    const double* h,
    const double* h2ev,
    const std::vector<std::vector<SpinFreeLink>>& links,
    const std::vector<double>& evecs,
    const std::vector<double>& t2,
    npy_intp n,
    npy_intp ndet,
    npy_intp ncore,
    double* energy
) {
    const npy_intp n2 = n * n;
    std::vector<double> left_vectors(static_cast<std::size_t>(n * ncore * ndet), 0.0);
    for (npy_intp j = 0; j < n; ++j) {
        for (npy_intp core = 0; core < ncore; ++core) {
            double* left = left_vectors.data() + (j * ncore + core) * ndet;
            for (npy_intp i = 0; i < n; ++i) {
                for (npy_intp k = 0; k < n; ++k) {
                    const double coeff = h2ev[idx_h2ev(n, ncore, j, i, core, k)];
                    if (coeff == 0.0) {
                        continue;
                    }
                    const double* ev = evecs.data() + (i * n + k) * ndet;
                    for (npy_intp det = 0; det < ndet; ++det) {
                        left[det] += coeff * ev[det];
                    }
                }
            }
        }
    }

    std::vector<double> term0_energy(static_cast<std::size_t>(ncore), 0.0);
    std::vector<double> term1_energy(static_cast<std::size_t>(ncore), 0.0);
    std::vector<double> term2_energy(static_cast<std::size_t>(ncore), 0.0);

    auto fill_term0 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        for (npy_intp a = 0; a < n; ++a) {
            for (npy_intp c = 0; c < n; ++c) {
                const npy_intp ac = a * n + c;
                for (npy_intp p = 0; p < n; ++p) {
                    for (npy_intp r = 0; r < n; ++r) {
                        const npy_intp pr = p * n + r;
                        const double* base = t2.data() + (pr * n2 + ac) * ndet;
                        for (npy_intp q = 0; q < n; ++q) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(q * n + j)], base, temp.data(), ndet);
                                for (npy_intp core = 0; core < ncore; ++core) {
                                    const double left = dot_product(
                                        left_vectors.data() + (j * ncore + core) * ndet,
                                        temp.data(),
                                        ndet
                                    );
                                    if (left == 0.0) {
                                        continue;
                                    }
                                    double right = 0.0;
                                    for (npy_intp b = 0; b < n; ++b) {
                                        right += h[idx4(n, p, q, r, b)] * h2ev[idx_h2ev(n, ncore, b, a, core, c)];
                                    }
                                    term0_energy[core] -= left * right;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto fill_term1 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        for (npy_intp p = 0; p < n; ++p) {
            for (npy_intp r = 0; r < n; ++r) {
                const npy_intp pr = p * n + r;
                for (npy_intp q = 0; q < n; ++q) {
                    for (npy_intp c = 0; c < n; ++c) {
                        const npy_intp qc = q * n + c;
                        const double* base = t2.data() + (qc * n2 + pr) * ndet;
                        for (npy_intp b = 0; b < n; ++b) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(b * n + j)], base, temp.data(), ndet);
                                for (npy_intp core = 0; core < ncore; ++core) {
                                    const double left = dot_product(
                                        left_vectors.data() + (j * ncore + core) * ndet,
                                        temp.data(),
                                        ndet
                                    );
                                    if (left == 0.0) {
                                        continue;
                                    }
                                    double right = 0.0;
                                    for (npy_intp a = 0; a < n; ++a) {
                                        right += h[idx4(n, p, q, r, a)] * h2ev[idx_h2ev(n, ncore, b, a, core, c)];
                                    }
                                    term1_energy[core] -= left * right;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto fill_term2 = [&]() {
        std::vector<double> temp(static_cast<std::size_t>(ndet), 0.0);
        for (npy_intp r = 0; r < n; ++r) {
            for (npy_intp p = 0; p < n; ++p) {
                const npy_intp rp = r * n + p;
                for (npy_intp a = 0; a < n; ++a) {
                    for (npy_intp q = 0; q < n; ++q) {
                        const npy_intp aq = a * n + q;
                        const double* base = t2.data() + (aq * n2 + rp) * ndet;
                        for (npy_intp b = 0; b < n; ++b) {
                            for (npy_intp j = 0; j < n; ++j) {
                                apply_spin_free_pair(links[static_cast<std::size_t>(b * n + j)], base, temp.data(), ndet);
                                for (npy_intp core = 0; core < ncore; ++core) {
                                    const double left = dot_product(
                                        left_vectors.data() + (j * ncore + core) * ndet,
                                        temp.data(),
                                        ndet
                                    );
                                    if (left == 0.0) {
                                        continue;
                                    }
                                    double right = 0.0;
                                    for (npy_intp c = 0; c < n; ++c) {
                                        right += h[idx4(n, r, c, p, q)] * h2ev[idx_h2ev(n, ncore, b, a, core, c)];
                                    }
                                    term2_energy[core] += left * right;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    if (n >= 5 && ndet >= 20) {
        std::thread worker0(fill_term0);
        std::thread worker1(fill_term1);
        fill_term2();
        worker0.join();
        worker1.join();
    } else {
        fill_term0();
        fill_term1();
        fill_term2();
    }
    for (npy_intp core = 0; core < ncore; ++core) {
        energy[core] += term0_energy[core] + term1_energy[core] + term2_energy[core];
    }
}

bool validate_nevpt_a22_4rdm_energy_inputs(
    PyArrayObject* h2e,
    PyArrayObject* h2e_v,
    PyArrayObject* ci,
    PyArrayObject* binary
) {
    if (!validate_nevpt_contract_inputs(h2e, ci, binary)) {
        return false;
    }
    const npy_intp n = PyArray_DIM(h2e, 0);
    if (
        PyArray_NDIM(h2e_v) != 4 ||
        PyArray_DIM(h2e_v, 0) != n ||
        PyArray_DIM(h2e_v, 1) != n ||
        PyArray_DIM(h2e_v, 3) != n
    ) {
        PyErr_SetString(PyExc_ValueError, "nevpt_a22_4rdm_energy expects h2e_v(n,n,ncore,n).");
        return false;
    }
    return true;
}

PyObject* nevpt_a22_4rdm_energy(PyObject*, PyObject* args) {
    PyObject* h2e_obj = nullptr;
    PyObject* h2e_v_obj = nullptr;
    PyObject* ci_obj = nullptr;
    PyObject* binary_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &h2e_obj, &h2e_v_obj, &ci_obj, &binary_obj)) {
        return nullptr;
    }

    ArrayRef h2e(h2e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef h2e_v(h2e_v_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!h2e || !h2e_v || !ci || !binary) {
        return nullptr;
    }
    if (!validate_nevpt_a22_4rdm_energy_inputs(h2e.obj, h2e_v.obj, ci.obj, binary.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h2e.obj, 0);
    const npy_intp ncore = PyArray_DIM(h2e_v.obj, 2);
    const npy_intp ndet = PyArray_DIM(ci.obj, 0);
    npy_intp dims[1] = {ncore};
    PyObject* energy_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (energy_obj == nullptr) {
        return nullptr;
    }

    std::vector<std::vector<SpinFreeLink>> links;
    std::vector<std::uint64_t> det_bits;
    if (!build_spin_free_links(binary.obj, links, det_bits)) {
        Py_DECREF(energy_obj);
        return nullptr;
    }
    std::vector<double> evecs;
    std::vector<double> t2;
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    build_evecs_and_t2(ci_data, links, n, ndet, evecs, t2);

    const auto* h = static_cast<const double*>(PyArray_DATA(h2e.obj));
    const auto* h2v = static_cast<const double*>(PyArray_DATA(h2e_v.obj));
    auto* energy = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(energy_obj)));

    Py_BEGIN_ALLOW_THREADS
    accumulate_nevpt_a22_4rdm_energy(h, h2v, links, evecs, t2, n, ndet, ncore, energy);
    Py_END_ALLOW_THREADS

    return energy_obj;
}

PyObject* nevpt_a16_4rdm_terms(PyObject*, PyObject* args) {
    PyObject* h2e_obj = nullptr;
    PyObject* ci_obj = nullptr;
    PyObject* binary_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &h2e_obj, &ci_obj, &binary_obj)) {
        return nullptr;
    }

    ArrayRef h2e(h2e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!h2e || !ci || !binary) {
        return nullptr;
    }
    if (!validate_nevpt_contract_inputs(h2e.obj, ci.obj, binary.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h2e.obj, 0);
    const npy_intp ndet = PyArray_DIM(ci.obj, 0);
    PyObject* term0_obj = nullptr;
    PyObject* term1_obj = nullptr;
    PyObject* term2_obj = nullptr;
    PyObject* result = make_nevpt_terms_result(n, &term0_obj, &term1_obj, &term2_obj);
    if (result == nullptr) {
        return nullptr;
    }

    std::vector<std::vector<SpinFreeLink>> links;
    std::vector<std::uint64_t> det_bits;
    if (!build_spin_free_links(binary.obj, links, det_bits)) {
        Py_DECREF(result);
        return nullptr;
    }
    std::vector<double> evecs;
    std::vector<double> t2;
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    build_evecs_and_t2(ci_data, links, n, ndet, evecs, t2);

    const auto* h = static_cast<const double*>(PyArray_DATA(h2e.obj));
    auto* term0 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term0_obj)));
    auto* term1 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term1_obj)));
    auto* term2 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term2_obj)));

    Py_BEGIN_ALLOW_THREADS
    fill_nevpt_a16_terms(h, links, evecs, t2, n, ndet, term0, term1, term2);
    Py_END_ALLOW_THREADS

    return result;
}

PyObject* nevpt_a22_4rdm_terms(PyObject*, PyObject* args) {
    PyObject* h2e_obj = nullptr;
    PyObject* ci_obj = nullptr;
    PyObject* binary_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &h2e_obj, &ci_obj, &binary_obj)) {
        return nullptr;
    }

    ArrayRef h2e(h2e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!h2e || !ci || !binary) {
        return nullptr;
    }
    if (!validate_nevpt_contract_inputs(h2e.obj, ci.obj, binary.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h2e.obj, 0);
    const npy_intp ndet = PyArray_DIM(ci.obj, 0);
    PyObject* term0_obj = nullptr;
    PyObject* term1_obj = nullptr;
    PyObject* term2_obj = nullptr;
    PyObject* result = make_nevpt_terms_result(n, &term0_obj, &term1_obj, &term2_obj);
    if (result == nullptr) {
        return nullptr;
    }

    std::vector<std::vector<SpinFreeLink>> links;
    std::vector<std::uint64_t> det_bits;
    if (!build_spin_free_links(binary.obj, links, det_bits)) {
        Py_DECREF(result);
        return nullptr;
    }
    std::vector<double> evecs;
    std::vector<double> t2;
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    build_evecs_and_t2(ci_data, links, n, ndet, evecs, t2);

    const auto* h = static_cast<const double*>(PyArray_DATA(h2e.obj));
    auto* term0 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term0_obj)));
    auto* term1 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term1_obj)));
    auto* term2 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(term2_obj)));

    Py_BEGIN_ALLOW_THREADS
    fill_nevpt_a22_terms(h, links, evecs, t2, n, ndet, term0, term1, term2);
    Py_END_ALLOW_THREADS

    return result;
}

PyObject* nevpt_a16_a22_4rdm_terms(PyObject*, PyObject* args) {
    PyObject* h2e_obj = nullptr;
    PyObject* ci_obj = nullptr;
    PyObject* binary_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &h2e_obj, &ci_obj, &binary_obj)) {
        return nullptr;
    }

    ArrayRef h2e(h2e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef ci(ci_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef binary(binary_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (!h2e || !ci || !binary) {
        return nullptr;
    }
    if (!validate_nevpt_contract_inputs(h2e.obj, ci.obj, binary.obj)) {
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(h2e.obj, 0);
    const npy_intp ndet = PyArray_DIM(ci.obj, 0);

    PyObject* a16_0_obj = nullptr;
    PyObject* a16_1_obj = nullptr;
    PyObject* a16_2_obj = nullptr;
    PyObject* a16_result = make_nevpt_terms_result(n, &a16_0_obj, &a16_1_obj, &a16_2_obj);
    if (a16_result == nullptr) {
        return nullptr;
    }

    PyObject* a22_0_obj = nullptr;
    PyObject* a22_1_obj = nullptr;
    PyObject* a22_2_obj = nullptr;
    PyObject* a22_result = make_nevpt_terms_result(n, &a22_0_obj, &a22_1_obj, &a22_2_obj);
    if (a22_result == nullptr) {
        Py_DECREF(a16_result);
        return nullptr;
    }

    std::vector<std::vector<SpinFreeLink>> links;
    std::vector<std::uint64_t> det_bits;
    if (!build_spin_free_links(binary.obj, links, det_bits)) {
        Py_DECREF(a16_result);
        Py_DECREF(a22_result);
        return nullptr;
    }
    std::vector<double> evecs;
    std::vector<double> t2;
    const auto* ci_data = static_cast<const double*>(PyArray_DATA(ci.obj));
    build_evecs_and_t2(ci_data, links, n, ndet, evecs, t2);

    const auto* h = static_cast<const double*>(PyArray_DATA(h2e.obj));
    auto* a16_0 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a16_0_obj)));
    auto* a16_1 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a16_1_obj)));
    auto* a16_2 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a16_2_obj)));
    auto* a22_0 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a22_0_obj)));
    auto* a22_1 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a22_1_obj)));
    auto* a22_2 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(a22_2_obj)));

    Py_BEGIN_ALLOW_THREADS
    fill_nevpt_a16_terms(h, links, evecs, t2, n, ndet, a16_0, a16_1, a16_2);
    fill_nevpt_a22_terms(h, links, evecs, t2, n, ndet, a22_0, a22_1, a22_2);
    Py_END_ALLOW_THREADS

    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(a16_result);
        Py_DECREF(a22_result);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, a16_result);
    PyTuple_SET_ITEM(result, 1, a22_result);
    return result;
}

PyObject* available(PyObject*, PyObject*) {
    Py_RETURN_TRUE;
}

PyObject* native_capabilities(PyObject*, PyObject*) {
#ifdef PYQED_HAVE_CBLAS
    const int have_cblas = 1;
#else
    const int have_cblas = 0;
#endif
#ifdef __APPLE__
    const char* blas_provider = "accelerate";
#elif defined(PYQED_USE_CBLAS)
    const char* blas_provider = "system-blas";
#else
    const char* blas_provider = "none";
#endif
    return Py_BuildValue(
        "{s:O,s:s,s:O,s:O,s:i}",
        "cblas", have_cblas ? Py_True : Py_False,
        "blas_provider", blas_provider,
        "rhf_davidson", have_cblas ? Py_True : Py_False,
        "rhf_multiroot", have_cblas ? Py_True : Py_False,
        "spin0_native_max_roots", 1
    );
}

PyMethodDef methods[] = {
    {"available", available, METH_NOARGS, "Return True when the C++ CASSCF helper extension is loaded."},
    {"native_capabilities", native_capabilities, METH_NOARGS, "Return compiled native CAS capability metadata."},
    {"ci_hamiltonian", ci_hamiltonian, METH_VARARGS, "Build the dense Slater-Condon CI Hamiltonian."},
    {"orbital_hessian_action_from_integrals", orbital_hessian_action_from_integrals, METH_VARARGS, "Apply the dense orbital Hessian action without materializing derivative ERIs."},
    {"scatter_opposite_spin_rdm2", scatter_opposite_spin_rdm2, METH_VARARGS, "Accumulate opposite-spin spin-string RDM2 contractions in place."},
    {"sigma_compact_spin_string", sigma_compact_spin_string, METH_VARARGS, "Apply the RHF spin-string compact direct-CI sigma kernel."},
    {"sigma_compact_spin0_pair", sigma_compact_spin0_pair, METH_VARARGS, "Apply the spin-adapted pair-space RHF direct-CI sigma kernel."},
    {"create_spin0_pair_workspace", create_spin0_pair_workspace, METH_VARARGS, "Create a persistent packed-BLAS restricted direct-CI workspace."},
    {"apply_spin0_pair_workspace_det", apply_spin0_pair_workspace_det, METH_VARARGS, "Apply a packed-BLAS workspace to a determinant-space CI vector."},
    {"davidson_spin0_pair", davidson_spin0_pair, METH_VARARGS, "Solve one-root spin0-pair direct CI with a native Davidson loop."},
    {"davidson_rhf_workspace", davidson_rhf_workspace, METH_VARARGS, "Solve low restricted direct-CI roots in a persistent packed-BLAS workspace."},
    {"sigma_values_conn", sigma_values_conn, METH_VARARGS, "Apply precomputed connection-value direct-CI sigma to one or more CI vectors."},
    {"caspt2_external_space", caspt2_external_space, METH_VARARGS, "Generate and classify the experimental CASPT2 external determinant space."},
    {"caspt2_external_kernel", caspt2_external_kernel, METH_VARARGS, "Evaluate experimental CASPT2 external determinant couplings and diagonal denominators."},
    {"caspt2_strong_contract", caspt2_strong_contract, METH_VARARGS, "Reduce CASPT2 determinant couplings into strongly contracted class energies."},
    {"caspt2_en_coupled_contract", caspt2_en_coupled_contract, METH_VARARGS, "Build the coupled strong-contracted EN CASPT2 metric and denominator matrix."},
    {"caspt2_solve_contracted", caspt2_solve_contracted, METH_VARARGS, "Solve a dense real-shifted contracted CASPT2 linear system."},
    {"nevpt_spin_free_rdms123", nevpt_spin_free_rdms123, METH_VARARGS, "Build spin-free active 1-, 2-, and 3-RDMs for native SC-NEVPT2."},
    {"nevpt_a22_4rdm_energy", nevpt_a22_4rdm_energy, METH_VARARGS, "Contract the SC-NEVPT2 Si A22 4-RDM numerator terms without materializing A22."},
    {"nevpt_a16_4rdm_terms", nevpt_a16_4rdm_terms, METH_VARARGS, "Build contracted A16 4-RDM terms without materializing the full 4-RDM."},
    {"nevpt_a22_4rdm_terms", nevpt_a22_4rdm_terms, METH_VARARGS, "Build contracted A22 4-RDM terms without materializing the full 4-RDM."},
    {"nevpt_a16_a22_4rdm_terms", nevpt_a16_a22_4rdm_terms, METH_VARARGS, "Build contracted A16 and A22 4-RDM terms using one shared determinant workspace."},
    {"single_string_links", single_string_links, METH_VARARGS, "Build single-excitation spin-string links."},
    {"double_string_links", double_string_links, METH_VARARGS, "Build double-excitation spin-string links."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_casscf_cpp",
    "C++ helper kernels for qchem CASSCF.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__casscf_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
