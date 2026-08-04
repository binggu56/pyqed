#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex>
#include <vector>

namespace {

template <typename scalar_t>
void apply_conditional_gauges_impl(
    scalar_t* left,
    scalar_t* right,
    const npy_intp* states,
    const npy_intp* group_offsets,
    const npy_intp* indices,
    const scalar_t* gauges,
    const scalar_t* gauge_inverses,
    npy_intp ngroups,
    npy_intp left_rows,
    npy_intp left_physical,
    npy_intp shared_dim,
    npy_intp bond_dim,
    npy_intp right_physical,
    npy_intp right_columns,
    bool terminal,
    scalar_t* work
) {
    npy_intp matrix_offset = 0;
    for (npy_intp record = 0; record < ngroups; ++record) {
        const npy_intp start = group_offsets[record];
        const npy_intp stop = group_offsets[record + 1];
        const npy_intp size = stop - start;
        const npy_intp state = states[record];
        const scalar_t* gauge = gauges + matrix_offset;
        const scalar_t* gauge_inverse = gauge_inverses + matrix_offset;

        for (npy_intp row = 0; row < left_rows; ++row) {
            for (npy_intp physical = 0; physical < left_physical; ++physical) {
                const npy_intp base =
                    ((row * left_physical + physical) * shared_dim + state) * bond_dim;
                for (npy_intp output = 0; output < size; ++output) {
                    scalar_t value{};
                    for (npy_intp input = 0; input < size; ++input) {
                        value +=
                            left[base + indices[start + input]]
                            * gauge[input * size + output];
                    }
                    work[output] = value;
                }
                for (npy_intp output = 0; output < size; ++output) {
                    left[base + indices[start + output]] = work[output];
                }
            }
        }

        if (terminal) {
            const npy_intp base = state * bond_dim;
            for (npy_intp output = 0; output < size; ++output) {
                scalar_t value{};
                for (npy_intp input = 0; input < size; ++input) {
                    value +=
                        gauge_inverse[output * size + input]
                        * right[base + indices[start + input]];
                }
                work[output] = value;
            }
            for (npy_intp output = 0; output < size; ++output) {
                right[base + indices[start + output]] = work[output];
            }
        } else {
            for (npy_intp physical = 0; physical < right_physical; ++physical) {
                for (npy_intp column = 0; column < right_columns; ++column) {
                    for (npy_intp output = 0; output < size; ++output) {
                        scalar_t value{};
                        for (npy_intp input = 0; input < size; ++input) {
                            const npy_intp index =
                                (((indices[start + input] * shared_dim + state)
                                  * right_physical + physical)
                                 * right_columns + column);
                            value +=
                                gauge_inverse[output * size + input] * right[index];
                        }
                        work[output] = value;
                    }
                    for (npy_intp output = 0; output < size; ++output) {
                        const npy_intp index =
                            (((indices[start + output] * shared_dim + state)
                              * right_physical + physical)
                             * right_columns + column);
                        right[index] = work[output];
                    }
                }
            }
        }
        matrix_offset += size * size;
    }
}

void resolve_and_decref(PyArrayObject* array) {
    if (array == nullptr) {
        return;
    }
    PyArray_ResolveWritebackIfCopy(array);
    Py_DECREF(array);
}

PyObject* apply_conditional_gauges_inplace(PyObject*, PyObject* args) {
    PyObject* left_object = nullptr;
    PyObject* right_object = nullptr;
    PyObject* states_object = nullptr;
    PyObject* offsets_object = nullptr;
    PyObject* indices_object = nullptr;
    PyObject* gauges_object = nullptr;
    PyObject* inverse_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOO",
            &left_object,
            &right_object,
            &states_object,
            &offsets_object,
            &indices_object,
            &gauges_object,
            &inverse_object
        )) {
        return nullptr;
    }

    PyArrayObject* left = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(left_object, NPY_NOTYPE, NPY_ARRAY_INOUT_ARRAY2)
    );
    PyArrayObject* right = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(right_object, NPY_NOTYPE, NPY_ARRAY_INOUT_ARRAY2)
    );
    PyArrayObject* states = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(states_object, NPY_INTP, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* offsets = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(offsets_object, NPY_INTP, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* indices = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(indices_object, NPY_INTP, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* gauges = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(gauges_object, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* inverses = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(inverse_object, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY)
    );
    if (
        left == nullptr || right == nullptr || states == nullptr || offsets == nullptr
        || indices == nullptr || gauges == nullptr || inverses == nullptr
    ) {
        resolve_and_decref(left);
        resolve_and_decref(right);
        Py_XDECREF(states);
        Py_XDECREF(offsets);
        Py_XDECREF(indices);
        Py_XDECREF(gauges);
        Py_XDECREF(inverses);
        return nullptr;
    }

    auto fail = [&](const char* message) -> PyObject* {
        PyErr_SetString(PyExc_ValueError, message);
        resolve_and_decref(left);
        resolve_and_decref(right);
        Py_DECREF(states);
        Py_DECREF(offsets);
        Py_DECREF(indices);
        Py_DECREF(gauges);
        Py_DECREF(inverses);
        return nullptr;
    };

    const int typenum = PyArray_TYPE(left);
    if (
        (typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE)
        || PyArray_TYPE(right) != typenum
        || PyArray_TYPE(gauges) != typenum
        || PyArray_TYPE(inverses) != typenum
    ) {
        return fail(
            "left, right, gauges, and gauge inverses must share float64 or complex128 dtype."
        );
    }
    if (
        PyArray_NDIM(left) != 4
        || (PyArray_NDIM(right) != 2 && PyArray_NDIM(right) != 4)
        || PyArray_NDIM(states) != 1
        || PyArray_NDIM(offsets) != 1
        || PyArray_NDIM(indices) != 1
        || PyArray_NDIM(gauges) != 1
        || PyArray_NDIM(inverses) != 1
    ) {
        return fail("conditional gauge arrays have incompatible ranks.");
    }

    const npy_intp ngroups = PyArray_DIM(states, 0);
    if (PyArray_DIM(offsets, 0) != ngroups + 1) {
        return fail("group_offsets must contain one more entry than states.");
    }
    const npy_intp left_rows = PyArray_DIM(left, 0);
    const npy_intp left_physical = PyArray_DIM(left, 1);
    const npy_intp shared_dim = PyArray_DIM(left, 2);
    const npy_intp bond_dim = PyArray_DIM(left, 3);
    const bool terminal = PyArray_NDIM(right) == 2;
    npy_intp right_physical = 1;
    npy_intp right_columns = 1;
    if (terminal) {
        if (
            PyArray_DIM(right, 0) != shared_dim
            || PyArray_DIM(right, 1) != bond_dim
        ) {
            return fail("terminal tensor dimensions do not match the left tensor.");
        }
    } else {
        if (
            PyArray_DIM(right, 0) != bond_dim
            || PyArray_DIM(right, 1) != shared_dim
        ) {
            return fail("pair tensor dimensions do not match the left tensor.");
        }
        right_physical = PyArray_DIM(right, 2);
        right_columns = PyArray_DIM(right, 3);
    }

    const auto* state_data = static_cast<const npy_intp*>(PyArray_DATA(states));
    const auto* offset_data = static_cast<const npy_intp*>(PyArray_DATA(offsets));
    const auto* index_data = static_cast<const npy_intp*>(PyArray_DATA(indices));
    if (offset_data[0] != 0 || offset_data[ngroups] != PyArray_DIM(indices, 0)) {
        return fail("group_offsets do not span group_indices.");
    }
    npy_intp matrix_size = 0;
    npy_intp maximum_group = 0;
    for (npy_intp record = 0; record < ngroups; ++record) {
        const npy_intp start = offset_data[record];
        const npy_intp stop = offset_data[record + 1];
        if (
            start < 0 || stop <= start || start > stop
            || state_data[record] < 0 || state_data[record] >= shared_dim
        ) {
            return fail("invalid conditional gauge group offsets or shared state.");
        }
        const npy_intp size = stop - start;
        maximum_group = size > maximum_group ? size : maximum_group;
        matrix_size += size * size;
        for (npy_intp position = start; position < stop; ++position) {
            if (index_data[position] < 0 || index_data[position] >= bond_dim) {
                return fail("conditional gauge group index is out of range.");
            }
        }
    }
    if (
        PyArray_DIM(gauges, 0) != matrix_size
        || PyArray_DIM(inverses, 0) != matrix_size
    ) {
        return fail("packed gauge matrix sizes do not match the groups.");
    }

    if (typenum == NPY_DOUBLE) {
        std::vector<double> work(static_cast<std::size_t>(maximum_group));
        Py_BEGIN_ALLOW_THREADS
        apply_conditional_gauges_impl(
            static_cast<double*>(PyArray_DATA(left)),
            static_cast<double*>(PyArray_DATA(right)),
            state_data,
            offset_data,
            index_data,
            static_cast<const double*>(PyArray_DATA(gauges)),
            static_cast<const double*>(PyArray_DATA(inverses)),
            ngroups,
            left_rows,
            left_physical,
            shared_dim,
            bond_dim,
            right_physical,
            right_columns,
            terminal,
            work.data()
        );
        Py_END_ALLOW_THREADS
    } else {
        using complex_t = std::complex<double>;
        std::vector<complex_t> work(static_cast<std::size_t>(maximum_group));
        Py_BEGIN_ALLOW_THREADS
        apply_conditional_gauges_impl(
            reinterpret_cast<complex_t*>(PyArray_DATA(left)),
            reinterpret_cast<complex_t*>(PyArray_DATA(right)),
            state_data,
            offset_data,
            index_data,
            reinterpret_cast<const complex_t*>(PyArray_DATA(gauges)),
            reinterpret_cast<const complex_t*>(PyArray_DATA(inverses)),
            ngroups,
            left_rows,
            left_physical,
            shared_dim,
            bond_dim,
            right_physical,
            right_columns,
            terminal,
            work.data()
        );
        Py_END_ALLOW_THREADS
    }

    resolve_and_decref(left);
    resolve_and_decref(right);
    Py_DECREF(states);
    Py_DECREF(offsets);
    Py_DECREF(indices);
    Py_DECREF(gauges);
    Py_DECREF(inverses);
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {
        "apply_conditional_gauges_inplace",
        apply_conditional_gauges_inplace,
        METH_VARARGS,
        "Apply a packed batch of physical-conditioned virtual gauges in place.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_conditional_gauge_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__conditional_gauge_cpp() {
    import_array();
    return PyModule_Create(&module);
}
