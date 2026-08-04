#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex>

namespace {

template <typename scalar_t>
void block_matvec_impl(
    const scalar_t* blocks,
    const npy_intp* rows,
    const npy_intp* columns,
    const scalar_t* inputs,
    scalar_t* outputs,
    npy_intp nblocks,
    npy_intp physical_blocks,
    npy_intp virtual_size,
    bool transpose_conj
) {
    const npy_intp block_stride = virtual_size * virtual_size;
    const npy_intp input_stride = virtual_size;
    for (npy_intp entry = 0; entry < nblocks; ++entry) {
        const npy_intp row = rows[entry];
        const npy_intp column = columns[entry];
        if (row < 0 || row >= physical_blocks || column < 0 || column >= physical_blocks) {
            continue;
        }
        const scalar_t* block = blocks + entry * block_stride;
        if (!transpose_conj) {
            const scalar_t* input = inputs + column * input_stride;
            scalar_t* output = outputs + row * input_stride;
            for (npy_intp i = 0; i < virtual_size; ++i) {
                scalar_t acc = scalar_t{};
                const scalar_t* block_row = block + i * virtual_size;
                for (npy_intp j = 0; j < virtual_size; ++j) {
                    acc += block_row[j] * input[j];
                }
                output[i] += acc;
            }
        } else {
            const scalar_t* input = inputs + row * input_stride;
            scalar_t* output = outputs + column * input_stride;
            for (npy_intp j = 0; j < virtual_size; ++j) {
                scalar_t acc = scalar_t{};
                for (npy_intp i = 0; i < virtual_size; ++i) {
                    acc += std::conj(block[i * virtual_size + j]) * input[i];
                }
                output[j] += acc;
            }
        }
    }
}

template <>
void block_matvec_impl<double>(
    const double* blocks,
    const npy_intp* rows,
    const npy_intp* columns,
    const double* inputs,
    double* outputs,
    npy_intp nblocks,
    npy_intp physical_blocks,
    npy_intp virtual_size,
    bool transpose_conj
) {
    const npy_intp block_stride = virtual_size * virtual_size;
    const npy_intp input_stride = virtual_size;
    for (npy_intp entry = 0; entry < nblocks; ++entry) {
        const npy_intp row = rows[entry];
        const npy_intp column = columns[entry];
        if (row < 0 || row >= physical_blocks || column < 0 || column >= physical_blocks) {
            continue;
        }
        const double* block = blocks + entry * block_stride;
        if (!transpose_conj) {
            const double* input = inputs + column * input_stride;
            double* output = outputs + row * input_stride;
            for (npy_intp i = 0; i < virtual_size; ++i) {
                double acc = 0.0;
                const double* block_row = block + i * virtual_size;
                for (npy_intp j = 0; j < virtual_size; ++j) {
                    acc += block_row[j] * input[j];
                }
                output[i] += acc;
            }
        } else {
            const double* input = inputs + row * input_stride;
            double* output = outputs + column * input_stride;
            for (npy_intp j = 0; j < virtual_size; ++j) {
                double acc = 0.0;
                for (npy_intp i = 0; i < virtual_size; ++i) {
                    acc += block[i * virtual_size + j] * input[i];
                }
                output[j] += acc;
            }
        }
    }
}

PyObject* block_matvec(PyObject*, PyObject* args) {
    PyObject* blocks_obj = nullptr;
    PyObject* rows_obj = nullptr;
    PyObject* columns_obj = nullptr;
    PyObject* inputs_obj = nullptr;
    int transpose_conj_int = 0;
    if (!PyArg_ParseTuple(
            args,
            "OOOO|p",
            &blocks_obj,
            &rows_obj,
            &columns_obj,
            &inputs_obj,
            &transpose_conj_int
        )) {
        return nullptr;
    }

    PyArrayObject* blocks = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(blocks_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* rows = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(rows_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* columns = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(columns_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* inputs = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(inputs_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY)
    );
    if (blocks == nullptr || rows == nullptr || columns == nullptr || inputs == nullptr) {
        Py_XDECREF(blocks);
        Py_XDECREF(rows);
        Py_XDECREF(columns);
        Py_XDECREF(inputs);
        return nullptr;
    }

    const int typenum = PyArray_TYPE(blocks);
    if (typenum != PyArray_TYPE(inputs) || (typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE)) {
        PyErr_SetString(PyExc_TypeError, "blocks and inputs must both be float64 or complex128.");
        Py_DECREF(blocks);
        Py_DECREF(rows);
        Py_DECREF(columns);
        Py_DECREF(inputs);
        return nullptr;
    }
    if (PyArray_NDIM(blocks) != 3 || PyArray_NDIM(inputs) != 2 ||
        PyArray_NDIM(rows) != 1 || PyArray_NDIM(columns) != 1) {
        PyErr_SetString(PyExc_ValueError, "expected blocks=(k,v,v), rows=(k,), columns=(k,), inputs=(p,v).");
        Py_DECREF(blocks);
        Py_DECREF(rows);
        Py_DECREF(columns);
        Py_DECREF(inputs);
        return nullptr;
    }
    const npy_intp nblocks = PyArray_DIM(blocks, 0);
    const npy_intp virtual_rows = PyArray_DIM(blocks, 1);
    const npy_intp virtual_columns = PyArray_DIM(blocks, 2);
    const npy_intp physical_blocks = PyArray_DIM(inputs, 0);
    const npy_intp virtual_size = PyArray_DIM(inputs, 1);
    if (virtual_rows != virtual_size || virtual_columns != virtual_size ||
        PyArray_DIM(rows, 0) != nblocks || PyArray_DIM(columns, 0) != nblocks) {
        PyErr_SetString(PyExc_ValueError, "block/operator dimensions are inconsistent.");
        Py_DECREF(blocks);
        Py_DECREF(rows);
        Py_DECREF(columns);
        Py_DECREF(inputs);
        return nullptr;
    }

    npy_intp dims[2] = {physical_blocks, virtual_size};
    PyArrayObject* outputs = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, typenum, 0)
    );
    if (outputs == nullptr) {
        Py_DECREF(blocks);
        Py_DECREF(rows);
        Py_DECREF(columns);
        Py_DECREF(inputs);
        return nullptr;
    }

    const bool transpose_conj = transpose_conj_int != 0;
    if (typenum == NPY_DOUBLE) {
        block_matvec_impl(
            static_cast<const double*>(PyArray_DATA(blocks)),
            static_cast<const npy_intp*>(PyArray_DATA(rows)),
            static_cast<const npy_intp*>(PyArray_DATA(columns)),
            static_cast<const double*>(PyArray_DATA(inputs)),
            static_cast<double*>(PyArray_DATA(outputs)),
            nblocks,
            physical_blocks,
            virtual_size,
            transpose_conj
        );
    } else {
        using complex_t = std::complex<double>;
        block_matvec_impl(
            reinterpret_cast<const complex_t*>(PyArray_DATA(blocks)),
            static_cast<const npy_intp*>(PyArray_DATA(rows)),
            static_cast<const npy_intp*>(PyArray_DATA(columns)),
            reinterpret_cast<const complex_t*>(PyArray_DATA(inputs)),
            reinterpret_cast<complex_t*>(PyArray_DATA(outputs)),
            nblocks,
            physical_blocks,
            virtual_size,
            transpose_conj
        );
    }

    Py_DECREF(blocks);
    Py_DECREF(rows);
    Py_DECREF(columns);
    Py_DECREF(inputs);
    return reinterpret_cast<PyObject*>(outputs);
}

PyMethodDef methods[] = {
    {"block_matvec", block_matvec, METH_VARARGS, "Apply physical-block operator blocks."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_physical_blocks_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__physical_blocks_cpp() {
    import_array();
    return PyModule_Create(&module);
}
