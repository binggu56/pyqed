#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <complex>
#include <vector>

using Complex = std::complex<double>;

static void compose_on_right(
    const Complex* left,
    const Complex* right,
    Complex* out,
    int nstates,
    bool conjugate_transpose
) {
    for (int i = 0; i < nstates; ++i) {
        for (int j = 0; j < nstates; ++j) {
            Complex value = 0.0;
            for (int k = 0; k < nstates; ++k) {
                Complex right_value = conjugate_transpose
                    ? std::conj(right[static_cast<size_t>(j) * nstates + k])
                    : right[static_cast<size_t>(k) * nstates + j];
                value += left[static_cast<size_t>(i) * nstates + k] * right_value;
            }
            out[static_cast<size_t>(i) * nstates + j] = value;
        }
    }
}

static int parse_shape(PyObject* shape_obj, std::vector<npy_intp>& shape) {
    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return -1;
    }

    Py_ssize_t ndim = PySequence_Length(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must be non-empty and positive");
        return -1;
    }

    shape.clear();
    shape.reserve(static_cast<size_t>(ndim));
    for (Py_ssize_t i = 0; i < ndim; ++i) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (item == nullptr) {
            return -1;
        }

        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "shape entries must be integers");
            Py_DECREF(item);
            return -1;
        }

        npy_intp value = (npy_intp)PyLong_AsLongLong(item);
        Py_DECREF(item);
        if (value <= 0 || PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "shape entries must be positive");
            return -1;
        }

        shape.push_back(value);
    }

    return 0;
}

static npy_intp flatten_coords(const std::vector<npy_intp>& coords,
                              const std::vector<npy_intp>& strides) {
    npy_intp flat = 0;
    for (size_t axis = 0; axis < coords.size(); ++axis) {
        flat += coords[axis] * strides[axis];
    }
    return flat;
}

static PyObject* linked_overlap_dense(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* shape_obj = nullptr;
    PyObject* axes_obj = nullptr;
    PyObject* indices_obj = nullptr;
    PyObject* values_obj = nullptr;
    int nstates = 0;
    int average_paths = 0;

    static const char* kwlist[] = {
        "shape",
        "axes",
        "indices",
        "values",
        "nstates",
        "average_paths",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOi|p",
            const_cast<char**>(kwlist),
            &shape_obj,
            &axes_obj,
            &indices_obj,
            &values_obj,
            &nstates,
            &average_paths
    )) {
        return nullptr;
    }

    if (average_paths != 0) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "average_paths is not implemented in the C++ kernel"
        );
        return nullptr;
    }
    if (nstates <= 0) {
        PyErr_SetString(PyExc_ValueError, "nstates must be positive");
        return nullptr;
    }

    std::vector<npy_intp> shape;
    if (parse_shape(shape_obj, shape) < 0) {
        return nullptr;
    }

    PyArrayObject* axes_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(axes_obj, NPY_INTP, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
    );
    PyArrayObject* indices_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(
            indices_obj,
            NPY_INTP,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
        )
    );
    PyArrayObject* values_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(
            values_obj,
            NPY_COMPLEX128,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
        )
    );

    if (axes_array == nullptr || indices_array == nullptr || values_array == nullptr) {
        Py_XDECREF(axes_array);
        Py_XDECREF(indices_array);
        Py_XDECREF(values_array);
        return nullptr;
    }

    const npy_intp ndim = static_cast<npy_intp>(shape.size());
    const npy_intp nlinks = PyArray_DIM(axes_array, 0);

    if (PyArray_NDIM(axes_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "axes must be one-dimensional");
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }
    if (PyArray_NDIM(indices_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "indices must be two-dimensional");
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }
    if (PyArray_DIM(indices_array, 0) != nlinks) {
        PyErr_SetString(PyExc_ValueError, "axes and indices lengths must match");
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }
    if (PyArray_DIM(indices_array, 1) != ndim) {
        PyErr_SetString(PyExc_ValueError, "indices second dimension must match shape rank");
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }
    if (PyArray_NDIM(values_array) != 3 ||
        PyArray_DIM(values_array, 0) != nlinks ||
        PyArray_DIM(values_array, 1) != nstates ||
        PyArray_DIM(values_array, 2) != nstates) {
        PyErr_SetString(
            PyExc_ValueError,
            "values must have shape (nlink, nstates, nstates)"
        );
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }

    npy_intp total = 1;
    for (npy_intp dim : shape) {
        total *= dim;
    }

    std::vector<npy_intp> strides(shape.size());
    npy_intp running = 1;
    for (npy_intp axis = ndim - 1; axis >= 0; --axis) {
        strides[static_cast<size_t>(axis)] = running;
        running *= shape[static_cast<size_t>(axis)];
    }

    std::vector<std::vector<npy_intp>> lookup(static_cast<size_t>(ndim));
    for (auto& axis_table : lookup) {
        axis_table.assign(static_cast<size_t>(total), -1);
    }

    const npy_intp* axes = reinterpret_cast<npy_intp*>(PyArray_DATA(axes_array));
    const npy_intp* flat_indices = reinterpret_cast<npy_intp*>(PyArray_DATA(indices_array));

    for (npy_intp link = 0; link < nlinks; ++link) {
        const npy_intp axis = axes[static_cast<size_t>(link)];
        if (axis < 0 || axis >= ndim) {
            PyErr_SetString(PyExc_ValueError, "axis index out of bounds");
            Py_DECREF(axes_array);
            Py_DECREF(indices_array);
            Py_DECREF(values_array);
            return nullptr;
        }

        std::vector<npy_intp> idx(static_cast<size_t>(ndim));
        const npy_intp offset = link * ndim;
        for (npy_intp dim = 0; dim < ndim; ++dim) {
            const npy_intp value = flat_indices[static_cast<size_t>(offset + dim)];
            if (value < 0 || value >= shape[static_cast<size_t>(dim)]) {
                PyErr_SetString(PyExc_ValueError, "link index outside grid shape");
                Py_DECREF(axes_array);
                Py_DECREF(indices_array);
                Py_DECREF(values_array);
                return nullptr;
            }
            idx[static_cast<size_t>(dim)] = value;
        }

        const npy_intp flat = flatten_coords(idx, strides);
        lookup[static_cast<size_t>(axis)][static_cast<size_t>(flat)] = link;
    }

    const Complex* raw_data = reinterpret_cast<Complex*>(PyArray_DATA(values_array));

    npy_intp out_dims[4] = {total, nstates, total, nstates};
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(4, out_dims, NPY_COMPLEX128, 0)
    );
    if (output == nullptr) {
        Py_DECREF(axes_array);
        Py_DECREF(indices_array);
        Py_DECREF(values_array);
        return nullptr;
    }
    Complex* out = reinterpret_cast<Complex*>(PyArray_DATA(output));

    auto out_index = [&](npy_intp i, int a, npy_intp j, int b) -> Complex& {
        const size_t base =
            (static_cast<size_t>(i) * static_cast<size_t>(nstates) + static_cast<size_t>(a))
            * static_cast<size_t>(total) * static_cast<size_t>(nstates)
            + static_cast<size_t>(j) * static_cast<size_t>(nstates)
            + static_cast<size_t>(b);
        return out[base];
    };

    std::vector<std::vector<npy_intp>> coords(static_cast<size_t>(total));
    for (npy_intp flat = 0; flat < total; ++flat) {
        npy_intp rem = flat;
        std::vector<npy_intp> point(static_cast<size_t>(ndim));
        for (size_t axis = 0; axis < static_cast<size_t>(ndim); ++axis) {
            point[axis] = rem / strides[axis];
            rem -= point[axis] * strides[axis];
        }
        coords[static_cast<size_t>(flat)] = std::move(point);

        for (int a = 0; a < nstates; ++a) {
            out_index(flat, a, flat, a) = Complex(1.0, 0.0);
        }
    }

    std::vector<Complex> working(static_cast<size_t>(nstates) * nstates);
    std::vector<Complex> scratch(static_cast<size_t>(nstates) * nstates);

    for (npy_intp i = 0; i < total; ++i) {
        const auto& i_coords = coords[static_cast<size_t>(i)];
        for (npy_intp j = i + 1; j < total; ++j) {
            const auto& j_coords = coords[static_cast<size_t>(j)];

            for (int a = 0; a < nstates; ++a) {
                for (int b = 0; b < nstates; ++b) {
                    working[static_cast<size_t>(a) * nstates + b] =
                        (a == b ? Complex(1.0, 0.0) : Complex(0.0, 0.0));
                }
            }

            npy_intp flat = i;
            std::vector<npy_intp> current = i_coords;
            for (size_t axis = 0; axis < static_cast<size_t>(ndim); ++axis) {
                const npy_intp target = j_coords[axis];
                while (current[axis] < target) {
                    const npy_intp link_index = lookup[axis][static_cast<size_t>(flat)];
                    if (link_index < 0) {
                        PyErr_SetString(PyExc_RuntimeError, "missing forward link");
                        Py_DECREF(axes_array);
                        Py_DECREF(indices_array);
                        Py_DECREF(values_array);
                        Py_DECREF(output);
                        return nullptr;
                    }
                    const Complex* link = raw_data
                        + static_cast<size_t>(link_index) * static_cast<size_t>(nstates) * nstates;
                    compose_on_right(working.data(), link, scratch.data(), nstates, false);
                    working.swap(scratch);
                    current[axis] += 1;
                    flat += strides[axis];
                }
                while (current[axis] > target) {
                    current[axis] -= 1;
                    flat -= strides[axis];
                    const npy_intp link_index = lookup[axis][static_cast<size_t>(flat)];
                    if (link_index < 0) {
                        PyErr_SetString(PyExc_RuntimeError, "missing backward link");
                        Py_DECREF(axes_array);
                        Py_DECREF(indices_array);
                        Py_DECREF(values_array);
                        Py_DECREF(output);
                        return nullptr;
                    }
                    const Complex* link = raw_data
                        + static_cast<size_t>(link_index) * static_cast<size_t>(nstates) * nstates;
                    compose_on_right(working.data(), link, scratch.data(), nstates, true);
                    working.swap(scratch);
                }
            }

            for (int a = 0; a < nstates; ++a) {
                for (int b = 0; b < nstates; ++b) {
                    const Complex value = working[static_cast<size_t>(a) * nstates + b];
                    out_index(i, a, j, b) = value;
                    out_index(j, b, i, a) = std::conj(value);
                }
            }
        }
    }

    Py_DECREF(axes_array);
    Py_DECREF(indices_array);
    Py_DECREF(values_array);
    return reinterpret_cast<PyObject*>(output);
}

static PyMethodDef methods[] = {
    {
        "linked_overlap_dense",
        reinterpret_cast<PyCFunction>(linked_overlap_dense),
        METH_VARARGS | METH_KEYWORDS,
        "Build linked-overlap tensor for a product grid.",
    },
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_kernels_cpp",
    "Native LDR kernels",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__kernels_cpp(void) {
    import_array();
    return PyModule_Create(&module_def);
}
