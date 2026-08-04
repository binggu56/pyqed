#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <complex>
#include <vector>

namespace {

using complex_t = std::complex<double>;

class ArrayHandle {
public:
    explicit ArrayHandle(PyArrayObject* value = nullptr) : value_(value) {}
    ~ArrayHandle() { Py_XDECREF(value_); }
    ArrayHandle(const ArrayHandle&) = delete;
    ArrayHandle& operator=(const ArrayHandle&) = delete;
    ArrayHandle(ArrayHandle&& other) noexcept : value_(other.release()) {}
    ArrayHandle& operator=(ArrayHandle&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(value_);
            value_ = other.release();
        }
        return *this;
    }
    PyArrayObject* get() const { return value_; }
    PyArrayObject* release() {
        PyArrayObject* result = value_;
        value_ = nullptr;
        return result;
    }

private:
    PyArrayObject* value_;
};

PyArrayObject* as_array(PyObject* object, int typenum) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, typenum, NPY_ARRAY_IN_ARRAY)
    );
}

struct KernelInputs {
    ArrayHandle coords;
    ArrayHandle left;
    ArrayHandle right;
    ArrayHandle bra_i;
    ArrayHandle ket_i;
    ArrayHandle bra_j;
    ArrayHandle ket_j;
    ArrayHandle entry_starts;
    ArrayHandle entry_m;
    ArrayHandle entry_n;
    ArrayHandle entry_values;
    npy_intp support_size = 0;
    npy_intp transitions = 0;
    npy_intp entries = 0;
    npy_intp di = 0;
    npy_intp dj = 0;
};

bool require_rank(PyArrayObject* array, int rank, const char* name) {
    if (PyArray_NDIM(array) == rank) {
        return true;
    }
    PyErr_Format(PyExc_ValueError, "%s must be %d-dimensional.", name, rank);
    return false;
}

bool parse_kernel_inputs(PyObject* const* objects, KernelInputs& input) {
    input.coords = ArrayHandle(as_array(objects[0], NPY_INT64));
    input.left = ArrayHandle(as_array(objects[1], NPY_CDOUBLE));
    input.right = ArrayHandle(as_array(objects[2], NPY_CDOUBLE));
    input.bra_i = ArrayHandle(as_array(objects[3], NPY_INT64));
    input.ket_i = ArrayHandle(as_array(objects[4], NPY_INT64));
    input.bra_j = ArrayHandle(as_array(objects[5], NPY_INT64));
    input.ket_j = ArrayHandle(as_array(objects[6], NPY_INT64));
    input.entry_starts = ArrayHandle(as_array(objects[7], NPY_INT64));
    input.entry_m = ArrayHandle(as_array(objects[8], NPY_INT64));
    input.entry_n = ArrayHandle(as_array(objects[9], NPY_INT64));
    input.entry_values = ArrayHandle(as_array(objects[10], NPY_CDOUBLE));
    if (
        input.coords.get() == nullptr || input.left.get() == nullptr ||
        input.right.get() == nullptr || input.bra_i.get() == nullptr ||
        input.ket_i.get() == nullptr || input.bra_j.get() == nullptr ||
        input.ket_j.get() == nullptr || input.entry_starts.get() == nullptr ||
        input.entry_m.get() == nullptr || input.entry_n.get() == nullptr ||
        input.entry_values.get() == nullptr
    ) {
        return false;
    }

    if (
        !require_rank(input.coords.get(), 2, "coords") ||
        !require_rank(input.left.get(), 5, "left") ||
        !require_rank(input.right.get(), 5, "right") ||
        !require_rank(input.bra_i.get(), 1, "bra_i") ||
        !require_rank(input.ket_i.get(), 1, "ket_i") ||
        !require_rank(input.bra_j.get(), 1, "bra_j") ||
        !require_rank(input.ket_j.get(), 1, "ket_j") ||
        !require_rank(input.entry_starts.get(), 1, "entry_starts") ||
        !require_rank(input.entry_m.get(), 1, "entry_m") ||
        !require_rank(input.entry_n.get(), 1, "entry_n") ||
        !require_rank(input.entry_values.get(), 1, "entry_values")
    ) {
        return false;
    }
    if (PyArray_DIM(input.coords.get(), 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "coords must have shape (support_size, 4).");
        return false;
    }

    input.support_size = PyArray_DIM(input.coords.get(), 0);
    input.transitions = PyArray_DIM(input.bra_i.get(), 0);
    input.entries = PyArray_DIM(input.entry_m.get(), 0);
    if (
        PyArray_DIM(input.ket_i.get(), 0) != input.transitions ||
        PyArray_DIM(input.bra_j.get(), 0) != input.transitions ||
        PyArray_DIM(input.ket_j.get(), 0) != input.transitions
    ) {
        PyErr_SetString(PyExc_ValueError, "physical transition arrays must have equal lengths.");
        return false;
    }
    if (PyArray_DIM(input.entry_starts.get(), 0) != input.transitions + 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "entry_starts must have one more item than the transitions."
        );
        return false;
    }
    if (
        PyArray_DIM(input.entry_n.get(), 0) != input.entries ||
        PyArray_DIM(input.entry_values.get(), 0) != input.entries
    ) {
        PyErr_SetString(PyExc_ValueError, "compact MPO entry arrays must have equal lengths.");
        return false;
    }

    input.di = PyArray_DIM(input.left.get(), 3);
    input.dj = PyArray_DIM(input.right.get(), 3);
    if (
        input.di != PyArray_DIM(input.left.get(), 4) ||
        input.dj != PyArray_DIM(input.right.get(), 4)
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "environment bra and ket physical dimensions must agree."
        );
        return false;
    }

    const auto* starts = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_starts.get())
    );
    if (starts[0] != 0 || starts[input.transitions] != input.entries) {
        PyErr_SetString(PyExc_ValueError, "entry_starts does not delimit the MPO entries.");
        return false;
    }
    for (npy_intp transition = 0; transition < input.transitions; ++transition) {
        if (starts[transition] > starts[transition + 1]) {
            PyErr_SetString(PyExc_ValueError, "entry_starts must be nondecreasing.");
            return false;
        }
    }

    const auto* coords = static_cast<const npy_int64*>(
        PyArray_DATA(input.coords.get())
    );
    const npy_intp left_limit = std::min(
        PyArray_DIM(input.left.get(), 0),
        PyArray_DIM(input.left.get(), 1)
    );
    const npy_intp right_limit = std::min(
        PyArray_DIM(input.right.get(), 0),
        PyArray_DIM(input.right.get(), 1)
    );
    for (npy_intp position = 0; position < input.support_size; ++position) {
        const auto* coordinate = coords + 4 * position;
        if (
            coordinate[0] < 0 || coordinate[0] >= left_limit ||
            coordinate[1] < 0 || coordinate[1] >= input.di ||
            coordinate[2] < 0 || coordinate[2] >= input.dj ||
            coordinate[3] < 0 || coordinate[3] >= right_limit
        ) {
            PyErr_SetString(PyExc_ValueError, "coords contains an out-of-range coordinate.");
            return false;
        }
    }

    const auto* bra_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_i.get())
    );
    const auto* ket_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_i.get())
    );
    const auto* bra_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_j.get())
    );
    const auto* ket_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_j.get())
    );
    for (npy_intp transition = 0; transition < input.transitions; ++transition) {
        if (
            bra_i[transition] < 0 || bra_i[transition] >= input.di ||
            ket_i[transition] < 0 || ket_i[transition] >= input.di ||
            bra_j[transition] < 0 || bra_j[transition] >= input.dj ||
            ket_j[transition] < 0 || ket_j[transition] >= input.dj
        ) {
            PyErr_SetString(PyExc_ValueError, "a physical transition index is out of range.");
            return false;
        }
    }

    const auto* entry_m = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_m.get())
    );
    const auto* entry_n = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_n.get())
    );
    const npy_intp m_dim = PyArray_DIM(input.left.get(), 2);
    const npy_intp n_dim = PyArray_DIM(input.right.get(), 2);
    for (npy_intp entry = 0; entry < input.entries; ++entry) {
        if (
            entry_m[entry] < 0 || entry_m[entry] >= m_dim ||
            entry_n[entry] < 0 || entry_n[entry] >= n_dim
        ) {
            PyErr_SetString(PyExc_ValueError, "an MPO environment index is out of range.");
            return false;
        }
    }
    return true;
}

std::vector<std::vector<npy_intp>> physical_groups(const KernelInputs& input) {
    std::vector<std::vector<npy_intp>> groups(
        static_cast<std::size_t>(input.di * input.dj)
    );
    const auto* coords = static_cast<const npy_int64*>(
        PyArray_DATA(input.coords.get())
    );
    for (npy_intp position = 0; position < input.support_size; ++position) {
        const auto* coordinate = coords + 4 * position;
        const npy_intp group = coordinate[1] * input.dj + coordinate[2];
        groups[static_cast<std::size_t>(group)].push_back(position);
    }
    return groups;
}

inline npy_intp left_offset(
    const KernelInputs& input,
    npy_intp bra_virtual,
    npy_intp ket_virtual,
    npy_intp mpo,
    npy_intp bra_physical,
    npy_intp ket_physical
) {
    const npy_intp d1 = PyArray_DIM(input.left.get(), 1);
    const npy_intp d2 = PyArray_DIM(input.left.get(), 2);
    const npy_intp d3 = PyArray_DIM(input.left.get(), 3);
    const npy_intp d4 = PyArray_DIM(input.left.get(), 4);
    return ((((bra_virtual * d1 + ket_virtual) * d2 + mpo) * d3 + bra_physical)
            * d4 + ket_physical);
}

inline npy_intp right_offset(
    const KernelInputs& input,
    npy_intp bra_virtual,
    npy_intp ket_virtual,
    npy_intp mpo,
    npy_intp bra_physical,
    npy_intp ket_physical
) {
    const npy_intp d1 = PyArray_DIM(input.right.get(), 1);
    const npy_intp d2 = PyArray_DIM(input.right.get(), 2);
    const npy_intp d3 = PyArray_DIM(input.right.get(), 3);
    const npy_intp d4 = PyArray_DIM(input.right.get(), 4);
    return ((((bra_virtual * d1 + ket_virtual) * d2 + mpo) * d3 + bra_physical)
            * d4 + ket_physical);
}

PyObject* assemble_dense(PyObject*, PyObject* args) {
    PyObject* objects[11] = {};
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOO",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7],
            &objects[8], &objects[9], &objects[10]
        )) {
        return nullptr;
    }

    KernelInputs input;
    if (!parse_kernel_inputs(objects, input)) {
        return nullptr;
    }
    npy_intp dims[2] = {input.support_size, input.support_size};
    ArrayHandle output(reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(2, dims, NPY_CDOUBLE, 0)
    ));
    if (output.get() == nullptr) {
        return nullptr;
    }

    const auto groups = physical_groups(input);
    const auto* coords = static_cast<const npy_int64*>(
        PyArray_DATA(input.coords.get())
    );
    const auto* left = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.left.get())
    );
    const auto* right = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.right.get())
    );
    const auto* bra_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_i.get())
    );
    const auto* ket_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_i.get())
    );
    const auto* bra_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_j.get())
    );
    const auto* ket_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_j.get())
    );
    const auto* starts = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_starts.get())
    );
    const auto* entry_m = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_m.get())
    );
    const auto* entry_n = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_n.get())
    );
    const auto* entry_values = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.entry_values.get())
    );
    auto* result = reinterpret_cast<complex_t*>(PyArray_DATA(output.get()));

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp transition = 0; transition < input.transitions; ++transition) {
        const auto& rows = groups[static_cast<std::size_t>(
            bra_i[transition] * input.dj + bra_j[transition]
        )];
        const auto& columns = groups[static_cast<std::size_t>(
            ket_i[transition] * input.dj + ket_j[transition]
        )];
        for (const npy_intp row : rows) {
            const auto* row_coord = coords + 4 * row;
            for (const npy_intp column : columns) {
                const auto* column_coord = coords + 4 * column;
                complex_t value = 0.0;
                for (
                    npy_intp entry = starts[transition];
                    entry < starts[transition + 1];
                    ++entry
                ) {
                    value += (
                        left[left_offset(
                            input,
                            row_coord[0],
                            column_coord[0],
                            entry_m[entry],
                            bra_i[transition],
                            ket_i[transition]
                        )]
                        * entry_values[entry]
                        * right[right_offset(
                            input,
                            row_coord[3],
                            column_coord[3],
                            entry_n[entry],
                            bra_j[transition],
                            ket_j[transition]
                        )]
                    );
                }
                result[row * input.support_size + column] += value;
            }
        }
    }
    Py_END_ALLOW_THREADS
    return reinterpret_cast<PyObject*>(output.release());
}

PyObject* apply_batched(PyObject*, PyObject* args) {
    PyObject* objects[12] = {};
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOO",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &objects[6], &objects[7],
            &objects[8], &objects[9], &objects[10], &objects[11]
        )) {
        return nullptr;
    }

    KernelInputs input;
    if (!parse_kernel_inputs(objects, input)) {
        return nullptr;
    }
    ArrayHandle vectors(as_array(objects[11], NPY_CDOUBLE));
    if (vectors.get() == nullptr) {
        return nullptr;
    }
    const int vector_rank = PyArray_NDIM(vectors.get());
    if (vector_rank != 1 && vector_rank != 2) {
        PyErr_SetString(
            PyExc_ValueError,
            "vectors must have shape (support_size,) or (support_size, nvec)."
        );
        return nullptr;
    }
    if (PyArray_DIM(vectors.get(), 0) != input.support_size) {
        PyErr_SetString(PyExc_ValueError, "vectors has the wrong support dimension.");
        return nullptr;
    }
    const npy_intp nrhs = vector_rank == 1 ? 1 : PyArray_DIM(vectors.get(), 1);
    npy_intp dims[2] = {input.support_size, nrhs};
    ArrayHandle output(reinterpret_cast<PyArrayObject*>(
        PyArray_ZEROS(vector_rank, dims, NPY_CDOUBLE, 0)
    ));
    if (output.get() == nullptr) {
        return nullptr;
    }

    const auto groups = physical_groups(input);
    const auto* coords = static_cast<const npy_int64*>(
        PyArray_DATA(input.coords.get())
    );
    const auto* left = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.left.get())
    );
    const auto* right = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.right.get())
    );
    const auto* bra_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_i.get())
    );
    const auto* ket_i = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_i.get())
    );
    const auto* bra_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.bra_j.get())
    );
    const auto* ket_j = static_cast<const npy_int64*>(
        PyArray_DATA(input.ket_j.get())
    );
    const auto* starts = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_starts.get())
    );
    const auto* entry_m = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_m.get())
    );
    const auto* entry_n = static_cast<const npy_int64*>(
        PyArray_DATA(input.entry_n.get())
    );
    const auto* entry_values = reinterpret_cast<const complex_t*>(
        PyArray_DATA(input.entry_values.get())
    );
    const auto* x = reinterpret_cast<const complex_t*>(PyArray_DATA(vectors.get()));
    auto* result = reinterpret_cast<complex_t*>(PyArray_DATA(output.get()));

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp transition = 0; transition < input.transitions; ++transition) {
        const auto& rows = groups[static_cast<std::size_t>(
            bra_i[transition] * input.dj + bra_j[transition]
        )];
        const auto& columns = groups[static_cast<std::size_t>(
            ket_i[transition] * input.dj + ket_j[transition]
        )];
        for (const npy_intp row : rows) {
            const auto* row_coord = coords + 4 * row;
            for (const npy_intp column : columns) {
                const auto* column_coord = coords + 4 * column;
                complex_t value = 0.0;
                for (
                    npy_intp entry = starts[transition];
                    entry < starts[transition + 1];
                    ++entry
                ) {
                    value += (
                        left[left_offset(
                            input,
                            row_coord[0],
                            column_coord[0],
                            entry_m[entry],
                            bra_i[transition],
                            ket_i[transition]
                        )]
                        * entry_values[entry]
                        * right[right_offset(
                            input,
                            row_coord[3],
                            column_coord[3],
                            entry_n[entry],
                            bra_j[transition],
                            ket_j[transition]
                        )]
                    );
                }
                const npy_intp row_offset = row * nrhs;
                const npy_intp column_offset = column * nrhs;
                for (npy_intp vector = 0; vector < nrhs; ++vector) {
                    result[row_offset + vector] += value * x[column_offset + vector];
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    return reinterpret_cast<PyObject*>(output.release());
}

PyMethodDef methods[] = {
    {
        "assemble_dense",
        assemble_dense,
        METH_VARARGS,
        "Assemble the dense native-pair LETTA support Hamiltonian."
    },
    {
        "apply_batched",
        apply_batched,
        METH_VARARGS,
        "Apply the support Hamiltonian to one or more column vectors."
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_support_kernels_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__support_kernels_cpp() {
    import_array();
    return PyModule_Create(&module);
}
