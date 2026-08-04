#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <complex>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

namespace {

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

PyArrayObject* as_input_array(PyObject* object, int typenum = NPY_NOTYPE) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, typenum, NPY_ARRAY_IN_ARRAY)
    );
}

class UnionFind {
public:
    explicit UnionFind(std::size_t size) : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int value) {
        int root = value;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        while (parent_[value] != value) {
            const int next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void unite(int left, int right) {
        left = find(left);
        right = find(right);
        if (left == right) {
            return;
        }
        if (rank_[left] < rank_[right]) {
            std::swap(left, right);
        }
        parent_[right] = left;
        if (rank_[left] == rank_[right]) {
            ++rank_[left];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<unsigned char> rank_;
};

bool checked_product(
    const std::vector<int>& classes,
    const std::vector<npy_intp>& dimensions,
    npy_intp& result
) {
    result = 1;
    for (const int value : classes) {
        const npy_intp dimension = dimensions[value];
        if (
            dimension < 0 ||
            (dimension != 0 &&
             result > std::numeric_limits<npy_intp>::max() / dimension)
        ) {
            return false;
        }
        result *= dimension;
    }
    return true;
}

void build_offsets(
    const std::vector<int>& classes,
    const std::vector<npy_intp>& dimensions,
    const std::vector<npy_intp>& left_strides,
    const std::vector<npy_intp>& right_strides,
    const std::vector<npy_intp>& operator_strides,
    std::vector<npy_intp>& left_offsets,
    std::vector<npy_intp>& right_offsets,
    std::vector<npy_intp>& operator_offsets
) {
    const npy_intp count = static_cast<npy_intp>(left_offsets.size());
    std::vector<npy_intp> values(classes.size(), 0);
    npy_intp left = 0;
    npy_intp right = 0;
    npy_intp local_operator = 0;
    for (npy_intp linear = 0; linear < count; ++linear) {
        left_offsets[linear] = left;
        right_offsets[linear] = right;
        operator_offsets[linear] = local_operator;
        for (std::size_t reverse = classes.size(); reverse > 0; --reverse) {
            const std::size_t axis = reverse - 1;
            const int cls = classes[axis];
            ++values[axis];
            left += left_strides[cls];
            right += right_strides[cls];
            local_operator += operator_strides[cls];
            if (values[axis] < dimensions[cls]) {
                break;
            }
            values[axis] = 0;
            left -= dimensions[cls] * left_strides[cls];
            right -= dimensions[cls] * right_strides[cls];
            local_operator -= dimensions[cls] * operator_strides[cls];
        }
    }
}

template <typename scalar_t>
inline bool is_zero(const scalar_t& value) {
    return value == scalar_t{};
}

template <typename scalar_t>
void contract_impl(
    const scalar_t* left,
    const scalar_t* right,
    const scalar_t* local_operator,
    scalar_t* output,
    const std::vector<int>& output_classes,
    const std::vector<npy_intp>& dimensions,
    const std::vector<npy_intp>& left_strides,
    const std::vector<npy_intp>& right_strides,
    const std::vector<npy_intp>& operator_strides,
    const std::vector<npy_intp>& output_strides,
    const std::vector<npy_intp>& contracted_left,
    const std::vector<npy_intp>& contracted_right,
    const std::vector<npy_intp>& contracted_operator,
    std::vector<npy_intp>& values,
    npy_intp output_count
) {
    std::fill(values.begin(), values.end(), 0);
    npy_intp left_base = 0;
    npy_intp right_base = 0;
    npy_intp operator_base = 0;
    npy_intp output_base = 0;
    const npy_intp contracted_count =
        static_cast<npy_intp>(contracted_left.size());

    for (npy_intp outer = 0; outer < output_count; ++outer) {
        scalar_t accumulated{};
        for (npy_intp inner = 0; inner < contracted_count; ++inner) {
            const scalar_t operator_value =
                local_operator[operator_base + contracted_operator[inner]];
            if (is_zero(operator_value)) {
                continue;
            }
            accumulated +=
                (left[left_base + contracted_left[inner]]
                 * right[right_base + contracted_right[inner]])
                * operator_value;
        }
        output[output_base] += accumulated;

        for (
            std::size_t reverse = output_classes.size();
            reverse > 0;
            --reverse
        ) {
            const std::size_t axis = reverse - 1;
            const int cls = output_classes[axis];
            ++values[axis];
            left_base += left_strides[cls];
            right_base += right_strides[cls];
            operator_base += operator_strides[cls];
            output_base += output_strides[cls];
            if (values[axis] < dimensions[cls]) {
                break;
            }
            values[axis] = 0;
            left_base -= dimensions[cls] * left_strides[cls];
            right_base -= dimensions[cls] * right_strides[cls];
            operator_base -= dimensions[cls] * operator_strides[cls];
            output_base -= dimensions[cls] * output_strides[cls];
        }
    }
}

PyObject* contract_copy_einsum(PyObject*, PyObject* args) {
    PyObject* left_object = nullptr;
    PyObject* right_object = nullptr;
    PyObject* operator_object = nullptr;
    PyObject* left_labels_object = nullptr;
    PyObject* right_labels_object = nullptr;
    PyObject* operator_labels_object = nullptr;
    PyObject* output_labels_object = nullptr;
    PyObject* copy_labels_object = nullptr;
    PyObject* copy_dimensions_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOO",
            &left_object,
            &right_object,
            &operator_object,
            &left_labels_object,
            &right_labels_object,
            &operator_labels_object,
            &output_labels_object,
            &copy_labels_object,
            &copy_dimensions_object
        )) {
        return nullptr;
    }

    ArrayHandle left(as_input_array(left_object));
    ArrayHandle right(as_input_array(right_object));
    ArrayHandle local_operator(as_input_array(operator_object));
    ArrayHandle left_labels(as_input_array(left_labels_object, NPY_INTP));
    ArrayHandle right_labels(as_input_array(right_labels_object, NPY_INTP));
    ArrayHandle operator_labels(
        as_input_array(operator_labels_object, NPY_INTP)
    );
    ArrayHandle output_labels(as_input_array(output_labels_object, NPY_INTP));
    ArrayHandle copy_labels(as_input_array(copy_labels_object, NPY_INTP));
    ArrayHandle copy_dimensions(
        as_input_array(copy_dimensions_object, NPY_INTP)
    );
    if (
        left.get() == nullptr || right.get() == nullptr ||
        local_operator.get() == nullptr || left_labels.get() == nullptr ||
        right_labels.get() == nullptr || operator_labels.get() == nullptr ||
        output_labels.get() == nullptr || copy_labels.get() == nullptr ||
        copy_dimensions.get() == nullptr
    ) {
        return nullptr;
    }

    auto fail = [](const char* message) -> PyObject* {
        PyErr_SetString(PyExc_ValueError, message);
        return nullptr;
    };

    const int typenum = PyArray_TYPE(left.get());
    if (
        (typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE) ||
        PyArray_TYPE(right.get()) != typenum ||
        PyArray_TYPE(local_operator.get()) != typenum
    ) {
        return fail(
            "left, right, and operator must share float64 or complex128 dtype."
        );
    }
    if (
        PyArray_NDIM(left_labels.get()) != 1 ||
        PyArray_NDIM(right_labels.get()) != 1 ||
        PyArray_NDIM(operator_labels.get()) != 1 ||
        PyArray_NDIM(output_labels.get()) != 1 ||
        PyArray_NDIM(copy_labels.get()) != 2 ||
        PyArray_NDIM(copy_dimensions.get()) != 1 ||
        PyArray_DIM(copy_labels.get(), 1) != 3
    ) {
        return fail("copy-einsum label metadata has incompatible ranks.");
    }
    if (
        PyArray_DIM(left_labels.get(), 0) != PyArray_NDIM(left.get()) ||
        PyArray_DIM(right_labels.get(), 0) != PyArray_NDIM(right.get()) ||
        PyArray_DIM(operator_labels.get(), 0) !=
            PyArray_NDIM(local_operator.get()) ||
        PyArray_DIM(copy_labels.get(), 0) !=
            PyArray_DIM(copy_dimensions.get(), 0)
    ) {
        return fail("copy-einsum labels do not match their operand ranks.");
    }

    const auto* left_label_data = static_cast<const npy_intp*>(
        PyArray_DATA(left_labels.get())
    );
    const auto* right_label_data = static_cast<const npy_intp*>(
        PyArray_DATA(right_labels.get())
    );
    const auto* operator_label_data = static_cast<const npy_intp*>(
        PyArray_DATA(operator_labels.get())
    );
    const auto* output_label_data = static_cast<const npy_intp*>(
        PyArray_DATA(output_labels.get())
    );
    const auto* copy_label_data = static_cast<const npy_intp*>(
        PyArray_DATA(copy_labels.get())
    );
    const auto* copy_dimension_data = static_cast<const npy_intp*>(
        PyArray_DATA(copy_dimensions.get())
    );

    std::vector<npy_intp> labels;
    const auto append_labels = [&labels](const npy_intp* values, npy_intp size) {
        labels.insert(labels.end(), values, values + size);
    };
    append_labels(
        left_label_data,
        PyArray_DIM(left_labels.get(), 0)
    );
    append_labels(
        right_label_data,
        PyArray_DIM(right_labels.get(), 0)
    );
    append_labels(
        operator_label_data,
        PyArray_DIM(operator_labels.get(), 0)
    );
    append_labels(
        output_label_data,
        PyArray_DIM(output_labels.get(), 0)
    );
    append_labels(
        copy_label_data,
        PyArray_SIZE(copy_labels.get())
    );
    if (labels.empty()) {
        return fail("copy-einsum requires at least one label.");
    }
    if (std::any_of(labels.begin(), labels.end(), [](npy_intp value) {
            return value < 0;
        })) {
        return fail("copy-einsum labels must be nonnegative.");
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

    const auto label_id = [&labels](npy_intp label) {
        return static_cast<int>(
            std::lower_bound(labels.begin(), labels.end(), label) -
            labels.begin()
        );
    };
    const int nlabels = static_cast<int>(labels.size());
    std::vector<npy_intp> label_dimensions(nlabels, -1);
    const auto assign_dimensions = [&](
        PyArrayObject* array,
        const npy_intp* axis_labels
    ) -> bool {
        for (int axis = 0; axis < PyArray_NDIM(array); ++axis) {
            const int label = label_id(axis_labels[axis]);
            const npy_intp dimension = PyArray_DIM(array, axis);
            if (
                label_dimensions[label] >= 0 &&
                label_dimensions[label] != dimension
            ) {
                return false;
            }
            label_dimensions[label] = dimension;
        }
        return true;
    };
    if (
        !assign_dimensions(left.get(), left_label_data) ||
        !assign_dimensions(right.get(), right_label_data) ||
        !assign_dimensions(local_operator.get(), operator_label_data)
    ) {
        return fail("a repeated copy-einsum label has inconsistent dimensions.");
    }

    UnionFind unions(labels.size());
    const npy_intp ncopies = PyArray_DIM(copy_labels.get(), 0);
    for (npy_intp copy = 0; copy < ncopies; ++copy) {
        const npy_intp dimension = copy_dimension_data[copy];
        if (dimension < 1) {
            return fail("copy dimensions must be positive.");
        }
        const int first = label_id(copy_label_data[3 * copy]);
        const int second = label_id(copy_label_data[3 * copy + 1]);
        const int third = label_id(copy_label_data[3 * copy + 2]);
        for (const int label : {first, second, third}) {
            if (
                label_dimensions[label] >= 0 &&
                label_dimensions[label] != dimension
            ) {
                return fail(
                    "a copy constraint has an inconsistent label dimension."
                );
            }
            label_dimensions[label] = dimension;
        }
        unions.unite(first, second);
        unions.unite(first, third);
    }
    if (std::any_of(
            label_dimensions.begin(),
            label_dimensions.end(),
            [](npy_intp value) { return value < 1; }
        )) {
        return fail("every copy-einsum label must have a known dimension.");
    }

    std::vector<int> roots(nlabels);
    std::vector<int> root_to_class(nlabels, -1);
    int nclasses = 0;
    for (int label = 0; label < nlabels; ++label) {
        roots[label] = unions.find(label);
        if (root_to_class[roots[label]] < 0) {
            root_to_class[roots[label]] = nclasses++;
        }
    }
    std::vector<int> label_classes(nlabels);
    std::vector<npy_intp> class_dimensions(nclasses, -1);
    for (int label = 0; label < nlabels; ++label) {
        const int cls = root_to_class[roots[label]];
        label_classes[label] = cls;
        const npy_intp dimension = label_dimensions[label];
        if (
            class_dimensions[cls] >= 0 &&
            class_dimensions[cls] != dimension
        ) {
            return fail(
                "copy-connected labels must have identical dimensions."
            );
        }
        class_dimensions[cls] = dimension;
    }

    std::vector<npy_intp> left_strides(nclasses, 0);
    std::vector<npy_intp> right_strides(nclasses, 0);
    std::vector<npy_intp> operator_strides(nclasses, 0);
    const auto collect_strides = [&](
        PyArrayObject* array,
        const npy_intp* axis_labels,
        std::vector<npy_intp>& strides
    ) {
        const npy_intp itemsize = PyArray_ITEMSIZE(array);
        for (int axis = 0; axis < PyArray_NDIM(array); ++axis) {
            const int label = label_id(axis_labels[axis]);
            const int cls = label_classes[label];
            strides[cls] += PyArray_STRIDE(array, axis) / itemsize;
        }
    };
    collect_strides(left.get(), left_label_data, left_strides);
    collect_strides(right.get(), right_label_data, right_strides);
    collect_strides(
        local_operator.get(),
        operator_label_data,
        operator_strides
    );

    const npy_intp output_rank = PyArray_DIM(output_labels.get(), 0);
    std::vector<npy_intp> output_dimensions(output_rank);
    std::vector<int> output_classes;
    std::vector<bool> class_is_output(nclasses, false);
    for (npy_intp axis = 0; axis < output_rank; ++axis) {
        const int label = label_id(output_label_data[axis]);
        const int cls = label_classes[label];
        output_dimensions[axis] = label_dimensions[label];
        if (!class_is_output[cls]) {
            class_is_output[cls] = true;
            output_classes.push_back(cls);
        }
    }
    ArrayHandle output(
        reinterpret_cast<PyArrayObject*>(
            PyArray_ZEROS(
                static_cast<int>(output_rank),
                output_dimensions.data(),
                typenum,
                0
            )
        )
    );
    if (output.get() == nullptr) {
        return nullptr;
    }
    std::vector<npy_intp> output_strides(nclasses, 0);
    const npy_intp output_itemsize = PyArray_ITEMSIZE(output.get());
    for (npy_intp axis = 0; axis < output_rank; ++axis) {
        const int label = label_id(output_label_data[axis]);
        const int cls = label_classes[label];
        output_strides[cls] +=
            PyArray_STRIDE(output.get(), axis) / output_itemsize;
    }

    std::sort(
        output_classes.begin(),
        output_classes.end(),
        [&output_strides](int left_class, int right_class) {
            return output_strides[left_class] > output_strides[right_class];
        }
    );
    std::vector<int> contracted_classes;
    for (int cls = 0; cls < nclasses; ++cls) {
        if (!class_is_output[cls]) {
            contracted_classes.push_back(cls);
        }
    }
    const auto contraction_stride_score = [&](
        int cls
    ) -> npy_intp {
        return std::max(
            {
                left_strides[cls],
                right_strides[cls],
                operator_strides[cls],
            }
        );
    };
    std::sort(
        contracted_classes.begin(),
        contracted_classes.end(),
        [&contraction_stride_score](int left_class, int right_class) {
            return contraction_stride_score(left_class) >
                contraction_stride_score(right_class);
        }
    );

    npy_intp output_count = 0;
    npy_intp contracted_count = 0;
    if (
        !checked_product(output_classes, class_dimensions, output_count) ||
        !checked_product(
            contracted_classes,
            class_dimensions,
            contracted_count
        )
    ) {
        PyErr_SetString(PyExc_OverflowError, "copy-einsum iteration space overflow.");
        return nullptr;
    }
    constexpr std::size_t maximum_workspace_bytes =
        static_cast<std::size_t>(512) * 1024 * 1024;
    if (
        static_cast<std::size_t>(contracted_count) >
        maximum_workspace_bytes / (3 * sizeof(npy_intp))
    ) {
        PyErr_SetString(
            PyExc_MemoryError,
            "copy-einsum contracted-offset workspace exceeds 512 MiB."
        );
        return nullptr;
    }
    std::vector<npy_intp> contracted_left;
    std::vector<npy_intp> contracted_right;
    std::vector<npy_intp> contracted_operator;
    try {
        contracted_left.resize(contracted_count);
        contracted_right.resize(contracted_count);
        contracted_operator.resize(contracted_count);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }
    build_offsets(
        contracted_classes,
        class_dimensions,
        left_strides,
        right_strides,
        operator_strides,
        contracted_left,
        contracted_right,
        contracted_operator
    );
    std::vector<npy_intp> output_values;
    try {
        output_values.resize(output_classes.size());
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }

    if (typenum == NPY_DOUBLE) {
        Py_BEGIN_ALLOW_THREADS
        contract_impl(
            static_cast<const double*>(PyArray_DATA(left.get())),
            static_cast<const double*>(PyArray_DATA(right.get())),
            static_cast<const double*>(PyArray_DATA(local_operator.get())),
            static_cast<double*>(PyArray_DATA(output.get())),
            output_classes,
            class_dimensions,
            left_strides,
            right_strides,
            operator_strides,
            output_strides,
            contracted_left,
            contracted_right,
            contracted_operator,
            output_values,
            output_count
        );
        Py_END_ALLOW_THREADS
    } else {
        using complex_t = std::complex<double>;
        Py_BEGIN_ALLOW_THREADS
        contract_impl(
            reinterpret_cast<const complex_t*>(PyArray_DATA(left.get())),
            reinterpret_cast<const complex_t*>(PyArray_DATA(right.get())),
            reinterpret_cast<const complex_t*>(
                PyArray_DATA(local_operator.get())
            ),
            reinterpret_cast<complex_t*>(PyArray_DATA(output.get())),
            output_classes,
            class_dimensions,
            left_strides,
            right_strides,
            operator_strides,
            output_strides,
            contracted_left,
            contracted_right,
            contracted_operator,
            output_values,
            output_count
        );
        Py_END_ALLOW_THREADS
    }
    return reinterpret_cast<PyObject*>(output.release());
}

template <typename scalar_t>
void contract_class_impl(
    const std::vector<const scalar_t*>& operands,
    scalar_t* output,
    const std::vector<int>& output_classes,
    const std::vector<npy_intp>& dimensions,
    const std::vector<std::vector<npy_intp>>& operand_strides,
    const std::vector<npy_intp>& output_strides,
    const std::vector<std::vector<npy_intp>>& contracted_offsets,
    std::vector<npy_intp>& values,
    std::vector<npy_intp>& bases,
    npy_intp output_count
) {
    const std::size_t noperands = operands.size();
    const npy_intp contracted_count = static_cast<npy_intp>(
        contracted_offsets.front().size()
    );
    std::fill(values.begin(), values.end(), 0);
    std::fill(bases.begin(), bases.end(), 0);
    npy_intp output_base = 0;

    for (npy_intp outer = 0; outer < output_count; ++outer) {
        scalar_t accumulated{};
        if (noperands == 2) {
            for (npy_intp inner = 0; inner < contracted_count; ++inner) {
                const scalar_t last =
                    operands[1][bases[1] + contracted_offsets[1][inner]];
                if (!is_zero(last)) {
                    accumulated +=
                        operands[0][bases[0] + contracted_offsets[0][inner]]
                        * last;
                }
            }
        } else if (noperands == 4) {
            for (npy_intp inner = 0; inner < contracted_count; ++inner) {
                const scalar_t last =
                    operands[3][bases[3] + contracted_offsets[3][inner]];
                if (!is_zero(last)) {
                    accumulated +=
                        (
                            (
                                operands[0][
                                    bases[0] + contracted_offsets[0][inner]
                                ]
                                * operands[1][
                                    bases[1] + contracted_offsets[1][inner]
                                ]
                            )
                            * operands[2][
                                bases[2] + contracted_offsets[2][inner]
                            ]
                        )
                        * last;
                }
            }
        } else {
            for (npy_intp inner = 0; inner < contracted_count; ++inner) {
                scalar_t term{1};
                for (std::size_t operand = 0; operand < noperands; ++operand) {
                    term *= operands[operand][
                        bases[operand] + contracted_offsets[operand][inner]
                    ];
                    if (is_zero(term)) {
                        break;
                    }
                }
                accumulated += term;
            }
        }
        output[output_base] += accumulated;

        for (
            std::size_t reverse = output_classes.size();
            reverse > 0;
            --reverse
        ) {
            const std::size_t axis = reverse - 1;
            const int cls = output_classes[axis];
            ++values[axis];
            for (std::size_t operand = 0; operand < noperands; ++operand) {
                bases[operand] += operand_strides[operand][cls];
            }
            output_base += output_strides[cls];
            if (values[axis] < dimensions[cls]) {
                break;
            }
            values[axis] = 0;
            for (std::size_t operand = 0; operand < noperands; ++operand) {
                bases[operand] -=
                    dimensions[cls] * operand_strides[operand][cls];
            }
            output_base -= dimensions[cls] * output_strides[cls];
        }
    }
}

PyObject* contract_class_einsum(PyObject*, PyObject* args) {
    PyObject* operands_object = nullptr;
    PyObject* operand_classes_object = nullptr;
    PyObject* output_classes_object = nullptr;
    PyObject* class_dimensions_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOOO",
            &operands_object,
            &operand_classes_object,
            &output_classes_object,
            &class_dimensions_object
        )) {
        return nullptr;
    }

    PyObject* operand_sequence = PySequence_Fast(
        operands_object,
        "operands must be a sequence."
    );
    if (operand_sequence == nullptr) {
        return nullptr;
    }
    PyObject* class_sequence = PySequence_Fast(
        operand_classes_object,
        "operand_classes must be a sequence."
    );
    if (class_sequence == nullptr) {
        Py_DECREF(operand_sequence);
        return nullptr;
    }
    const Py_ssize_t noperands = PySequence_Fast_GET_SIZE(operand_sequence);
    if (
        noperands < 1 ||
        PySequence_Fast_GET_SIZE(class_sequence) != noperands
    ) {
        Py_DECREF(operand_sequence);
        Py_DECREF(class_sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "operands and operand_classes must have equal nonzero lengths."
        );
        return nullptr;
    }

    std::vector<ArrayHandle> operands;
    std::vector<ArrayHandle> operand_classes;
    operands.reserve(static_cast<std::size_t>(noperands));
    operand_classes.reserve(static_cast<std::size_t>(noperands));
    for (Py_ssize_t index = 0; index < noperands; ++index) {
        operands.emplace_back(
            as_input_array(PySequence_Fast_GET_ITEM(operand_sequence, index))
        );
        operand_classes.emplace_back(
            as_input_array(
                PySequence_Fast_GET_ITEM(class_sequence, index),
                NPY_INTP
            )
        );
        if (
            operands.back().get() == nullptr ||
            operand_classes.back().get() == nullptr
        ) {
            Py_DECREF(operand_sequence);
            Py_DECREF(class_sequence);
            return nullptr;
        }
    }
    Py_DECREF(operand_sequence);
    Py_DECREF(class_sequence);
    ArrayHandle output_class_array(
        as_input_array(output_classes_object, NPY_INTP)
    );
    ArrayHandle class_dimension_array(
        as_input_array(class_dimensions_object, NPY_INTP)
    );
    if (
        output_class_array.get() == nullptr ||
        class_dimension_array.get() == nullptr
    ) {
        return nullptr;
    }

    const int typenum = PyArray_TYPE(operands.front().get());
    if (typenum != NPY_DOUBLE && typenum != NPY_CDOUBLE) {
        PyErr_SetString(
            PyExc_TypeError,
            "class-einsum operands must be float64 or complex128."
        );
        return nullptr;
    }
    if (
        PyArray_NDIM(output_class_array.get()) != 1 ||
        PyArray_NDIM(class_dimension_array.get()) != 1
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "class-einsum output classes and dimensions must be vectors."
        );
        return nullptr;
    }
    const npy_intp nclasses = PyArray_DIM(class_dimension_array.get(), 0);
    if (nclasses < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "class-einsum requires at least one logical class."
        );
        return nullptr;
    }
    const auto* class_dimensions = static_cast<const npy_intp*>(
        PyArray_DATA(class_dimension_array.get())
    );
    if (std::any_of(
            class_dimensions,
            class_dimensions + nclasses,
            [](npy_intp value) { return value < 1; }
        )) {
        PyErr_SetString(
            PyExc_ValueError,
            "class-einsum dimensions must be positive."
        );
        return nullptr;
    }

    std::vector<std::vector<npy_intp>> operand_strides(
        static_cast<std::size_t>(noperands),
        std::vector<npy_intp>(nclasses, 0)
    );
    for (Py_ssize_t operand = 0; operand < noperands; ++operand) {
        PyArrayObject* array = operands[operand].get();
        PyArrayObject* classes = operand_classes[operand].get();
        if (
            PyArray_TYPE(array) != typenum ||
            PyArray_NDIM(classes) != 1 ||
            PyArray_DIM(classes, 0) != PyArray_NDIM(array)
        ) {
            PyErr_SetString(
                PyExc_ValueError,
                "class-einsum operands must share dtype and match class ranks."
            );
            return nullptr;
        }
        const auto* values = static_cast<const npy_intp*>(
            PyArray_DATA(classes)
        );
        const npy_intp itemsize = PyArray_ITEMSIZE(array);
        for (int axis = 0; axis < PyArray_NDIM(array); ++axis) {
            const npy_intp cls = values[axis];
            if (
                cls < 0 || cls >= nclasses ||
                PyArray_DIM(array, axis) != class_dimensions[cls]
            ) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "an operand class has an incompatible dimension."
                );
                return nullptr;
            }
            operand_strides[operand][cls] +=
                PyArray_STRIDE(array, axis) / itemsize;
        }
    }

    const npy_intp output_rank = PyArray_DIM(output_class_array.get(), 0);
    const auto* output_class_values = static_cast<const npy_intp*>(
        PyArray_DATA(output_class_array.get())
    );
    std::vector<npy_intp> output_dimensions(output_rank);
    std::vector<int> unique_output_classes;
    std::vector<bool> class_is_output(nclasses, false);
    for (npy_intp axis = 0; axis < output_rank; ++axis) {
        const npy_intp cls = output_class_values[axis];
        if (cls < 0 || cls >= nclasses) {
            PyErr_SetString(
                PyExc_ValueError,
                "an output class is out of range."
            );
            return nullptr;
        }
        output_dimensions[axis] = class_dimensions[cls];
        if (!class_is_output[cls]) {
            class_is_output[cls] = true;
            unique_output_classes.push_back(static_cast<int>(cls));
        }
    }
    ArrayHandle output(
        reinterpret_cast<PyArrayObject*>(
            PyArray_ZEROS(
                static_cast<int>(output_rank),
                output_dimensions.data(),
                typenum,
                0
            )
        )
    );
    if (output.get() == nullptr) {
        return nullptr;
    }
    std::vector<npy_intp> output_strides(nclasses, 0);
    const npy_intp output_itemsize = PyArray_ITEMSIZE(output.get());
    for (npy_intp axis = 0; axis < output_rank; ++axis) {
        output_strides[output_class_values[axis]] +=
            PyArray_STRIDE(output.get(), axis) / output_itemsize;
    }
    std::sort(
        unique_output_classes.begin(),
        unique_output_classes.end(),
        [&output_strides](int left_class, int right_class) {
            return output_strides[left_class] > output_strides[right_class];
        }
    );

    std::vector<int> contracted_classes;
    for (npy_intp cls = 0; cls < nclasses; ++cls) {
        if (!class_is_output[cls]) {
            contracted_classes.push_back(static_cast<int>(cls));
        }
    }
    const auto contraction_stride_score = [&operand_strides](
        int cls
    ) -> npy_intp {
        npy_intp score = 0;
        for (const auto& strides : operand_strides) {
            score = std::max(score, strides[cls]);
        }
        return score;
    };
    std::sort(
        contracted_classes.begin(),
        contracted_classes.end(),
        [&contraction_stride_score](int left_class, int right_class) {
            return contraction_stride_score(left_class) >
                contraction_stride_score(right_class);
        }
    );
    const std::vector<npy_intp> dimensions(
        class_dimensions,
        class_dimensions + nclasses
    );
    npy_intp output_count = 0;
    npy_intp contracted_count = 0;
    if (
        !checked_product(
            unique_output_classes,
            dimensions,
            output_count
        ) ||
        !checked_product(
            contracted_classes,
            dimensions,
            contracted_count
        )
    ) {
        PyErr_SetString(
            PyExc_OverflowError,
            "class-einsum iteration space overflow."
        );
        return nullptr;
    }
    constexpr std::size_t maximum_workspace_bytes =
        static_cast<std::size_t>(512) * 1024 * 1024;
    if (
        static_cast<std::size_t>(contracted_count) >
        maximum_workspace_bytes /
            (static_cast<std::size_t>(noperands) * sizeof(npy_intp))
    ) {
        PyErr_SetString(
            PyExc_MemoryError,
            "class-einsum contracted-offset workspace exceeds 512 MiB."
        );
        return nullptr;
    }

    std::vector<std::vector<npy_intp>> contracted_offsets;
    try {
        contracted_offsets.assign(
            static_cast<std::size_t>(noperands),
            std::vector<npy_intp>(contracted_count)
        );
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }
    std::vector<npy_intp> values(contracted_classes.size(), 0);
    std::vector<npy_intp> offsets(static_cast<std::size_t>(noperands), 0);
    for (npy_intp linear = 0; linear < contracted_count; ++linear) {
        for (Py_ssize_t operand = 0; operand < noperands; ++operand) {
            contracted_offsets[operand][linear] = offsets[operand];
        }
        for (
            std::size_t reverse = contracted_classes.size();
            reverse > 0;
            --reverse
        ) {
            const std::size_t axis = reverse - 1;
            const int cls = contracted_classes[axis];
            ++values[axis];
            for (Py_ssize_t operand = 0; operand < noperands; ++operand) {
                offsets[operand] += operand_strides[operand][cls];
            }
            if (values[axis] < dimensions[cls]) {
                break;
            }
            values[axis] = 0;
            for (Py_ssize_t operand = 0; operand < noperands; ++operand) {
                offsets[operand] -=
                    dimensions[cls] * operand_strides[operand][cls];
            }
        }
    }
    std::vector<npy_intp> output_values;
    std::vector<npy_intp> operand_bases;
    try {
        output_values.resize(unique_output_classes.size());
        operand_bases.resize(static_cast<std::size_t>(noperands));
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    }

    if (typenum == NPY_DOUBLE) {
        std::vector<const double*> pointers;
        pointers.reserve(static_cast<std::size_t>(noperands));
        for (const auto& operand : operands) {
            pointers.push_back(
                static_cast<const double*>(PyArray_DATA(operand.get()))
            );
        }
        Py_BEGIN_ALLOW_THREADS
        contract_class_impl(
            pointers,
            static_cast<double*>(PyArray_DATA(output.get())),
            unique_output_classes,
            dimensions,
            operand_strides,
            output_strides,
            contracted_offsets,
            output_values,
            operand_bases,
            output_count
        );
        Py_END_ALLOW_THREADS
    } else {
        using complex_t = std::complex<double>;
        std::vector<const complex_t*> pointers;
        pointers.reserve(static_cast<std::size_t>(noperands));
        for (const auto& operand : operands) {
            pointers.push_back(
                reinterpret_cast<const complex_t*>(
                    PyArray_DATA(operand.get())
                )
            );
        }
        Py_BEGIN_ALLOW_THREADS
        contract_class_impl(
            pointers,
            reinterpret_cast<complex_t*>(PyArray_DATA(output.get())),
            unique_output_classes,
            dimensions,
            operand_strides,
            output_strides,
            contracted_offsets,
            output_values,
            operand_bases,
            output_count
        );
        Py_END_ALLOW_THREADS
    }
    return reinterpret_cast<PyObject*>(output.release());
}

PyMethodDef methods[] = {
    {
        "contract_copy_einsum",
        contract_copy_einsum,
        METH_VARARGS,
        "Contract three dense operands with three-way equality constraints.",
    },
    {
        "contract_class_einsum",
        contract_class_einsum,
        METH_VARARGS,
        "Contract dense operands whose axes reference equality classes.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_copy_einsum_cpp",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__copy_einsum_cpp() {
    import_array();
    return PyModule_Create(&module);
}
