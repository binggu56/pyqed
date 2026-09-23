#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyqed::linalg {

using Complex = std::complex<double>;

#ifdef __APPLE__
extern "C" void cblas_dgemm(int, int, int, int, int, int, double,
    const double*, int, const double*, int, double, double*, int);
extern "C" void dsyevd_(char*, char*, int*, double*, int*, double*,
    double*, int*, int*, int*, int*);
extern "C" void dpotrf_(char*, int*, double*, int*, int*);
extern "C" void dtrsm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);
extern "C" void dgeqp3_(int*, int*, double*, int*, int*, double*, double*, int*, int*);
extern "C" void dorgqr_(int*, int*, int*, double*, int*, double*, double*, int*, int*);
extern "C" void zheev_(
    char* jobz,
    char* uplo,
    int* n,
    Complex* a,
    int* lda,
    double* w,
    Complex* work,
    int* lwork,
    double* rwork,
    int* info
);
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
extern "C" void zgesvd_(
    char* jobu,
    char* jobvt,
    int* m,
    int* n,
    Complex* a,
    int* lda,
    double* s,
    Complex* u,
    int* ldu,
    Complex* vt,
    int* ldvt,
    Complex* work,
    int* lwork,
    double* rwork,
    int* info
);
extern "C" void cblas_zgemv(
    const int order,
    const int transpose,
    const int rows,
    const int cols,
    const void* alpha,
    const void* matrix,
    const int leading_dimension,
    const void* vector,
    const int vector_stride,
    const void* beta,
    void* output,
    const int output_stride
);
extern "C" void cblas_dgemv(
    const int order,
    const int transpose,
    const int rows,
    const int cols,
    const double alpha,
    const double* matrix,
    const int leading_dimension,
    const double* vector,
    const int vector_stride,
    const double beta,
    double* output,
    const int output_stride
);
#else
inline void cblas_dgemv(
    const int,
    const int transpose,
    const int rows,
    const int cols,
    const double alpha,
    const double* matrix,
    const int leading_dimension,
    const double* vector,
    const int vector_stride,
    const double beta,
    double* output,
    const int output_stride
) {
    const bool transposed = transpose == 112;
    const int output_size = transposed ? cols : rows;
    const int inner_size = transposed ? rows : cols;
    for (int outer = 0; outer < output_size; ++outer) {
        double value = 0.0;
        for (int inner = 0; inner < inner_size; ++inner) {
            const int row = transposed ? inner : outer;
            const int column = transposed ? outer : inner;
            value += matrix[row * leading_dimension + column]
                * vector[inner * vector_stride];
        }
        output[outer * output_stride] =
            alpha * value + beta * output[outer * output_stride];
    }
}
#endif

struct ComplexThinSVDWorkspace {
    std::vector<Complex> matrix;
    std::vector<Complex> left;
    std::vector<Complex> right;
    std::vector<Complex> work;
    std::vector<double> singular_values;
    std::vector<double> real_work;
    std::uint64_t growths = 0;
};

inline void complex_thin_svd(
    const Complex* matrix,
    std::size_t rows,
    std::size_t cols,
    ComplexThinSVDWorkspace& workspace
) {
    if (
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || cols > static_cast<std::size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::overflow_error("Thin SVD dimensions exceed LAPACK integer range.");
    }
    const std::size_t rank = std::min(rows, cols);
    const std::size_t matrix_elements = rows * cols;
    bool grew = false;
    auto reserve_at_least = [&grew](auto& values, std::size_t size) {
        if (values.capacity() < size) {
            values.reserve(size);
            grew = true;
        }
        values.resize(size);
    };
    reserve_at_least(workspace.matrix, matrix_elements);
    reserve_at_least(workspace.left, rows * rank);
    reserve_at_least(workspace.right, rank * cols);
    reserve_at_least(workspace.singular_values, rank);
    reserve_at_least(workspace.real_work, std::max<std::size_t>(1, 5 * rank));
    if (grew) {
        ++workspace.growths;
    }
    if (rows == 0 || cols == 0) {
        return;
    }
#ifdef __APPLE__
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(rank);
    const int lda = std::max(1, m);
    const int ldu = std::max(1, m);
    const int ldvt = std::max(1, k);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            workspace.matrix[row + col * rows] = matrix[row * cols + col];
        }
    }
    char jobu = 'S';
    char jobvt = 'S';
    int m_arg = m;
    int n_arg = n;
    int lda_arg = lda;
    int ldu_arg = ldu;
    int ldvt_arg = ldvt;
    int info = 0;
    int lwork = -1;
    Complex work_query = 0.0;
    zgesvd_(
        &jobu,
        &jobvt,
        &m_arg,
        &n_arg,
        workspace.matrix.data(),
        &lda_arg,
        workspace.singular_values.data(),
        workspace.left.data(),
        &ldu_arg,
        workspace.right.data(),
        &ldvt_arg,
        &work_query,
        &lwork,
        workspace.real_work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error("Thin SVD workspace query failed.");
    }
    const std::size_t required_work = static_cast<std::size_t>(
        std::max(1, static_cast<int>(std::real(work_query)))
    );
    if (workspace.work.capacity() < required_work) {
        workspace.work.reserve(required_work);
        ++workspace.growths;
    }
    workspace.work.resize(required_work);
    lwork = static_cast<int>(required_work);
    zgesvd_(
        &jobu,
        &jobvt,
        &m_arg,
        &n_arg,
        workspace.matrix.data(),
        &lda_arg,
        workspace.singular_values.data(),
        workspace.left.data(),
        &ldu_arg,
        workspace.right.data(),
        &ldvt_arg,
        workspace.work.data(),
        &lwork,
        workspace.real_work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error("Thin SVD failed to converge.");
    }
#else
    (void)matrix;
    throw std::runtime_error(
        "Thin SVD currently requires an Accelerate LAPACK build."
    );
#endif
}

inline double norm2(const std::vector<Complex>& vector) {
    double value = 0.0;
    for (const Complex& element : vector) {
        value += std::norm(element);
    }
    return std::sqrt(value);
}

inline Complex dotc(
    const std::vector<Complex>& left,
    const std::vector<Complex>& right
) {
    Complex value = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        value += std::conj(left[index]) * right[index];
    }
    return value;
}

inline void axpy(
    std::vector<Complex>& target,
    Complex scale,
    const std::vector<Complex>& source
) {
    for (std::size_t index = 0; index < target.size(); ++index) {
        target[index] += scale * source[index];
    }
}

inline void normalize_phase(std::vector<Complex>& vector) {
    std::size_t pivot = 0;
    double magnitude = 0.0;
    for (std::size_t index = 0; index < vector.size(); ++index) {
        const double candidate = std::abs(vector[index]);
        if (candidate > magnitude) {
            magnitude = candidate;
            pivot = index;
        }
    }
    if (magnitude == 0.0) {
        return;
    }
    const Complex phase = vector[pivot] / magnitude;
    for (Complex& value : vector) {
        value /= phase;
    }
}

struct LowestEigenpair {
    double value = 0.0;
    std::vector<Complex> vector;
};

struct RealLowestEigenpair {
    double value = 0.0;
    std::vector<double> vector;
};

struct RealSymmetricEigendecomposition {
    std::vector<double> values;
    // Row-major matrix whose columns are the eigenvectors.
    std::vector<double> vectors;
};

inline RealSymmetricEigendecomposition symmetric_eigh(
    const std::vector<double>& matrix,
    std::size_t dimension
) {
    if (dimension == 0 || matrix.size() != dimension * dimension) {
        throw std::invalid_argument(
            "Symmetric eigensolver dimension mismatch."
        );
    }
    if (dimension == 1) {
        return {{matrix[0]}, {1.0}};
    }
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    if (
        dimension >
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::overflow_error(
            "Symmetric eigensolver dimension exceeds LAPACK integer range."
        );
    }
    const int n = static_cast<int>(dimension);
    std::vector<double> column_major(dimension * dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            column_major[row + col * dimension] =
                matrix[row * dimension + col];
        }
    }
    std::vector<double> eigenvalues(dimension);
    char jobz = 'V';
    char uplo = 'U';
    int n_arg = n;
    int lda_arg = n;
    int info = 0;
    int work_size = -1;
    double work_query = 0.0;
    dsyev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        &work_query,
        &work_size,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Symmetric eigensolver workspace query failed."
        );
    }
    work_size = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(work_size));
    dsyev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        work.data(),
        &work_size,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Symmetric eigensolver failed to converge."
        );
    }
    std::vector<double> eigenvectors(dimension * dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            eigenvectors[row * dimension + col] =
                column_major[row + col * dimension];
        }
    }
    return {std::move(eigenvalues), std::move(eigenvectors)};
#else
    std::vector<double> values = matrix;
    std::vector<double> vectors(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        vectors[index * dimension + index] = 1.0;
    }
    const std::size_t max_rotations =
        std::max<std::size_t>(64, 8 * dimension * dimension);
    for (std::size_t rotation = 0; rotation < max_rotations; ++rotation) {
        std::size_t pivot_row = 0;
        std::size_t pivot_col = 1;
        double maximum = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (
                std::size_t col = row + 1;
                col < dimension;
                ++col
            ) {
                const double candidate = std::abs(
                    values[row * dimension + col]
                );
                if (candidate > maximum) {
                    maximum = candidate;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if (maximum < 1.0e-13) {
            break;
        }
        const double app =
            values[pivot_row * dimension + pivot_row];
        const double aqq =
            values[pivot_col * dimension + pivot_col];
        const double apq =
            values[pivot_row * dimension + pivot_col];
        const double angle = 0.5 * std::atan2(
            2.0 * apq,
            aqq - app
        );
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t index = 0; index < dimension; ++index) {
            const double left =
                values[index * dimension + pivot_row];
            const double right =
                values[index * dimension + pivot_col];
            values[index * dimension + pivot_row] =
                cosine * left - sine * right;
            values[index * dimension + pivot_col] =
                sine * left + cosine * right;
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            const double top =
                values[pivot_row * dimension + index];
            const double bottom =
                values[pivot_col * dimension + index];
            values[pivot_row * dimension + index] =
                cosine * top - sine * bottom;
            values[pivot_col * dimension + index] =
                sine * top + cosine * bottom;
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            const double top =
                vectors[index * dimension + pivot_row];
            const double bottom =
                vectors[index * dimension + pivot_col];
            vectors[index * dimension + pivot_row] =
                cosine * top - sine * bottom;
            vectors[index * dimension + pivot_col] =
                sine * top + cosine * bottom;
        }
    }
    std::vector<std::size_t> order(dimension);
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&values, dimension](std::size_t left, std::size_t right) {
            return values[left * dimension + left]
                < values[right * dimension + right];
        }
    );
    std::vector<double> eigenvalues(dimension);
    std::vector<double> eigenvectors(dimension * dimension);
    for (std::size_t col = 0; col < dimension; ++col) {
        const std::size_t source = order[col];
        eigenvalues[col] = values[source * dimension + source];
        for (std::size_t row = 0; row < dimension; ++row) {
            eigenvectors[row * dimension + col] =
                vectors[row * dimension + source];
        }
    }
    return {std::move(eigenvalues), std::move(eigenvectors)};
#endif
}

inline RealLowestEigenpair lowest_projected_eigenpair(
    const std::vector<double>& matrix,
    std::size_t dimension
) {
    if (dimension == 0 || matrix.size() != dimension * dimension) {
        throw std::invalid_argument(
            "Projected symmetric eigensolver dimension mismatch."
        );
    }
    if (dimension == 1) {
        return {matrix[0], {1.0}};
    }
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    const int n = static_cast<int>(dimension);
    std::vector<double> column_major(dimension * dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            column_major[row + col * dimension] =
                matrix[row * dimension + col];
        }
    }
    std::vector<double> eigenvalues(dimension);
    char jobz = 'V';
    char uplo = 'U';
    int n_arg = n;
    int lda_arg = n;
    int info = 0;
    int work_size = -1;
    double work_query = 0.0;
    dsyev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        &work_query,
        &work_size,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Projected symmetric eigensolver workspace query failed."
        );
    }
    work_size = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(work_size));
    dsyev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        work.data(),
        &work_size,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Projected symmetric eigensolver failed to converge."
        );
    }
    std::vector<double> vector(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        vector[row] = column_major[row];
    }
    return {eigenvalues[0], std::move(vector)};
#else
    std::vector<double> values = matrix;
    std::vector<double> vectors(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        vectors[index * dimension + index] = 1.0;
    }
    const std::size_t max_rotations =
        std::max<std::size_t>(64, 8 * dimension * dimension);
    for (std::size_t rotation = 0; rotation < max_rotations; ++rotation) {
        std::size_t pivot_row = 0;
        std::size_t pivot_col = 1;
        double maximum = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t col = row + 1; col < dimension; ++col) {
                const double candidate = std::abs(
                    values[row * dimension + col]
                );
                if (candidate > maximum) {
                    maximum = candidate;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if (maximum < 1.0e-13) {
            break;
        }
        const double diagonal_row =
            values[pivot_row * dimension + pivot_row];
        const double diagonal_col =
            values[pivot_col * dimension + pivot_col];
        const double off_diagonal =
            values[pivot_row * dimension + pivot_col];
        const double tau =
            (diagonal_col - diagonal_row) / (2.0 * off_diagonal);
        const double tangent =
            (tau >= 0.0 ? 1.0 : -1.0)
            / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        for (std::size_t index = 0; index < dimension; ++index) {
            if (index == pivot_row || index == pivot_col) {
                continue;
            }
            const double row_value =
                values[index * dimension + pivot_row];
            const double col_value =
                values[index * dimension + pivot_col];
            values[index * dimension + pivot_row] =
                cosine * row_value - sine * col_value;
            values[pivot_row * dimension + index] =
                values[index * dimension + pivot_row];
            values[index * dimension + pivot_col] =
                sine * row_value + cosine * col_value;
            values[pivot_col * dimension + index] =
                values[index * dimension + pivot_col];
        }
        values[pivot_row * dimension + pivot_row] =
            cosine * cosine * diagonal_row
            - 2.0 * sine * cosine * off_diagonal
            + sine * sine * diagonal_col;
        values[pivot_col * dimension + pivot_col] =
            sine * sine * diagonal_row
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * diagonal_col;
        values[pivot_row * dimension + pivot_col] = 0.0;
        values[pivot_col * dimension + pivot_row] = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            const double row_value =
                vectors[row * dimension + pivot_row];
            const double col_value =
                vectors[row * dimension + pivot_col];
            vectors[row * dimension + pivot_row] =
                cosine * row_value - sine * col_value;
            vectors[row * dimension + pivot_col] =
                sine * row_value + cosine * col_value;
        }
    }
    std::size_t minimum_index = 0;
    for (std::size_t index = 1; index < dimension; ++index) {
        if (
            values[index * dimension + index]
            < values[minimum_index * dimension + minimum_index]
        ) {
            minimum_index = index;
        }
    }
    std::vector<double> vector(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        vector[row] = vectors[row * dimension + minimum_index];
    }
    return {
        values[minimum_index * dimension + minimum_index],
        std::move(vector),
    };
#endif
}

inline LowestEigenpair lowest_projected_eigenpair(
    const std::vector<Complex>& matrix,
    std::size_t dimension
) {
    if (dimension == 0 || matrix.size() != dimension * dimension) {
        throw std::invalid_argument(
            "Projected Hermitian eigensolver dimension mismatch."
        );
    }
    if (dimension == 1) {
        return {matrix[0].real(), {Complex(1.0, 0.0)}};
    }
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    const int n = static_cast<int>(dimension);
    const int lda = n;
    std::vector<Complex> column_major(dimension * dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            column_major[row + col * dimension] =
                matrix[row * dimension + col];
        }
    }
    std::vector<double> eigenvalues(dimension);
    std::vector<double> real_work(
        static_cast<std::size_t>(std::max(1, 3 * n - 2))
    );
    char jobz = 'V';
    char uplo = 'U';
    int n_arg = n;
    int lda_arg = lda;
    int info = 0;
    int work_size = -1;
    Complex work_query;
    zheev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        &work_query,
        &work_size,
        real_work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Projected Hermitian eigensolver workspace query failed."
        );
    }
    work_size = std::max(1, static_cast<int>(work_query.real()));
    std::vector<Complex> work(static_cast<std::size_t>(work_size));
    zheev_(
        &jobz,
        &uplo,
        &n_arg,
        column_major.data(),
        &lda_arg,
        eigenvalues.data(),
        work.data(),
        &work_size,
        real_work.data(),
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "Projected Hermitian eigensolver failed to converge."
        );
    }
    std::vector<Complex> vector(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        vector[row] = column_major[row];
    }
    normalize_phase(vector);
    return {eigenvalues[0], std::move(vector)};
#else
    const std::size_t real_dimension = 2 * dimension;
    std::vector<double> values(real_dimension * real_dimension, 0.0);
    std::vector<double> vectors(real_dimension * real_dimension, 0.0);
    for (std::size_t row = 0; row < real_dimension; ++row) {
        vectors[row * real_dimension + row] = 1.0;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            const Complex element = matrix[row * dimension + col];
            values[row * real_dimension + col] = element.real();
            values[row * real_dimension + col + dimension] = -element.imag();
            values[(row + dimension) * real_dimension + col] = element.imag();
            values[
                (row + dimension) * real_dimension + col + dimension
            ] = element.real();
        }
    }
    const std::size_t max_rotations =
        std::max<std::size_t>(64, 8 * real_dimension * real_dimension);
    for (std::size_t rotation = 0; rotation < max_rotations; ++rotation) {
        std::size_t pivot_row = 0;
        std::size_t pivot_col = 1;
        double maximum = 0.0;
        for (std::size_t row = 0; row < real_dimension; ++row) {
            for (std::size_t col = row + 1; col < real_dimension; ++col) {
                const double candidate = std::abs(
                    values[row * real_dimension + col]
                );
                if (candidate > maximum) {
                    maximum = candidate;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if (maximum < 1.0e-13) {
            break;
        }
        const double diagonal_row =
            values[pivot_row * real_dimension + pivot_row];
        const double diagonal_col =
            values[pivot_col * real_dimension + pivot_col];
        const double off_diagonal =
            values[pivot_row * real_dimension + pivot_col];
        const double tau =
            (diagonal_col - diagonal_row) / (2.0 * off_diagonal);
        const double tangent =
            (tau >= 0.0 ? 1.0 : -1.0) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        for (std::size_t index = 0; index < real_dimension; ++index) {
            if (index == pivot_row || index == pivot_col) {
                continue;
            }
            const double row_value =
                values[index * real_dimension + pivot_row];
            const double col_value =
                values[index * real_dimension + pivot_col];
            values[index * real_dimension + pivot_row] =
                cosine * row_value - sine * col_value;
            values[pivot_row * real_dimension + index] =
                values[index * real_dimension + pivot_row];
            values[index * real_dimension + pivot_col] =
                sine * row_value + cosine * col_value;
            values[pivot_col * real_dimension + index] =
                values[index * real_dimension + pivot_col];
        }
        values[pivot_row * real_dimension + pivot_row] =
            cosine * cosine * diagonal_row
            - 2.0 * sine * cosine * off_diagonal
            + sine * sine * diagonal_col;
        values[pivot_col * real_dimension + pivot_col] =
            sine * sine * diagonal_row
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * diagonal_col;
        values[pivot_row * real_dimension + pivot_col] = 0.0;
        values[pivot_col * real_dimension + pivot_row] = 0.0;
        for (std::size_t row = 0; row < real_dimension; ++row) {
            const double row_value =
                vectors[row * real_dimension + pivot_row];
            const double col_value =
                vectors[row * real_dimension + pivot_col];
            vectors[row * real_dimension + pivot_row] =
                cosine * row_value - sine * col_value;
            vectors[row * real_dimension + pivot_col] =
                sine * row_value + cosine * col_value;
        }
    }
    std::size_t minimum_index = 0;
    for (std::size_t index = 1; index < real_dimension; ++index) {
        if (
            values[index * real_dimension + index]
            < values[minimum_index * real_dimension + minimum_index]
        ) {
            minimum_index = index;
        }
    }
    std::vector<Complex> vector(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        vector[row] = Complex(
            vectors[row * real_dimension + minimum_index],
            vectors[(row + dimension) * real_dimension + minimum_index]
        );
    }
    const double vector_norm = norm2(vector);
    if (vector_norm < 1.0e-14) {
        throw std::runtime_error(
            "Projected Hermitian eigensolver returned a null vector."
        );
    }
    for (Complex& value : vector) {
        value /= vector_norm;
    }
    normalize_phase(vector);
    return {
        values[minimum_index * real_dimension + minimum_index],
        std::move(vector),
    };
#endif
}

struct DavidsonWorkspace {
    std::vector<std::vector<Complex>> basis;
    std::vector<std::vector<Complex>> h_basis;
    std::vector<Complex> projected;
    std::vector<Complex> projected_dense;
    std::vector<Complex> coefficients;
    std::vector<Complex> ritz;
    std::vector<Complex> h_ritz;
    std::vector<Complex> residual;
    std::vector<Complex> correction;
    std::vector<Complex> best_vector;
    std::size_t dimension = 0;
    std::size_t dimension_capacity = 0;
    std::size_t capacity = 0;
    bool initialized = false;

    bool ensure(std::size_t requested_dimension, std::size_t requested_capacity) {
        if (requested_dimension == 0 || requested_capacity == 0) {
            throw std::invalid_argument(
                "Davidson workspace dimensions must be positive."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity;
        if (!reused) {
            *this = DavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            basis.resize(capacity);
            h_basis.resize(capacity);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            coefficients.resize(capacity);
            initialized = true;
        }
        dimension = requested_dimension;
        for (std::size_t index = 0; index < capacity; ++index) {
            basis[index].reserve(dimension_capacity);
            h_basis[index].reserve(dimension_capacity);
            basis[index].resize(dimension);
            h_basis[index].resize(dimension);
        }
        ritz.reserve(dimension_capacity);
        h_ritz.reserve(dimension_capacity);
        residual.reserve(dimension_capacity);
        correction.reserve(dimension_capacity);
        best_vector.reserve(dimension_capacity);
        ritz.resize(dimension);
        h_ritz.resize(dimension);
        residual.resize(dimension);
        correction.resize(dimension);
        best_vector.resize(dimension);
        return reused;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t elements =
            projected.capacity()
            + projected_dense.capacity()
            + coefficients.capacity()
            + ritz.capacity()
            + h_ritz.capacity()
            + residual.capacity()
            + correction.capacity()
            + best_vector.capacity();
        for (const auto& vector : basis) {
            elements += vector.capacity();
        }
        for (const auto& vector : h_basis) {
            elements += vector.capacity();
        }
        return elements * sizeof(Complex);
    }
};

struct DavidsonResult {
    bool accepted = false;
    double energy = 0.0;
    std::vector<Complex> vector;
    double residual_norm = std::numeric_limits<double>::infinity();
    int iterations = 0;
    std::size_t basis_size = 0;
    int restarts = 0;
    bool converged = false;
    bool workspace_reused = false;
    std::uint64_t matvec_calls = 0;
    std::uint64_t norm_matvec_calls = 0;
};

template <typename Matvec>
DavidsonResult davidson(
    const std::vector<Complex>& diagonal,
    std::vector<Complex> guess,
    double tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    DavidsonWorkspace& workspace,
    Matvec&& matvec
) {
    const std::size_t dimension = diagonal.size();
    if (dimension == 0 || guess.size() != dimension) {
        throw std::invalid_argument("Davidson vector dimensions differ.");
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument("Davidson max_iterations must be positive.");
    }
    const double guess_norm = norm2(guess);
    if (guess_norm < 1.0e-14) {
        throw std::invalid_argument(
            "Initial Davidson vector has near-zero norm."
        );
    }
    for (Complex& value : guess) {
        value /= guess_norm;
    }

    const std::size_t requested_capacity = static_cast<std::size_t>(
        restart_dimension > 0 ? restart_dimension : max_iterations
    );
    DavidsonResult result;
    result.workspace_reused = workspace.ensure(
        dimension,
        std::max<std::size_t>(
            std::min<std::size_t>(dimension, 2),
            requested_capacity
        )
    );
    std::fill(workspace.projected.begin(), workspace.projected.end(), 0.0);
    std::fill(workspace.best_vector.begin(), workspace.best_vector.end(), 0.0);
    workspace.basis[0] = std::move(guess);
    std::size_t basis_size = 1;
    if (dimension > 1 && workspace.capacity > 1) {
        const std::size_t diagonal_index = static_cast<std::size_t>(
            std::min_element(diagonal.begin(), diagonal.end(), [](
                const Complex& left,
                const Complex& right
            ) {
                return left.real() < right.real();
            }) - diagonal.begin()
        );
        std::fill(
            workspace.basis[1].begin(),
            workspace.basis[1].end(),
            Complex{0.0, 0.0}
        );
        workspace.basis[1][diagonal_index] = Complex{1.0, 0.0};
        const Complex overlap = dotc(
            workspace.basis[0],
            workspace.basis[1]
        );
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.basis[1][index] -= workspace.basis[0][index] * overlap;
        }
        const double seed_norm = norm2(workspace.basis[1]);
        if (seed_norm > 1.0e-12) {
            for (Complex& value : workspace.basis[1]) {
                value /= seed_norm;
            }
            basis_size = 2;
        }
    }
    std::size_t computed_basis_size = 0;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        for (
            std::size_t newest = computed_basis_size;
            newest < basis_size;
            ++newest
        ) {
            workspace.h_basis[newest] = matvec(workspace.basis[newest]);
            ++result.matvec_calls;
            if (workspace.h_basis[newest].size() != dimension) {
                throw std::runtime_error(
                    "Davidson matvec returned an incompatible vector."
                );
            }
            for (std::size_t index = 0; index <= newest; ++index) {
                const Complex element = dotc(
                    workspace.basis[index],
                    workspace.h_basis[newest]
                );
                workspace.projected[
                    index * workspace.capacity + newest
                ] = element;
                workspace.projected[
                    newest * workspace.capacity + index
                ] = std::conj(element);
            }
        }
        computed_basis_size = basis_size;
        for (std::size_t row = 0; row < basis_size; ++row) {
            for (std::size_t col = 0; col < basis_size; ++col) {
                workspace.projected_dense[row * basis_size + col] =
                    workspace.projected[row * workspace.capacity + col];
            }
        }
        const LowestEigenpair eigenpair = lowest_projected_eigenpair(
            std::vector<Complex>(
                workspace.projected_dense.begin(),
                workspace.projected_dense.begin() + basis_size * basis_size
            ),
            basis_size
        );
        result.energy = eigenpair.value;
        for (std::size_t index = 0; index < basis_size; ++index) {
            workspace.coefficients[index] = eigenpair.vector[index];
        }
        std::fill(workspace.ritz.begin(), workspace.ritz.end(), 0.0);
        std::fill(workspace.h_ritz.begin(), workspace.h_ritz.end(), 0.0);
        for (std::size_t index = 0; index < basis_size; ++index) {
            axpy(
                workspace.ritz,
                workspace.coefficients[index],
                workspace.basis[index]
            );
            axpy(
                workspace.h_ritz,
                workspace.coefficients[index],
                workspace.h_basis[index]
            );
        }
        workspace.residual = workspace.h_ritz;
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.residual[index] -= result.energy * workspace.ritz[index];
        }
        result.residual_norm = norm2(workspace.residual);
        workspace.best_vector = workspace.ritz;
        if (result.residual_norm < tolerance) {
            result.converged = true;
            break;
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            Complex denominator = result.energy - diagonal[index];
            if (std::abs(denominator) < 1.0e-12) {
                denominator = Complex(
                    denominator.real() >= 0.0 ? 1.0e-12 : -1.0e-12,
                    denominator.imag()
                );
            }
            workspace.correction[index] =
                workspace.residual[index] / denominator;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t basis_index = 0;
                basis_index < basis_size;
                ++basis_index
            ) {
                const Complex overlap = dotc(
                    workspace.basis[basis_index],
                    workspace.correction
                );
                for (std::size_t index = 0; index < dimension; ++index) {
                    workspace.correction[index] -=
                        workspace.basis[basis_index][index] * overlap;
                }
            }
        }
        const double correction_norm = norm2(workspace.correction);
        if (correction_norm < 1.0e-10 || basis_size >= workspace.capacity) {
            if (
                correction_norm < 1.0e-10
                || workspace.capacity < 2
            ) {
                break;
            }
        }
        for (Complex& value : workspace.correction) {
            value /= correction_norm;
        }
        if (basis_size >= workspace.capacity) {
            const double ritz_norm = norm2(workspace.ritz);
            if (ritz_norm < 1.0e-14) {
                break;
            }
            for (Complex& value : workspace.ritz) {
                value /= ritz_norm;
            }
            const Complex overlap = dotc(
                workspace.ritz,
                workspace.correction
            );
            for (std::size_t index = 0; index < dimension; ++index) {
                workspace.correction[index] -= workspace.ritz[index] * overlap;
            }
            const double restarted_correction_norm =
                norm2(workspace.correction);
            if (restarted_correction_norm < 1.0e-10) {
                break;
            }
            for (Complex& value : workspace.correction) {
                value /= restarted_correction_norm;
            }
            workspace.basis[0] = workspace.ritz;
            workspace.basis[1] = workspace.correction;
            std::fill(
                workspace.projected.begin(),
                workspace.projected.end(),
                0.0
            );
            basis_size = 2;
            computed_basis_size = 0;
            ++result.restarts;
            continue;
        }
        workspace.basis[basis_size] = workspace.correction;
        ++basis_size;
    }

    result.basis_size = basis_size;
    result.accepted = result.converged || accept_unconverged;
    if (result.accepted) {
        result.vector = workspace.best_vector;
    }
    return result;
}

struct GeneralizedDavidsonWorkspace {
    std::vector<std::vector<Complex>> basis;
    std::vector<std::vector<Complex>> h_basis;
    std::vector<std::vector<Complex>> n_basis;
    std::vector<Complex> projected;
    std::vector<Complex> projected_dense;
    std::vector<Complex> coefficients;
    std::vector<Complex> ritz;
    std::vector<Complex> h_ritz;
    std::vector<Complex> n_ritz;
    std::vector<Complex> residual;
    std::vector<Complex> correction;
    std::vector<Complex> n_correction;
    std::vector<Complex> best_vector;
    std::size_t dimension = 0;
    std::size_t dimension_capacity = 0;
    std::size_t capacity = 0;
    bool initialized = false;

    bool ensure(std::size_t requested_dimension, std::size_t requested_capacity) {
        if (requested_dimension == 0 || requested_capacity == 0) {
            throw std::invalid_argument(
                "Generalized Davidson workspace dimensions must be positive."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity;
        if (!reused) {
            *this = GeneralizedDavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            basis.resize(capacity);
            h_basis.resize(capacity);
            n_basis.resize(capacity);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            coefficients.resize(capacity);
            initialized = true;
        }
        dimension = requested_dimension;
        for (std::size_t index = 0; index < capacity; ++index) {
            basis[index].resize(dimension);
            h_basis[index].resize(dimension);
            n_basis[index].resize(dimension);
        }
        ritz.resize(dimension);
        h_ritz.resize(dimension);
        n_ritz.resize(dimension);
        residual.resize(dimension);
        correction.resize(dimension);
        n_correction.resize(dimension);
        best_vector.resize(dimension);
        return reused;
    }

    std::size_t memory_bytes() const noexcept {
        std::size_t elements =
            projected.capacity()
            + projected_dense.capacity()
            + coefficients.capacity()
            + ritz.capacity()
            + h_ritz.capacity()
            + n_ritz.capacity()
            + residual.capacity()
            + correction.capacity()
            + n_correction.capacity()
            + best_vector.capacity();
        for (const auto& vector : basis) {
            elements += vector.capacity();
        }
        for (const auto& vector : h_basis) {
            elements += vector.capacity();
        }
        for (const auto& vector : n_basis) {
            elements += vector.capacity();
        }
        return elements * sizeof(Complex);
    }
};

template <typename HMatvec, typename NMatvec>
DavidsonResult generalized_davidson(
    const std::vector<Complex>& h_diagonal,
    const std::vector<Complex>& n_diagonal,
    std::vector<Complex> guess,
    double energy_tolerance,
    double residual_tolerance,
    double linear_dependence_tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    GeneralizedDavidsonWorkspace& workspace,
    HMatvec&& h_matvec,
    NMatvec&& n_matvec
) {
    const std::size_t dimension = h_diagonal.size();
    if (
        dimension == 0
        || n_diagonal.size() != dimension
        || guess.size() != dimension
    ) {
        throw std::invalid_argument(
            "Generalized Davidson vector dimensions differ."
        );
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "Generalized Davidson max_iterations must be positive."
        );
    }
    const std::size_t requested_capacity = std::max<std::size_t>(
        std::min<std::size_t>(dimension, 2),
        static_cast<std::size_t>(
            restart_dimension > 0 ? restart_dimension : max_iterations
        )
    );
    DavidsonResult result;
    result.workspace_reused = workspace.ensure(
        dimension,
        requested_capacity
    );
    std::fill(workspace.projected.begin(), workspace.projected.end(), 0.0);

    auto apply_h = [&](const std::vector<Complex>& input) {
        auto output = h_matvec(input);
        ++result.matvec_calls;
        if (output.size() != dimension) {
            throw std::runtime_error(
                "Generalized Davidson Hamiltonian matvec dimension mismatch."
            );
        }
        return output;
    };
    auto apply_n = [&](const std::vector<Complex>& input) {
        auto output = n_matvec(input);
        ++result.norm_matvec_calls;
        if (output.size() != dimension) {
            throw std::runtime_error(
                "Generalized Davidson metric matvec dimension mismatch."
            );
        }
        return output;
    };
    auto metric_normalize = [&](
        std::vector<Complex>& vector,
        std::vector<Complex>& n_vector
    ) {
        const double norm_squared = std::real(dotc(vector, n_vector));
        if (
            !std::isfinite(norm_squared)
            || norm_squared
                <= linear_dependence_tolerance * linear_dependence_tolerance
        ) {
            return false;
        }
        const double inverse_norm = 1.0 / std::sqrt(norm_squared);
        for (std::size_t index = 0; index < dimension; ++index) {
            vector[index] *= inverse_norm;
            n_vector[index] *= inverse_norm;
        }
        return true;
    };
    auto metric_orthogonalize = [&](
        std::vector<Complex>& vector,
        std::vector<Complex>& n_vector,
        std::size_t basis_size
    ) {
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t basis_index = 0;
                basis_index < basis_size;
                ++basis_index
            ) {
                const Complex overlap = dotc(
                    workspace.basis[basis_index],
                    n_vector
                );
                for (std::size_t index = 0; index < dimension; ++index) {
                    vector[index] -=
                        workspace.basis[basis_index][index] * overlap;
                    n_vector[index] -=
                        workspace.n_basis[basis_index][index] * overlap;
                }
            }
        }
        return metric_normalize(vector, n_vector);
    };

    workspace.basis[0] = guess;
    workspace.n_basis[0] = apply_n(workspace.basis[0]);
    if (!metric_normalize(workspace.basis[0], workspace.n_basis[0])) {
        throw std::invalid_argument(
            "Initial Davidson vector is singular in the local metric."
        );
    }
    const std::vector<Complex> normalized_guess = workspace.basis[0];
    std::size_t basis_size = 1;
    std::size_t computed_basis_size = 0;
    double previous_energy = std::numeric_limits<double>::infinity();
    const std::size_t minimum_explored_dimension = std::min<std::size_t>(
        std::min<std::size_t>(dimension, workspace.capacity),
        16
    );

    auto append_seed = [&](std::size_t used_basis) {
        std::vector<std::size_t> order(dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            order[index] = index;
        }
        std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            const double left_norm =
                std::abs(n_diagonal[left]) > 1.0e-12
                ? n_diagonal[left].real()
                : 1.0;
            const double right_norm =
                std::abs(n_diagonal[right]) > 1.0e-12
                ? n_diagonal[right].real()
                : 1.0;
            return h_diagonal[left].real() / left_norm
                < h_diagonal[right].real() / right_norm;
        });
        for (const std::size_t seed_index : order) {
            std::fill(
                workspace.correction.begin(),
                workspace.correction.end(),
                Complex{0.0, 0.0}
            );
            workspace.correction[seed_index] = Complex{1.0, 0.0};
            workspace.n_correction = apply_n(workspace.correction);
            if (metric_orthogonalize(
                workspace.correction,
                workspace.n_correction,
                used_basis
            )) {
                return true;
            }
        }
        return false;
    };

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        for (
            std::size_t newest = computed_basis_size;
            newest < basis_size;
            ++newest
        ) {
            workspace.h_basis[newest] = apply_h(workspace.basis[newest]);
            for (std::size_t index = 0; index <= newest; ++index) {
                const Complex element = dotc(
                    workspace.basis[index],
                    workspace.h_basis[newest]
                );
                workspace.projected[
                    index * workspace.capacity + newest
                ] = element;
                workspace.projected[
                    newest * workspace.capacity + index
                ] = std::conj(element);
            }
        }
        computed_basis_size = basis_size;
        for (std::size_t row = 0; row < basis_size; ++row) {
            for (std::size_t col = 0; col < basis_size; ++col) {
                workspace.projected_dense[row * basis_size + col] =
                    workspace.projected[row * workspace.capacity + col];
            }
        }
        const LowestEigenpair eigenpair = lowest_projected_eigenpair(
            std::vector<Complex>(
                workspace.projected_dense.begin(),
                workspace.projected_dense.begin()
                    + basis_size * basis_size
            ),
            basis_size
        );
        result.energy = eigenpair.value;
        for (std::size_t index = 0; index < basis_size; ++index) {
            workspace.coefficients[index] = eigenpair.vector[index];
        }
        std::fill(workspace.ritz.begin(), workspace.ritz.end(), 0.0);
        std::fill(workspace.h_ritz.begin(), workspace.h_ritz.end(), 0.0);
        std::fill(workspace.n_ritz.begin(), workspace.n_ritz.end(), 0.0);
        for (std::size_t index = 0; index < basis_size; ++index) {
            axpy(
                workspace.ritz,
                workspace.coefficients[index],
                workspace.basis[index]
            );
            axpy(
                workspace.h_ritz,
                workspace.coefficients[index],
                workspace.h_basis[index]
            );
            axpy(
                workspace.n_ritz,
                workspace.coefficients[index],
                workspace.n_basis[index]
            );
        }
        workspace.residual = workspace.h_ritz;
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.residual[index] -=
                result.energy * workspace.n_ritz[index];
        }
        result.residual_norm = norm2(workspace.residual);
        workspace.best_vector = workspace.ritz;
        const double energy_change = std::abs(
            result.energy - previous_energy
        );
        if (
            result.residual_norm <= residual_tolerance
            && energy_change <= energy_tolerance
            && basis_size >= minimum_explored_dimension
        ) {
            result.converged = true;
            break;
        }

        for (std::size_t index = 0; index < dimension; ++index) {
            Complex denominator =
                result.energy * n_diagonal[index] - h_diagonal[index];
            if (std::abs(denominator) < 1.0e-12) {
                denominator = Complex(
                    denominator.real() >= 0.0 ? 1.0e-12 : -1.0e-12,
                    denominator.imag()
                );
            }
            workspace.correction[index] =
                workspace.residual[index] / denominator;
        }
        workspace.n_correction = apply_n(workspace.correction);
        bool correction_valid = metric_orthogonalize(
            workspace.correction,
            workspace.n_correction,
            basis_size
        );
        if (!correction_valid) {
            correction_valid = append_seed(basis_size);
        }
        if (!correction_valid) {
            result.converged =
                result.residual_norm <= residual_tolerance;
            break;
        }

        if (basis_size >= workspace.capacity) {
            if (workspace.capacity < 2) {
                break;
            }
            workspace.basis[0] = workspace.ritz;
            workspace.n_basis[0] = workspace.n_ritz;
            if (!metric_normalize(
                workspace.basis[0],
                workspace.n_basis[0]
            )) {
                break;
            }
            workspace.basis[1] = workspace.correction;
            workspace.n_basis[1] = workspace.n_correction;
            std::fill(
                workspace.projected.begin(),
                workspace.projected.end(),
                Complex{0.0, 0.0}
            );
            basis_size = 2;
            computed_basis_size = 0;
            ++result.restarts;
        } else {
            workspace.basis[basis_size] = workspace.correction;
            workspace.n_basis[basis_size] = workspace.n_correction;
            ++basis_size;
        }
        previous_energy = result.energy;
    }

    result.basis_size = basis_size;
    result.accepted = result.converged || accept_unconverged;
    if (result.accepted) {
        result.vector = workspace.best_vector;
        const Complex reference_overlap = dotc(
            normalized_guess,
            result.vector
        );
        if (std::abs(reference_overlap) > 1.0e-12) {
            const Complex phase =
                reference_overlap / std::abs(reference_overlap);
            for (Complex& value : result.vector) {
                value /= phase;
            }
        } else {
            normalize_phase(result.vector);
        }
    }
    return result;
}

struct RealGeneralizedDavidsonWorkspace {
    std::vector<double> basis;
    std::vector<double> h_basis;
    std::vector<double> n_basis;
    std::vector<double> projected;
    std::vector<double> projected_dense;
    std::vector<double> coefficients;
    std::vector<double> ritz;
    std::vector<double> h_ritz;
    std::vector<double> n_ritz;
    std::vector<double> residual;
    std::vector<double> correction;
    std::vector<double> n_correction;
    std::vector<double> best_vector;
    std::vector<double> reference;
    std::size_t dimension = 0;
    std::size_t dimension_capacity = 0;
    std::size_t capacity = 0;
    bool initialized = false;

    bool ensure(std::size_t requested_dimension, std::size_t requested_capacity) {
        if (requested_dimension == 0 || requested_capacity == 0) {
            throw std::invalid_argument(
                "Real generalized Davidson workspace dimensions must be positive."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity;
        if (!reused) {
            *this = RealGeneralizedDavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            const std::size_t arena_elements =
                dimension_capacity * capacity;
            basis.resize(arena_elements);
            h_basis.resize(arena_elements);
            n_basis.resize(arena_elements);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            coefficients.resize(capacity);
            ritz.resize(dimension_capacity);
            h_ritz.resize(dimension_capacity);
            n_ritz.resize(dimension_capacity);
            residual.resize(dimension_capacity);
            correction.resize(dimension_capacity);
            n_correction.resize(dimension_capacity);
            best_vector.resize(dimension_capacity);
            reference.resize(dimension_capacity);
            initialized = true;
        }
        dimension = requested_dimension;
        return reused;
    }

    double* basis_column(std::size_t column) {
        return basis.data() + column * dimension_capacity;
    }
    const double* basis_column(std::size_t column) const {
        return basis.data() + column * dimension_capacity;
    }
    double* h_basis_column(std::size_t column) {
        return h_basis.data() + column * dimension_capacity;
    }
    const double* h_basis_column(std::size_t column) const {
        return h_basis.data() + column * dimension_capacity;
    }
    double* n_basis_column(std::size_t column) {
        return n_basis.data() + column * dimension_capacity;
    }
    const double* n_basis_column(std::size_t column) const {
        return n_basis.data() + column * dimension_capacity;
    }

    std::size_t memory_bytes() const noexcept {
        return (
            basis.capacity()
            + h_basis.capacity()
            + n_basis.capacity()
            + projected.capacity()
            + projected_dense.capacity()
            + coefficients.capacity()
            + ritz.capacity()
            + h_ritz.capacity()
            + n_ritz.capacity()
            + residual.capacity()
            + correction.capacity()
            + n_correction.capacity()
            + best_vector.capacity()
            + reference.capacity()
        ) * sizeof(double);
    }
};

inline double real_dot(
    const double* left,
    const double* right,
    std::size_t dimension
) {
    double value = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        value += left[index] * right[index];
    }
    return value;
}

inline double real_norm(const double* vector, std::size_t dimension) {
    return std::sqrt(real_dot(vector, vector, dimension));
}

inline void real_axpy(
    double* target,
    double scale,
    const double* source,
    std::size_t dimension
) {
    for (std::size_t index = 0; index < dimension; ++index) {
        target[index] += scale * source[index];
    }
}

struct RealDavidsonWorkspace {
    std::vector<double> basis;
    std::vector<double> h_basis;
    std::vector<double> projected;
    std::vector<double> projected_dense;
    std::vector<double> coefficients;
    std::vector<double> ritz;
    std::vector<double> h_ritz;
    std::vector<double> residual;
    std::vector<double> correction;
    std::vector<double> best_vector;
    std::vector<double> restart_basis;
    std::vector<double> restart_h_basis;
    std::size_t dimension = 0;
    std::size_t dimension_capacity = 0;
    std::size_t capacity = 0;
    bool initialized = false;

    bool ensure(std::size_t requested_dimension, std::size_t requested_capacity) {
        if (requested_dimension == 0 || requested_capacity == 0) {
            throw std::invalid_argument(
                "Real Davidson workspace dimensions must be positive."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity;
        if (!reused) {
            *this = RealDavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            const std::size_t arena_elements =
                dimension_capacity * capacity;
            basis.resize(arena_elements);
            h_basis.resize(arena_elements);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            coefficients.resize(capacity);
            ritz.resize(dimension_capacity);
            h_ritz.resize(dimension_capacity);
            residual.resize(dimension_capacity);
            correction.resize(dimension_capacity);
            best_vector.resize(dimension_capacity);
            restart_basis.resize(4 * dimension_capacity);
            restart_h_basis.resize(4 * dimension_capacity);
            initialized = true;
        }
        dimension = requested_dimension;
        return reused;
    }

    double* basis_column(std::size_t column) {
        return basis.data() + column * dimension_capacity;
    }
    const double* basis_column(std::size_t column) const {
        return basis.data() + column * dimension_capacity;
    }
    double* h_basis_column(std::size_t column) {
        return h_basis.data() + column * dimension_capacity;
    }
    const double* h_basis_column(std::size_t column) const {
        return h_basis.data() + column * dimension_capacity;
    }
    double* restart_basis_column(std::size_t column) {
        return restart_basis.data() + column * dimension_capacity;
    }
    double* restart_h_basis_column(std::size_t column) {
        return restart_h_basis.data() + column * dimension_capacity;
    }

    std::size_t memory_bytes() const noexcept {
        return (
            basis.capacity()
            + h_basis.capacity()
            + projected.capacity()
            + projected_dense.capacity()
            + coefficients.capacity()
            + ritz.capacity()
            + h_ritz.capacity()
            + residual.capacity()
            + correction.capacity()
            + best_vector.capacity()
            + restart_basis.capacity()
            + restart_h_basis.capacity()
        ) * sizeof(double);
    }
};

template <typename Scalar>
struct BlockDavidsonWorkspace {
    std::vector<Scalar> basis;
    std::vector<Scalar> h_basis;
    std::vector<Scalar> projected;
    std::vector<Scalar> projected_dense;
    std::vector<Scalar> ritz;
    std::vector<Scalar> h_ritz;
    std::vector<Scalar> residual;
    std::vector<Scalar> correction;
    std::vector<Scalar> best_vectors;
    std::vector<double> best_energies;
    std::vector<double> best_residuals;
    std::size_t dimension = 0;
    std::size_t dimension_capacity = 0;
    std::size_t capacity = 0;
    std::size_t roots = 0;
    std::size_t roots_capacity = 0;
    bool initialized = false;

    bool ensure(
        std::size_t requested_dimension,
        std::size_t requested_capacity,
        std::size_t requested_roots
    ) {
        if (
            requested_dimension == 0
            || requested_capacity == 0
            || requested_roots == 0
            || requested_roots > requested_dimension
        ) {
            throw std::invalid_argument(
                "Block Davidson workspace dimensions are invalid."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity
            && roots_capacity >= requested_roots;
        if (!reused) {
            *this = BlockDavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            roots_capacity = requested_roots;
            basis.resize(dimension_capacity * capacity);
            h_basis.resize(dimension_capacity * capacity);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            const std::size_t root_elements =
                dimension_capacity * roots_capacity;
            ritz.resize(dimension_capacity * std::min(capacity, 3*roots_capacity));
            h_ritz.resize(dimension_capacity * std::min(capacity, 3*roots_capacity));
            residual.resize(root_elements);
            correction.resize(root_elements);
            best_vectors.resize(root_elements);
            best_energies.resize(roots_capacity);
            best_residuals.resize(roots_capacity);
            initialized = true;
        }
        dimension = requested_dimension;
        roots = requested_roots;
        return reused;
    }

    Scalar* basis_column(std::size_t column) {
        return basis.data() + column * dimension_capacity;
    }
    const Scalar* basis_column(std::size_t column) const {
        return basis.data() + column * dimension_capacity;
    }
    Scalar* h_basis_column(std::size_t column) {
        return h_basis.data() + column * dimension_capacity;
    }
    const Scalar* h_basis_column(std::size_t column) const {
        return h_basis.data() + column * dimension_capacity;
    }
    Scalar* root_vector(std::vector<Scalar>& arena, std::size_t root) {
        return arena.data() + root * dimension_capacity;
    }
    const Scalar* root_vector(
        const std::vector<Scalar>& arena,
        std::size_t root
    ) const {
        return arena.data() + root * dimension_capacity;
    }

    std::size_t memory_bytes() const noexcept {
        return (
            basis.capacity()
            + h_basis.capacity()
            + projected.capacity()
            + projected_dense.capacity()
            + ritz.capacity()
            + h_ritz.capacity()
            + residual.capacity()
            + correction.capacity()
            + best_vectors.capacity()
        ) * sizeof(Scalar)
            + (best_energies.capacity() + best_residuals.capacity()) * sizeof(double);
    }
};

using RealBlockDavidsonWorkspace = BlockDavidsonWorkspace<double>;
using HermitianBlockDavidsonWorkspace = BlockDavidsonWorkspace<Complex>;

struct BlockDavidsonResult {
    bool accepted = false;
    std::vector<double> energies;
    std::vector<std::vector<Complex>> vectors;
    std::vector<double> residual_norms;
    int iterations = 0;
    std::size_t basis_size = 0;
    int restarts = 0;
    std::size_t restart_history_vectors = 0;
    std::size_t locked_roots = 0;
    int deflated_iterations = 0;
    int deflation_unlocks = 0;
    bool converged = false;
    bool workspace_reused = false;
    std::uint64_t matvec_calls = 0;
    double action_seconds = 0.;
    double projection_seconds = 0.;
    double diagonalization_seconds = 0.;
    double ritz_seconds = 0.;
    double correction_seconds = 0.;
    std::size_t cholesky_qr_blocks = 0;
    std::size_t householder_qr_blocks = 0;
    double restart_seconds = 0.;
};

inline double davidson_conj(double x) { return x; }
inline Complex davidson_conj(Complex x) { return std::conj(x); }
template <typename Scalar>
inline Scalar davidson_dot(const Scalar* a, const Scalar* b, std::size_t n) {
    Scalar value=0.;
    for (std::size_t i=0;i<n;++i) value+=davidson_conj(a[i])*b[i];
    return value;
}
template <typename Scalar>
inline double davidson_norm(const Scalar* a, std::size_t n) {
    return std::sqrt(std::real(davidson_dot(a,a,n)));
}
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
extern "C" void cblas_zgemm(int,int,int,int,int,int,const void*,const void*,int,
    const void*,int,const void*,void*,int);
extern "C" void zheevd_(char*,char*,int*,Complex*,int*,double*,Complex*,int*,
    double*,int*,int*,int*,int*);
extern "C" void zgeqp3_(int*,int*,Complex*,int*,int*,Complex*,Complex*,int*,double*,int*);
extern "C" void zungqr_(int*,int*,int*,Complex*,int*,Complex*,Complex*,int*,int*);
inline void davidson_gemm(int order,int a,int b,int m,int n,int k,double alpha,
    const double* x,int ldx,const double* y,int ldy,double beta,double* z,int ldz) {
    cblas_dgemm(order,a==113?112:a,b==113?112:b,m,n,k,alpha,x,ldx,y,ldy,beta,z,ldz);
}
inline void davidson_gemm(int order,int a,int b,int m,int n,int k,double alpha,
    const Complex* x,int ldx,const Complex* y,int ldy,double beta,Complex* z,int ldz) {
    const Complex ca=alpha, cb=beta;
    cblas_zgemm(order,a,b,m,n,k,&ca,x,ldx,y,ldy,&cb,z,ldz);
}
#endif

// Columns of the large bases are contiguous; coefficients are row-major.
// Keep the contraction blocked so the large arrays are read by GEMM rather
// than once for each requested root. The portable path has identical layout.
template <typename Scalar>
inline void davidson_rotate(const Scalar* basis, std::size_t stride,
    std::size_t dimension, std::size_t columns, const Scalar* coefficients,
    std::size_t coefficient_stride, std::size_t roots, Scalar* output) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    davidson_gemm(101, 112, 111, roots, dimension, columns, 1.,
        coefficients, coefficient_stride, basis, stride, 0., output, stride);
#else
    for (std::size_t root=0; root<roots; ++root) {
        std::fill(output+root*stride, output+root*stride+dimension, 0.);
        for (std::size_t col=0; col<columns; ++col)
            for (std::size_t i=0; i<dimension; ++i)
                output[root*stride+i] += coefficients[col*coefficient_stride+root]*basis[col*stride+i];
    }
#endif
}

inline RealSymmetricEigendecomposition davidson_projected_eigh(
    const std::vector<double>& matrix, std::size_t dimension) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    if (!dimension || dimension > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || matrix.size() < dimension*dimension)
        throw std::invalid_argument("Invalid Davidson projected matrix");
    int n=static_cast<int>(dimension), work_size=-1, integer_size=-1, info=0;
    char job='V', triangle='U';
    // Symmetry makes the input row/column-major layouts identical.
    std::vector<double> vectors(matrix.begin(), matrix.begin()+dimension*dimension), values(n);
    double work_query=0.;
    int integer_query=0;
    dsyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(),
        &work_query, &work_size, &integer_query, &integer_size, &info);
    if (info) throw std::runtime_error("Davidson eigensolver workspace query failed");
    work_size=static_cast<int>(work_query); integer_size=integer_query;
    std::vector<double> work(work_size);
    std::vector<int> integer_work(integer_size);
    dsyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(),
        work.data(), &work_size, integer_work.data(), &integer_size, &info);
    if (info) throw std::runtime_error("Davidson projected eigensolver failed");
    for (int i=0; i<n; ++i)
        for (int j=0; j<i; ++j) std::swap(vectors[i*n+j], vectors[j*n+i]);
    return {std::move(values), std::move(vectors)};
#else
    return symmetric_eigh(std::vector<double>(matrix.begin(), matrix.begin()+dimension*dimension), dimension);
#endif
}

// Adapted CholeskyQR2: Fukaya et al., ScalA 2014, pp. 31-38,
// doi:10.1109/ScalA.2014.11. Column scaling and conservative pivot/orthogonality
// checks are additions; this is not shifted CholeskyQR3 or a condition estimate.
// The input is committed only after success; rejected blocks use pivoted QR.
inline bool davidson_cholesky_qr(double* columns, std::size_t dimension,
    std::size_t count, std::size_t stride) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    if (count<8) return false;
    std::vector<double> work(dimension*count), gram(count*count);
    for (std::size_t j=0;j<count;++j) {
        const double norm=real_norm(columns+j*stride,dimension);
        if (norm<=1.e-14 || !std::isfinite(norm)) return false;
        for (std::size_t i=0;i<dimension;++i) work[j*dimension+i]=columns[j*stride+i]/norm;
    }
    int n=static_cast<int>(dimension), k=static_cast<int>(count), info=0;
    char upper='U', right='R', normal='N'; double one=1.;
    for (int pass=0;pass<2;++pass) {
        cblas_dgemm(101,111,112,k,k,n,1.,work.data(),n,work.data(),n,0.,gram.data(),k);
        dpotrf_(&upper,&k,gram.data(),&k,&info);
        if (info) return false;
        for (int j=0;j<k;++j)
            if (!std::isfinite(gram[j*k+j]) || gram[j*k+j]<1.e-4) return false;
        dtrsm_(&right,&upper,&normal,&normal,&n,&k,&one,gram.data(),&k,work.data(),&n);
    }
    cblas_dgemm(101,111,112,k,k,n,1.,work.data(),n,work.data(),n,0.,gram.data(),k);
    for (int i=0;i<k;++i) {
        double error=0.;
        for (int j=0;j<k;++j) error+=std::abs(gram[i*k+j]-(i==j ? 1. : 0.));
        if (!std::isfinite(error) || error>1.e-11) return false;
    }
    for (std::size_t j=0;j<count;++j)
        std::copy(work.data()+j*dimension,work.data()+(j+1)*dimension,columns+j*stride);
    return true;
#else
    return false;
#endif
}

#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
inline std::size_t davidson_correction_qr(double* columns, std::size_t dimension,
    std::size_t corrections, std::size_t stride, BlockDavidsonResult& result) {
    std::size_t independent=0;
    if (davidson_cholesky_qr(columns, dimension,
            corrections, stride)) {
        independent=corrections;
        ++result.cholesky_qr_blocks;
    } else if (corrections) {
        ++result.householder_qr_blocks;
        int rows=static_cast<int>(dimension), cols=static_cast<int>(corrections);
        int leading=static_cast<int>(stride), work_size=-1, info=0;
        std::vector<int> pivots(cols,0);
        std::vector<double> tau(std::min(rows,cols));
        double query=0.;
        dgeqp3_(&rows,&cols,columns,&leading,pivots.data(),tau.data(),
            &query,&work_size,&info);
        if (info) throw std::runtime_error("Davidson correction QR query failed");
        work_size=static_cast<int>(query);
        std::vector<double> work(work_size);
        dgeqp3_(&rows,&cols,columns,&leading,pivots.data(),tau.data(),
            work.data(),&work_size,&info);
        if (info) throw std::runtime_error("Davidson correction QR failed");
        while (independent<tau.size() &&
            std::abs(columns[independent*leading+independent])>1.e-14)
            ++independent;
        if (independent) {
            int rank=static_cast<int>(independent);
            work_size=-1;
            dorgqr_(&rows,&rank,&rank,columns,&leading,tau.data(),
                &query,&work_size,&info);
            if (info) throw std::runtime_error("Davidson correction Q query failed");
            work_size=static_cast<int>(query); work.resize(work_size);
            dorgqr_(&rows,&rank,&rank,columns,&leading,tau.data(),
                work.data(),&work_size,&info);
            if (info) throw std::runtime_error("Davidson correction Q failed");
        }
    }
    return independent;
}
#endif

// The projected matrix is small; its eigenvectors use row-major storage.
inline auto davidson_projected_eigh(const std::vector<Complex>& matrix, std::size_t dimension) {
    struct Eigenpairs { std::vector<double> values; std::vector<Complex> vectors; };
    if (!dimension || matrix.size()<dimension*dimension ||
        dimension>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid Davidson projected matrix");
    Eigenpairs result{std::vector<double>(dimension),std::vector<Complex>(dimension*dimension)};
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    int n=static_cast<int>(dimension), lwork=-1, lrwork=-1, liwork=-1, info=0, iq=0;
    char job='V', triangle='U';
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) result.vectors[j*n+i]=matrix[i*n+j];
    Complex query; double rq=0.;
    zheevd_(&job,&triangle,&n,result.vectors.data(),&n,result.values.data(),
        &query,&lwork,&rq,&lrwork,&iq,&liwork,&info);
    if (info) throw std::runtime_error("Hermitian Davidson eigensolver query failed");
    lwork=static_cast<int>(query.real()); lrwork=static_cast<int>(rq); liwork=iq;
    std::vector<Complex> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    zheevd_(&job,&triangle,&n,result.vectors.data(),&n,result.values.data(),
        work.data(),&lwork,rwork.data(),&lrwork,iwork.data(),&liwork,&info);
    if (info) throw std::runtime_error("Hermitian Davidson eigensolver failed");
    for (int i=0;i<n;++i) for (int j=0;j<i;++j)
        std::swap(result.vectors[i*n+j],result.vectors[j*n+i]);
#else
    // Cyclic complex Jacobi rotations, without doubling the problem dimension.
    const std::size_t n=dimension;
    std::vector<Complex> a(matrix.begin(),matrix.begin()+n*n);
    for (std::size_t i=0;i<n;++i) result.vectors[i*n+i]=1.;
    double scale=0.;
    for (const auto value:a) scale=std::max(scale,std::abs(value));
    const double threshold=std::numeric_limits<double>::epsilon()*scale;
    bool converged=false;
    for (int sweep=0;sweep<100;++sweep) {
        double largest=0.;
        for (std::size_t p=0;p<n;++p) for (std::size_t q=p+1;q<n;++q) {
            const double magnitude=std::abs(a[p*n+q]);
            largest=std::max(largest,magnitude);
            if (magnitude<=threshold) continue;
            const Complex phase=a[p*n+q]/magnitude;
            const double tau=(a[q*n+q].real()-a[p*n+p].real())/(2*magnitude);
            const double t=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau));
            const double c=1/std::sqrt(1+t*t), s=t*c;
            for (std::size_t i=0;i<n;++i) {
                if (i!=p && i!=q) {
                    const Complex ip=a[i*n+p], iq=a[i*n+q];
                    a[i*n+p]=c*ip-s*std::conj(phase)*iq;
                    a[i*n+q]=s*phase*ip+c*iq;
                    a[p*n+i]=std::conj(a[i*n+p]); a[q*n+i]=std::conj(a[i*n+q]);
                }
                const Complex vp=result.vectors[i*n+p], vq=result.vectors[i*n+q];
                result.vectors[i*n+p]=c*vp-s*std::conj(phase)*vq;
                result.vectors[i*n+q]=s*phase*vp+c*vq;
            }
            a[p*n+p]=a[p*n+p].real()-t*magnitude;
            a[q*n+q]=a[q*n+q].real()+t*magnitude;
            a[p*n+q]=a[q*n+p]=0.;
        }
        if (largest<=threshold) { converged=true; break; }
    }
    if (!converged) throw std::runtime_error("Hermitian Davidson Jacobi eigensolver failed");
    std::vector<std::size_t> order(n); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto i,auto j){return a[i*n+i].real()<a[j*n+j].real();});
    auto vectors=result.vectors;
    for (std::size_t j=0;j<n;++j) {
        result.values[j]=a[order[j]*n+order[j]].real();
        for (std::size_t i=0;i<n;++i) result.vectors[i*n+j]=vectors[i*n+order[j]];
    }
#endif
    return result;
}

#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
inline std::size_t davidson_correction_qr(Complex* columns, std::size_t dimension,
    std::size_t corrections, std::size_t stride, BlockDavidsonResult& result) {
    if (!corrections) return 0;
    ++result.householder_qr_blocks;
    int rows=static_cast<int>(dimension), cols=static_cast<int>(corrections);
    int leading=static_cast<int>(stride), lwork=-1, info=0;
    std::vector<int> pivots(cols,0);
    std::vector<Complex> tau(std::min(rows,cols));
    std::vector<double> rwork(2*cols);
    Complex query;
    zgeqp3_(&rows,&cols,columns,&leading,pivots.data(),tau.data(),&query,&lwork,rwork.data(),&info);
    if (info) throw std::runtime_error("Hermitian Davidson QR query failed");
    lwork=static_cast<int>(query.real()); std::vector<Complex> work(lwork);
    zgeqp3_(&rows,&cols,columns,&leading,pivots.data(),tau.data(),work.data(),&lwork,rwork.data(),&info);
    if (info) throw std::runtime_error("Hermitian Davidson QR failed");
    int rank=0;
    while (rank<static_cast<int>(tau.size()) && std::abs(columns[rank*stride+rank])>1.e-14) ++rank;
    if (rank) {
        lwork=-1;
        zungqr_(&rows,&rank,&rank,columns,&leading,tau.data(),&query,&lwork,&info);
        if (info) throw std::runtime_error("Hermitian Davidson Q query failed");
        lwork=static_cast<int>(query.real()); work.resize(lwork);
        zungqr_(&rows,&rank,&rank,columns,&leading,tau.data(),work.data(),&lwork,&info);
        if (info) throw std::runtime_error("Hermitian Davidson Q failed");
    }
    return rank;
}
#endif

/** Real symmetric / complex Hermitian multi-root block Davidson, adapted from E. R. Davidson,
 * J. Comput. Phys. 17, 87-94 (1975), doi:10.1016/0021-9991(75)90065-0.
 * This is a thick-restarted block adaptation, not an exact reproduction:
 * seed up to 2*nroots and retain roots plus guard Ritz vectors (half
 * the root count, bounded between 8 and nroots), use diagonal preconditioning and explicit
 * residual tests. Restarts additionally retain independent previous Ritz
 * directions (GD+k adaptation of Stathopoulos and McCombs, ACM TOMS 37(2),
 * Article 21, 2010, doi:10.1145/1731022.1731031). Unlike PRIMME, this keeps
 * up to min(nroots,16) history vectors, prioritizing the lowest unconverged
 * roots. Converged lowest roots are temporarily locked at restarts; a final
 * coupled Rayleigh-Ritz check restores their cross couplings. Discovery of a
 * lower active root disables locking. This heuristic deflation is not PRIMME's
 * locking policy; no inner JD solve is implemented.
 * No universal iteration or performance guarantee is made.
 * Matvec may accept (x,y,dimension) or a contiguous column block described
 * by (x,y,dimension,columns,stride). Apple builds use BLAS/LAPACK block
 * contractions; real blocks use guarded CholeskyQR2 (Fukaya et al., ScalA 2014,
 * doi:10.1109/ScalA.2014.11) with pivoted QR fallback, and divide-and-conquer
 * projected solves. Complex blocks use pivoted Householder QR directly.
 * The QR adaptation adds column scaling and explicit checks; portable builds
 * use two-pass Gram-Schmidt and symmetric/complex Jacobi projected solves.
 */
template <typename Scalar, typename Matvec>
BlockDavidsonResult block_davidson(
    const std::vector<double>& diagonal,
    const std::vector<std::vector<Scalar>>& guesses,
    std::size_t nroots,
    double tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    BlockDavidsonWorkspace<Scalar>& workspace,
    Matvec&& matvec
) {
    const std::size_t dimension = diagonal.size();
    if (
        dimension == 0
        || nroots == 0
        || nroots > dimension
    ) {
        throw std::invalid_argument(
            "Block Davidson dimensions are invalid."
        );
    }
    if (!std::isfinite(tolerance) || tolerance <= 0) {
        throw std::invalid_argument("Davidson tolerance must be finite and positive");
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "Block Davidson max_iterations must be positive."
        );
    }
    const std::size_t requested_capacity = std::min<std::size_t>(
        dimension,
        std::max<std::size_t>(
            nroots,
            restart_dimension > 0
                ? static_cast<std::size_t>(restart_dimension)
                : static_cast<std::size_t>(max_iterations)
        )
    );
    BlockDavidsonResult result;
    result.workspace_reused = workspace.ensure(
        dimension,
        requested_capacity,
        nroots
    );
    std::fill(workspace.projected.begin(), workspace.projected.end(), 0.0);
    std::fill(
        workspace.best_residuals.begin(),
        workspace.best_residuals.begin() + nroots,
        std::numeric_limits<double>::infinity()
    );

    auto orthonormalize = [dimension](
        Scalar* vector,
        const BlockDavidsonWorkspace<Scalar>& owner,
        std::size_t basis_size
    ) {
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t column = 0;
                column < basis_size;
                ++column
            ) {
                const Scalar overlap = davidson_dot(
                    owner.basis_column(column),
                    vector,
                    dimension
                );
                for (std::size_t index = 0; index < dimension; ++index) {
                    vector[index] -=
                        overlap * owner.basis_column(column)[index];
                }
            }
        }
        const double norm = davidson_norm(vector, dimension);
        if (norm <= 1.0e-12) {
            return false;
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            vector[index] /= norm;
        }
        return true;
    };

    std::size_t basis_size = 0;
    const std::size_t initial_size = std::min(requested_capacity, 2*nroots);
    for (const auto& guess : guesses) {
        if (basis_size >= initial_size || guess.size() != dimension) {
            continue;
        }
        std::copy(
            guess.begin(),
            guess.end(),
            workspace.basis_column(basis_size)
        );
        if (orthonormalize(
            workspace.basis_column(basis_size),
            workspace,
            basis_size
        )) {
            ++basis_size;
        }
    }
    std::vector<std::size_t> diagonal_order(dimension);
    std::iota(diagonal_order.begin(), diagonal_order.end(), 0);
    std::stable_sort(
        diagonal_order.begin(),
        diagonal_order.end(),
        [&diagonal](std::size_t left, std::size_t right) {
            return diagonal[left] < diagonal[right];
        }
    );
    for (const std::size_t seed : diagonal_order) {
        if (basis_size >= initial_size) {
            break;
        }
        Scalar* vector = workspace.basis_column(basis_size);
        std::fill(vector, vector + dimension, 0.0);
        vector[seed] = 1.0;
        if (guesses.empty() || orthonormalize(vector, workspace, basis_size)) {
            ++basis_size;
        }
    }
    if (basis_size < nroots) {
        throw std::runtime_error(
            "Block Davidson could not construct the root seed space."
        );
    }

    std::size_t computed_basis_size = 0;
    std::size_t previous_size = 0;
    std::size_t locked = 0;
    bool allow_locking = true;
    std::vector<Scalar> previous_coefficients(requested_capacity*nroots, 0.);
    std::vector<double> current_energies(nroots, 0.0);
    std::vector<double> current_residuals(
        nroots,
        std::numeric_limits<double>::infinity()
    );
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [&](auto start) {
        return std::chrono::duration<double>(clock_now()-start).count();
    };
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        auto started = clock_now();
        if constexpr (std::is_invocable_v<Matvec, const Scalar*, Scalar*,
                                         std::size_t, std::size_t, std::size_t>) {
            if (basis_size > computed_basis_size)
                matvec(workspace.basis_column(computed_basis_size),
                    workspace.h_basis_column(computed_basis_size), dimension,
                    basis_size-computed_basis_size, workspace.dimension_capacity);
        } else {
            for (std::size_t newest=computed_basis_size; newest<basis_size; ++newest)
                matvec(workspace.basis_column(newest), workspace.h_basis_column(newest), dimension);
        }
        result.matvec_calls += basis_size-computed_basis_size;
        result.action_seconds += elapsed(started);
        started = clock_now();
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
        const std::size_t added = basis_size-computed_basis_size;
        if (added) {
            davidson_gemm(101, 111, 113, basis_size, added, dimension, 1.,
                workspace.basis.data(), workspace.dimension_capacity,
                workspace.h_basis_column(computed_basis_size), workspace.dimension_capacity,
                0., workspace.projected.data()+computed_basis_size, workspace.capacity);
            for (std::size_t i=0;i<basis_size;++i)
                for (std::size_t j=computed_basis_size;j<basis_size;++j)
                    workspace.projected[i*workspace.capacity+j]=davidson_conj(workspace.projected[i*workspace.capacity+j]);
            for (std::size_t newest=computed_basis_size; newest<basis_size; ++newest)
                for (std::size_t index=0; index<newest; ++index)
                    workspace.projected[newest*workspace.capacity+index] =
                        davidson_conj(workspace.projected[index*workspace.capacity+newest]);
        }
#else
        for (
            std::size_t newest = computed_basis_size;
            newest < basis_size;
            ++newest
        ) {
            for (std::size_t index = 0; index <= newest; ++index) {
                const Scalar element = davidson_dot(
                    workspace.basis_column(index),
                    workspace.h_basis_column(newest),
                    dimension
                );
                workspace.projected[
                    index * workspace.capacity + newest
                ] = element;
                workspace.projected[
                    newest * workspace.capacity + index
                ] = davidson_conj(element);
            }
        }
#endif
        computed_basis_size = basis_size;
        for (std::size_t row = 0; row < basis_size; ++row) {
            for (std::size_t col = 0; col < basis_size; ++col) {
                workspace.projected_dense[row * basis_size + col] =
                    workspace.projected[
                        row * workspace.capacity + col
                    ];
            }
        }
        result.projection_seconds += elapsed(started);
        started = clock_now();
        auto eigenpairs = [&] {
            if (!locked)
                return davidson_projected_eigh(workspace.projected_dense, basis_size);
            const std::size_t active = basis_size-locked;
            std::vector<Scalar> projected(active*active);
            for (std::size_t i=0;i<active;++i)
                for (std::size_t j=0;j<active;++j)
                    projected[i*active+j]=workspace.projected_dense[(i+locked)*basis_size+j+locked];
            auto reduced=davidson_projected_eigh(projected,active);
            // A newly discovered lower state invalidates prefix locking.
            if (reduced.values.front()<current_energies[locked-1]) {
                locked=0;
                allow_locking=false;
                ++result.deflation_unlocks;
                return davidson_projected_eigh(workspace.projected_dense,basis_size);
            }
            decltype(reduced) full;
            full.values.resize(basis_size);
            full.vectors.assign(basis_size*basis_size,Scalar(0.));
            for (std::size_t i=0;i<locked;++i) {
                full.values[i]=current_energies[i];
                full.vectors[i*basis_size+i]=1.;
            }
            for (std::size_t i=0;i<active;++i) {
                full.values[i+locked]=reduced.values[i];
                for (std::size_t j=0;j<active;++j)
                    full.vectors[(i+locked)*basis_size+j+locked]=reduced.vectors[i*active+j];
            }
            ++result.deflated_iterations;
            return full;
        }();
        result.diagonalization_seconds += elapsed(started);
        started = clock_now();
        davidson_rotate(workspace.basis_column(locked), workspace.dimension_capacity,
            dimension, basis_size-locked, eigenpairs.vectors.data()+locked*basis_size+locked, basis_size,
            nroots-locked, workspace.root_vector(workspace.ritz,locked));
        davidson_rotate(workspace.h_basis_column(locked), workspace.dimension_capacity,
            dimension, basis_size-locked, eigenpairs.vectors.data()+locked*basis_size+locked, basis_size,
            nroots-locked, workspace.root_vector(workspace.h_ritz,locked));
        bool all_converged = true;
        for (std::size_t root = 0; root < nroots; ++root) {
            Scalar* ritz = workspace.root_vector(workspace.ritz, root);
            Scalar* h_ritz = workspace.root_vector(
                workspace.h_ritz,
                root
            );
            Scalar* residual = workspace.root_vector(
                workspace.residual,
                root
            );
            current_energies[root] = eigenpairs.values[root];
            for (std::size_t index = 0; index < dimension; ++index) {
                residual[index] = h_ritz[index]
                    - current_energies[root] * ritz[index];
            }
            current_residuals[root] = davidson_norm(residual, dimension);
            // Return one common Ritz space, not independently selected roots
            // from different iterations (which need not be orthogonal).
            {
                workspace.best_residuals[root] = current_residuals[root];
                workspace.best_energies[root] = current_energies[root];
                std::copy(
                    ritz,
                    ritz + dimension,
                    workspace.root_vector(workspace.best_vectors, root)
                );
            }
            all_converged = all_converged
                && current_residuals[root] < tolerance;
        }
        result.ritz_seconds += elapsed(started);
        if (all_converged) {
            // Restore all cross couplings and perform a common Rayleigh-Ritz
            // check before accepting independently frozen vectors.
            if (locked) {
                locked=0;
                allow_locking=false;
                continue;
            }
            result.converged = true;
            break;
        }

        started = clock_now();
        std::size_t corrections = 0;
        for (std::size_t root = 0; root < nroots; ++root) {
            if (current_residuals[root] < tolerance) {
                continue;
            }
            Scalar* correction = workspace.root_vector(
                workspace.correction,
                corrections
            );
            const Scalar* residual = workspace.root_vector(
                workspace.residual,
                root
            );
            for (std::size_t index = 0; index < dimension; ++index) {
                double denominator = current_energies[root]
                    - diagonal[index];
                if (std::abs(denominator) < 1.0e-12) {
                    denominator = denominator >= 0.0
                        ? 1.0e-12
                        : -1.0e-12;
                }
                correction[index] = residual[index] / denominator;
            }
            ++corrections;
        }
        // Project the complete correction block against the search space.
        std::vector<Scalar> overlaps(corrections*basis_size);
        for (int pass=0; pass<2 && corrections; ++pass) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
            davidson_gemm(101, 111, 113, corrections, basis_size, dimension, 1.,
                workspace.correction.data(), workspace.dimension_capacity,
                workspace.basis.data(), workspace.dimension_capacity,
                0., overlaps.data(), basis_size);
            davidson_gemm(101, 111, 111, corrections, dimension, basis_size, -1.,
                overlaps.data(), basis_size, workspace.basis.data(), workspace.dimension_capacity,
                1., workspace.correction.data(), workspace.dimension_capacity);
#else
            for (std::size_t root=0; root<corrections; ++root)
                for (std::size_t col=0; col<basis_size; ++col) {
                    Scalar* v=workspace.root_vector(workspace.correction, root);
                    const Scalar* b=workspace.basis_column(col);
                    const Scalar overlap=davidson_dot(b,v,dimension);
                    for (std::size_t i=0;i<dimension;++i) v[i]-=overlap*b[i];
                }
#endif
        }
        std::size_t independent = 0;
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
        independent=davidson_correction_qr(workspace.correction.data(),dimension,
            corrections,workspace.dimension_capacity,result);
#else
        for (std::size_t candidate=0; candidate<corrections; ++candidate) {
            Scalar* correction=workspace.root_vector(workspace.correction,candidate);
            for (int pass = 0; pass < 2; ++pass) {
                for (
                    std::size_t previous = 0;
                    previous < independent;
                    ++previous
                ) {
                    const Scalar* prior = workspace.root_vector(
                        workspace.correction,
                        previous
                    );
                    const Scalar overlap = davidson_dot(
                        prior,
                        correction,
                        dimension
                    );
                    for (
                        std::size_t index = 0;
                        index < dimension;
                        ++index
                    ) {
                        correction[index] -= overlap * prior[index];
                    }
                }
            }
            const double norm = davidson_norm(correction, dimension);
            if (norm <= 1.0e-14) {
                continue;
            }
            for (std::size_t index = 0; index < dimension; ++index) {
                correction[index] /= norm;
            }
            if (independent != candidate)
                std::copy(correction, correction+dimension,
                    workspace.root_vector(workspace.correction, independent));
            ++independent;
        }
#endif
        corrections = independent;
        result.correction_seconds += elapsed(started);
        if (corrections == 0) {
            break;
        }

        if (basis_size + corrections > requested_capacity) {
            started = clock_now();
            const std::size_t old_size=basis_size;
            const std::size_t limit=std::min({basis_size,3*nroots,
                std::max(nroots,requested_capacity-corrections)});
            std::size_t retained=std::min(limit,
                nroots+std::min(nroots,std::max<std::size_t>(8,nroots/2)));
            const std::size_t current_retained=retained;
            // Bound history so it does not crowd out fresh correction blocks.
            const std::size_t history_limit=std::min(
                limit,retained+std::min<std::size_t>(nroots,16));
            std::vector<Scalar> coefficients=eigenpairs.vectors;
            std::vector<Scalar> candidate(basis_size);
            // Keep previous Ritz directions orthogonal to the retained current
            // Ritz space: locally optimal (+k) restart, in small coordinates.
            for (std::size_t root=0;root<nroots && retained<history_limit && previous_size;++root) {
                if (current_residuals[root]<tolerance) continue;
                std::fill(candidate.begin(),candidate.end(),0.);
                for (std::size_t i=0;i<previous_size;++i)
                    candidate[i]=previous_coefficients[i*nroots+root];
                for (int pass=0;pass<2;++pass) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
                    std::vector<Scalar> overlaps(retained);
                    davidson_gemm(101,113,111,retained,1,basis_size,1.,
                        coefficients.data(),basis_size,candidate.data(),1,0.,overlaps.data(),1);
                    davidson_gemm(101,111,111,basis_size,1,retained,-1.,
                        coefficients.data(),basis_size,overlaps.data(),1,1.,candidate.data(),1);
#else
                    for (std::size_t j=0;j<retained;++j) {
                        Scalar dot=0.;
                        for (std::size_t i=0;i<basis_size;++i) dot+=davidson_conj(coefficients[i*basis_size+j])*candidate[i];
                        for (std::size_t i=0;i<basis_size;++i) candidate[i]-=dot*coefficients[i*basis_size+j];
                    }
#endif
                }
                const double norm=davidson_norm(candidate.data(),basis_size);
                if (norm<=1.e-10) continue;
                for (std::size_t i=0;i<basis_size;++i) coefficients[i*basis_size+retained]=candidate[i]/norm;
                ++retained;
            }
            result.restart_history_vectors += retained-current_retained;
            // The requested Ritz vectors and their images already exist.
            // Rotate only guard/history columns, preserving those buffers.
            davidson_rotate(workspace.basis.data(), workspace.dimension_capacity,
                dimension, basis_size, coefficients.data()+nroots, basis_size,
                retained-nroots, workspace.root_vector(workspace.ritz,nroots));
            davidson_rotate(workspace.h_basis.data(), workspace.dimension_capacity,
                dimension, basis_size, coefficients.data()+nroots, basis_size,
                retained-nroots, workspace.root_vector(workspace.h_ritz,nroots));
            for (std::size_t root=0;root<retained;++root) {
                std::copy(workspace.root_vector(workspace.ritz,root),
                    workspace.root_vector(workspace.ritz,root)+dimension,workspace.basis_column(root));
                std::copy(workspace.root_vector(workspace.h_ritz,root),
                    workspace.root_vector(workspace.h_ritz,root)+dimension,workspace.h_basis_column(root));
            }
            // History directions are not Ritz vectors, so carry their full
            // projected Hamiltonian instead of resetting it to a diagonal.
            std::vector<Scalar> product(old_size*retained,0.);
            std::fill(workspace.projected.begin(),workspace.projected.end(),0.);
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
            davidson_gemm(101,111,111,old_size,retained,old_size,1.,
                workspace.projected_dense.data(),old_size,coefficients.data(),old_size,
                0.,product.data(),retained);
            davidson_gemm(101,113,111,retained,retained,old_size,1.,
                coefficients.data(),old_size,product.data(),retained,
                0.,workspace.projected.data(),workspace.capacity);
#else
            for (std::size_t i=0;i<old_size;++i)
                for (std::size_t j=0;j<retained;++j)
                    for (std::size_t l=0;l<old_size;++l)
                        product[i*retained+j]+=workspace.projected_dense[i*old_size+l]*coefficients[l*old_size+j];
            for (std::size_t i=0;i<retained;++i)
                for (std::size_t j=0;j<retained;++j)
                    for (std::size_t l=0;l<old_size;++l)
                        workspace.projected[i*workspace.capacity+j]+=davidson_conj(coefficients[l*old_size+i])*product[l*retained+j];
#endif
            for (std::size_t i=0;i<retained;++i)
                for (std::size_t j=0;j<i;++j) {
                    const Scalar value=.5*(workspace.projected[i*workspace.capacity+j]+davidson_conj(workspace.projected[j*workspace.capacity+i]));
                    workspace.projected[i*workspace.capacity+j]=value;
                    workspace.projected[j*workspace.capacity+i]=davidson_conj(value);
                }
            basis_size=retained;
            computed_basis_size=retained;
            if (allow_locking) {
                // Only lock a lowest contiguous prefix at a restart, when
                // those Ritz vectors are explicit basis columns. A tighter
                // gate leaves room for the final coupled residual check.
                while (locked+1<nroots && current_residuals[locked]<0.9*tolerance)
                    ++locked;
                result.locked_roots=std::max(result.locked_roots,locked);
            }
            previous_size=retained;
            std::fill(previous_coefficients.begin(),previous_coefficients.end(),0.);
            for (std::size_t root=0;root<nroots;++root) previous_coefficients[root*nroots+root]=1.;
            ++result.restarts;
            result.restart_seconds += elapsed(started);
        } else {
            previous_size=basis_size;
            std::fill(previous_coefficients.begin(),previous_coefficients.end(),0.);
            for (std::size_t i=0;i<basis_size;++i)
                for (std::size_t root=0;root<nroots;++root)
                    previous_coefficients[i*nroots+root]=eigenpairs.vectors[i*basis_size+root];
        }
        const std::size_t available = requested_capacity - basis_size;
        const std::size_t append = std::min(corrections, available);
        for (std::size_t correction = 0; correction < append; ++correction) {
            std::copy(
                workspace.root_vector(workspace.correction, correction),
                workspace.root_vector(workspace.correction, correction)
                    + dimension,
                workspace.basis_column(basis_size + correction)
            );
        }
        basis_size += append;
        if (append == 0) {
            break;
        }
    }

    result.basis_size = basis_size;
    result.accepted = result.converged || accept_unconverged;
    result.energies.assign(
        workspace.best_energies.begin(),
        workspace.best_energies.begin() + nroots
    );
    result.residual_norms.assign(
        workspace.best_residuals.begin(),
        workspace.best_residuals.begin() + nroots
    );
    if (result.accepted) {
        result.vectors.resize(nroots);
        for (std::size_t root = 0; root < nroots; ++root) {
            const Scalar* source = workspace.root_vector(
                workspace.best_vectors,
                root
            );
            auto& vector = result.vectors[root];
            vector.resize(dimension);
            std::size_t pivot = 0;
            double maximum = 0.0;
            for (std::size_t index = 0; index < dimension; ++index) {
                if (std::abs(source[index]) > maximum) {
                    maximum = std::abs(source[index]);
                    pivot = index;
                }
            }
            const Scalar phase = maximum > 0.0 ? davidson_conj(source[pivot])/maximum : Scalar(1.);
            for (std::size_t index = 0; index < dimension; ++index) {
                vector[index] = phase * source[index];
            }
        }
    }
    return result;
}

template <typename Matvec>
BlockDavidsonResult real_block_davidson(const std::vector<double>& diagonal,
    const std::vector<std::vector<double>>& guesses, std::size_t roots, double tolerance,
    int iterations, int space, bool accept_unconverged, RealBlockDavidsonWorkspace& workspace,
    Matvec&& matvec) {
    return block_davidson(diagonal,guesses,roots,tolerance,iterations,space,
        accept_unconverged,workspace,std::forward<Matvec>(matvec));
}

template <typename Matvec>
DavidsonResult real_davidson(
    const std::vector<double>& diagonal,
    std::vector<double> guess,
    double tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    RealDavidsonWorkspace& workspace,
    Matvec&& matvec
) {
    const std::size_t dimension = diagonal.size();
    if (dimension == 0 || guess.size() != dimension) {
        throw std::invalid_argument(
            "Real Davidson vector dimensions differ."
        );
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "Real Davidson max_iterations must be positive."
        );
    }
    const double guess_norm = real_norm(guess.data(), dimension);
    if (guess_norm < 1.0e-14) {
        throw std::invalid_argument(
            "Initial real Davidson vector has near-zero norm."
        );
    }
    for (double& value : guess) {
        value /= guess_norm;
    }

    const std::size_t requested_capacity = static_cast<std::size_t>(
        restart_dimension > 0 ? restart_dimension : max_iterations
    );
    DavidsonResult result;
    result.workspace_reused = workspace.ensure(
        dimension,
        std::max<std::size_t>(
            std::min<std::size_t>(dimension, 2),
            requested_capacity
        )
    );
    std::fill(workspace.projected.begin(), workspace.projected.end(), 0.0);
    std::fill(
        workspace.best_vector.begin(),
        workspace.best_vector.begin() + dimension,
        0.0
    );
    std::copy(
        guess.begin(),
        guess.end(),
        workspace.basis_column(0)
    );
    std::size_t basis_size = 1;
    if (dimension > 1 && workspace.capacity > 1) {
        const std::size_t diagonal_index = static_cast<std::size_t>(
            std::min_element(diagonal.begin(), diagonal.end())
                - diagonal.begin()
        );
        double* seed = workspace.basis_column(1);
        std::fill(seed, seed + dimension, 0.0);
        seed[diagonal_index] = 1.0;
        const double overlap = real_dot(
            workspace.basis_column(0),
            seed,
            dimension
        );
        for (std::size_t index = 0; index < dimension; ++index) {
            seed[index] -= workspace.basis_column(0)[index] * overlap;
        }
        const double seed_norm = real_norm(seed, dimension);
        if (seed_norm > 1.0e-12) {
            for (std::size_t index = 0; index < dimension; ++index) {
                seed[index] /= seed_norm;
            }
            basis_size = 2;
        }
    }
    std::size_t computed_basis_size = 0;
    double best_residual = std::numeric_limits<double>::infinity();
    double best_energy = 0.0;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        for (
            std::size_t newest = computed_basis_size;
            newest < basis_size;
            ++newest
        ) {
            matvec(
                workspace.basis_column(newest),
                workspace.h_basis_column(newest),
                dimension
            );
            ++result.matvec_calls;
            for (std::size_t index = 0; index <= newest; ++index) {
                const double element = real_dot(
                    workspace.basis_column(index),
                    workspace.h_basis_column(newest),
                    dimension
                );
                workspace.projected[
                    index * workspace.capacity + newest
                ] = element;
                workspace.projected[
                    newest * workspace.capacity + index
                ] = element;
            }
        }
        computed_basis_size = basis_size;
        std::vector<double> projected(basis_size * basis_size, 0.0);
        for (std::size_t row = 0; row < basis_size; ++row)
        for (std::size_t col = 0; col < basis_size; ++col) {
            projected[row * basis_size + col] = workspace.projected[
                row * workspace.capacity + col
            ];
        }
        const RealSymmetricEigendecomposition eigenpairs = symmetric_eigh(
            projected,
            basis_size
        );
        result.energy = eigenpairs.values[0];
        for (std::size_t index = 0; index < basis_size; ++index) {
            workspace.coefficients[index] =
                eigenpairs.vectors[index * basis_size];
        }
        std::fill(
            workspace.ritz.begin(),
            workspace.ritz.begin() + dimension,
            0.0
        );
        std::fill(
            workspace.h_ritz.begin(),
            workspace.h_ritz.begin() + dimension,
            0.0
        );
        for (std::size_t index = 0; index < basis_size; ++index) {
            real_axpy(
                workspace.ritz.data(),
                workspace.coefficients[index],
                workspace.basis_column(index),
                dimension
            );
            real_axpy(
                workspace.h_ritz.data(),
                workspace.coefficients[index],
                workspace.h_basis_column(index),
                dimension
            );
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.residual[index] =
                workspace.h_ritz[index]
                - result.energy * workspace.ritz[index];
        }
        result.residual_norm = real_norm(
            workspace.residual.data(),
            dimension
        );
        if (result.residual_norm < best_residual) {
            best_residual = result.residual_norm;
            best_energy = result.energy;
            std::copy(
                workspace.ritz.begin(),
                workspace.ritz.begin() + dimension,
                workspace.best_vector.begin()
            );
        }
        if (result.residual_norm < tolerance) {
            result.converged = true;
            break;
        }
        double olsen_numerator = 0.0;
        double olsen_denominator = 0.0;
        for (std::size_t index = 0; index < dimension; ++index) {
            double denominator = result.energy - diagonal[index];
            if (std::abs(denominator) < 1.0e-12) {
                denominator = denominator >= 0.0 ? 1.0e-12 : -1.0e-12;
            }
            workspace.correction[index] =
                workspace.residual[index] / denominator;
            workspace.h_ritz[index] =
                workspace.ritz[index] / denominator;
            olsen_numerator +=
                workspace.ritz[index] * workspace.correction[index];
            olsen_denominator +=
                workspace.ritz[index] * workspace.h_ritz[index];
        }
        if (std::abs(olsen_denominator) > 1.0e-14) {
            const double olsen_scale =
                olsen_numerator / olsen_denominator;
            for (std::size_t index = 0; index < dimension; ++index) {
                workspace.correction[index] -=
                    olsen_scale * workspace.h_ritz[index];
            }
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t basis_index = 0;
                basis_index < basis_size;
                ++basis_index
            ) {
                const double overlap = real_dot(
                    workspace.basis_column(basis_index),
                    workspace.correction.data(),
                    dimension
                );
                for (std::size_t index = 0; index < dimension; ++index) {
                    workspace.correction[index] -=
                        workspace.basis_column(basis_index)[index]
                        * overlap;
                }
            }
        }
        const double correction_norm = real_norm(
            workspace.correction.data(),
            dimension
        );
        if (
            correction_norm < 1.0e-10
            || basis_size >= workspace.capacity
        ) {
            if (
                correction_norm < 1.0e-10
                || workspace.capacity < 2
            ) {
                break;
            }
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.correction[index] /= correction_norm;
        }
        if (basis_size >= workspace.capacity) {
            const std::size_t restart_keep = std::min<std::size_t>(
                {
                    basis_size,
                    workspace.capacity - 1,
                    4,
                    std::max<std::size_t>(2, workspace.capacity / 32),
                }
            );
            for (std::size_t retained = 0; retained < restart_keep; ++retained) {
                double* restart_vector =
                    workspace.restart_basis_column(retained);
                double* restart_h_vector =
                    workspace.restart_h_basis_column(retained);
                std::fill(restart_vector, restart_vector + dimension, 0.0);
                std::fill(restart_h_vector, restart_h_vector + dimension, 0.0);
                for (std::size_t source = 0; source < basis_size; ++source) {
                    const double coefficient = eigenpairs.vectors[
                        source * basis_size + retained
                    ];
                    real_axpy(
                        restart_vector,
                        coefficient,
                        workspace.basis_column(source),
                        dimension
                    );
                    real_axpy(
                        restart_h_vector,
                        coefficient,
                        workspace.h_basis_column(source),
                        dimension
                    );
                }
            }
            for (std::size_t retained = 0; retained < restart_keep; ++retained) {
                std::copy(
                    workspace.restart_basis_column(retained),
                    workspace.restart_basis_column(retained) + dimension,
                    workspace.basis_column(retained)
                );
                std::copy(
                    workspace.restart_h_basis_column(retained),
                    workspace.restart_h_basis_column(retained) + dimension,
                    workspace.h_basis_column(retained)
                );
            }
            std::copy(
                workspace.correction.begin(),
                workspace.correction.begin() + dimension,
                workspace.basis_column(restart_keep)
            );
            std::fill(
                workspace.projected.begin(),
                workspace.projected.end(),
                0.0
            );
            for (std::size_t retained = 0; retained < restart_keep; ++retained) {
                workspace.projected[
                    retained * workspace.capacity + retained
                ] = eigenpairs.values[retained];
            }
            basis_size = restart_keep + 1;
            computed_basis_size = restart_keep;
            ++result.restarts;
            continue;
        }
        std::copy(
            workspace.correction.begin(),
            workspace.correction.begin() + dimension,
            workspace.basis_column(basis_size)
        );
        ++basis_size;
    }

    result.basis_size = basis_size;
    if (std::isfinite(best_residual)) {
        result.energy = best_energy;
        result.residual_norm = best_residual;
    }
    result.accepted = result.converged || accept_unconverged;
    if (result.accepted) {
        result.vector.resize(dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            result.vector[index] = Complex{
                workspace.best_vector[index],
                0.0,
            };
        }
    }
    return result;
}

template <typename HMatvec, typename NMatvec>
DavidsonResult real_generalized_davidson(
    const std::vector<double>& h_diagonal,
    const std::vector<double>& n_diagonal,
    std::vector<double> guess,
    double energy_tolerance,
    double residual_tolerance,
    double linear_dependence_tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    RealGeneralizedDavidsonWorkspace& workspace,
    HMatvec&& h_matvec,
    NMatvec&& n_matvec
) {
    const std::size_t dimension = h_diagonal.size();
    if (
        dimension == 0
        || n_diagonal.size() != dimension
        || guess.size() != dimension
    ) {
        throw std::invalid_argument(
            "Real generalized Davidson vector dimensions differ."
        );
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "Real generalized Davidson max_iterations must be positive."
        );
    }
    const std::size_t requested_capacity = std::max<std::size_t>(
        std::min<std::size_t>(dimension, 2),
        static_cast<std::size_t>(
            restart_dimension > 0 ? restart_dimension : max_iterations
        )
    );
    DavidsonResult result;
    result.workspace_reused = workspace.ensure(
        dimension,
        requested_capacity
    );
    std::fill(workspace.projected.begin(), workspace.projected.end(), 0.0);

    auto apply_h = [&](const double* input, double* output) {
        h_matvec(input, output, dimension);
        ++result.matvec_calls;
    };
    auto apply_n = [&](const double* input, double* output) {
        n_matvec(input, output, dimension);
        ++result.norm_matvec_calls;
    };
    auto metric_normalize = [&](double* vector, double* n_vector) {
        const double norm_squared = real_dot(vector, n_vector, dimension);
        if (
            !std::isfinite(norm_squared)
            || norm_squared
                <= linear_dependence_tolerance * linear_dependence_tolerance
        ) {
            return false;
        }
        const double inverse_norm = 1.0 / std::sqrt(norm_squared);
        for (std::size_t index = 0; index < dimension; ++index) {
            vector[index] *= inverse_norm;
            n_vector[index] *= inverse_norm;
        }
        return true;
    };
    auto metric_orthogonalize = [&](
        double* vector,
        double* n_vector,
        std::size_t basis_size
    ) {
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t basis_index = 0;
                basis_index < basis_size;
                ++basis_index
            ) {
                const double overlap = real_dot(
                    workspace.basis_column(basis_index),
                    n_vector,
                    dimension
                );
                for (std::size_t index = 0; index < dimension; ++index) {
                    vector[index] -=
                        workspace.basis_column(basis_index)[index] * overlap;
                    n_vector[index] -=
                        workspace.n_basis_column(basis_index)[index] * overlap;
                }
            }
        }
        return metric_normalize(vector, n_vector);
    };

    std::copy(
        guess.begin(),
        guess.end(),
        workspace.basis_column(0)
    );
    apply_n(
        workspace.basis_column(0),
        workspace.n_basis_column(0)
    );
    if (
        !metric_normalize(
            workspace.basis_column(0),
            workspace.n_basis_column(0)
        )
    ) {
        throw std::invalid_argument(
            "Initial real Davidson vector is singular in the local metric."
        );
    }
    std::copy(
        workspace.basis_column(0),
        workspace.basis_column(0) + dimension,
        workspace.reference.begin()
    );
    std::size_t basis_size = 1;
    std::size_t computed_basis_size = 0;
    double previous_energy = std::numeric_limits<double>::infinity();
    const std::size_t minimum_explored_dimension = std::min<std::size_t>(
        std::min<std::size_t>(dimension, workspace.capacity),
        16
    );

    auto append_seed = [&](std::size_t used_basis) {
        std::vector<std::size_t> order(dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            order[index] = index;
        }
        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t left, std::size_t right) {
                const double left_norm =
                    std::abs(n_diagonal[left]) > 1.0e-12
                    ? n_diagonal[left]
                    : 1.0;
                const double right_norm =
                    std::abs(n_diagonal[right]) > 1.0e-12
                    ? n_diagonal[right]
                    : 1.0;
                return h_diagonal[left] / left_norm
                    < h_diagonal[right] / right_norm;
            }
        );
        for (const std::size_t seed_index : order) {
            std::fill(
                workspace.correction.begin(),
                workspace.correction.begin() + dimension,
                0.0
            );
            workspace.correction[seed_index] = 1.0;
            apply_n(
                workspace.correction.data(),
                workspace.n_correction.data()
            );
            if (
                metric_orthogonalize(
                    workspace.correction.data(),
                    workspace.n_correction.data(),
                    used_basis
                )
            ) {
                return true;
            }
        }
        return false;
    };

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        for (
            std::size_t newest = computed_basis_size;
            newest < basis_size;
            ++newest
        ) {
            apply_h(
                workspace.basis_column(newest),
                workspace.h_basis_column(newest)
            );
            for (std::size_t index = 0; index <= newest; ++index) {
                const double element = real_dot(
                    workspace.basis_column(index),
                    workspace.h_basis_column(newest),
                    dimension
                );
                workspace.projected[
                    index * workspace.capacity + newest
                ] = element;
                workspace.projected[
                    newest * workspace.capacity + index
                ] = element;
            }
        }
        computed_basis_size = basis_size;
        for (std::size_t row = 0; row < basis_size; ++row) {
            for (std::size_t col = 0; col < basis_size; ++col) {
                workspace.projected_dense[row * basis_size + col] =
                    workspace.projected[row * workspace.capacity + col];
            }
        }
        std::vector<Complex> projected_complex(
            basis_size * basis_size
        );
        for (std::size_t index = 0; index < projected_complex.size(); ++index) {
            projected_complex[index] = Complex(
                workspace.projected_dense[index],
                0.0
            );
        }
        const LowestEigenpair eigenpair = lowest_projected_eigenpair(
            projected_complex,
            basis_size
        );
        result.energy = eigenpair.value;
        for (std::size_t index = 0; index < basis_size; ++index) {
            workspace.coefficients[index] =
                eigenpair.vector[index].real();
        }
        std::fill(
            workspace.ritz.begin(),
            workspace.ritz.begin() + dimension,
            0.0
        );
        std::fill(
            workspace.h_ritz.begin(),
            workspace.h_ritz.begin() + dimension,
            0.0
        );
        std::fill(
            workspace.n_ritz.begin(),
            workspace.n_ritz.begin() + dimension,
            0.0
        );
        for (std::size_t index = 0; index < basis_size; ++index) {
            real_axpy(
                workspace.ritz.data(),
                workspace.coefficients[index],
                workspace.basis_column(index),
                dimension
            );
            real_axpy(
                workspace.h_ritz.data(),
                workspace.coefficients[index],
                workspace.h_basis_column(index),
                dimension
            );
            real_axpy(
                workspace.n_ritz.data(),
                workspace.coefficients[index],
                workspace.n_basis_column(index),
                dimension
            );
        }
        std::copy(
            workspace.h_ritz.begin(),
            workspace.h_ritz.begin() + dimension,
            workspace.residual.begin()
        );
        for (std::size_t index = 0; index < dimension; ++index) {
            workspace.residual[index] -=
                result.energy * workspace.n_ritz[index];
        }
        result.residual_norm = real_norm(
            workspace.residual.data(),
            dimension
        );
        std::copy(
            workspace.ritz.begin(),
            workspace.ritz.begin() + dimension,
            workspace.best_vector.begin()
        );
        const double energy_change = std::abs(
            result.energy - previous_energy
        );
        if (
            result.residual_norm <= residual_tolerance
            && energy_change <= energy_tolerance
            && basis_size >= minimum_explored_dimension
        ) {
            result.converged = true;
            break;
        }

        for (std::size_t index = 0; index < dimension; ++index) {
            double denominator =
                result.energy * n_diagonal[index] - h_diagonal[index];
            if (std::abs(denominator) < 1.0e-12) {
                denominator = denominator >= 0.0 ? 1.0e-12 : -1.0e-12;
            }
            workspace.correction[index] =
                workspace.residual[index] / denominator;
        }
        apply_n(
            workspace.correction.data(),
            workspace.n_correction.data()
        );
        bool correction_valid = metric_orthogonalize(
            workspace.correction.data(),
            workspace.n_correction.data(),
            basis_size
        );
        if (!correction_valid) {
            correction_valid = append_seed(basis_size);
        }
        if (!correction_valid) {
            result.converged =
                result.residual_norm <= residual_tolerance;
            break;
        }

        if (basis_size >= workspace.capacity) {
            if (workspace.capacity < 2) {
                break;
            }
            std::copy(
                workspace.ritz.begin(),
                workspace.ritz.begin() + dimension,
                workspace.basis_column(0)
            );
            std::copy(
                workspace.n_ritz.begin(),
                workspace.n_ritz.begin() + dimension,
                workspace.n_basis_column(0)
            );
            if (
                !metric_normalize(
                    workspace.basis_column(0),
                    workspace.n_basis_column(0)
                )
            ) {
                break;
            }
            std::copy(
                workspace.correction.begin(),
                workspace.correction.begin() + dimension,
                workspace.basis_column(1)
            );
            std::copy(
                workspace.n_correction.begin(),
                workspace.n_correction.begin() + dimension,
                workspace.n_basis_column(1)
            );
            std::fill(
                workspace.projected.begin(),
                workspace.projected.end(),
                0.0
            );
            basis_size = 2;
            computed_basis_size = 0;
            ++result.restarts;
        } else {
            std::copy(
                workspace.correction.begin(),
                workspace.correction.begin() + dimension,
                workspace.basis_column(basis_size)
            );
            std::copy(
                workspace.n_correction.begin(),
                workspace.n_correction.begin() + dimension,
                workspace.n_basis_column(basis_size)
            );
            ++basis_size;
        }
        previous_energy = result.energy;
    }

    result.basis_size = basis_size;
    result.accepted = result.converged || accept_unconverged;
    if (result.accepted) {
        const double reference_overlap = real_dot(
            workspace.reference.data(),
            workspace.best_vector.data(),
            dimension
        );
        if (reference_overlap < 0.0) {
            for (std::size_t index = 0; index < dimension; ++index) {
                workspace.best_vector[index] *= -1.0;
            }
        } else if (std::abs(reference_overlap) <= 1.0e-12) {
            const auto pivot = std::max_element(
                workspace.best_vector.begin(),
                workspace.best_vector.begin() + dimension,
                [](double left, double right) {
                    return std::abs(left) < std::abs(right);
                }
            );
            if (pivot != workspace.best_vector.begin() + dimension
                && *pivot < 0.0) {
                for (std::size_t index = 0; index < dimension; ++index) {
                    workspace.best_vector[index] *= -1.0;
                }
            }
        }
        result.vector.resize(dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            result.vector[index] = Complex(
                workspace.best_vector[index],
                0.0
            );
        }
    }
    return result;
}

struct BlockView {
    const Complex* values = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t row_stride = 0;
    std::int64_t col_stride = 1;
    std::int64_t input_start = 0;
    std::int64_t output_start = 0;
};

inline std::vector<Complex> block_matvec(
    const std::vector<BlockView>& blocks,
    const std::vector<Complex>& vector,
    std::size_t dimension
) {
    if (vector.size() != dimension) {
        throw std::invalid_argument("Block matvec vector dimension mismatch.");
    }
    std::vector<Complex> output(dimension, 0.0);
    for (const BlockView& block : blocks) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
        if (
            block.row_stride == block.cols
            && block.col_stride == 1
            && block.rows * block.cols >= 512
            && block.rows <= std::numeric_limits<int>::max()
            && block.cols <= std::numeric_limits<int>::max()
        ) {
            const Complex alpha(1.0, 0.0);
            const Complex beta(1.0, 0.0);
            cblas_zgemv(
                101,
                111,
                static_cast<int>(block.rows),
                static_cast<int>(block.cols),
                &alpha,
                block.values,
                static_cast<int>(block.cols),
                vector.data() + block.input_start,
                1,
                &beta,
                output.data() + block.output_start,
                1
            );
            continue;
        }
#endif
        for (std::int64_t row = 0; row < block.rows; ++row) {
            Complex value = 0.0;
            const Complex* row_values =
                block.values + row * block.row_stride;
            for (std::int64_t col = 0; col < block.cols; ++col) {
                value +=
                    row_values[col * block.col_stride]
                    * vector[
                        static_cast<std::size_t>(block.input_start + col)
                    ];
            }
            output[
                static_cast<std::size_t>(block.output_start + row)
            ] += value;
        }
    }
    return output;
}

inline std::vector<Complex> block_diagonal(
    const std::vector<BlockView>& blocks,
    std::size_t dimension
) {
    std::vector<Complex> diagonal(dimension, 0.0);
    for (const BlockView& block : blocks) {
        const std::int64_t first =
            std::max(block.input_start, block.output_start);
        const std::int64_t last = std::min(
            block.input_start + block.cols,
            block.output_start + block.rows
        );
        for (std::int64_t global = first; global < last; ++global) {
            const std::int64_t row = global - block.output_start;
            const std::int64_t col = global - block.input_start;
            diagonal[static_cast<std::size_t>(global)] +=
                block.values[
                    row * block.row_stride + col * block.col_stride
                ];
        }
    }
    return diagonal;
}

}  // namespace pyqed::linalg
