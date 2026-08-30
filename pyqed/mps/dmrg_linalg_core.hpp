#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pyqed::dmrg {

using Complex = std::complex<double>;

#ifdef __APPLE__
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
#ifdef __APPLE__
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
#ifdef __APPLE__
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
#ifdef __APPLE__
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

struct RealBlockDavidsonWorkspace {
    std::vector<double> basis;
    std::vector<double> h_basis;
    std::vector<double> projected;
    std::vector<double> projected_dense;
    std::vector<double> ritz;
    std::vector<double> h_ritz;
    std::vector<double> residual;
    std::vector<double> correction;
    std::vector<double> best_vectors;
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
                "Real block Davidson workspace dimensions are invalid."
            );
        }
        const bool reused =
            initialized
            && dimension_capacity >= requested_dimension
            && capacity >= requested_capacity
            && roots_capacity >= requested_roots;
        if (!reused) {
            *this = RealBlockDavidsonWorkspace{};
            dimension_capacity = requested_dimension;
            capacity = requested_capacity;
            roots_capacity = requested_roots;
            basis.resize(dimension_capacity * capacity);
            h_basis.resize(dimension_capacity * capacity);
            projected.resize(capacity * capacity);
            projected_dense.resize(capacity * capacity);
            const std::size_t root_elements =
                dimension_capacity * roots_capacity;
            ritz.resize(root_elements);
            h_ritz.resize(root_elements);
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
    double* root_vector(std::vector<double>& arena, std::size_t root) {
        return arena.data() + root * dimension_capacity;
    }
    const double* root_vector(
        const std::vector<double>& arena,
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
            + best_energies.capacity()
            + best_residuals.capacity()
        ) * sizeof(double);
    }
};

struct BlockDavidsonResult {
    bool accepted = false;
    std::vector<double> energies;
    std::vector<std::vector<Complex>> vectors;
    std::vector<double> residual_norms;
    int iterations = 0;
    std::size_t basis_size = 0;
    int restarts = 0;
    bool converged = false;
    bool workspace_reused = false;
    std::uint64_t matvec_calls = 0;
};

template <typename Matvec>
BlockDavidsonResult real_block_davidson(
    const std::vector<double>& diagonal,
    const std::vector<std::vector<double>>& guesses,
    std::size_t nroots,
    double tolerance,
    int max_iterations,
    int restart_dimension,
    bool accept_unconverged,
    RealBlockDavidsonWorkspace& workspace,
    Matvec&& matvec
) {
    const std::size_t dimension = diagonal.size();
    if (
        dimension == 0
        || nroots == 0
        || nroots > dimension
        || guesses.empty()
    ) {
        throw std::invalid_argument(
            "Real block Davidson dimensions are invalid."
        );
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "Real block Davidson max_iterations must be positive."
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
        double* vector,
        const RealBlockDavidsonWorkspace& owner,
        std::size_t basis_size
    ) {
        for (int pass = 0; pass < 2; ++pass) {
            for (
                std::size_t column = 0;
                column < basis_size;
                ++column
            ) {
                const double overlap = real_dot(
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
        const double norm = real_norm(vector, dimension);
        if (norm <= 1.0e-12) {
            return false;
        }
        for (std::size_t index = 0; index < dimension; ++index) {
            vector[index] /= norm;
        }
        return true;
    };

    std::size_t basis_size = 0;
    for (const auto& guess : guesses) {
        if (basis_size >= nroots || guess.size() != dimension) {
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
        if (basis_size >= nroots) {
            break;
        }
        double* vector = workspace.basis_column(basis_size);
        std::fill(vector, vector + dimension, 0.0);
        vector[seed] = 1.0;
        if (orthonormalize(vector, workspace, basis_size)) {
            ++basis_size;
        }
    }
    if (basis_size < nroots) {
        throw std::runtime_error(
            "Real block Davidson could not construct the root seed space."
        );
    }

    std::size_t computed_basis_size = 0;
    std::vector<double> current_energies(nroots, 0.0);
    std::vector<double> current_residuals(
        nroots,
        std::numeric_limits<double>::infinity()
    );
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
        for (std::size_t row = 0; row < basis_size; ++row) {
            for (std::size_t col = 0; col < basis_size; ++col) {
                workspace.projected_dense[row * basis_size + col] =
                    workspace.projected[
                        row * workspace.capacity + col
                    ];
            }
        }
        const RealSymmetricEigendecomposition eigenpairs = symmetric_eigh(
            std::vector<double>(
                workspace.projected_dense.begin(),
                workspace.projected_dense.begin()
                    + basis_size * basis_size
            ),
            basis_size
        );
        bool all_converged = true;
        for (std::size_t root = 0; root < nroots; ++root) {
            double* ritz = workspace.root_vector(workspace.ritz, root);
            double* h_ritz = workspace.root_vector(
                workspace.h_ritz,
                root
            );
            double* residual = workspace.root_vector(
                workspace.residual,
                root
            );
            std::fill(ritz, ritz + dimension, 0.0);
            std::fill(h_ritz, h_ritz + dimension, 0.0);
            for (std::size_t column = 0; column < basis_size; ++column) {
                const double coefficient = eigenpairs.vectors[
                    column * basis_size + root
                ];
                real_axpy(
                    ritz,
                    coefficient,
                    workspace.basis_column(column),
                    dimension
                );
                real_axpy(
                    h_ritz,
                    coefficient,
                    workspace.h_basis_column(column),
                    dimension
                );
            }
            current_energies[root] = eigenpairs.values[root];
            for (std::size_t index = 0; index < dimension; ++index) {
                residual[index] = h_ritz[index]
                    - current_energies[root] * ritz[index];
            }
            current_residuals[root] = real_norm(residual, dimension);
            if (
                current_residuals[root]
                < workspace.best_residuals[root]
            ) {
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
        if (all_converged) {
            result.converged = true;
            break;
        }

        std::size_t corrections = 0;
        for (std::size_t root = 0; root < nroots; ++root) {
            if (current_residuals[root] < tolerance) {
                continue;
            }
            double* correction = workspace.root_vector(
                workspace.correction,
                corrections
            );
            const double* residual = workspace.root_vector(
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
            for (int pass = 0; pass < 2; ++pass) {
                for (
                    std::size_t column = 0;
                    column < basis_size;
                    ++column
                ) {
                    const double overlap = real_dot(
                        workspace.basis_column(column),
                        correction,
                        dimension
                    );
                    for (
                        std::size_t index = 0;
                        index < dimension;
                        ++index
                    ) {
                        correction[index] -= overlap
                            * workspace.basis_column(column)[index];
                    }
                }
                for (
                    std::size_t previous = 0;
                    previous < corrections;
                    ++previous
                ) {
                    const double* prior = workspace.root_vector(
                        workspace.correction,
                        previous
                    );
                    const double overlap = real_dot(
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
            const double norm = real_norm(correction, dimension);
            if (norm <= 1.0e-10) {
                continue;
            }
            for (std::size_t index = 0; index < dimension; ++index) {
                correction[index] /= norm;
            }
            ++corrections;
        }
        if (corrections == 0) {
            break;
        }

        if (basis_size + corrections > workspace.capacity) {
            const std::size_t retained = std::min(nroots, workspace.capacity);
            for (std::size_t root = 0; root < retained; ++root) {
                std::copy(
                    workspace.root_vector(workspace.ritz, root),
                    workspace.root_vector(workspace.ritz, root) + dimension,
                    workspace.basis_column(root)
                );
                std::copy(
                    workspace.root_vector(workspace.h_ritz, root),
                    workspace.root_vector(workspace.h_ritz, root)
                        + dimension,
                    workspace.h_basis_column(root)
                );
            }
            std::fill(
                workspace.projected.begin(),
                workspace.projected.end(),
                0.0
            );
            for (std::size_t root = 0; root < retained; ++root) {
                workspace.projected[
                    root * workspace.capacity + root
                ] = current_energies[root];
            }
            basis_size = retained;
            computed_basis_size = retained;
            ++result.restarts;
        }
        const std::size_t available = workspace.capacity - basis_size;
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
            const double* source = workspace.root_vector(
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
            const double phase = maximum > 0.0 && source[pivot] < 0.0
                ? -1.0
                : 1.0;
            for (std::size_t index = 0; index < dimension; ++index) {
                vector[index] = Complex{phase * source[index], 0.0};
            }
        }
    }
    return result;
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
#ifdef __APPLE__
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

}  // namespace pyqed::dmrg
