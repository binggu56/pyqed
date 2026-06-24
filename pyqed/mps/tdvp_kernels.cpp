#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define PYQED_TDVP_HAS_BLAS 1
#else
#define PYQED_TDVP_HAS_BLAS 0
#endif

namespace py = pybind11;
using cdouble = std::complex<double>;

struct SiteRightGroup {
    ssize_t m;
    ssize_t q;
    std::vector<cdouble> right_eff;
};

static double vec_norm(const std::vector<cdouble>& v) {
    double total = 0.0;
    for (const auto& x : v) {
        total += std::norm(x);
    }
    return std::sqrt(total);
}

static cdouble dotc(const std::vector<cdouble>& a, const std::vector<cdouble>& b) {
    cdouble total = 0.0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        total += std::conj(a[i]) * b[i];
    }
    return total;
}

static void jacobi_symmetric(
    std::vector<double> a,
    int n,
    std::vector<double>& evals,
    std::vector<double>& evecs
) {
    evecs.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        evecs[static_cast<std::size_t>(i) * n + i] = 1.0;
    }
    if (n == 1) {
        evals = {a[0]};
        return;
    }

    const int max_iter = std::max(32, 64 * n * n);
    for (int iter = 0; iter < max_iter; ++iter) {
        int p = 0;
        int q = 1;
        double max_abs = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double value = std::abs(a[static_cast<std::size_t>(i) * n + j]);
                if (value > max_abs) {
                    max_abs = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_abs <= 1.0e-14) {
            break;
        }

        const double app = a[static_cast<std::size_t>(p) * n + p];
        const double aqq = a[static_cast<std::size_t>(q) * n + q];
        const double apq = a[static_cast<std::size_t>(p) * n + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double sign = tau >= 0.0 ? 1.0 : -1.0;
        const double t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const double akp = a[static_cast<std::size_t>(k) * n + p];
            const double akq = a[static_cast<std::size_t>(k) * n + q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            a[static_cast<std::size_t>(k) * n + p] = new_kp;
            a[static_cast<std::size_t>(p) * n + k] = new_kp;
            a[static_cast<std::size_t>(k) * n + q] = new_kq;
            a[static_cast<std::size_t>(q) * n + k] = new_kq;
        }

        a[static_cast<std::size_t>(p) * n + p] = app - t * apq;
        a[static_cast<std::size_t>(q) * n + q] = aqq + t * apq;
        a[static_cast<std::size_t>(p) * n + q] = 0.0;
        a[static_cast<std::size_t>(q) * n + p] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vkp = evecs[static_cast<std::size_t>(k) * n + p];
            const double vkq = evecs[static_cast<std::size_t>(k) * n + q];
            evecs[static_cast<std::size_t>(k) * n + p] = c * vkp - s * vkq;
            evecs[static_cast<std::size_t>(k) * n + q] = s * vkp + c * vkq;
        }
    }

    evals.resize(n);
    for (int i = 0; i < n; ++i) {
        evals[i] = a[static_cast<std::size_t>(i) * n + i];
    }
}

static std::vector<cdouble> lanczos_apply(
    const std::vector<cdouble>& vec,
    double dt,
    int krylov_dim,
    double tol,
    const std::function<void(const std::vector<cdouble>&, std::vector<cdouble>&)>& matvec
) {
    const double norm = vec_norm(vec);
    if (norm <= tol) {
        return vec;
    }

    const int size = static_cast<int>(vec.size());
    const int mmax = std::max(1, std::min(krylov_dim, size));
    std::vector<std::vector<cdouble>> basis;
    basis.reserve(static_cast<std::size_t>(mmax));
    basis.emplace_back(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        basis[0][i] = vec[i] / norm;
    }

    std::vector<double> alpha(static_cast<std::size_t>(mmax), 0.0);
    std::vector<double> beta(static_cast<std::size_t>(std::max(0, mmax - 1)), 0.0);
    std::vector<cdouble> trial(vec.size());
    std::vector<cdouble> q_prev(vec.size());
    double beta_prev = 0.0;
    int actual_dim = mmax;

    for (int j = 0; j < mmax; ++j) {
        matvec(basis[static_cast<std::size_t>(j)], trial);
        if (j > 0) {
            for (std::size_t i = 0; i < trial.size(); ++i) {
                trial[i] -= beta_prev * q_prev[i];
            }
        }
        const cdouble alpha_j = dotc(basis[static_cast<std::size_t>(j)], trial);
        alpha[static_cast<std::size_t>(j)] = std::real(alpha_j);
        for (std::size_t i = 0; i < trial.size(); ++i) {
            trial[i] -= alpha_j * basis[static_cast<std::size_t>(j)][i];
        }

        const double beta_j = vec_norm(trial);
        actual_dim = j + 1;
        if (beta_j <= tol || j + 1 == mmax) {
            break;
        }
        beta[static_cast<std::size_t>(j)] = beta_j;
        q_prev = basis[static_cast<std::size_t>(j)];
        beta_prev = beta_j;
        basis.emplace_back(vec.size());
        for (std::size_t i = 0; i < trial.size(); ++i) {
            basis.back()[i] = trial[i] / beta_j;
        }
    }

    std::vector<cdouble> small_action(static_cast<std::size_t>(actual_dim), 0.0);
    if (actual_dim == 1) {
        small_action[0] = norm * std::exp(cdouble(0.0, -dt * alpha[0]));
    } else {
        std::vector<double> tri(static_cast<std::size_t>(actual_dim) * actual_dim, 0.0);
        for (int i = 0; i < actual_dim; ++i) {
            tri[static_cast<std::size_t>(i) * actual_dim + i] = alpha[static_cast<std::size_t>(i)];
            if (i + 1 < actual_dim) {
                tri[static_cast<std::size_t>(i) * actual_dim + i + 1] = beta[static_cast<std::size_t>(i)];
                tri[static_cast<std::size_t>(i + 1) * actual_dim + i] = beta[static_cast<std::size_t>(i)];
            }
        }
        std::vector<double> evals;
        std::vector<double> evecs;
        jacobi_symmetric(tri, actual_dim, evals, evecs);
        for (int row = 0; row < actual_dim; ++row) {
            cdouble value = 0.0;
            for (int col = 0; col < actual_dim; ++col) {
                const double coeff = evecs[static_cast<std::size_t>(0) * actual_dim + col];
                const double row_coeff = evecs[static_cast<std::size_t>(row) * actual_dim + col];
                value += norm * row_coeff * coeff * std::exp(cdouble(0.0, -dt * evals[static_cast<std::size_t>(col)]));
            }
            small_action[static_cast<std::size_t>(row)] = value;
        }
    }

    std::vector<cdouble> out(vec.size(), 0.0);
    for (int j = 0; j < actual_dim; ++j) {
        const cdouble coeff = small_action[static_cast<std::size_t>(j)];
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] += coeff * basis[static_cast<std::size_t>(j)][i];
        }
    }
    return out;
}

#if PYQED_TDVP_HAS_BLAS
static std::vector<cdouble> grouped_sparse_site_lanczos_apply(
    const std::vector<cdouble>& vec,
    double dt,
    int krylov_dim,
    double tol,
    const cdouble* left_ptr,
    const cdouble* W_ptr,
    const cdouble* right_ptr,
    ssize_t A,
    ssize_t B,
    ssize_t Q,
    ssize_t S,
    ssize_t M,
    ssize_t N,
    ssize_t P,
    ssize_t R,
    double w_cutoff
) {
    std::vector<std::vector<SiteRightGroup>> groups(static_cast<std::size_t>(P));
    std::vector<int> group_index(static_cast<std::size_t>(M) * P * Q, -1);
    for (ssize_t m = 0; m < M; ++m) {
        for (ssize_t n = 0; n < N; ++n) {
            for (ssize_t p = 0; p < P; ++p) {
                for (ssize_t q = 0; q < Q; ++q) {
                    const cdouble wval = W_ptr[((m * N + n) * P + p) * Q + q];
                    if (std::abs(wval) <= w_cutoff) {
                        continue;
                    }
                    const std::size_t key = static_cast<std::size_t>((m * P + p) * Q + q);
                    int idx = group_index[key];
                    if (idx < 0) {
                        idx = static_cast<int>(groups[static_cast<std::size_t>(p)].size());
                        group_index[key] = idx;
                        groups[static_cast<std::size_t>(p)].push_back(
                            SiteRightGroup{
                                m,
                                q,
                                std::vector<cdouble>(static_cast<std::size_t>(R) * S, 0.0),
                            }
                        );
                    }
                    SiteRightGroup& group = groups[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)];
                    for (ssize_t r = 0; r < R; ++r) {
                        for (ssize_t s = 0; s < S; ++s) {
                            group.right_eff[static_cast<std::size_t>(r) * S + s] +=
                                wval * right_ptr[(r * N + n) * S + s];
                        }
                    }
                }
            }
        }
    }

    std::vector<cdouble> left_stack(static_cast<std::size_t>(M) * A * B, 0.0);
    for (ssize_t m = 0; m < M; ++m) {
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                left_stack[(m * A + a) * B + b] = left_ptr[(a * M + m) * B + b];
            }
        }
    }

    std::vector<std::vector<cdouble>> right_concat(static_cast<std::size_t>(P));
    std::size_t max_groups = 0;
    for (ssize_t p = 0; p < P; ++p) {
        const std::size_t group_count = groups[static_cast<std::size_t>(p)].size();
        max_groups = std::max(max_groups, group_count);
        const std::size_t kdim = group_count * static_cast<std::size_t>(S);
        right_concat[static_cast<std::size_t>(p)].assign(static_cast<std::size_t>(R) * kdim, 0.0);
        for (std::size_t g = 0; g < group_count; ++g) {
            const SiteRightGroup& group = groups[static_cast<std::size_t>(p)][g];
            for (ssize_t r = 0; r < R; ++r) {
                for (ssize_t s = 0; s < S; ++s) {
                    right_concat[static_cast<std::size_t>(p)][static_cast<std::size_t>(r) * kdim + g * S + s] =
                        group.right_eff[static_cast<std::size_t>(r) * S + s];
                }
            }
        }
    }

    const std::size_t dim = static_cast<std::size_t>(B) * Q * S;
    std::vector<cdouble> local_q(static_cast<std::size_t>(B) * S, 0.0);
    std::vector<cdouble> left_times_local(static_cast<std::size_t>(Q) * M * A * S, 0.0);
    std::vector<cdouble> left_concat(static_cast<std::size_t>(A) * max_groups * S, 0.0);
    std::vector<cdouble> out_p(static_cast<std::size_t>(A) * R, 0.0);
    const cdouble alpha_blas(1.0, 0.0);
    const cdouble beta_blas(0.0, 0.0);

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        out.assign(dim, cdouble(0.0, 0.0));
        for (ssize_t q = 0; q < Q; ++q) {
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t s = 0; s < S; ++s) {
                    local_q[static_cast<std::size_t>(b) * S + s] = local[(b * Q + q) * S + s];
                }
            }
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(M * A),
                static_cast<int>(S),
                static_cast<int>(B),
                &alpha_blas,
                left_stack.data(),
                static_cast<int>(B),
                local_q.data(),
                static_cast<int>(S),
                &beta_blas,
                left_times_local.data() + static_cast<std::size_t>(q) * M * A * S,
                static_cast<int>(S)
            );
        }

        for (ssize_t p = 0; p < P; ++p) {
            const std::size_t group_count = groups[static_cast<std::size_t>(p)].size();
            if (group_count == 0) {
                continue;
            }
            const std::size_t kdim = group_count * static_cast<std::size_t>(S);
            for (std::size_t g = 0; g < group_count; ++g) {
                const SiteRightGroup& group = groups[static_cast<std::size_t>(p)][g];
                const cdouble* src = left_times_local.data()
                    + static_cast<std::size_t>(group.q) * M * A * S
                    + static_cast<std::size_t>(group.m) * A * S;
                for (ssize_t a = 0; a < A; ++a) {
                    for (ssize_t s = 0; s < S; ++s) {
                        left_concat[static_cast<std::size_t>(a) * kdim + g * S + s] =
                            src[static_cast<std::size_t>(a) * S + s];
                    }
                }
            }
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(A),
                static_cast<int>(R),
                static_cast<int>(kdim),
                &alpha_blas,
                left_concat.data(),
                static_cast<int>(kdim),
                right_concat[static_cast<std::size_t>(p)].data(),
                static_cast<int>(kdim),
                &beta_blas,
                out_p.data(),
                static_cast<int>(R)
            );
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t r = 0; r < R; ++r) {
                    out[(a * P + p) * R + r] = out_p[static_cast<std::size_t>(a) * R + r];
                }
            }
        }
    };

    return lanczos_apply(vec, dt, krylov_dim, tol, matvec);
}
#endif

static py::array_t<cdouble> site_lanczos(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> theta,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> W,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> right,
    double dt,
    int krylov_dim,
    double tol
) {
    if (theta.ndim() != 3 || left.ndim() != 3 || W.ndim() != 4 || right.ndim() != 3) {
        throw std::runtime_error("site_lanczos expects theta(3), left(3), W(4), right(3)");
    }
    const ssize_t B = theta.shape(0);
    const ssize_t Q = theta.shape(1);
    const ssize_t S = theta.shape(2);
    const ssize_t A = left.shape(0);
    const ssize_t M = left.shape(1);
    const ssize_t B_left = left.shape(2);
    const ssize_t M_w = W.shape(0);
    const ssize_t N = W.shape(1);
    const ssize_t P = W.shape(2);
    const ssize_t Q_w = W.shape(3);
    const ssize_t R = right.shape(0);
    const ssize_t N_right = right.shape(1);
    const ssize_t S_right = right.shape(2);
    if (B_left != B || M_w != M || Q_w != Q || N_right != N || S_right != S) {
        throw std::runtime_error("site_lanczos input shapes are incompatible");
    }
    if (A != B || P != Q || R != S) {
        throw std::runtime_error("site_lanczos requires a square effective local space");
    }

    const cdouble* theta_ptr = theta.data();
    const cdouble* left_ptr = left.data();
    const cdouble* W_ptr = W.data();
    const cdouble* right_ptr = right.data();
    const std::size_t dim = static_cast<std::size_t>(B) * Q * S;
    std::vector<cdouble> vec(dim);
    std::copy(theta_ptr, theta_ptr + static_cast<ssize_t>(dim), vec.begin());
    std::vector<cdouble> result;

    {
        py::gil_scoped_release release;
#if PYQED_TDVP_HAS_BLAS
        const double w_cutoff = 1.0e-14;
        std::size_t nnz_w = 0;
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t p = 0; p < P; ++p) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        const cdouble wval = W_ptr[((m * N + n) * P + p) * Q + q];
                        if (std::abs(wval) > w_cutoff) {
                            ++nnz_w;
                        }
                    }
                }
            }
        }
        const bool use_grouped_sparse = (
            nnz_w > 0
            && nnz_w * 4 < static_cast<std::size_t>(M) * N * P * Q
            && M * A <= static_cast<ssize_t>(INT_MAX)
            && A <= static_cast<ssize_t>(INT_MAX)
            && B <= static_cast<ssize_t>(INT_MAX)
            && R <= static_cast<ssize_t>(INT_MAX)
            && S <= static_cast<ssize_t>(INT_MAX)
        );
        if (use_grouped_sparse) {
            result = grouped_sparse_site_lanczos_apply(
                vec,
                dt,
                krylov_dim,
                tol,
                left_ptr,
                W_ptr,
                right_ptr,
                A,
                B,
                Q,
                S,
                M,
                N,
                P,
                R,
                w_cutoff
            );
        } else {
        const ssize_t left_cols = B * N * Q;
        std::vector<cdouble> left_matrix(static_cast<std::size_t>(A) * P * left_cols, 0.0);
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t p = 0; p < P; ++p) {
                for (ssize_t b = 0; b < B; ++b) {
                    for (ssize_t n = 0; n < N; ++n) {
                        for (ssize_t q = 0; q < Q; ++q) {
                            cdouble sum = 0.0;
                            for (ssize_t m = 0; m < M; ++m) {
                                const cdouble lval = left_ptr[(a * M + m) * B + b];
                                const cdouble wval = W_ptr[((m * N + n) * P + p) * Q + q];
                                sum += lval * wval;
                            }
                            left_matrix[((a * P + p) * left_cols + (b * N + n) * Q + q)] = sum;
                        }
                    }
                }
            }
        }

        std::vector<cdouble> tmp_bq_rn(static_cast<std::size_t>(B) * Q * R * N, 0.0);
        std::vector<cdouble> tmp_bnq_r(static_cast<std::size_t>(B) * N * Q * R, 0.0);
        const cdouble alpha_blas(1.0, 0.0);
        const cdouble beta_blas(0.0, 0.0);
        auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            out.assign(dim, cdouble(0.0, 0.0));
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(B * Q),
                static_cast<int>(R * N),
                static_cast<int>(S),
                &alpha_blas,
                local.data(),
                static_cast<int>(S),
                right_ptr,
                static_cast<int>(S),
                &beta_blas,
                tmp_bq_rn.data(),
                static_cast<int>(R * N)
            );
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t n = 0; n < N; ++n) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        for (ssize_t r = 0; r < R; ++r) {
                            tmp_bnq_r[((b * N + n) * Q + q) * R + r] =
                                tmp_bq_rn[(b * Q + q) * (R * N) + r * N + n];
                        }
                    }
                }
            }
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(A * P),
                static_cast<int>(R),
                static_cast<int>(left_cols),
                &alpha_blas,
                left_matrix.data(),
                static_cast<int>(left_cols),
                tmp_bnq_r.data(),
                static_cast<int>(R),
                &beta_blas,
                out.data(),
                static_cast<int>(R)
            );
        };
            result = lanczos_apply(vec, dt, krylov_dim, tol, matvec);
        }
#else
        std::vector<cdouble> left_kernel(static_cast<std::size_t>(A) * B * N * P * Q, 0.0);
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t n = 0; n < N; ++n) {
                    for (ssize_t p = 0; p < P; ++p) {
                        for (ssize_t q = 0; q < Q; ++q) {
                            cdouble sum = 0.0;
                            for (ssize_t m = 0; m < M; ++m) {
                                const cdouble lval = left_ptr[(a * M + m) * B + b];
                                const cdouble wval = W_ptr[((m * N + n) * P + p) * Q + q];
                                sum += lval * wval;
                            }
                            left_kernel[((((a * B + b) * N + n) * P + p) * Q + q)] = sum;
                        }
                    }
                }
            }
        }

        std::vector<cdouble> tmp(static_cast<std::size_t>(B) * Q * R * N, 0.0);
        auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            std::fill(tmp.begin(), tmp.end(), cdouble(0.0, 0.0));
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t q = 0; q < Q; ++q) {
                    for (ssize_t r = 0; r < R; ++r) {
                        for (ssize_t n = 0; n < N; ++n) {
                            cdouble sum = 0.0;
                            for (ssize_t s = 0; s < S; ++s) {
                                const cdouble x = local[(b * Q + q) * S + s];
                                const cdouble rval = right_ptr[(r * N + n) * S + s];
                                sum += x * rval;
                            }
                            tmp[(((b * Q + q) * R + r) * N + n)] = sum;
                        }
                    }
                }
            }

            out.assign(dim, cdouble(0.0, 0.0));
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t p = 0; p < P; ++p) {
                    for (ssize_t r = 0; r < R; ++r) {
                        cdouble sum = 0.0;
                        for (ssize_t b = 0; b < B; ++b) {
                            for (ssize_t n = 0; n < N; ++n) {
                                for (ssize_t q = 0; q < Q; ++q) {
                                    const cdouble lk = left_kernel[((((a * B + b) * N + n) * P + p) * Q + q)];
                                    const cdouble tv = tmp[(((b * Q + q) * R + r) * N + n)];
                                    sum += lk * tv;
                                }
                            }
                        }
                        out[(a * P + p) * R + r] = sum;
                    }
                }
            }
        };
        result = lanczos_apply(vec, dt, krylov_dim, tol, matvec);
#endif
    }

    py::array_t<cdouble> out({A, P, R});
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

static py::array_t<cdouble> bond_lanczos(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> center,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> right,
    double dt,
    int krylov_dim,
    double tol
) {
    if (center.ndim() != 2 || left.ndim() != 3 || right.ndim() != 3) {
        throw std::runtime_error("bond_lanczos expects center(2), left(3), right(3)");
    }
    const ssize_t B = center.shape(0);
    const ssize_t S = center.shape(1);
    const ssize_t A = left.shape(0);
    const ssize_t M = left.shape(1);
    const ssize_t B_left = left.shape(2);
    const ssize_t R = right.shape(0);
    const ssize_t M_right = right.shape(1);
    const ssize_t S_right = right.shape(2);
    if (B_left != B || M_right != M || S_right != S) {
        throw std::runtime_error("bond_lanczos input shapes are incompatible");
    }
    if (A != B || R != S) {
        throw std::runtime_error("bond_lanczos requires a square effective bond space");
    }

    const cdouble* center_ptr = center.data();
    const cdouble* left_ptr = left.data();
    const cdouble* right_ptr = right.data();
    const std::size_t dim = static_cast<std::size_t>(B) * S;
    std::vector<cdouble> vec(dim);
    std::copy(center_ptr, center_ptr + static_cast<ssize_t>(dim), vec.begin());
    std::vector<cdouble> result;

    {
        py::gil_scoped_release release;
#if PYQED_TDVP_HAS_BLAS
        const ssize_t cols = M * S;
        std::vector<cdouble> left_stack(static_cast<std::size_t>(M) * A * B, 0.0);
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t b = 0; b < B; ++b) {
                    left_stack[(m * A + a) * B + b] = left_ptr[(a * M + m) * B + b];
                }
            }
        }
        std::vector<cdouble> left_times_center(static_cast<std::size_t>(M) * A * S, 0.0);
        std::vector<cdouble> left_concat(static_cast<std::size_t>(A) * cols, 0.0);
        const cdouble alpha_blas(1.0, 0.0);
        const cdouble beta_blas(0.0, 0.0);
        auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            out.assign(dim, cdouble(0.0, 0.0));
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(M * A),
                static_cast<int>(S),
                static_cast<int>(B),
                &alpha_blas,
                left_stack.data(),
                static_cast<int>(B),
                local.data(),
                static_cast<int>(S),
                &beta_blas,
                left_times_center.data(),
                static_cast<int>(S)
            );
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t m = 0; m < M; ++m) {
                    for (ssize_t s = 0; s < S; ++s) {
                        left_concat[static_cast<std::size_t>(a) * cols + m * S + s] =
                            left_times_center[(m * A + a) * S + s];
                    }
                }
            }
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(A),
                static_cast<int>(R),
                static_cast<int>(cols),
                &alpha_blas,
                left_concat.data(),
                static_cast<int>(cols),
                right_ptr,
                static_cast<int>(cols),
                &beta_blas,
                out.data(),
                static_cast<int>(R)
            );
        };
#else
        std::vector<cdouble> kernel(static_cast<std::size_t>(A) * B * R * S, 0.0);
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t r = 0; r < R; ++r) {
                    for (ssize_t s = 0; s < S; ++s) {
                        cdouble sum = 0.0;
                        for (ssize_t m = 0; m < M; ++m) {
                            const cdouble lval = left_ptr[(a * M + m) * B + b];
                            const cdouble rval = right_ptr[(r * M + m) * S + s];
                            sum += lval * rval;
                        }
                        kernel[(((a * B + b) * R + r) * S + s)] = sum;
                    }
                }
            }
        }

        auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            out.assign(dim, cdouble(0.0, 0.0));
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t r = 0; r < R; ++r) {
                    cdouble sum = 0.0;
                    for (ssize_t b = 0; b < B; ++b) {
                        for (ssize_t s = 0; s < S; ++s) {
                            sum += kernel[(((a * B + b) * R + r) * S + s)] * local[b * S + s];
                        }
                    }
                    out[a * R + r] = sum;
                }
            }
        };
#endif
        result = lanczos_apply(vec, dt, krylov_dim, tol, matvec);
    }

    py::array_t<cdouble> out({A, R});
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

PYBIND11_MODULE(_cpp_tdvp, m) {
    m.doc() = "C++ dense TDVP local evolution kernels";
    m.attr("HAS_BLAS") = bool(PYQED_TDVP_HAS_BLAS);
    m.def("site_lanczos", &site_lanczos, "Lanczos evolution for one TDVP site tensor");
    m.def("bond_lanczos", &bond_lanczos, "Lanczos evolution for one TDVP bond-center tensor");
}
