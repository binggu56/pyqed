#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <complex>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <malloc/malloc.h>
#define PYQED_TDVP_HAS_BLAS 1
#else
#define PYQED_TDVP_HAS_BLAS 0
#endif

namespace py = pybind11;
using cdouble = std::complex<double>;
using CArray = py::array_t<
    cdouble,
    py::array::c_style | py::array::forcecast
>;

static void release_free_numeric_pages() {
#ifdef __APPLE__
    malloc_zone_pressure_relief(malloc_default_zone(), 0);
#endif
}

static std::atomic<long long> lanczos_calls(0);
static std::atomic<long long> lanczos_matvecs(0);
static std::atomic<long long> low_memory_site_sum_calls(0);

struct SiteRightGroup {
    ssize_t m;
    ssize_t q;
    std::vector<cdouble> right_eff;
};

struct SiteSumComponentView {
    const cdouble* left;
    const cdouble* W;
    const cdouble* right;
    ssize_t M;
    ssize_t N;
};

struct SiteSumGroupKey {
    std::size_t component;
    ssize_t m;
    ssize_t global_m;
    ssize_t q;
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

static std::vector<cdouble> lanczos_coefficients(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    int actual_dim,
    double norm,
    double dt,
    bool imaginary_time
) {
    std::vector<cdouble> coefficients(
        static_cast<std::size_t>(actual_dim),
        0.0
    );
    if (actual_dim == 1) {
        const cdouble exponent = imaginary_time
            ? cdouble(-dt * alpha[0], 0.0)
            : cdouble(0.0, -dt * alpha[0]);
        coefficients[0] = norm * std::exp(exponent);
        return coefficients;
    }

    std::vector<double> tri(
        static_cast<std::size_t>(actual_dim) * actual_dim,
        0.0
    );
    for (int i = 0; i < actual_dim; ++i) {
        tri[static_cast<std::size_t>(i) * actual_dim + i] =
            alpha[static_cast<std::size_t>(i)];
        if (i + 1 < actual_dim) {
            tri[static_cast<std::size_t>(i) * actual_dim + i + 1] =
                beta[static_cast<std::size_t>(i)];
            tri[static_cast<std::size_t>(i + 1) * actual_dim + i] =
                beta[static_cast<std::size_t>(i)];
        }
    }
    std::vector<double> evals;
    std::vector<double> evecs;
    jacobi_symmetric(tri, actual_dim, evals, evecs);
    for (int row = 0; row < actual_dim; ++row) {
        cdouble value = 0.0;
        for (int col = 0; col < actual_dim; ++col) {
            const double first =
                evecs[static_cast<std::size_t>(col)];
            const double element = evecs[
                static_cast<std::size_t>(row) * actual_dim + col
            ];
            const double eigenvalue = evals[static_cast<std::size_t>(col)];
            const cdouble exponent = imaginary_time
                ? cdouble(-dt * eigenvalue, 0.0)
                : cdouble(0.0, -dt * eigenvalue);
            value += norm * element * first * std::exp(exponent);
        }
        coefficients[static_cast<std::size_t>(row)] = value;
    }
    return coefficients;
}

static std::vector<cdouble> lanczos_apply(
    const std::vector<cdouble>& vec,
    double dt,
    int krylov_dim,
    double tol,
    const std::function<void(const std::vector<cdouble>&, std::vector<cdouble>&)>& matvec,
    bool imaginary_time = false
) {
    lanczos_calls.fetch_add(1, std::memory_order_relaxed);
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
    int matvec_count = 0;
    std::vector<cdouble> previous_coefficients;
    std::vector<cdouble> coefficients;
    bool coefficients_ready = false;

    for (int j = 0; j < mmax; ++j) {
        matvec(basis[static_cast<std::size_t>(j)], trial);
        ++matvec_count;
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
        std::vector<cdouble> current_coefficients = lanczos_coefficients(
            alpha,
            beta,
            actual_dim,
            norm,
            dt,
            imaginary_time
        );
        if (!previous_coefficients.empty() && tol > 0.0) {
            double action_delta2 = 0.0;
            for (int k = 0; k < actual_dim - 1; ++k) {
                action_delta2 += std::norm(
                    current_coefficients[static_cast<std::size_t>(k)]
                    - previous_coefficients[static_cast<std::size_t>(k)]
                );
            }
            action_delta2 += std::norm(
                current_coefficients[static_cast<std::size_t>(actual_dim - 1)]
            );
            if (
                std::sqrt(action_delta2)
                <= tol * std::max(1.0, norm)
            ) {
                coefficients = std::move(current_coefficients);
                coefficients_ready = true;
                break;
            }
        }
        previous_coefficients = std::move(current_coefficients);
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
    lanczos_matvecs.fetch_add(matvec_count, std::memory_order_relaxed);

    if (!coefficients_ready) {
        coefficients = lanczos_coefficients(
            alpha,
            beta,
            actual_dim,
            norm,
            dt,
            imaginary_time
        );
    }

    std::vector<cdouble> out(vec.size(), 0.0);
    for (int j = 0; j < actual_dim; ++j) {
        const cdouble coeff = coefficients[static_cast<std::size_t>(j)];
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] += coeff * basis[static_cast<std::size_t>(j)][i];
        }
    }
    return out;
}

static void reset_kernel_stats() {
    lanczos_calls.store(0, std::memory_order_relaxed);
    lanczos_matvecs.store(0, std::memory_order_relaxed);
    low_memory_site_sum_calls.store(0, std::memory_order_relaxed);
}

static py::dict kernel_stats() {
    py::dict result;
    result["lanczos_calls"] = py::int_(
        lanczos_calls.load(std::memory_order_relaxed)
    );
    result["lanczos_matvecs"] = py::int_(
        lanczos_matvecs.load(std::memory_order_relaxed)
    );
    result["low_memory_site_sum_calls"] = py::int_(
        low_memory_site_sum_calls.load(std::memory_order_relaxed)
    );
    return result;
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
    double w_cutoff,
    int workers,
    const std::vector<SiteSumComponentView>* sum_components = nullptr,
    bool imaginary_time = false
) {
    std::vector<std::vector<SiteRightGroup>> groups(static_cast<std::size_t>(P));
    std::vector<int> group_index(static_cast<std::size_t>(M) * P * Q, -1);
    if (sum_components == nullptr) {
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
    } else {
        ssize_t m_offset = 0;
        for (const SiteSumComponentView& component : *sum_components) {
            for (ssize_t m = 0; m < component.M; ++m) {
                for (ssize_t n = 0; n < component.N; ++n) {
                    for (ssize_t p = 0; p < P; ++p) {
                        for (ssize_t q = 0; q < Q; ++q) {
                            const cdouble wval = component.W[
                                ((m * component.N + n) * P + p) * Q + q
                            ];
                            if (std::abs(wval) <= w_cutoff) {
                                continue;
                            }
                            const ssize_t global_m = m_offset + m;
                            const std::size_t key = static_cast<std::size_t>(
                                (global_m * P + p) * Q + q
                            );
                            int idx = group_index[key];
                            if (idx < 0) {
                                idx = static_cast<int>(
                                    groups[static_cast<std::size_t>(p)].size()
                                );
                                group_index[key] = idx;
                                groups[static_cast<std::size_t>(p)].push_back(
                                    SiteRightGroup{
                                        global_m,
                                        q,
                                        std::vector<cdouble>(
                                            static_cast<std::size_t>(R) * S,
                                            0.0
                                        ),
                                    }
                                );
                            }
                            SiteRightGroup& group = groups[
                                static_cast<std::size_t>(p)
                            ][static_cast<std::size_t>(idx)];
                            for (ssize_t r = 0; r < R; ++r) {
                                for (ssize_t s = 0; s < S; ++s) {
                                    group.right_eff[
                                        static_cast<std::size_t>(r) * S + s
                                    ] += wval * component.right[
                                        (r * component.N + n) * S + s
                                    ];
                                }
                            }
                        }
                    }
                }
            }
            m_offset += component.M;
        }
    }

    std::vector<cdouble> left_stack(static_cast<std::size_t>(M) * A * B, 0.0);
    if (sum_components == nullptr) {
    for (ssize_t m = 0; m < M; ++m) {
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                left_stack[(m * A + a) * B + b] = left_ptr[(a * M + m) * B + b];
            }
        }
    }
    } else {
        ssize_t m_offset = 0;
        for (const SiteSumComponentView& component : *sum_components) {
            for (ssize_t m = 0; m < component.M; ++m) {
                for (ssize_t a = 0; a < A; ++a) {
                    for (ssize_t b = 0; b < B; ++b) {
                        left_stack[((m_offset + m) * A + a) * B + b] =
                            component.left[(a * component.M + m) * B + b];
                    }
                }
            }
            m_offset += component.M;
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
            SiteRightGroup& group = groups[static_cast<std::size_t>(p)][g];
            for (ssize_t r = 0; r < R; ++r) {
                for (ssize_t s = 0; s < S; ++s) {
                    right_concat[static_cast<std::size_t>(p)][static_cast<std::size_t>(r) * kdim + g * S + s] =
                        group.right_eff[static_cast<std::size_t>(r) * S + s];
                }
            }
            std::vector<cdouble>().swap(group.right_eff);
        }
    }

    const std::size_t dim = static_cast<std::size_t>(B) * Q * S;
    const ssize_t local_columns = Q * S;
    std::vector<cdouble> left_times_local(
        static_cast<std::size_t>(M) * A * local_columns,
        0.0
    );
    const int q_threads = std::max(1, std::min(workers, static_cast<int>(Q)));
    const int p_threads = std::max(1, std::min(workers, static_cast<int>(P)));
    std::vector<std::vector<cdouble>> left_concat_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(static_cast<std::size_t>(A) * max_groups * S, 0.0)
    );
    std::vector<std::vector<cdouble>> out_p_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(static_cast<std::size_t>(A) * R, 0.0)
    );
    const cdouble alpha_blas(1.0, 0.0);
    const cdouble beta_blas(0.0, 0.0);

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        out.assign(dim, cdouble(0.0, 0.0));
        auto apply_q_range = [&](ssize_t begin, ssize_t end) {
            if (begin >= end) {
                return;
            }
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(M * A),
                static_cast<int>((end - begin) * S),
                static_cast<int>(B),
                &alpha_blas,
                left_stack.data(),
                static_cast<int>(B),
                local.data() + static_cast<std::size_t>(begin) * S,
                static_cast<int>(local_columns),
                &beta_blas,
                left_times_local.data() + static_cast<std::size_t>(begin) * S,
                static_cast<int>(local_columns)
            );
        };
        if (q_threads == 1) {
            apply_q_range(0, Q);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(q_threads));
            for (int worker = 0; worker < q_threads; ++worker) {
                const ssize_t begin = Q * worker / q_threads;
                const ssize_t end = Q * (worker + 1) / q_threads;
                threads.emplace_back(
                    [&, begin, end]() {
                        apply_q_range(begin, end);
                    }
                );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }

        auto apply_p_range = [&](ssize_t begin, ssize_t end, int worker) {
            std::vector<cdouble>& left_concat =
                left_concat_scratch[static_cast<std::size_t>(worker)];
            std::vector<cdouble>& out_p =
                out_p_scratch[static_cast<std::size_t>(worker)];
        for (ssize_t p = begin; p < end; ++p) {
            const std::size_t group_count = groups[static_cast<std::size_t>(p)].size();
            if (group_count == 0) {
                continue;
            }
            const std::size_t kdim = group_count * static_cast<std::size_t>(S);
            for (std::size_t g = 0; g < group_count; ++g) {
                const SiteRightGroup& group = groups[static_cast<std::size_t>(p)][g];
                for (ssize_t a = 0; a < A; ++a) {
                    for (ssize_t s = 0; s < S; ++s) {
                        left_concat[static_cast<std::size_t>(a) * kdim + g * S + s] =
                            left_times_local[
                                ((static_cast<std::size_t>(group.m) * A + a) * Q
                                 + group.q) * S + s
                            ];
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
        if (p_threads == 1) {
            apply_p_range(0, P, 0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(p_threads));
            for (int worker = 0; worker < p_threads; ++worker) {
                const ssize_t begin = P * worker / p_threads;
                const ssize_t end = P * (worker + 1) / p_threads;
                threads.emplace_back(
                    [&, begin, end, worker]() {
                        apply_p_range(begin, end, worker);
                    }
                );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }
    };

    return lanczos_apply(vec, dt, krylov_dim, tol, matvec, imaginary_time);
}

static std::vector<cdouble> blocked_low_memory_site_lanczos_sum_apply(
    const std::vector<cdouble>& vec,
    double dt,
    int krylov_dim,
    double tol,
    const std::vector<SiteSumComponentView>& components,
    ssize_t B,
    ssize_t Q,
    ssize_t S,
    ssize_t total_m,
    double w_cutoff,
    std::size_t max_workspace_elements,
    int workers,
    bool imaginary_time = false
) {
    low_memory_site_sum_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::vector<SiteSumGroupKey>> groups(
        static_cast<std::size_t>(Q)
    );
    ssize_t m_offset = 0;
    std::size_t max_groups = 0;
    for (std::size_t k = 0; k < components.size(); ++k) {
        const SiteSumComponentView& component = components[k];
        for (ssize_t m = 0; m < component.M; ++m) {
            for (ssize_t p = 0; p < Q; ++p) {
                for (ssize_t q = 0; q < Q; ++q) {
                    bool active = false;
                    for (ssize_t n = 0; n < component.N; ++n) {
                        if (
                            std::abs(component.W[
                                ((m * component.N + n) * Q + p) * Q + q
                            ]) > w_cutoff
                        ) {
                            active = true;
                            break;
                        }
                    }
                    if (active) {
                        groups[static_cast<std::size_t>(p)].push_back(
                            SiteSumGroupKey{k, m, m_offset + m, q}
                        );
                    }
                }
            }
        }
        m_offset += component.M;
    }
    for (const auto& local_groups : groups) {
        max_groups = std::max(max_groups, local_groups.size());
    }

    std::vector<cdouble> left_stack(
        static_cast<std::size_t>(total_m) * B * B,
        0.0
    );
    m_offset = 0;
    for (const SiteSumComponentView& component : components) {
        for (ssize_t m = 0; m < component.M; ++m) {
            for (ssize_t a = 0; a < B; ++a) {
                for (ssize_t b = 0; b < B; ++b) {
                    left_stack[
                        ((m_offset + m) * B + a) * B + b
                    ] = component.left[(a * component.M + m) * B + b];
                }
            }
        }
        m_offset += component.M;
    }

    const int q_threads = std::max(1, std::min(workers, static_cast<int>(Q)));
    const int p_threads = std::max(1, std::min(workers, static_cast<int>(Q)));
    const std::size_t elements_per_group =
        static_cast<std::size_t>(S) * static_cast<std::size_t>(B + S);
    const std::size_t per_thread_budget = std::max<std::size_t>(
        1,
        max_workspace_elements / static_cast<std::size_t>(p_threads)
    );
    const std::size_t block_groups = std::max<std::size_t>(
        1,
        std::min(
            max_groups,
            per_thread_budget / std::max<std::size_t>(1, elements_per_group)
        )
    );
    const std::size_t dim = static_cast<std::size_t>(B) * Q * S;
    std::vector<cdouble> left_times_local(
        static_cast<std::size_t>(Q) * total_m * B * S,
        0.0
    );
    std::vector<std::vector<cdouble>> local_q_scratch(
        static_cast<std::size_t>(q_threads),
        std::vector<cdouble>(static_cast<std::size_t>(B) * S, 0.0)
    );
    std::vector<std::vector<cdouble>> left_concat_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(
            static_cast<std::size_t>(B) * block_groups * S,
            0.0
        )
    );
    std::vector<std::vector<cdouble>> right_concat_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(
            static_cast<std::size_t>(S) * block_groups * S,
            0.0
        )
    );
    std::vector<std::vector<cdouble>> out_p_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(static_cast<std::size_t>(B) * S, 0.0)
    );
    const cdouble alpha_blas(1.0, 0.0);
    const cdouble beta_zero(0.0, 0.0);
    const cdouble beta_one(1.0, 0.0);

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        out.assign(dim, cdouble(0.0, 0.0));
        auto apply_q_range = [&](ssize_t begin, ssize_t end, int worker) {
            std::vector<cdouble>& local_q =
                local_q_scratch[static_cast<std::size_t>(worker)];
            for (ssize_t q = begin; q < end; ++q) {
                for (ssize_t b = 0; b < B; ++b) {
                    for (ssize_t s = 0; s < S; ++s) {
                        local_q[static_cast<std::size_t>(b) * S + s] =
                            local[(b * Q + q) * S + s];
                    }
                }
                cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(total_m * B),
                    static_cast<int>(S),
                    static_cast<int>(B),
                    &alpha_blas,
                    left_stack.data(),
                    static_cast<int>(B),
                    local_q.data(),
                    static_cast<int>(S),
                    &beta_zero,
                    left_times_local.data()
                        + static_cast<std::size_t>(q) * total_m * B * S,
                    static_cast<int>(S)
                );
            }
        };
        if (q_threads == 1) {
            apply_q_range(0, Q, 0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(q_threads));
            for (int worker = 0; worker < q_threads; ++worker) {
                const ssize_t begin = Q * worker / q_threads;
                const ssize_t end = Q * (worker + 1) / q_threads;
                threads.emplace_back(
                    [&, begin, end, worker]() {
                        apply_q_range(begin, end, worker);
                    }
                );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }

        auto apply_p_range = [&](ssize_t begin, ssize_t end, int worker) {
            std::vector<cdouble>& left_concat =
                left_concat_scratch[static_cast<std::size_t>(worker)];
            std::vector<cdouble>& right_concat =
                right_concat_scratch[static_cast<std::size_t>(worker)];
            std::vector<cdouble>& out_p =
                out_p_scratch[static_cast<std::size_t>(worker)];
            for (ssize_t p = begin; p < end; ++p) {
                std::fill(out_p.begin(), out_p.end(), cdouble(0.0, 0.0));
                const auto& local_groups = groups[static_cast<std::size_t>(p)];
                for (
                    std::size_t block_begin = 0;
                    block_begin < local_groups.size();
                    block_begin += block_groups
                ) {
                    const std::size_t count = std::min(
                        block_groups,
                        local_groups.size() - block_begin
                    );
                    const std::size_t kdim = count * static_cast<std::size_t>(S);
                    for (std::size_t j = 0; j < count; ++j) {
                        const SiteSumGroupKey& group =
                            local_groups[block_begin + j];
                        const SiteSumComponentView& component =
                            components[group.component];
                        const cdouble* left_src = left_times_local.data()
                            + static_cast<std::size_t>(group.q)
                                * total_m * B * S
                            + static_cast<std::size_t>(group.global_m) * B * S;
                        for (ssize_t a = 0; a < B; ++a) {
                            for (ssize_t s = 0; s < S; ++s) {
                                left_concat[
                                    static_cast<std::size_t>(a) * kdim
                                    + j * S + s
                                ] = left_src[static_cast<std::size_t>(a) * S + s];
                            }
                        }
                        for (ssize_t r = 0; r < S; ++r) {
                            for (ssize_t s = 0; s < S; ++s) {
                                cdouble value = 0.0;
                                for (ssize_t n = 0; n < component.N; ++n) {
                                    const cdouble wval = component.W[
                                        ((group.m * component.N + n) * Q + p)
                                            * Q + group.q
                                    ];
                                    if (std::abs(wval) > w_cutoff) {
                                        value += wval * component.right[
                                            (r * component.N + n) * S + s
                                        ];
                                    }
                                }
                                right_concat[
                                    static_cast<std::size_t>(r) * kdim
                                    + j * S + s
                                ] = value;
                            }
                        }
                    }
                    cblas_zgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        static_cast<int>(B),
                        static_cast<int>(S),
                        static_cast<int>(kdim),
                        &alpha_blas,
                        left_concat.data(),
                        static_cast<int>(kdim),
                        right_concat.data(),
                        static_cast<int>(kdim),
                        block_begin == 0 ? &beta_zero : &beta_one,
                        out_p.data(),
                        static_cast<int>(S)
                    );
                }
                for (ssize_t a = 0; a < B; ++a) {
                    for (ssize_t r = 0; r < S; ++r) {
                        out[(a * Q + p) * S + r] =
                            out_p[static_cast<std::size_t>(a) * S + r];
                    }
                }
            }
        };
        if (p_threads == 1) {
            apply_p_range(0, Q, 0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(p_threads));
            for (int worker = 0; worker < p_threads; ++worker) {
                const ssize_t begin = Q * worker / p_threads;
                const ssize_t end = Q * (worker + 1) / p_threads;
                threads.emplace_back(
                    [&, begin, end, worker]() {
                        apply_p_range(begin, end, worker);
                    }
                );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }
    };

    return lanczos_apply(vec, dt, krylov_dim, tol, matvec, imaginary_time);
}
#endif

static py::array_t<cdouble> site_lanczos(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> theta,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> W,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> right,
    double dt,
    int krylov_dim,
    double tol,
    int workers = 1,
    bool imaginary_time = false
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
                w_cutoff,
                workers,
                nullptr,
                imaginary_time
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
            result = lanczos_apply(
                vec, dt, krylov_dim, tol, matvec, imaginary_time
            );
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
        result = lanczos_apply(
            vec, dt, krylov_dim, tol, matvec, imaginary_time
        );
#endif
    }

    py::array_t<cdouble> out({A, P, R});
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

static py::array_t<cdouble> site_lanczos_sum(
    CArray theta,
    const std::vector<CArray>& lefts,
    const std::vector<CArray>& operators,
    const std::vector<CArray>& rights,
    double dt,
    int krylov_dim,
    double tol,
    std::size_t max_direct_sum_elements,
    int workers,
    bool imaginary_time = false
) {
    const std::size_t count = lefts.size();
    if (count == 0 || operators.size() != count || rights.size() != count) {
        throw std::runtime_error(
            "site_lanczos_sum expects equally sized, nonempty component lists"
        );
    }
    if (theta.ndim() != 3) {
        throw std::runtime_error("site_lanczos_sum expects theta(3)");
    }

    const ssize_t B = theta.shape(0);
    const ssize_t Q = theta.shape(1);
    const ssize_t S = theta.shape(2);
    ssize_t total_m = 0;
    ssize_t total_n = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& W = operators[k];
        const CArray& right = rights[k];
        if (
            left.ndim() != 3 || W.ndim() != 4 || right.ndim() != 3
            || left.shape(0) != B || left.shape(2) != B
            || W.shape(0) != left.shape(1)
            || W.shape(2) != Q || W.shape(3) != Q
            || right.shape(0) != S || right.shape(1) != W.shape(1)
            || right.shape(2) != S
        ) {
            throw std::runtime_error(
                "site_lanczos_sum component shapes are incompatible"
            );
        }
        total_m += left.shape(1);
        total_n += right.shape(1);
    }

#if PYQED_TDVP_HAS_BLAS
    std::vector<SiteSumComponentView> components;
    components.reserve(count);
    for (std::size_t k = 0; k < count; ++k) {
        components.push_back(
            SiteSumComponentView{
                lefts[k].data(),
                operators[k].data(),
                rights[k].data(),
                lefts[k].shape(1),
                rights[k].shape(1),
            }
        );
    }
    std::vector<cdouble> vec(static_cast<std::size_t>(theta.size()));
    std::copy(theta.data(), theta.data() + theta.size(), vec.begin());
    std::vector<cdouble> result;
    {
        py::gil_scoped_release release;
        const double w_cutoff = 1.0e-14;
        std::size_t active_groups = 0;
        for (const SiteSumComponentView& component : components) {
            for (ssize_t m = 0; m < component.M; ++m) {
                for (ssize_t p = 0; p < Q; ++p) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        for (ssize_t n = 0; n < component.N; ++n) {
                            if (
                                std::abs(component.W[
                                    ((m * component.N + n) * Q + p) * Q + q
                                ]) > w_cutoff
                            ) {
                                ++active_groups;
                                break;
                            }
                        }
                    }
                }
            }
        }
        const long double packed_right_elements =
            static_cast<long double>(active_groups)
            * static_cast<long double>(S)
            * static_cast<long double>(S);
        const bool use_low_memory =
            max_direct_sum_elements > 0
            && packed_right_elements
                > 4.0L * static_cast<long double>(max_direct_sum_elements);
        if (use_low_memory) {
            result = blocked_low_memory_site_lanczos_sum_apply(
                vec,
                dt,
                krylov_dim,
                tol,
                components,
                B,
                Q,
                S,
                total_m,
                w_cutoff,
                max_direct_sum_elements,
                workers,
                imaginary_time
            );
        } else {
            result = grouped_sparse_site_lanczos_apply(
                vec,
                dt,
                krylov_dim,
                tol,
                nullptr,
                nullptr,
                nullptr,
                B,
                B,
                Q,
                S,
                total_m,
                total_n,
                Q,
                S,
                w_cutoff,
                workers,
                &components,
                imaginary_time
            );
        }
    }
    py::array_t<cdouble> compact_out({B, Q, S});
    std::copy(result.begin(), result.end(), compact_out.mutable_data());
    return compact_out;
#endif

    const std::size_t operator_elements =
        static_cast<std::size_t>(total_m)
        * static_cast<std::size_t>(total_n)
        * static_cast<std::size_t>(Q)
        * static_cast<std::size_t>(Q);
    if (
        max_direct_sum_elements > 0
        && operator_elements > max_direct_sum_elements
    ) {
        throw std::runtime_error(
            "site_lanczos_sum direct-sum workspace exceeds configured limit"
        );
    }

    CArray left_sum({B, total_m, B});
    CArray operator_sum({total_m, total_n, Q, Q});
    CArray right_sum({S, total_n, S});
    std::fill(
        left_sum.mutable_data(),
        left_sum.mutable_data() + left_sum.size(),
        cdouble(0.0, 0.0)
    );
    std::fill(
        operator_sum.mutable_data(),
        operator_sum.mutable_data() + operator_sum.size(),
        cdouble(0.0, 0.0)
    );
    std::fill(
        right_sum.mutable_data(),
        right_sum.mutable_data() + right_sum.size(),
        cdouble(0.0, 0.0)
    );

    ssize_t m_offset = 0;
    ssize_t n_offset = 0;
    cdouble* left_dst = left_sum.mutable_data();
    cdouble* operator_dst = operator_sum.mutable_data();
    cdouble* right_dst = right_sum.mutable_data();
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& W = operators[k];
        const CArray& right = rights[k];
        const ssize_t M = left.shape(1);
        const ssize_t N = right.shape(1);
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t m = 0; m < M; ++m) {
                for (ssize_t b = 0; b < B; ++b) {
                    left_dst[(a * total_m + m_offset + m) * B + b] =
                        left.data()[(a * M + m) * B + b];
                }
            }
        }
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t p = 0; p < Q; ++p) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        operator_dst[
                            (((m_offset + m) * total_n + n_offset + n) * Q + p)
                            * Q + q
                        ] = W.data()[((m * N + n) * Q + p) * Q + q];
                    }
                }
            }
        }
        for (ssize_t r = 0; r < S; ++r) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t s = 0; s < S; ++s) {
                    right_dst[(r * total_n + n_offset + n) * S + s] =
                        right.data()[(r * N + n) * S + s];
                }
            }
        }
        m_offset += M;
        n_offset += N;
    }

    return site_lanczos(
        theta,
        left_sum,
        operator_sum,
        right_sum,
        dt,
        krylov_dim,
        tol,
        workers,
        imaginary_time
    );
}

#if PYQED_TDVP_HAS_BLAS
struct TwoSitePhysicalGroup {
    std::vector<ssize_t> outputs;
    std::vector<cdouble> blocks;
};

struct TwoSiteSumComponentView {
    const cdouble* left;
    const cdouble* first;
    const cdouble* second;
    const cdouble* right;
    ssize_t M;
    ssize_t N;
    ssize_t O;
};

static std::vector<cdouble> grouped_sparse_two_site_lanczos_apply(
    const std::vector<cdouble>& vec,
    double dt,
    int krylov_dim,
    double tol,
    const cdouble* left_ptr,
    const cdouble* W_left_ptr,
    const cdouble* W_right_ptr,
    const cdouble* right_ptr,
    ssize_t A,
    ssize_t B,
    ssize_t Q,
    ssize_t S,
    ssize_t D,
    ssize_t M,
    ssize_t N,
    ssize_t O,
    ssize_t P,
    ssize_t R,
    ssize_t C,
    double cutoff,
    int workers,
    const std::vector<TwoSiteSumComponentView>* sum_components = nullptr
) {
    std::vector<TwoSitePhysicalGroup> left_groups(
        static_cast<std::size_t>(N) * Q
    );
    std::vector<TwoSitePhysicalGroup> right_groups(
        static_cast<std::size_t>(N) * S
    );

    std::vector<TwoSiteSumComponentView> direct_component;
    if (sum_components == nullptr) {
        direct_component.push_back(
            TwoSiteSumComponentView{
                left_ptr, W_left_ptr, W_right_ptr, right_ptr, M, N, O,
            }
        );
        sum_components = &direct_component;
    }
    std::vector<const TwoSiteSumComponentView*> component_for_n(
        static_cast<std::size_t>(N)
    );
    std::vector<ssize_t> local_n(static_cast<std::size_t>(N));
    ssize_t n_offset = 0;
    for (const TwoSiteSumComponentView& component : *sum_components) {
        for (ssize_t n = 0; n < component.N; ++n) {
            component_for_n[static_cast<std::size_t>(n_offset + n)] = &component;
            local_n[static_cast<std::size_t>(n_offset + n)] = n;
        }
        n_offset += component.N;
    }
    if (n_offset != N) {
        throw std::runtime_error("two-site component bond dimensions are inconsistent");
    }

    auto build_groups_for_n = [&](ssize_t global_n) {
        const TwoSiteSumComponentView& component =
            *component_for_n[static_cast<std::size_t>(global_n)];
        const ssize_t n = local_n[static_cast<std::size_t>(global_n)];
        std::vector<cdouble> block_left(static_cast<std::size_t>(A) * B);
        std::vector<cdouble> block_right(static_cast<std::size_t>(C) * D);
        for (ssize_t q = 0; q < Q; ++q) {
            TwoSitePhysicalGroup& group = left_groups[
                static_cast<std::size_t>(global_n) * Q + q
            ];
            for (ssize_t p = 0; p < P; ++p) {
                double max_abs = 0.0;
                for (ssize_t a = 0; a < A; ++a) {
                    for (ssize_t b = 0; b < B; ++b) {
                        cdouble value = 0.0;
                        for (ssize_t m = 0; m < component.M; ++m) {
                            value +=
                                component.left[(a * component.M + m) * B + b]
                                * component.first[
                                    ((m * component.N + n) * P + p) * Q + q
                                ];
                        }
                        block_left[static_cast<std::size_t>(a) * B + b] = value;
                        max_abs = std::max(max_abs, std::abs(value));
                    }
                }
                if (max_abs > cutoff) {
                    group.outputs.push_back(p);
                    group.blocks.insert(
                        group.blocks.end(), block_left.begin(), block_left.end()
                    );
                }
            }
        }
        for (ssize_t s = 0; s < S; ++s) {
            TwoSitePhysicalGroup& group = right_groups[
                static_cast<std::size_t>(global_n) * S + s
            ];
            for (ssize_t r = 0; r < R; ++r) {
                double max_abs = 0.0;
                for (ssize_t c = 0; c < C; ++c) {
                    for (ssize_t d = 0; d < D; ++d) {
                        cdouble value = 0.0;
                        for (ssize_t o = 0; o < component.O; ++o) {
                            value +=
                                component.second[
                                    ((n * component.O + o) * R + r) * S + s
                                ]
                                * component.right[(c * component.O + o) * D + d];
                        }
                        block_right[static_cast<std::size_t>(c) * D + d] = value;
                        max_abs = std::max(max_abs, std::abs(value));
                    }
                }
                if (max_abs > cutoff) {
                    group.outputs.push_back(r);
                    group.blocks.insert(
                        group.blocks.end(), block_right.begin(), block_right.end()
                    );
                }
            }
        }
    };

    const int setup_threads = std::max(1, std::min(workers, static_cast<int>(N)));
    if (setup_threads == 1) {
        for (ssize_t n = 0; n < N; ++n) {
            build_groups_for_n(n);
        }
    } else {
        std::atomic<ssize_t> next_n(0);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(setup_threads));
        for (int worker = 0; worker < setup_threads; ++worker) {
            threads.emplace_back([&]() {
                while (true) {
                    const ssize_t n = next_n.fetch_add(1, std::memory_order_relaxed);
                    if (n >= N) {
                        return;
                    }
                    build_groups_for_n(n);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    std::size_t max_left_rows = 0;
    std::size_t max_right_rows = 0;
    for (const auto& group : left_groups) {
        max_left_rows = std::max(
            max_left_rows,
            group.outputs.size() * static_cast<std::size_t>(A)
        );
    }
    for (const auto& group : right_groups) {
        max_right_rows = std::max(
            max_right_rows,
            group.outputs.size() * static_cast<std::size_t>(C)
        );
    }

    const std::size_t dim = static_cast<std::size_t>(B) * Q * S * D;
    const ssize_t AP = A * P;
    const ssize_t SD = S * D;
    std::vector<cdouble> x_by_q(
        static_cast<std::size_t>(Q) * B * SD,
        0.0
    );
    struct Scratch {
        std::vector<cdouble> intermediate;
        std::vector<cdouble> left_contribution;
        std::vector<cdouble> intermediate_s;
        std::vector<cdouble> right_contribution;
        std::vector<cdouble> out;

        Scratch(
            std::size_t intermediate_size,
            std::size_t left_size,
            std::size_t intermediate_s_size,
            std::size_t right_size,
            std::size_t out_size
        ) :
            intermediate(intermediate_size, 0.0),
            left_contribution(left_size, 0.0),
            intermediate_s(intermediate_s_size, 0.0),
            right_contribution(right_size, 0.0),
            out(out_size, 0.0) {}
    };
    const int thread_count = std::max(
        1,
        std::min(workers, static_cast<int>(N))
    );
    std::vector<Scratch> scratches;
    scratches.reserve(static_cast<std::size_t>(thread_count));
    for (int worker = 0; worker < thread_count; ++worker) {
        scratches.emplace_back(
            static_cast<std::size_t>(AP) * SD,
            max_left_rows * static_cast<std::size_t>(SD),
            static_cast<std::size_t>(AP) * D,
            static_cast<std::size_t>(AP) * max_right_rows,
            dim
        );
    }
    const cdouble one(1.0, 0.0);
    const cdouble zero(0.0, 0.0);

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        for (ssize_t q = 0; q < Q; ++q) {
            cdouble* x_q =
                x_by_q.data() + static_cast<std::size_t>(q) * B * SD;
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t s = 0; s < S; ++s) {
                    for (ssize_t d = 0; d < D; ++d) {
                        x_q[(b * S + s) * D + d] =
                            local[((b * Q + q) * S + s) * D + d];
                    }
                }
            }
        }
        for (Scratch& scratch : scratches) {
            std::fill(scratch.out.begin(), scratch.out.end(), cdouble(0.0, 0.0));
        }
        auto apply_range = [&](ssize_t begin, ssize_t end, Scratch& scratch) {
        for (ssize_t n = begin; n < end; ++n) {
            std::fill(
                scratch.intermediate.begin(),
                scratch.intermediate.end(),
                cdouble(0.0, 0.0)
            );
            for (ssize_t q = 0; q < Q; ++q) {
                const TwoSitePhysicalGroup& left_group =
                    left_groups[static_cast<std::size_t>(n) * Q + q];
                if (left_group.outputs.empty()) {
                    continue;
                }
                const ssize_t left_rows =
                    static_cast<ssize_t>(left_group.outputs.size()) * A;
                cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(left_rows),
                    static_cast<int>(SD),
                    static_cast<int>(B),
                    &one,
                    left_group.blocks.data(),
                    static_cast<int>(B),
                    x_by_q.data() + static_cast<std::size_t>(q) * B * SD,
                    static_cast<int>(SD),
                    &zero,
                    scratch.left_contribution.data(),
                    static_cast<int>(SD)
                );
                for (
                    std::size_t lp = 0;
                    lp < left_group.outputs.size();
                    ++lp
                ) {
                    const ssize_t p = left_group.outputs[lp];
                    for (ssize_t a = 0; a < A; ++a) {
                        const cdouble* source =
                            scratch.left_contribution.data()
                            + (static_cast<ssize_t>(lp) * A + a) * SD;
                        cdouble* target =
                            scratch.intermediate.data() + (p * A + a) * SD;
                        for (ssize_t sd = 0; sd < SD; ++sd) {
                            target[sd] += source[sd];
                        }
                    }
                }
            }
            for (ssize_t s = 0; s < S; ++s) {
                const TwoSitePhysicalGroup& right_group =
                    right_groups[static_cast<std::size_t>(n) * S + s];
                if (right_group.outputs.empty()) {
                    continue;
                }
                const ssize_t right_rows =
                    static_cast<ssize_t>(right_group.outputs.size()) * C;
                for (ssize_t ap = 0; ap < AP; ++ap) {
                    const cdouble* source =
                        scratch.intermediate.data() + ap * SD + s * D;
                    cdouble* target = scratch.intermediate_s.data() + ap * D;
                    for (ssize_t d = 0; d < D; ++d) {
                        target[d] = source[d];
                    }
                }
                cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(AP),
                    static_cast<int>(right_rows),
                    static_cast<int>(D),
                    &one,
                    scratch.intermediate_s.data(),
                    static_cast<int>(D),
                    right_group.blocks.data(),
                    static_cast<int>(D),
                    &zero,
                    scratch.right_contribution.data(),
                    static_cast<int>(right_rows)
                );
                for (ssize_t p = 0; p < P; ++p) {
                    for (ssize_t a = 0; a < A; ++a) {
                        const ssize_t ap = p * A + a;
                        for (
                            std::size_t rr = 0;
                            rr < right_group.outputs.size();
                            ++rr
                        ) {
                            const ssize_t r = right_group.outputs[rr];
                            for (ssize_t c = 0; c < C; ++c) {
                                scratch.out[((a * P + p) * R + r) * C + c] +=
                                    scratch.right_contribution[
                                        static_cast<std::size_t>(ap) * right_rows
                                        + static_cast<ssize_t>(rr) * C
                                        + c
                                    ];
                            }
                        }
                    }
                }
            }
        }
        };

        if (thread_count == 1) {
            apply_range(0, N, scratches[0]);
        } else {
            std::atomic<ssize_t> next_n(0);
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(thread_count));
            for (int worker = 0; worker < thread_count; ++worker) {
                threads.emplace_back(
                    [&, worker]() {
                        Scratch& scratch = scratches[static_cast<std::size_t>(worker)];
                        while (true) {
                            const ssize_t n = next_n.fetch_add(
                                1, std::memory_order_relaxed
                            );
                            if (n >= N) {
                                return;
                            }
                            apply_range(n, n + 1, scratch);
                        }
                    }
                );
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }
        out.assign(dim, cdouble(0.0, 0.0));
        for (const Scratch& scratch : scratches) {
            for (std::size_t index = 0; index < dim; ++index) {
                out[index] += scratch.out[index];
            }
        }
    };

    return lanczos_apply(vec, dt, krylov_dim, tol, matvec);
}
#endif

static py::array_t<cdouble> two_site_lanczos(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> theta,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> W_left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> W_right,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> right,
    double dt,
    int krylov_dim,
    double tol,
    int workers = 1
) {
    if (
        theta.ndim() != 4
        || left.ndim() != 3
        || W_left.ndim() != 4
        || W_right.ndim() != 4
        || right.ndim() != 3
    ) {
        throw std::runtime_error(
            "two_site_lanczos expects theta(4), left(3), W_left(4), "
            "W_right(4), right(3)"
        );
    }

    const ssize_t B = theta.shape(0);
    const ssize_t Q = theta.shape(1);
    const ssize_t S = theta.shape(2);
    const ssize_t D = theta.shape(3);
    const ssize_t A = left.shape(0);
    const ssize_t M = left.shape(1);
    const ssize_t N = W_left.shape(1);
    const ssize_t P = W_left.shape(2);
    const ssize_t O = W_right.shape(1);
    const ssize_t R = W_right.shape(2);
    const ssize_t C = right.shape(0);

    if (
        left.shape(2) != B
        || W_left.shape(0) != M
        || W_left.shape(3) != Q
        || W_right.shape(0) != N
        || W_right.shape(3) != S
        || right.shape(1) != O
        || right.shape(2) != D
    ) {
        throw std::runtime_error("two_site_lanczos input shapes are incompatible");
    }
    if (A != B || P != Q || R != S || C != D) {
        throw std::runtime_error("two_site_lanczos requires a square effective local space");
    }

    const cdouble* theta_ptr = theta.data();
    const cdouble* left_ptr = left.data();
    const cdouble* W_left_ptr = W_left.data();
    const cdouble* W_right_ptr = W_right.data();
    const cdouble* right_ptr = right.data();
    const ssize_t AP = A * P;
    const ssize_t BQ = B * Q;
    const ssize_t SD = S * D;
    const ssize_t RC = R * C;
    const std::size_t dim = static_cast<std::size_t>(BQ) * SD;

    std::vector<cdouble> vec(dim);
    std::copy(theta_ptr, theta_ptr + static_cast<ssize_t>(dim), vec.begin());
    std::vector<cdouble> result;

    {
        py::gil_scoped_release release;
#if PYQED_TDVP_HAS_BLAS
        const double physical_cutoff = 1.0e-14;
        std::vector<std::size_t> left_transition_counts(
            static_cast<std::size_t>(N),
            0
        );
        std::vector<std::size_t> right_transition_counts(
            static_cast<std::size_t>(N),
            0
        );
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t p = 0; p < P; ++p) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        if (
                            std::abs(
                                W_left_ptr[((m * N + n) * P + p) * Q + q]
                            ) > physical_cutoff
                        ) {
                            ++left_transition_counts[static_cast<std::size_t>(n)];
                        }
                    }
                }
            }
        }
        for (ssize_t n = 0; n < N; ++n) {
            for (ssize_t o = 0; o < O; ++o) {
                for (ssize_t r = 0; r < R; ++r) {
                    for (ssize_t s = 0; s < S; ++s) {
                        if (
                            std::abs(
                                W_right_ptr[((n * O + o) * R + r) * S + s]
                            ) > physical_cutoff
                        ) {
                            ++right_transition_counts[static_cast<std::size_t>(n)];
                        }
                    }
                }
            }
        }
        std::size_t sparse_routes = 0;
        for (ssize_t n = 0; n < N; ++n) {
            sparse_routes +=
                left_transition_counts[static_cast<std::size_t>(n)]
                * right_transition_counts[static_cast<std::size_t>(n)];
        }
        const std::size_t dense_routes =
            static_cast<std::size_t>(N) * M * P * Q * O * R * S;
        if (
            sparse_routes > 0
            && sparse_routes * 4 < dense_routes
            && A * P <= static_cast<ssize_t>(INT_MAX)
            && B <= static_cast<ssize_t>(INT_MAX)
            && C * R <= static_cast<ssize_t>(INT_MAX)
            && D <= static_cast<ssize_t>(INT_MAX)
        ) {
            result = grouped_sparse_two_site_lanczos_apply(
                vec,
                dt,
                krylov_dim,
                tol,
                left_ptr,
                W_left_ptr,
                W_right_ptr,
                right_ptr,
                A,
                B,
                Q,
                S,
                D,
                M,
                N,
                O,
                P,
                R,
                C,
                physical_cutoff,
                workers
            );
        } else {
#endif
        std::vector<cdouble> left_eff(
            static_cast<std::size_t>(N) * AP * BQ,
            0.0
        );
        std::vector<cdouble> right_eff(
            static_cast<std::size_t>(N) * RC * SD,
            0.0
        );

        for (ssize_t n = 0; n < N; ++n) {
            cdouble* block = left_eff.data() + static_cast<std::size_t>(n) * AP * BQ;
            for (ssize_t a = 0; a < A; ++a) {
                for (ssize_t p = 0; p < P; ++p) {
                    const ssize_t row = a * P + p;
                    for (ssize_t b = 0; b < B; ++b) {
                        for (ssize_t q = 0; q < Q; ++q) {
                            cdouble value = 0.0;
                            for (ssize_t m = 0; m < M; ++m) {
                                value +=
                                    left_ptr[(a * M + m) * B + b]
                                    * W_left_ptr[((m * N + n) * P + p) * Q + q];
                            }
                            block[row * BQ + b * Q + q] = value;
                        }
                    }
                }
            }
        }

        for (ssize_t n = 0; n < N; ++n) {
            cdouble* block = right_eff.data() + static_cast<std::size_t>(n) * RC * SD;
            for (ssize_t r = 0; r < R; ++r) {
                for (ssize_t c = 0; c < C; ++c) {
                    const ssize_t row = r * C + c;
                    for (ssize_t s = 0; s < S; ++s) {
                        for (ssize_t d = 0; d < D; ++d) {
                            cdouble value = 0.0;
                            for (ssize_t o = 0; o < O; ++o) {
                                value +=
                                    W_right_ptr[((n * O + o) * R + r) * S + s]
                                    * right_ptr[(c * O + o) * D + d];
                            }
                            block[row * SD + s * D + d] = value;
                        }
                    }
                }
            }
        }

        std::vector<cdouble> temporary(static_cast<std::size_t>(AP) * SD, 0.0);
        auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            out.assign(static_cast<std::size_t>(AP) * RC, cdouble(0.0, 0.0));
#if PYQED_TDVP_HAS_BLAS
            const cdouble one(1.0, 0.0);
            const cdouble zero(0.0, 0.0);
            for (ssize_t n = 0; n < N; ++n) {
                const cdouble* left_block =
                    left_eff.data() + static_cast<std::size_t>(n) * AP * BQ;
                const cdouble* right_block =
                    right_eff.data() + static_cast<std::size_t>(n) * RC * SD;
                cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(AP),
                    static_cast<int>(SD),
                    static_cast<int>(BQ),
                    &one,
                    left_block,
                    static_cast<int>(BQ),
                    local.data(),
                    static_cast<int>(SD),
                    &zero,
                    temporary.data(),
                    static_cast<int>(SD)
                );
                cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(AP),
                    static_cast<int>(RC),
                    static_cast<int>(SD),
                    &one,
                    temporary.data(),
                    static_cast<int>(SD),
                    right_block,
                    static_cast<int>(SD),
                    n == 0 ? &zero : &one,
                    out.data(),
                    static_cast<int>(RC)
                );
            }
#else
            for (ssize_t n = 0; n < N; ++n) {
                const cdouble* left_block =
                    left_eff.data() + static_cast<std::size_t>(n) * AP * BQ;
                const cdouble* right_block =
                    right_eff.data() + static_cast<std::size_t>(n) * RC * SD;
                for (ssize_t ap = 0; ap < AP; ++ap) {
                    for (ssize_t sd = 0; sd < SD; ++sd) {
                        cdouble value = 0.0;
                        for (ssize_t bq = 0; bq < BQ; ++bq) {
                            value += left_block[ap * BQ + bq] * local[bq * SD + sd];
                        }
                        temporary[ap * SD + sd] = value;
                    }
                }
                for (ssize_t ap = 0; ap < AP; ++ap) {
                    for (ssize_t rc = 0; rc < RC; ++rc) {
                        cdouble value = 0.0;
                        for (ssize_t sd = 0; sd < SD; ++sd) {
                            value += temporary[ap * SD + sd] * right_block[rc * SD + sd];
                        }
                        out[ap * RC + rc] += value;
                    }
                }
            }
#endif
        };
        result = lanczos_apply(vec, dt, krylov_dim, tol, matvec);
#if PYQED_TDVP_HAS_BLAS
        }
#endif
    }

    py::array_t<cdouble> out({A, P, R, C});
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

static py::array_t<cdouble> two_site_lanczos_sum(
    CArray theta,
    const std::vector<CArray>& lefts,
    const std::vector<CArray>& first_operators,
    const std::vector<CArray>& second_operators,
    const std::vector<CArray>& rights,
    double dt,
    int krylov_dim,
    double tol,
    std::size_t max_direct_sum_elements,
    int workers
) {
    const std::size_t count = lefts.size();
    if (
        count == 0 || first_operators.size() != count
        || second_operators.size() != count || rights.size() != count
    ) {
        throw std::runtime_error(
            "two_site_lanczos_sum expects equally sized, nonempty component lists"
        );
    }
    if (theta.ndim() != 4) {
        throw std::runtime_error("two_site_lanczos_sum expects theta(4)");
    }

    const ssize_t B = theta.shape(0);
    const ssize_t Q = theta.shape(1);
    const ssize_t S = theta.shape(2);
    const ssize_t D = theta.shape(3);
    ssize_t total_m = 0;
    ssize_t total_n = 0;
    ssize_t total_o = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& first = first_operators[k];
        const CArray& second = second_operators[k];
        const CArray& right = rights[k];
        if (
            left.ndim() != 3 || first.ndim() != 4 || second.ndim() != 4
            || right.ndim() != 3
            || left.shape(0) != B || left.shape(2) != B
            || first.shape(0) != left.shape(1)
            || first.shape(2) != Q || first.shape(3) != Q
            || second.shape(0) != first.shape(1)
            || second.shape(2) != S || second.shape(3) != S
            || right.shape(0) != D || right.shape(1) != second.shape(1)
            || right.shape(2) != D
        ) {
            throw std::runtime_error(
                "two_site_lanczos_sum component shapes are incompatible"
            );
        }
        total_m += left.shape(1);
        total_n += first.shape(1);
        total_o += second.shape(1);
    }

#if PYQED_TDVP_HAS_BLAS
    std::vector<TwoSiteSumComponentView> components;
    components.reserve(count);
    for (std::size_t k = 0; k < count; ++k) {
        components.push_back(
            TwoSiteSumComponentView{
                lefts[k].data(),
                first_operators[k].data(),
                second_operators[k].data(),
                rights[k].data(),
                lefts[k].shape(1),
                first_operators[k].shape(1),
                second_operators[k].shape(1),
            }
        );
    }
    std::vector<cdouble> vec(static_cast<std::size_t>(theta.size()));
    std::copy(theta.data(), theta.data() + theta.size(), vec.begin());
    std::vector<cdouble> result;
    {
        py::gil_scoped_release release;
        result = grouped_sparse_two_site_lanczos_apply(
            vec,
            dt,
            krylov_dim,
            tol,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            B,
            B,
            Q,
            S,
            D,
            total_m,
            total_n,
            total_o,
            Q,
            S,
            D,
            1.0e-14,
            workers,
            &components
        );
    }
    py::array_t<cdouble> compact_out({B, Q, S, D});
    std::copy(result.begin(), result.end(), compact_out.mutable_data());
    return compact_out;
#endif

    const std::size_t first_elements =
        static_cast<std::size_t>(total_m)
        * static_cast<std::size_t>(total_n)
        * static_cast<std::size_t>(Q)
        * static_cast<std::size_t>(Q);
    const std::size_t second_elements =
        static_cast<std::size_t>(total_n)
        * static_cast<std::size_t>(total_o)
        * static_cast<std::size_t>(S)
        * static_cast<std::size_t>(S);
    if (
        max_direct_sum_elements > 0
        && (first_elements > max_direct_sum_elements
            || second_elements > max_direct_sum_elements)
    ) {
        throw std::runtime_error(
            "two_site_lanczos_sum direct-sum workspace exceeds configured limit"
        );
    }

    CArray left_sum({B, total_m, B});
    CArray first_sum({total_m, total_n, Q, Q});
    CArray second_sum({total_n, total_o, S, S});
    CArray right_sum({D, total_o, D});
    std::fill(
        left_sum.mutable_data(),
        left_sum.mutable_data() + left_sum.size(),
        cdouble(0.0, 0.0)
    );
    std::fill(
        first_sum.mutable_data(),
        first_sum.mutable_data() + first_sum.size(),
        cdouble(0.0, 0.0)
    );
    std::fill(
        second_sum.mutable_data(),
        second_sum.mutable_data() + second_sum.size(),
        cdouble(0.0, 0.0)
    );
    std::fill(
        right_sum.mutable_data(),
        right_sum.mutable_data() + right_sum.size(),
        cdouble(0.0, 0.0)
    );

    ssize_t m_offset = 0;
    ssize_t n_offset = 0;
    ssize_t o_offset = 0;
    cdouble* left_dst = left_sum.mutable_data();
    cdouble* first_dst = first_sum.mutable_data();
    cdouble* second_dst = second_sum.mutable_data();
    cdouble* right_dst = right_sum.mutable_data();
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& first = first_operators[k];
        const CArray& second = second_operators[k];
        const CArray& right = rights[k];
        const ssize_t M = left.shape(1);
        const ssize_t N = first.shape(1);
        const ssize_t O = second.shape(1);
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t m = 0; m < M; ++m) {
                for (ssize_t b = 0; b < B; ++b) {
                    left_dst[(a * total_m + m_offset + m) * B + b] =
                        left.data()[(a * M + m) * B + b];
                }
            }
        }
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t p = 0; p < Q; ++p) {
                    for (ssize_t q = 0; q < Q; ++q) {
                        first_dst[
                            (((m_offset + m) * total_n + n_offset + n) * Q + p)
                            * Q + q
                        ] = first.data()[((m * N + n) * Q + p) * Q + q];
                    }
                }
            }
        }
        for (ssize_t n = 0; n < N; ++n) {
            for (ssize_t o = 0; o < O; ++o) {
                for (ssize_t r = 0; r < S; ++r) {
                    for (ssize_t s = 0; s < S; ++s) {
                        second_dst[
                            (((n_offset + n) * total_o + o_offset + o) * S + r)
                            * S + s
                        ] = second.data()[((n * O + o) * S + r) * S + s];
                    }
                }
            }
        }
        for (ssize_t c = 0; c < D; ++c) {
            for (ssize_t o = 0; o < O; ++o) {
                for (ssize_t d = 0; d < D; ++d) {
                    right_dst[(c * total_o + o_offset + o) * D + d] =
                        right.data()[(c * O + o) * D + d];
                }
            }
        }
        m_offset += M;
        n_offset += N;
        o_offset += O;
    }

    return two_site_lanczos(
        theta,
        left_sum,
        first_sum,
        second_sum,
        right_sum,
        dt,
        krylov_dim,
        tol,
        workers
    );
}

static py::array_t<cdouble> bond_lanczos(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> center,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> left,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> right,
    double dt,
    int krylov_dim,
    double tol,
    bool imaginary_time = false
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
        result = lanczos_apply(
            vec, dt, krylov_dim, tol, matvec, imaginary_time
        );
    }

    py::array_t<cdouble> out({A, R});
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

static py::array_t<cdouble> bond_lanczos_sum(
    CArray center,
    const std::vector<CArray>& lefts,
    const std::vector<CArray>& rights,
    double dt,
    int krylov_dim,
    double tol,
    std::size_t max_direct_sum_elements,
    bool imaginary_time = false
) {
    const std::size_t count = lefts.size();
    if (count == 0 || rights.size() != count || center.ndim() != 2) {
        throw std::runtime_error(
            "bond_lanczos_sum expects equally sized, nonempty environment lists"
        );
    }
    const ssize_t B = center.shape(0);
    const ssize_t S = center.shape(1);
    ssize_t total_m = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& right = rights[k];
        if (
            left.ndim() != 3 || right.ndim() != 3
            || left.shape(0) != B || left.shape(2) != B
            || right.shape(0) != S || right.shape(2) != S
            || right.shape(1) != left.shape(1)
        ) {
            throw std::runtime_error(
                "bond_lanczos_sum environment shapes are incompatible"
            );
        }
        total_m += left.shape(1);
    }
    const std::size_t left_elements =
        static_cast<std::size_t>(B) * total_m * B;
    const std::size_t right_elements =
        static_cast<std::size_t>(S) * total_m * S;
    if (
        max_direct_sum_elements > 0
        && (left_elements > max_direct_sum_elements
            || right_elements > max_direct_sum_elements)
    ) {
        throw std::runtime_error(
            "bond_lanczos_sum workspace exceeds configured limit"
        );
    }

    CArray left_sum({B, total_m, B});
    CArray right_sum({S, total_m, S});
    ssize_t offset = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const CArray& left = lefts[k];
        const CArray& right = rights[k];
        const ssize_t M = left.shape(1);
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t m = 0; m < M; ++m) {
                for (ssize_t b = 0; b < B; ++b) {
                    left_sum.mutable_data()[
                        (a * total_m + offset + m) * B + b
                    ] = left.data()[(a * M + m) * B + b];
                }
            }
        }
        for (ssize_t r = 0; r < S; ++r) {
            for (ssize_t m = 0; m < M; ++m) {
                for (ssize_t s = 0; s < S; ++s) {
                    right_sum.mutable_data()[
                        (r * total_m + offset + m) * S + s
                    ] = right.data()[(r * M + m) * S + s];
                }
            }
        }
        offset += M;
    }
    return bond_lanczos(
        center,
        left_sum,
        right_sum,
        dt,
        krylov_dim,
        tol,
        imaginary_time
    );
}

struct DenseQRResult {
    CArray q;
    CArray r;
};

static DenseQRResult dense_thin_qr(
    const cdouble* matrix,
    ssize_t rows,
    ssize_t cols
) {
#if !PYQED_TDVP_HAS_BLAS
    throw std::runtime_error(
        "the compiled sum-TDVP sweep requires a LAPACK-enabled build"
    );
#else
    if (
        rows <= 0 || cols <= 0
        || rows > std::numeric_limits<int>::max()
        || cols > std::numeric_limits<int>::max()
    ) {
        throw std::runtime_error("dense QR dimensions are unsupported");
    }
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = std::min(m, n);
    const int lda = std::max(1, m);
    std::vector<cdouble> a(static_cast<std::size_t>(lda) * n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            a[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * lda] =
                matrix[static_cast<std::size_t>(row) * n + col];
        }
    }

    std::vector<cdouble> tau(static_cast<std::size_t>(k));
    int m_arg = m;
    int n_arg = n;
    int lda_arg = lda;
    int info = 0;
    int lwork = -1;
    cdouble work_query;
    zgeqrf_(
        &m_arg,
        &n_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(a.data()),
        &lda_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(tau.data()),
        reinterpret_cast<__CLPK_doublecomplex*>(&work_query),
        &lwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "zgeqrf workspace query failed with info=" + std::to_string(info)
        );
    }
    lwork = std::max(1, static_cast<int>(std::real(work_query)));
    std::vector<cdouble> work(static_cast<std::size_t>(lwork));
    zgeqrf_(
        &m_arg,
        &n_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(a.data()),
        &lda_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(tau.data()),
        reinterpret_cast<__CLPK_doublecomplex*>(work.data()),
        &lwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error("zgeqrf failed with info=" + std::to_string(info));
    }

    CArray r({static_cast<ssize_t>(k), cols});
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            r.mutable_data()[static_cast<std::size_t>(row) * n + col] =
                row <= col
                ? a[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * lda]
                : cdouble(0.0, 0.0);
        }
    }

    int q_m = m;
    int q_n = k;
    int q_k = k;
    lwork = -1;
    zungqr_(
        &q_m,
        &q_n,
        &q_k,
        reinterpret_cast<__CLPK_doublecomplex*>(a.data()),
        &lda_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(tau.data()),
        reinterpret_cast<__CLPK_doublecomplex*>(&work_query),
        &lwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error(
            "zungqr workspace query failed with info=" + std::to_string(info)
        );
    }
    lwork = std::max(1, static_cast<int>(std::real(work_query)));
    work.assign(static_cast<std::size_t>(lwork), cdouble(0.0, 0.0));
    zungqr_(
        &q_m,
        &q_n,
        &q_k,
        reinterpret_cast<__CLPK_doublecomplex*>(a.data()),
        &lda_arg,
        reinterpret_cast<__CLPK_doublecomplex*>(tau.data()),
        reinterpret_cast<__CLPK_doublecomplex*>(work.data()),
        &lwork,
        &info
    );
    if (info != 0) {
        throw std::runtime_error("zungqr failed with info=" + std::to_string(info));
    }

    CArray q({rows, static_cast<ssize_t>(k)});
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            q.mutable_data()[static_cast<std::size_t>(row) * k + col] =
                a[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * lda];
        }
    }
    return DenseQRResult{std::move(q), std::move(r)};
#endif
}

static std::pair<CArray, CArray> dense_left_qr_site(const CArray& tensor) {
    if (tensor.ndim() != 3) {
        throw std::runtime_error("left QR expects a rank-3 MPS tensor");
    }
    const ssize_t left = tensor.shape(0);
    const ssize_t physical = tensor.shape(1);
    const ssize_t right = tensor.shape(2);
    DenseQRResult qr = dense_thin_qr(tensor.data(), left * physical, right);
    const ssize_t rank = qr.q.shape(1);
    CArray site({left, physical, rank});
    std::copy(qr.q.data(), qr.q.data() + qr.q.size(), site.mutable_data());
    return {std::move(site), std::move(qr.r)};
}

static std::pair<CArray, CArray> dense_right_rq_site(const CArray& tensor) {
    if (tensor.ndim() != 3) {
        throw std::runtime_error("right RQ expects a rank-3 MPS tensor");
    }
    const ssize_t left = tensor.shape(0);
    const ssize_t physical = tensor.shape(1);
    const ssize_t right = tensor.shape(2);
    const ssize_t columns = physical * right;
    std::vector<cdouble> transposed(
        static_cast<std::size_t>(columns) * left
    );
    for (ssize_t row = 0; row < left; ++row) {
        for (ssize_t col = 0; col < columns; ++col) {
            transposed[static_cast<std::size_t>(col) * left + row] =
                tensor.data()[static_cast<std::size_t>(row) * columns + col];
        }
    }
    DenseQRResult qr = dense_thin_qr(transposed.data(), columns, left);
    const ssize_t rank = qr.q.shape(1);
    CArray center({left, rank});
    for (ssize_t row = 0; row < left; ++row) {
        for (ssize_t col = 0; col < rank; ++col) {
            center.mutable_data()[row * rank + col] = qr.r.data()[col * left + row];
        }
    }
    CArray site({rank, physical, right});
    for (ssize_t row = 0; row < rank; ++row) {
        for (ssize_t col = 0; col < columns; ++col) {
            site.mutable_data()[row * columns + col] = qr.q.data()[col * rank + row];
        }
    }
    return {std::move(center), std::move(site)};
}

static CArray allocate_left_environment(
    const CArray& left,
    const CArray& site,
    const CArray& W
) {
    if (
        left.ndim() != 3 || site.ndim() != 3 || W.ndim() != 4
        || left.shape(0) != site.shape(0)
        || left.shape(2) != site.shape(0)
        || left.shape(1) != W.shape(0)
        || site.shape(1) != W.shape(2)
        || site.shape(1) != W.shape(3)
    ) {
        throw std::runtime_error("left environment shapes are incompatible");
    }
    return CArray({site.shape(2), W.shape(1), site.shape(2)});
}

static CArray allocate_right_environment(
    const CArray& right,
    const CArray& site,
    const CArray& W
) {
    if (
        right.ndim() != 3 || site.ndim() != 3 || W.ndim() != 4
        || right.shape(0) != site.shape(2)
        || right.shape(2) != site.shape(2)
        || right.shape(1) != W.shape(1)
        || site.shape(1) != W.shape(2)
        || site.shape(1) != W.shape(3)
    ) {
        throw std::runtime_error("right environment shapes are incompatible");
    }
    return CArray({site.shape(0), W.shape(0), site.shape(0)});
}

static void fill_left_environment(
    const CArray& left,
    const CArray& site,
    const CArray& W,
    CArray& out
) {
    const ssize_t B = site.shape(0);
    const ssize_t Q = site.shape(1);
    const ssize_t S = site.shape(2);
    const ssize_t M = W.shape(0);
    const ssize_t N = W.shape(1);
    std::fill(out.mutable_data(), out.mutable_data() + out.size(), cdouble(0.0, 0.0));
#if PYQED_TDVP_HAS_BLAS
    const ssize_t flat = Q * S;
    std::vector<cdouble> environment(static_cast<std::size_t>(B) * B);
    std::vector<cdouble> first(static_cast<std::size_t>(B) * flat);
    std::vector<cdouble> left_site(
        static_cast<std::size_t>(B) * M * Q * S
    );
    std::vector<cdouble> wmat(static_cast<std::size_t>(N) * M * Q);
    std::vector<cdouble> contracted(static_cast<std::size_t>(N) * S);
    std::vector<cdouble> operator_site(
        static_cast<std::size_t>(B) * N * Q * S
    );
    std::vector<cdouble> ket(static_cast<std::size_t>(B) * Q * S);
    std::vector<cdouble> result(static_cast<std::size_t>(S) * S);
    const cdouble alpha(1.0, 0.0);
    const cdouble beta(0.0, 0.0);
    for (ssize_t m = 0; m < M; ++m) {
        for (ssize_t i = 0; i < B; ++i) {
            for (ssize_t k = 0; k < B; ++k) {
                environment[i * B + k] = left.data()[(i * M + m) * B + k];
            }
        }
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(B), static_cast<int>(flat), static_cast<int>(B),
            &alpha, environment.data(), static_cast<int>(B),
            site.data(), static_cast<int>(flat),
            &beta, first.data(), static_cast<int>(flat)
        );
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t qket = 0; qket < Q; ++qket) {
                for (ssize_t s = 0; s < S; ++s) {
                    left_site[((a * M + m) * Q + qket) * S + s] =
                        first[(a * Q + qket) * S + s];
                }
            }
        }
    }
    for (ssize_t p = 0; p < Q; ++p) {
        for (ssize_t n = 0; n < N; ++n) {
            for (ssize_t m = 0; m < M; ++m) {
                for (ssize_t qket = 0; qket < Q; ++qket) {
                    wmat[n * M * Q + m * Q + qket] =
                        W.data()[((m * N + n) * Q + p) * Q + qket];
                }
            }
        }
        for (ssize_t a = 0; a < B; ++a) {
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(N), static_cast<int>(S), static_cast<int>(M * Q),
                &alpha, wmat.data(), static_cast<int>(M * Q),
                left_site.data() + static_cast<std::size_t>(a) * M * Q * S,
                static_cast<int>(S),
                &beta, contracted.data(), static_cast<int>(S)
        );
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t s = 0; s < S; ++s) {
                    operator_site[((a * N + n) * Q + p) * S + s] =
                        contracted[n * S + s];
                }
            }
        }
    }
    for (ssize_t n = 0; n < N; ++n) {
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t p = 0; p < Q; ++p) {
                for (ssize_t s = 0; s < S; ++s) {
                    ket[(a * Q + p) * S + s] =
                        operator_site[((a * N + n) * Q + p) * S + s];
                }
            }
        }
        cblas_zgemm(
            CblasRowMajor, CblasConjTrans, CblasNoTrans,
            static_cast<int>(S), static_cast<int>(S), static_cast<int>(B * Q),
            &alpha, site.data(), static_cast<int>(S),
            ket.data(), static_cast<int>(S),
            &beta, result.data(), static_cast<int>(S)
        );
        for (ssize_t j = 0; j < S; ++j) {
            for (ssize_t s = 0; s < S; ++s) {
                out.mutable_data()[(j * N + n) * S + s] = result[j * S + s];
            }
        }
    }
#else
    for (ssize_t j = 0; j < S; ++j) {
        for (ssize_t n = 0; n < N; ++n) {
            for (ssize_t l = 0; l < S; ++l) {
                cdouble total = 0.0;
                for (ssize_t i = 0; i < B; ++i) {
                    for (ssize_t m = 0; m < M; ++m) {
                        for (ssize_t k = 0; k < B; ++k) {
                            for (ssize_t p = 0; p < Q; ++p) {
                                for (ssize_t qket = 0; qket < Q; ++qket) {
                                    total += std::conj(site.data()[(i * Q + p) * S + j])
                                        * left.data()[(i * M + m) * B + k]
                                        * W.data()[((m * N + n) * Q + p) * Q + qket]
                                        * site.data()[(k * Q + qket) * S + l];
                                }
                            }
                        }
                    }
                }
                out.mutable_data()[(j * N + n) * S + l] = total;
            }
        }
    }
#endif
}

static void fill_right_environment(
    const CArray& right,
    const CArray& site,
    const CArray& W,
    CArray& out
) {
    const ssize_t B = site.shape(0);
    const ssize_t Q = site.shape(1);
    const ssize_t S = site.shape(2);
    const ssize_t M = W.shape(0);
    const ssize_t N = W.shape(1);
    std::fill(out.mutable_data(), out.mutable_data() + out.size(), cdouble(0.0, 0.0));
#if PYQED_TDVP_HAS_BLAS
    const ssize_t state_rows = B * Q;
    const ssize_t state_pairs = B * S;
    const ssize_t operator_rows = N * Q;
    std::vector<cdouble> environment(static_cast<std::size_t>(S) * S);
    std::vector<cdouble> first(static_cast<std::size_t>(state_rows) * S);
    std::vector<cdouble> right_site(
        static_cast<std::size_t>(operator_rows) * state_pairs
    );
    std::vector<cdouble> wmat(static_cast<std::size_t>(M) * operator_rows);
    std::vector<cdouble> contracted(static_cast<std::size_t>(M) * state_pairs);
    std::vector<cdouble> operator_site(
        static_cast<std::size_t>(B) * M * Q * S
    );
    std::vector<cdouble> bra(static_cast<std::size_t>(B) * Q * S);
    std::vector<cdouble> ket(static_cast<std::size_t>(Q) * S * B);
    std::vector<cdouble> result(static_cast<std::size_t>(B) * B);
    const cdouble alpha(1.0, 0.0);
    const cdouble beta(0.0, 0.0);
    for (ssize_t a = 0; a < B; ++a) {
        for (ssize_t p = 0; p < Q; ++p) {
            for (ssize_t r = 0; r < S; ++r) {
                bra[(a * Q + p) * S + r] =
                    std::conj(site.data()[(a * Q + p) * S + r]);
            }
        }
    }
    for (ssize_t n = 0; n < N; ++n) {
        for (ssize_t j = 0; j < S; ++j) {
            for (ssize_t l = 0; l < S; ++l) {
                environment[j * S + l] = right.data()[(j * N + n) * S + l];
            }
        }
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            static_cast<int>(state_rows), static_cast<int>(S), static_cast<int>(S),
            &alpha, site.data(), static_cast<int>(S),
            environment.data(), static_cast<int>(S),
            &beta, first.data(), static_cast<int>(S)
        );
        for (ssize_t qket = 0; qket < Q; ++qket) {
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t r = 0; r < S; ++r) {
                    right_site[(n * Q + qket) * state_pairs + b * S + r] =
                        first[(b * Q + qket) * S + r];
                }
            }
        }
    }
    for (ssize_t p = 0; p < Q; ++p) {
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t n = 0; n < N; ++n) {
                for (ssize_t qket = 0; qket < Q; ++qket) {
                    wmat[m * operator_rows + n * Q + qket] =
                        W.data()[((m * N + n) * Q + p) * Q + qket];
                }
            }
        }
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(state_pairs),
            static_cast<int>(operator_rows),
            &alpha, wmat.data(), static_cast<int>(operator_rows),
            right_site.data(), static_cast<int>(state_pairs),
            &beta, contracted.data(), static_cast<int>(state_pairs)
        );
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t b = 0; b < B; ++b) {
                for (ssize_t r = 0; r < S; ++r) {
                    operator_site[((b * M + m) * Q + p) * S + r] =
                        contracted[m * state_pairs + b * S + r];
                }
            }
        }
    }
    for (ssize_t m = 0; m < M; ++m) {
        for (ssize_t p = 0; p < Q; ++p) {
            for (ssize_t r = 0; r < S; ++r) {
                for (ssize_t b = 0; b < B; ++b) {
                    ket[(p * S + r) * B + b] =
                        operator_site[((b * M + m) * Q + p) * S + r];
                }
            }
        }
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(B), static_cast<int>(B), static_cast<int>(Q * S),
            &alpha, bra.data(), static_cast<int>(Q * S),
            ket.data(), static_cast<int>(B),
            &beta, result.data(), static_cast<int>(B)
        );
        for (ssize_t a = 0; a < B; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                out.mutable_data()[(a * M + m) * B + b] = result[a * B + b];
            }
        }
    }
#else
    for (ssize_t i = 0; i < B; ++i) {
        for (ssize_t m = 0; m < M; ++m) {
            for (ssize_t k = 0; k < B; ++k) {
                cdouble total = 0.0;
                for (ssize_t p = 0; p < Q; ++p) {
                    for (ssize_t j = 0; j < S; ++j) {
                        for (ssize_t n = 0; n < N; ++n) {
                            for (ssize_t l = 0; l < S; ++l) {
                                for (ssize_t qket = 0; qket < Q; ++qket) {
                                    total += std::conj(site.data()[(i * Q + p) * S + j])
                                        * W.data()[((m * N + n) * Q + p) * Q + qket]
                                        * right.data()[(j * N + n) * S + l]
                                        * site.data()[(k * Q + qket) * S + l];
                                }
                            }
                        }
                    }
                }
                out.mutable_data()[(i * M + m) * B + k] = total;
            }
        }
    }
#endif
}

template <typename Function>
static void parallel_components(std::size_t count, int workers, Function function) {
    const int thread_count = std::max(
        1,
        std::min({workers, 4, static_cast<int>(count)})
    );
    if (thread_count == 1) {
        for (std::size_t item = 0; item < count; ++item) {
            function(item);
        }
        return;
    }
    std::atomic<std::size_t> next(0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(thread_count));
    for (int worker = 0; worker < thread_count; ++worker) {
        threads.emplace_back([&]() {
            while (true) {
                const std::size_t item = next.fetch_add(1, std::memory_order_relaxed);
                if (item >= count) {
                    break;
                }
                try {
                    function(item);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!failure) {
                        failure = std::current_exception();
                    }
                    break;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

static CArray absorb_center_left(const CArray& center, const CArray& site) {
    if (center.ndim() != 2 || site.ndim() != 3 || center.shape(1) != site.shape(0)) {
        throw std::runtime_error("left center and following site are incompatible");
    }
    const ssize_t L = center.shape(0);
    const ssize_t B = center.shape(1);
    const ssize_t Q = site.shape(1);
    const ssize_t R = site.shape(2);
    CArray out({L, Q, R});
#if PYQED_TDVP_HAS_BLAS
    const cdouble alpha(1.0, 0.0);
    const cdouble beta(0.0, 0.0);
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        static_cast<int>(L), static_cast<int>(Q * R), static_cast<int>(B),
        &alpha, center.data(), static_cast<int>(B),
        site.data(), static_cast<int>(Q * R),
        &beta, out.mutable_data(), static_cast<int>(Q * R)
    );
#else
    for (ssize_t l = 0; l < L; ++l) {
        for (ssize_t q = 0; q < Q; ++q) {
            for (ssize_t r = 0; r < R; ++r) {
                cdouble total = 0.0;
                for (ssize_t b = 0; b < B; ++b) {
                    total += center.data()[l * B + b] * site.data()[(b * Q + q) * R + r];
                }
                out.mutable_data()[(l * Q + q) * R + r] = total;
            }
        }
    }
#endif
    return out;
}

static CArray absorb_center_right(const CArray& site, const CArray& center) {
    if (site.ndim() != 3 || center.ndim() != 2 || site.shape(2) != center.shape(0)) {
        throw std::runtime_error("preceding site and right center are incompatible");
    }
    const ssize_t L = site.shape(0);
    const ssize_t Q = site.shape(1);
    const ssize_t B = site.shape(2);
    const ssize_t R = center.shape(1);
    CArray out({L, Q, R});
#if PYQED_TDVP_HAS_BLAS
    const cdouble alpha(1.0, 0.0);
    const cdouble beta(0.0, 0.0);
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        static_cast<int>(L * Q), static_cast<int>(R), static_cast<int>(B),
        &alpha, site.data(), static_cast<int>(B),
        center.data(), static_cast<int>(R),
        &beta, out.mutable_data(), static_cast<int>(R)
    );
#else
    for (ssize_t l = 0; l < L; ++l) {
        for (ssize_t q = 0; q < Q; ++q) {
            for (ssize_t r = 0; r < R; ++r) {
                cdouble total = 0.0;
                for (ssize_t b = 0; b < B; ++b) {
                    total += site.data()[(l * Q + q) * B + b] * center.data()[b * R + r];
                }
                out.mutable_data()[(l * Q + q) * R + r] = total;
            }
        }
    }
#endif
    return out;
}

static py::dict one_site_lanczos_sum_sweep(
    const std::vector<CArray>& factors_in,
    const std::vector<std::vector<CArray>>& operators,
    const std::vector<std::vector<CArray>>& cached_right_environments,
    double dt,
    int krylov_dim,
    double tol,
    std::size_t max_direct_sum_elements,
    int workers,
    bool normalize,
    bool imaginary_time = false
) {
    const std::size_t nsites = factors_in.size();
    const std::size_t components = operators.size();
    if (nsites == 0 || components == 0) {
        throw std::runtime_error("compiled sum-TDVP sweep requires sites and components");
    }
    for (const auto& mpo : operators) {
        if (mpo.size() != nsites) {
            throw std::runtime_error("MPS and MPO lengths differ in compiled sum-TDVP sweep");
        }
    }
    workers = std::max(1, workers);
    std::vector<CArray> factors;
    factors.reserve(nsites);
    for (std::size_t site = 0; site < nsites; ++site) {
        if (factors_in[site].ndim() != 3) {
            throw std::runtime_error("compiled sum-TDVP sweep expects rank-3 MPS factors");
        }
        CArray factor({
            factors_in[site].shape(0),
            factors_in[site].shape(1),
            factors_in[site].shape(2),
        });
        std::copy(
            factors_in[site].data(),
            factors_in[site].data() + factors_in[site].size(),
            factor.mutable_data()
        );
        factors.push_back(std::move(factor));
    }
    CArray identity({1, 1, 1});
    identity.mutable_data()[0] = cdouble(1.0, 0.0);

    auto site_operators = [&](std::size_t site) {
        std::vector<CArray> local;
        local.reserve(components);
        for (const auto& mpo : operators) {
            local.push_back(mpo[site]);
        }
        return local;
    };
    std::vector<std::vector<CArray>> right_environments;
    bool reused_right_environments = false;

    if (nsites == 1) {
        std::vector<CArray> boundaries(components, identity);
        factors[0] = site_lanczos_sum(
            factors[0], boundaries, site_operators(0), boundaries,
            dt, krylov_dim, tol, max_direct_sum_elements, workers,
            imaginary_time
        );
        release_free_numeric_pages();
    } else {
        const double half_dt = 0.5 * dt;
        reused_right_environments =
            cached_right_environments.size() == components;
        if (reused_right_environments) {
            for (const auto& environments : cached_right_environments) {
                if (environments.size() != nsites + 1) {
                    reused_right_environments = false;
                    break;
                }
            }
        }
        if (reused_right_environments) {
            right_environments = cached_right_environments;
        } else {
            right_environments.assign(
                components,
                std::vector<CArray>(nsites + 1)
            );
            for (std::size_t term = 0; term < components; ++term) {
                right_environments[term][nsites] = identity;
            }
            for (std::size_t offset = 0; offset < nsites; ++offset) {
                const std::size_t site = nsites - 1 - offset;
                for (std::size_t term = 0; term < components; ++term) {
                    right_environments[term][site] = allocate_right_environment(
                        right_environments[term][site + 1],
                        factors[site],
                        operators[term][site]
                    );
                }
                parallel_components(components, workers, [&](std::size_t term) {
                    fill_right_environment(
                        right_environments[term][site + 1],
                        factors[site],
                        operators[term][site],
                        right_environments[term][site]
                    );
                });
            }
        }

        std::vector<std::vector<CArray>> left_environments(
            components,
            std::vector<CArray>(nsites)
        );
        for (std::size_t term = 0; term < components; ++term) {
            left_environments[term][0] = identity;
        }
        for (std::size_t site = 0; site + 1 < nsites; ++site) {
            std::vector<CArray> lefts;
            std::vector<CArray> rights;
            lefts.reserve(components);
            rights.reserve(components);
            for (std::size_t term = 0; term < components; ++term) {
                lefts.push_back(left_environments[term][site]);
                rights.push_back(right_environments[term][site + 1]);
            }
            factors[site] = site_lanczos_sum(
                factors[site], lefts, site_operators(site), rights,
                half_dt, krylov_dim, tol, max_direct_sum_elements, workers,
                imaginary_time
            );
            release_free_numeric_pages();
            auto qr = dense_left_qr_site(factors[site]);
            factors[site] = std::move(qr.first);
            CArray center = std::move(qr.second);
            for (std::size_t term = 0; term < components; ++term) {
                left_environments[term][site + 1] = allocate_left_environment(
                    left_environments[term][site], factors[site], operators[term][site]
                );
            }
            parallel_components(components, workers, [&](std::size_t term) {
                fill_left_environment(
                    left_environments[term][site],
                    factors[site],
                    operators[term][site],
                    left_environments[term][site + 1]
                );
            });
            lefts.clear();
            rights.clear();
            for (std::size_t term = 0; term < components; ++term) {
                lefts.push_back(left_environments[term][site + 1]);
                rights.push_back(right_environments[term][site + 1]);
            }
            center = bond_lanczos_sum(
                center, lefts, rights, -half_dt,
                krylov_dim, tol, max_direct_sum_elements, imaginary_time
            );
            factors[site + 1] = absorb_center_left(center, factors[site + 1]);
        }

        std::vector<CArray> lefts;
        std::vector<CArray> boundaries(components, identity);
        lefts.reserve(components);
        for (std::size_t term = 0; term < components; ++term) {
            lefts.push_back(left_environments[term][nsites - 1]);
        }
        factors[nsites - 1] = site_lanczos_sum(
            factors[nsites - 1], lefts, site_operators(nsites - 1), boundaries,
            dt, krylov_dim, tol, max_direct_sum_elements, workers,
            imaginary_time
        );
        release_free_numeric_pages();

        std::vector<CArray> rights(components, identity);
        for (std::size_t offset = 0; offset + 1 < nsites; ++offset) {
            const std::size_t site = nsites - 1 - offset;
            if (site != nsites - 1) {
                lefts.clear();
                for (std::size_t term = 0; term < components; ++term) {
                    lefts.push_back(left_environments[term][site]);
                }
                factors[site] = site_lanczos_sum(
                    factors[site], lefts, site_operators(site), rights,
                    half_dt, krylov_dim, tol, max_direct_sum_elements, workers,
                    imaginary_time
                );
                release_free_numeric_pages();
            }
            auto rq = dense_right_rq_site(factors[site]);
            CArray center = std::move(rq.first);
            factors[site] = std::move(rq.second);
            std::vector<CArray> updated;
            updated.reserve(components);
            for (std::size_t term = 0; term < components; ++term) {
                updated.push_back(allocate_right_environment(
                    rights[term], factors[site], operators[term][site]
                ));
            }
            parallel_components(components, workers, [&](std::size_t term) {
                fill_right_environment(
                    rights[term], factors[site], operators[term][site], updated[term]
                );
            });
            rights = std::move(updated);
            for (std::size_t term = 0; term < components; ++term) {
                right_environments[term][site] = rights[term];
            }
            lefts.clear();
            for (std::size_t term = 0; term < components; ++term) {
                lefts.push_back(left_environments[term][site]);
            }
            center = bond_lanczos_sum(
                center, lefts, rights, -half_dt,
                krylov_dim, tol, max_direct_sum_elements, imaginary_time
            );
            factors[site - 1] = absorb_center_right(factors[site - 1], center);
        }
        factors[0] = site_lanczos_sum(
            factors[0], boundaries, site_operators(0), rights,
            half_dt, krylov_dim, tol, max_direct_sum_elements, workers,
            imaginary_time
        );
        release_free_numeric_pages();
    }

    double norm2 = 0.0;
    for (ssize_t index = 0; index < factors[0].size(); ++index) {
        norm2 += std::norm(factors[0].data()[index]);
    }
    if (normalize && norm2 > 0.0) {
        const double scale = 1.0 / std::sqrt(norm2);
        for (ssize_t index = 0; index < factors[0].size(); ++index) {
            factors[0].mutable_data()[index] *= scale;
        }
    }
    py::list output;
    for (CArray& factor : factors) {
        output.append(std::move(factor));
    }
    py::dict result;
    result["factors"] = std::move(output);
    result["pre_normalization_norm2"] = norm2;
    result["components"] = components;
    result["reused_right_environments"] = reused_right_environments;
    if (nsites > 1) {
        py::list cached;
        for (const auto& term_environments : right_environments) {
            py::list term;
            for (const CArray& environment : term_environments) {
                term.append(environment);
            }
            cached.append(std::move(term));
        }
        result["right_environments"] = std::move(cached);
    } else {
        result["right_environments"] = py::list();
    }
    return result;
}

PYBIND11_MODULE(_cpp_tdvp, m) {
    m.doc() = "C++ dense TDVP local evolution kernels";
    m.attr("HAS_BLAS") = bool(PYQED_TDVP_HAS_BLAS);
    m.def("reset_kernel_stats", &reset_kernel_stats);
    m.def("kernel_stats", &kernel_stats);
    m.def(
        "site_lanczos",
        &site_lanczos,
        "Lanczos evolution for one TDVP site tensor",
        py::arg("theta"),
        py::arg("left"),
        py::arg("W"),
        py::arg("right"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("workers") = 1,
        py::arg("imaginary_time") = false
    );
    m.def(
        "site_lanczos_sum",
        &site_lanczos_sum,
        "Lanczos evolution for a sum of one-site effective MPO operators",
        py::arg("theta"),
        py::arg("lefts"),
        py::arg("operators"),
        py::arg("rights"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("max_direct_sum_elements"),
        py::arg("workers") = 1,
        py::arg("imaginary_time") = false
    );
    m.def(
        "two_site_lanczos",
        &two_site_lanczos,
        "Lanczos evolution for one TDVP two-site tensor",
        py::arg("theta"),
        py::arg("left"),
        py::arg("W_left"),
        py::arg("W_right"),
        py::arg("right"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("workers") = 1
    );
    m.def(
        "two_site_lanczos_sum",
        &two_site_lanczos_sum,
        "Lanczos evolution for a sum of two-site effective MPO operators"
    );
    m.def(
        "bond_lanczos",
        &bond_lanczos,
        "Lanczos evolution for one TDVP bond-center tensor",
        py::arg("center"),
        py::arg("left"),
        py::arg("right"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("imaginary_time") = false
    );
    m.def(
        "bond_lanczos_sum",
        &bond_lanczos_sum,
        "Lanczos evolution for a sum of TDVP bond-center operators",
        py::arg("center"),
        py::arg("lefts"),
        py::arg("rights"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("max_direct_sum_elements"),
        py::arg("imaginary_time") = false
    );
    m.def(
        "one_site_lanczos_sum_sweep",
        &one_site_lanczos_sum_sweep,
        "Complete compiled one-site TDVP sweep for a sum of MPOs",
        py::arg("factors"),
        py::arg("operators"),
        py::arg("right_environments"),
        py::arg("dt"),
        py::arg("krylov_dim"),
        py::arg("tol"),
        py::arg("max_direct_sum_elements"),
        py::arg("workers") = 1,
        py::arg("normalize") = true,
        py::arg("imaginary_time") = false
    );
}
