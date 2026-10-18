#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <complex>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <malloc/malloc.h>
#define PYQED_TDVP_HAS_BLAS 1
#elif defined(PYQED_TDVP_USE_CBLAS)
#include <cblas.h>
#define PYQED_TDVP_HAS_BLAS 1
#else
#define PYQED_TDVP_HAS_BLAS 0
#endif

namespace py = pybind11;
using cdouble = std::complex<double>;

#ifdef __APPLE__
using lapack_complex = __CLPK_doublecomplex;
#elif defined(PYQED_TDVP_USE_CBLAS)
using lapack_complex = cdouble;
extern "C" {
void dstev_(const char*, const int*, double*, double*, double*, const int*, double*, int*);
void zgeqrf_(
    const int*, const int*, lapack_complex*, const int*, lapack_complex*,
    lapack_complex*, const int*, int*
);
void zungqr_(
    const int*, const int*, const int*, lapack_complex*, const int*,
    const lapack_complex*, lapack_complex*, const int*, int*
);
}
#endif
using CArray = py::array_t<
    cdouble,
    py::array::c_style | py::array::forcecast
>;

static std::atomic<long long> reused_action_worker_batches(0);
// A team belongs to one local solve; callbacks finish before its next matvec.
class ActionWorkers {
    std::mutex mutex;
    std::condition_variable ready, finished;
    std::vector<std::thread> threads;
    std::function<void(int)> job;
    std::exception_ptr error;
    std::size_t generation = 0, remaining = 0;
    bool stopping = false;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto& thread : threads) thread.join();
    }
public:
    explicit ActionWorkers(int count, bool enabled_by_default = false) {
        const char* flag = std::getenv("PYQED_TDVP_REUSE_ACTION_WORKERS");
        if (count <= 1 || (flag ? flag[0] != '1' : !enabled_by_default)) return;
        try {
            for (int worker = 0; worker < count; ++worker) threads.emplace_back([&, worker] {
                std::size_t seen = 0;
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    ready.wait(lock, [&] { return stopping || generation != seen; });
                    if (stopping) return;
                    seen = generation;
                    lock.unlock();
                    try { job(worker); }
                    catch (...) {
                        std::lock_guard<std::mutex> failure_lock(mutex);
                        if (!error) error = std::current_exception();
                    }
                    lock.lock();
                    if (--remaining == 0) finished.notify_one();
                }
            });
        } catch (...) { stop(); throw; }
    }
    ~ActionWorkers() { stop(); }
    bool active() const { return !threads.empty(); }
    void run(std::function<void(int)> callback) {
        reused_action_worker_batches.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::mutex> lock(mutex);
        job = std::move(callback);
        error = nullptr;
        remaining = threads.size();
        ++generation;
        ready.notify_all();
        finished.wait(lock, [&] { return remaining == 0; });
        job = {};
        if (error) std::rethrow_exception(error);
    }
};

static void release_free_numeric_pages() {
#ifdef __APPLE__
    malloc_zone_pressure_relief(malloc_default_zone(), 0);
#endif
}

static std::atomic<long long> lanczos_calls(0);
static std::atomic<long long> lanczos_action_ns(0);
static std::atomic<long long> lanczos_projection_ns(0);
static std::atomic<long long> lanczos_subdivisions(0);
static std::atomic<long long> lanczos_extra_intervals(0);
static std::atomic<long long> lanczos_matvecs(0);
static std::atomic<long long> low_memory_site_sum_calls(0);
static std::atomic<long long> shared_pair_channels(0);
static std::atomic<long long> compressed_site_channels(0);
static std::atomic<long long> diagonal_pair_channels(0);
static std::atomic<long long> compressed_boundary_channels(0);
static std::atomic<long long> factorized_prefix_channels(0);
static std::atomic<long long> real_pair_factors(0);
static std::atomic<long long> matrix_unit_site_calls(0);
static std::atomic<long long> split_complex_pair_factors(0);
static std::atomic<long long> interleaved_right_factors(0);
static std::atomic<long long> interleaved_right_cache_hits(0);
static std::atomic<long long> screened_environment_values(0);
static std::atomic<long long> packed_left_batch_channels(0);
static std::atomic<long long> packed_left_bank_rejected(0), packed_left_product_rejected(0);
static std::atomic<long long> diagonal_left_field_channels(0);
static std::atomic<long long> coordinate_prefix_field_channels(0);
static std::atomic<long long> point_prefix_field_channels(0);
static std::atomic<long long> coordinate_prefix_real_products(0);
static std::atomic<long long> compact_pair_right_products(0);
static std::atomic<long long> direct_pair_right_factors(0);
static std::atomic<long long> shared_prefix_input_components(0);
static std::atomic<long long> diagonal_left_field_candidates(0);
static std::atomic<long long> diagonal_left_field_budget_rejected(0);
static std::atomic<long long> real_coordinate_fields(0);
static std::atomic<long long> interleaved_coordinate_fields(0);
static std::atomic<long long> batched_site_field_tasks(0), site_field_batch_products(0);
static std::atomic<long long> fused_site_components(0), fused_site_saved_groups(0);
static std::atomic<long long> fused_site_tiles(0);
static std::atomic<long long> peak_streamed_site_bytes(0);
static std::atomic<long long> fused_site_frame_rejected(0), fused_site_temporary_rejected(0), fused_site_staging_rejected(0);
static std::atomic<long long> split_left_pair_factors(0);
static std::atomic<long long> diagonal_prefix_factors(0);
static std::atomic<long long> spectator_frame_channels(0);
static std::atomic<long long> shared_pair_ket_frames(0), shared_pair_bra_frames(0);
static std::atomic<long long> compact_spectator_channels(0);
static std::atomic<long long> site_spectator_components(0);
static std::atomic<long long> projected_site_components(0), shared_site_projections(0);
static std::atomic<long long> real_site_projections(0);
static std::atomic<long long> shared_site_right_components(0), streamed_site_right_calls(0);
static std::atomic<long long> compact_residual_right_components(0);
static std::atomic<long long> real_streamed_site_components(0);
static std::atomic<long long> peak_streamed_right_scratch_bytes(0);
static std::atomic<long long> peak_projected_site_bytes(0);
static std::atomic<long long> site_spectator_groups(0);
static std::atomic<long long> shared_prefix_outputs(0);
static std::atomic<long long> reused_pair_actions(0);
static std::atomic<long long> split_sparse_left_factors(0), split_sparse_right_factors(0);
static std::atomic<long long> diagonal_left_sparse_channels(0);
static std::atomic<long long> diagonal_left_state_diagonal_channels(0);
static std::atomic<long long> diagonal_left_batched_channels(0);
static std::atomic<long long> point_diagonal_batched_channels(0);
static std::atomic<long long> compact_boundary_diagonal_channels(0);
static std::atomic<long long> residual_left_saved_elements(0);
static std::atomic<long long> compact_site_physical_components(0);
static std::atomic<long long> compact_site_right_components(0);
static std::atomic<long long> real_environment_products(0);
static std::atomic<long long> sparse_spectator_channels(0);
static std::atomic<long long> batched_prefix_channels(0);
static std::atomic<long long> prefix_field_channels(0);
static std::atomic<long long> batched_diagonal_channels(0);
static std::atomic<long long> diagonal_right_states(0);
static std::atomic<long long> diagonal_environment_products(0);
static std::atomic<long long> diagonal_site_aggregations(0);
static std::atomic<long long> compact_site_cores(0);
static std::atomic<long long> compact_pair_cores(0);
static std::atomic<long long> diagonal_unit_factors(0);
static std::atomic<long long> partial_site_unit_calls(0);
static std::atomic<long long> physical_site_unit_calls(0);
static std::atomic<long long> direct_pair_factors(0);
struct ContractionRecord {
    bool pair;
    std::vector<ssize_t> shape;
    ssize_t channels;
    double setup_seconds, solve_seconds;
    std::vector<std::pair<std::string, double>> phases;
    std::vector<std::pair<std::string, double>> worker_times;
    std::vector<std::pair<std::string, double>> preparation_times;
    int preparation_workers;
};
static std::mutex contraction_records_mutex;
static std::vector<ContractionRecord> contraction_records;
static void record_contraction(bool pair, std::vector<ssize_t> shape, ssize_t channels,
    std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point solve,
    std::vector<std::pair<std::string, double>> phases = {},
    std::vector<std::pair<std::string, double>> worker_times = {},
    std::vector<std::pair<std::string, double>> preparation_times = {}, int preparation_workers = 1) {
    const char* enabled = std::getenv("PYQED_TDVP_PROFILE_CONTRACTIONS");
    if (!enabled || enabled[0] != '1') return;
    const auto end = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> guard(contraction_records_mutex);
    if (contraction_records.size() >= 64) return;
    contraction_records.push_back({pair, std::move(shape), channels,
        std::chrono::duration<double>(solve - start).count(),
        std::chrono::duration<double>(end - solve).count(), std::move(phases), std::move(worker_times), std::move(preparation_times), preparation_workers});
}

struct SiteRightGroup {
    ssize_t m;
    ssize_t q;
    std::vector<cdouble> right_eff;
    bool diagonal = false;
    bool compact = false;
};

struct SiteSumComponentView {
    const cdouble* left;
    const cdouble* W;
    const cdouble* right;
    ssize_t M;
    ssize_t N;
    bool diagonal = false;
    const cdouble* spectator_bra = nullptr;
    const cdouble* spectator_ket = nullptr;
    const cdouble* spectator_coefficients = nullptr;
    ssize_t spectator_size = 0;
    cdouble coefficient(ssize_t m, ssize_t n, ssize_t p, ssize_t q, ssize_t physical) const {
        return diagonal ? (p == q ? W[(m * N + n) * physical + p] : cdouble(0.0, 0.0))
                        : W[((m * N + n) * physical + p) * physical + q];
    }
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

    std::vector<double> evals;
    std::vector<double> evecs;
#if PYQED_TDVP_HAS_BLAS
    evals.assign(alpha.begin(), alpha.begin() + actual_dim);
    std::vector<double> off_diagonal(beta.begin(), beta.begin() + actual_dim - 1);
    std::vector<double> columns(static_cast<std::size_t>(actual_dim) * actual_dim);
    std::vector<double> work(std::max(1, 2 * actual_dim - 2));
    char vectors = 'V';
    int info = 0;
    dstev_(&vectors, &actual_dim, evals.data(), off_diagonal.data(), columns.data(),
           &actual_dim, work.data(), &info);
    if (info != 0) throw std::runtime_error("Lanczos tridiagonal eigensolver failed");
    evecs.resize(columns.size());
    for (int row = 0; row < actual_dim; ++row)
        for (int col = 0; col < actual_dim; ++col)
            evecs[row * actual_dim + col] = columns[row + col * actual_dim];
#else
    // Small non-BLAS reference builds only; production TDVP requires BLAS.
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
    jacobi_symmetric(tri, actual_dim, evals, evecs);
#endif
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
    bool imaginary_time = false,
    int subdivision = 0
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
    bool converged = false;
    const char* profile = std::getenv("PYQED_TDVP_PROFILE_CONTRACTIONS");
    const bool timed = profile && profile[0] == '1';

    for (int j = 0; j < mmax; ++j) {
        const auto action_start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        matvec(basis[static_cast<std::size_t>(j)], trial);
        if (timed) lanczos_action_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - action_start).count(), std::memory_order_relaxed);
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
        const auto projection_start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::vector<cdouble> current_coefficients = lanczos_coefficients(
            alpha,
            beta,
            actual_dim,
            norm,
            dt,
            imaginary_time
        );
        if (timed) lanczos_projection_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - projection_start).count(), std::memory_order_relaxed);
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
                converged = true;
                break;
            }
        }
        previous_coefficients = std::move(current_coefficients);
        if (beta_j <= tol || j + 1 == mmax) {
            converged = beta_j <= tol || actual_dim == size;
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

    int pieces = 1;
    if (tol > 0.0 && !converged) {
        if (mmax < 2 || subdivision >= 10) {
            throw std::runtime_error("Lanczos action did not converge after local subdivision; "
                "increase krylov_dim or reduce the time step");
        }
        // Reuse the capped projection to choose and advance the first interval.
        for (int level = 1; level <= 10; ++level) {
            pieces *= 2;
            const double interval = dt / pieces;
            coefficients = lanczos_coefficients(alpha, beta, actual_dim, norm, interval, imaginary_time);
            const auto shorter = lanczos_coefficients(alpha, beta, actual_dim - 1, norm, interval, imaginary_time);
            double delta2 = std::norm(coefficients.back());
            for (int k = 0; k < actual_dim - 1; ++k) {
                delta2 += std::norm(coefficients[k] - shorter[k]);
            }
            if (std::sqrt(delta2) <= tol * std::max(1.0, norm) / pieces) {
                coefficients_ready = true;
                converged = true;
                break;
            }
        }
        if (!converged) {
            throw std::runtime_error("Lanczos action did not converge after local subdivision; "
                "increase krylov_dim or reduce the time step");
        }
    }

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
    if (pieces > 1) {
        lanczos_subdivisions.fetch_add(1, std::memory_order_relaxed);
        lanczos_extra_intervals.fetch_add(pieces - 1, std::memory_order_relaxed);
        // Release the large projection before evolving the remaining intervals.
        basis.clear(); basis.shrink_to_fit();
        trial.clear(); trial.shrink_to_fit();
        q_prev.clear(); q_prev.shrink_to_fit();
        for (int k = 1; k < pieces; ++k) {
            out = lanczos_apply(out, dt / pieces, krylov_dim, tol / pieces,
                                matvec, imaginary_time, subdivision + 1);
        }
    }
    return out;
}

static void reset_kernel_stats() {
    lanczos_calls.store(0, std::memory_order_relaxed);
    lanczos_action_ns.store(0, std::memory_order_relaxed);
    lanczos_projection_ns.store(0, std::memory_order_relaxed);
    lanczos_subdivisions.store(0, std::memory_order_relaxed);
    lanczos_extra_intervals.store(0, std::memory_order_relaxed);
    lanczos_matvecs.store(0, std::memory_order_relaxed);
    low_memory_site_sum_calls.store(0, std::memory_order_relaxed);
    shared_pair_channels.store(0, std::memory_order_relaxed);
    compressed_site_channels.store(0, std::memory_order_relaxed);
    diagonal_pair_channels.store(0, std::memory_order_relaxed);
    compressed_boundary_channels.store(0, std::memory_order_relaxed);
    factorized_prefix_channels.store(0, std::memory_order_relaxed);
    real_pair_factors.store(0, std::memory_order_relaxed);
    matrix_unit_site_calls.store(0, std::memory_order_relaxed);
    split_complex_pair_factors.store(0, std::memory_order_relaxed);
    interleaved_right_factors.store(0, std::memory_order_relaxed);
    interleaved_right_cache_hits.store(0, std::memory_order_relaxed);
    screened_environment_values.store(0, std::memory_order_relaxed);
    packed_left_batch_channels.store(0, std::memory_order_relaxed);
    packed_left_bank_rejected.store(0, std::memory_order_relaxed);
    packed_left_product_rejected.store(0, std::memory_order_relaxed);
    diagonal_left_field_channels.store(0, std::memory_order_relaxed);
    coordinate_prefix_field_channels.store(0, std::memory_order_relaxed);
    point_prefix_field_channels.store(0, std::memory_order_relaxed);
    coordinate_prefix_real_products.store(0, std::memory_order_relaxed);
    compact_pair_right_products.store(0, std::memory_order_relaxed);
    direct_pair_right_factors.store(0, std::memory_order_relaxed);
    shared_prefix_input_components.store(0, std::memory_order_relaxed);
    diagonal_left_field_candidates.store(0, std::memory_order_relaxed);
    diagonal_left_field_budget_rejected.store(0, std::memory_order_relaxed);
    real_coordinate_fields.store(0, std::memory_order_relaxed);
    interleaved_coordinate_fields.store(0, std::memory_order_relaxed);
    batched_site_field_tasks.store(0, std::memory_order_relaxed);
    site_field_batch_products.store(0, std::memory_order_relaxed);
    fused_site_components.store(0, std::memory_order_relaxed);
    fused_site_saved_groups.store(0, std::memory_order_relaxed);
    fused_site_tiles.store(0, std::memory_order_relaxed);
    fused_site_frame_rejected.store(0, std::memory_order_relaxed);
    fused_site_temporary_rejected.store(0, std::memory_order_relaxed);
    fused_site_staging_rejected.store(0, std::memory_order_relaxed);
    peak_streamed_site_bytes.store(0, std::memory_order_relaxed);
    split_left_pair_factors.store(0, std::memory_order_relaxed);
    diagonal_prefix_factors.store(0, std::memory_order_relaxed);
    spectator_frame_channels.store(0, std::memory_order_relaxed);
    reused_action_worker_batches.store(0, std::memory_order_relaxed);
    shared_pair_ket_frames.store(0, std::memory_order_relaxed);
    shared_pair_bra_frames.store(0, std::memory_order_relaxed);
    compact_spectator_channels.store(0, std::memory_order_relaxed);
    site_spectator_components.store(0, std::memory_order_relaxed);
    projected_site_components.store(0, std::memory_order_relaxed);
    shared_site_projections.store(0, std::memory_order_relaxed);
    real_site_projections.store(0, std::memory_order_relaxed);
    shared_site_right_components.store(0, std::memory_order_relaxed);
    streamed_site_right_calls.store(0, std::memory_order_relaxed);
    compact_residual_right_components.store(0, std::memory_order_relaxed);
    real_streamed_site_components.store(0, std::memory_order_relaxed);
    peak_streamed_right_scratch_bytes.store(0, std::memory_order_relaxed);
    peak_projected_site_bytes.store(0, std::memory_order_relaxed);
    site_spectator_groups.store(0, std::memory_order_relaxed);
    shared_prefix_outputs.store(0, std::memory_order_relaxed);
    reused_pair_actions.store(0, std::memory_order_relaxed);
    split_sparse_left_factors.store(0, std::memory_order_relaxed);
    split_sparse_right_factors.store(0, std::memory_order_relaxed);
    diagonal_left_sparse_channels.store(0, std::memory_order_relaxed);
    diagonal_left_state_diagonal_channels.store(0, std::memory_order_relaxed);
    diagonal_left_batched_channels.store(0, std::memory_order_relaxed);
    point_diagonal_batched_channels.store(0, std::memory_order_relaxed);
    compact_boundary_diagonal_channels.store(0, std::memory_order_relaxed);
    residual_left_saved_elements.store(0, std::memory_order_relaxed);
    compact_site_physical_components.store(0, std::memory_order_relaxed);
    compact_site_right_components.store(0, std::memory_order_relaxed);
    real_environment_products.store(0, std::memory_order_relaxed);
    sparse_spectator_channels.store(0, std::memory_order_relaxed);
    batched_prefix_channels.store(0, std::memory_order_relaxed);
    prefix_field_channels.store(0, std::memory_order_relaxed);
    batched_diagonal_channels.store(0, std::memory_order_relaxed);
    diagonal_right_states.store(0, std::memory_order_relaxed);
    diagonal_environment_products.store(0, std::memory_order_relaxed);
    diagonal_site_aggregations.store(0, std::memory_order_relaxed);
    compact_site_cores.store(0, std::memory_order_relaxed);
    compact_pair_cores.store(0, std::memory_order_relaxed);
    diagonal_unit_factors.store(0, std::memory_order_relaxed);
    partial_site_unit_calls.store(0, std::memory_order_relaxed);
    physical_site_unit_calls.store(0, std::memory_order_relaxed);
    direct_pair_factors.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(contraction_records_mutex);
    contraction_records.clear();
}

static py::dict kernel_stats() {
    py::dict result;
    py::list records;
    {
        std::lock_guard<std::mutex> guard(contraction_records_mutex);
        for (const auto& record : contraction_records) {
            py::dict item;
            item["pair"] = record.pair;
            item["shape"] = record.shape;
            item["channels"] = record.channels;
            item["setup_seconds"] = record.setup_seconds;
            item["solve_seconds"] = record.solve_seconds;
            py::dict phases;
            for (const auto& phase : record.phases) phases[py::str(phase.first)] = phase.second;
            item["phase_seconds"] = phases;
            py::dict worker_times;
            for (const auto& phase : record.worker_times) worker_times[py::str(phase.first)] = phase.second;
            item["worker_seconds"] = worker_times;
            py::dict preparation;
            for (const auto& entry : record.preparation_times) preparation[py::str(entry.first)] = entry.second;
            item["preparation_seconds"] = preparation;
            item["preparation_workers"] = record.preparation_workers;
            records.append(item);
        }
    }
    result["contractions"] = records;
    result["shared_pair_channels"] = py::int_(shared_pair_channels.load(std::memory_order_relaxed));
    result["compressed_site_channels"] = py::int_(compressed_site_channels.load(std::memory_order_relaxed));
    result["diagonal_pair_channels"] = py::int_(diagonal_pair_channels.load(std::memory_order_relaxed));
    result["compressed_boundary_channels"] = py::int_(compressed_boundary_channels.load(std::memory_order_relaxed));
    result["factorized_prefix_channels"] = py::int_(factorized_prefix_channels.load(std::memory_order_relaxed));
    result["real_pair_factors"] = py::int_(real_pair_factors.load(std::memory_order_relaxed));
    result["matrix_unit_site_calls"] = py::int_(matrix_unit_site_calls.load(std::memory_order_relaxed));
    result["split_complex_pair_factors"] = py::int_(split_complex_pair_factors.load(std::memory_order_relaxed));
    result["interleaved_right_factors"] = py::int_(interleaved_right_factors.load(std::memory_order_relaxed));
    result["interleaved_right_cache_hits"] = py::int_(interleaved_right_cache_hits.load(std::memory_order_relaxed));
    result["screened_environment_values"] = py::int_(screened_environment_values.load(std::memory_order_relaxed));
    result["packed_left_batch_channels"] = py::int_(packed_left_batch_channels.load(std::memory_order_relaxed));
    result["packed_left_bank_rejected"] = py::int_(packed_left_bank_rejected.load(std::memory_order_relaxed));
    result["packed_left_product_rejected"] = py::int_(packed_left_product_rejected.load(std::memory_order_relaxed));
    result["diagonal_left_field_channels"] = py::int_(diagonal_left_field_channels.load(std::memory_order_relaxed));
    result["coordinate_prefix_field_channels"] = py::int_(coordinate_prefix_field_channels.load(std::memory_order_relaxed));
    result["point_prefix_field_channels"] = py::int_(point_prefix_field_channels.load(std::memory_order_relaxed));
    result["coordinate_prefix_real_products"] = py::int_(coordinate_prefix_real_products.load(std::memory_order_relaxed));
    result["compact_pair_right_products"] = py::int_(compact_pair_right_products.load(std::memory_order_relaxed));
    result["direct_pair_right_factors"] = py::int_(direct_pair_right_factors.load(std::memory_order_relaxed));
    result["shared_prefix_input_components"] = py::int_(shared_prefix_input_components.load(std::memory_order_relaxed));
    result["diagonal_left_field_candidates"] = py::int_(diagonal_left_field_candidates.load(std::memory_order_relaxed));
    result["diagonal_left_field_budget_rejected"] = py::int_(diagonal_left_field_budget_rejected.load(std::memory_order_relaxed));
    result["real_coordinate_fields"] = py::int_(real_coordinate_fields.load(std::memory_order_relaxed));
    result["interleaved_coordinate_fields"] = py::int_(interleaved_coordinate_fields.load(std::memory_order_relaxed));
    result["batched_site_field_tasks"] = py::int_(batched_site_field_tasks.load(std::memory_order_relaxed));
    result["site_field_batch_products"] = py::int_(site_field_batch_products.load(std::memory_order_relaxed));
    result["fused_site_components"] = py::int_(fused_site_components.load(std::memory_order_relaxed));
    result["fused_site_saved_groups"] = py::int_(fused_site_saved_groups.load(std::memory_order_relaxed));
    result["fused_site_tiles"] = py::int_(fused_site_tiles.load(std::memory_order_relaxed));
    result["fused_site_frame_rejected"] = py::int_(fused_site_frame_rejected.load(std::memory_order_relaxed));
    result["fused_site_temporary_rejected"] = py::int_(fused_site_temporary_rejected.load(std::memory_order_relaxed));
    result["fused_site_staging_rejected"] = py::int_(fused_site_staging_rejected.load(std::memory_order_relaxed));
    result["peak_streamed_site_bytes"] = py::int_(peak_streamed_site_bytes.load(std::memory_order_relaxed));
    result["split_left_pair_factors"] = py::int_(split_left_pair_factors.load(std::memory_order_relaxed));
    result["diagonal_prefix_factors"] = py::int_(diagonal_prefix_factors.load(std::memory_order_relaxed));
    result["spectator_frame_channels"] = py::int_(spectator_frame_channels.load(std::memory_order_relaxed));
    result["reused_action_worker_batches"] = py::int_(reused_action_worker_batches.load(std::memory_order_relaxed));
    result["shared_pair_ket_frames"] = py::int_(shared_pair_ket_frames.load(std::memory_order_relaxed));
    result["shared_pair_bra_frames"] = py::int_(shared_pair_bra_frames.load(std::memory_order_relaxed));
    result["compact_spectator_channels"] = py::int_(compact_spectator_channels.load(std::memory_order_relaxed));
    result["site_spectator_components"] = py::int_(site_spectator_components.load(std::memory_order_relaxed));
    result["projected_site_components"] = py::int_(projected_site_components.load(std::memory_order_relaxed));
    result["shared_site_projections"] = py::int_(shared_site_projections.load(std::memory_order_relaxed));
    result["real_site_projections"] = py::int_(real_site_projections.load(std::memory_order_relaxed));
    result["shared_site_right_components"] = py::int_(shared_site_right_components.load(std::memory_order_relaxed));
    result["streamed_site_right_calls"] = py::int_(streamed_site_right_calls.load(std::memory_order_relaxed));
    result["compact_residual_right_components"] = py::int_(compact_residual_right_components.load(std::memory_order_relaxed));
    result["real_streamed_site_components"] = py::int_(real_streamed_site_components.load(std::memory_order_relaxed));
    result["peak_streamed_right_scratch_bytes"] = py::int_(peak_streamed_right_scratch_bytes.load(std::memory_order_relaxed));
    result["peak_projected_site_bytes"] = py::int_(peak_projected_site_bytes.load(std::memory_order_relaxed));
    result["site_spectator_groups"] = py::int_(site_spectator_groups.load(std::memory_order_relaxed));
    result["shared_prefix_outputs"] = py::int_(shared_prefix_outputs.load(std::memory_order_relaxed));
    result["reused_pair_actions"] = py::int_(reused_pair_actions.load(std::memory_order_relaxed));
    result["split_sparse_left_factors"] = py::int_(split_sparse_left_factors.load(std::memory_order_relaxed));
    result["split_sparse_right_factors"] = py::int_(split_sparse_right_factors.load(std::memory_order_relaxed));
    result["diagonal_left_sparse_channels"] = py::int_(diagonal_left_sparse_channels.load(std::memory_order_relaxed));
    result["diagonal_left_state_diagonal_channels"] = py::int_(diagonal_left_state_diagonal_channels.load(std::memory_order_relaxed));
    result["diagonal_left_batched_channels"] = py::int_(diagonal_left_batched_channels.load(std::memory_order_relaxed));
    result["point_diagonal_batched_channels"] = py::int_(point_diagonal_batched_channels.load(std::memory_order_relaxed));
    result["compact_boundary_diagonal_channels"] = py::int_(compact_boundary_diagonal_channels.load(std::memory_order_relaxed));
    result["residual_left_saved_elements"] = py::int_(residual_left_saved_elements.load(std::memory_order_relaxed));
    result["compact_site_physical_components"] = py::int_(compact_site_physical_components.load(std::memory_order_relaxed));
    result["compact_site_right_components"] = py::int_(compact_site_right_components.load(std::memory_order_relaxed));
    result["real_environment_products"] = py::int_(real_environment_products.load(std::memory_order_relaxed));
    result["sparse_spectator_channels"] = py::int_(sparse_spectator_channels.load(std::memory_order_relaxed));
    result["batched_prefix_channels"] = py::int_(batched_prefix_channels.load(std::memory_order_relaxed));
    result["prefix_field_channels"] = py::int_(prefix_field_channels.load(std::memory_order_relaxed));
    result["batched_diagonal_channels"] = py::int_(batched_diagonal_channels.load(std::memory_order_relaxed));
    result["diagonal_right_states"] = py::int_(diagonal_right_states.load(std::memory_order_relaxed));
    result["diagonal_site_aggregations"] = py::int_(diagonal_site_aggregations.load(std::memory_order_relaxed));
    result["compact_site_cores"] = py::int_(compact_site_cores.load(std::memory_order_relaxed));
    result["compact_pair_cores"] = py::int_(compact_pair_cores.load(std::memory_order_relaxed));
    result["diagonal_unit_factors"] = py::int_(diagonal_unit_factors.load(std::memory_order_relaxed));
    result["partial_site_unit_calls"] = py::int_(partial_site_unit_calls.load(std::memory_order_relaxed));
    result["physical_site_unit_calls"] = py::int_(physical_site_unit_calls.load(std::memory_order_relaxed));
    result["direct_pair_factors"] = py::int_(direct_pair_factors.load(std::memory_order_relaxed));
    result["diagonal_environment_products"] = py::int_(diagonal_environment_products.load(std::memory_order_relaxed));
    result["lanczos_calls"] = py::int_(
        lanczos_calls.load(std::memory_order_relaxed)
    );
    result["lanczos_action_seconds"] = py::float_(lanczos_action_ns.load(std::memory_order_relaxed) * 1e-9);
    result["lanczos_projection_seconds"] = py::float_(lanczos_projection_ns.load(std::memory_order_relaxed) * 1e-9);
    result["lanczos_subdivisions"] = py::int_(lanczos_subdivisions.load(std::memory_order_relaxed));
    result["lanczos_extra_intervals"] = py::int_(lanczos_extra_intervals.load(std::memory_order_relaxed));
    result["lanczos_matvecs"] = py::int_(
        lanczos_matvecs.load(std::memory_order_relaxed)
    );
    result["low_memory_site_sum_calls"] = py::int_(
        low_memory_site_sum_calls.load(std::memory_order_relaxed)
    );
    return result;
}

#if PYQED_TDVP_HAS_BLAS
static std::vector<cdouble> diagonal_environment(const std::vector<cdouble>& environment,
                                               ssize_t inner, ssize_t rows, ssize_t columns) {
    if (rows != columns || rows < 2 || std::getenv("PYQED_TDVP_EXPANDED_ENVIRONMENT_PRODUCT")) return {};
    for (ssize_t n = 0; n < inner; ++n)
        for (ssize_t r = 0; r < rows; ++r)
            for (ssize_t s = 0; s < columns; ++s)
                if (r != s && environment[(n * rows + r) * columns + s] != cdouble(0.0, 0.0)) return {};
    std::vector<cdouble> result(inner * rows);
    for (ssize_t n = 0; n < inner; ++n)
        for (ssize_t r = 0; r < rows; ++r)
            result[n * rows + r] = environment[(n * rows + r) * columns + r];
    return result;
}

static std::shared_ptr<const std::vector<double>> interleaved_right_block(
        const std::vector<double>& parts, ssize_t rows, ssize_t columns, bool reusable) {
    struct Entry {
        ssize_t rows, columns;
        std::vector<double> source;
        std::shared_ptr<const std::vector<double>> block;
    };
    thread_local std::unordered_multimap<std::size_t, Entry> cache;
    thread_local std::size_t cache_bytes = 0;
    std::size_t key = 0;
    if (reusable) {
        key = std::hash<std::string_view>{}(std::string_view(
            reinterpret_cast<const char*>(parts.data()), parts.size() * sizeof(double)));
        const auto candidates = cache.equal_range(key);
        for (auto it = candidates.first; it != candidates.second; ++it)
            if (it->second.rows == rows && it->second.columns == columns && it->second.source == parts) {
                interleaved_right_cache_hits.fetch_add(1, std::memory_order_relaxed);
                return it->second.block;
            }
    }
    const std::size_t size = static_cast<std::size_t>(rows) * columns;
    auto block = std::make_shared<std::vector<double>>(4 * size);
    for (ssize_t column = 0; column < columns; ++column)
        for (ssize_t row = 0; row < rows; ++row) {
            const auto i = row * columns + column;
            const double real = parts[i], imag = parts[size + i];
            (*block)[(2 * column) * (2 * rows) + 2 * row] = real;
            (*block)[(2 * column) * (2 * rows) + 2 * row + 1] = imag;
            (*block)[(2 * column + 1) * (2 * rows) + 2 * row] = -imag;
            (*block)[(2 * column + 1) * (2 * rows) + 2 * row + 1] = real;
        }
    const std::size_t charge = (parts.size() + block->size()) * sizeof(double);
    if (reusable && charge <= (64ULL << 20)) {
        if (cache_bytes + charge > (64ULL << 20)) { cache.clear(); cache_bytes = 0; }
        cache.emplace(key, Entry{rows, columns, parts, block});
        cache_bytes += charge;
    }
    return block;
}

static void environment_product(const cdouble* coefficients, const cdouble* environment, cdouble* out,
                                ssize_t rows, ssize_t inner, ssize_t columns,
                                const std::vector<cdouble>& diagonal) {
    const cdouble one(1.0, 0.0), zero(0.0, 0.0);
    auto multiply = [&](const cdouble* right, cdouble* target, ssize_t width) {
        const char* real_mode = std::getenv("PYQED_TDVP_REAL_ENVIRONMENT_PRODUCTS");
        if (real_mode && std::string(real_mode) == "1"
            && static_cast<long double>(rows) * inner * sizeof(double) <= (16ULL << 20)) {
            bool real = true, imaginary = true;
            for (ssize_t i = 0; i < rows * inner && (real || imaginary); ++i) {
                real = real && coefficients[i].imag() == 0.0;
                imaginary = imaginary && coefficients[i].real() == 0.0;
            }
            if (real || imaginary) {
                std::vector<double> values(rows * inner);
                for (ssize_t i = 0; i < rows * inner; ++i)
                    values[i] = real ? coefficients[i].real() : coefficients[i].imag();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(rows), static_cast<int>(2 * width), static_cast<int>(inner),
                    1.0, values.data(), static_cast<int>(inner),
                    reinterpret_cast<const double*>(right), static_cast<int>(2 * width),
                    0.0, reinterpret_cast<double*>(target), static_cast<int>(2 * width));
                if (!real) for (ssize_t i = 0; i < rows * width; ++i)
                    target[i] = cdouble(-target[i].imag(), target[i].real());
                real_environment_products.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(rows), static_cast<int>(width), static_cast<int>(inner),
            &one, coefficients, static_cast<int>(inner), right, static_cast<int>(width),
            &zero, target, static_cast<int>(width));
    };
    if (diagonal.empty()) {
        multiply(environment, out, columns);
        return;
    }
    const ssize_t state = diagonal.size() / inner;
    std::vector<cdouble> product(rows * state);
    multiply(diagonal.data(), product.data(), state);
    std::fill_n(out, rows * columns, zero);
    for (ssize_t row = 0; row < rows; ++row)
        for (ssize_t s = 0; s < state; ++s) out[row * columns + s * state + s] = product[row * state + s];
    diagonal_environment_products.fetch_add(1, std::memory_order_relaxed);
}

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
    const auto contraction_start = std::chrono::steady_clock::now();
    const char* site_profile = std::getenv("PYQED_TDVP_PROFILE_CONTRACTIONS");
    const bool profile_site = site_profile && site_profile[0] == '1';
    double coefficient_seconds = 0, product_seconds = 0, storage_seconds = 0;
    int preparation_workers = 1;
    struct FusedSiteField {
        bool valid = false, active = false, diagonal = false, general = false;
        std::vector<cdouble> values;
    };
    std::vector<FusedSiteField> fused_fields(sum_components ? sum_components->size() : 0);
    std::vector<char> fuse_selected(fused_fields.size(),false);
    std::vector<ssize_t> fused_physical_tiles(fused_fields.size(),0);
    const char* fused_site_flag = std::getenv("PYQED_TDVP_FUSED_SITE_FIELDS");
    const char* stream_site_flag = std::getenv("PYQED_TDVP_STREAM_SITE_FIELDS");
    const bool requested_stream_site = !stream_site_flag || stream_site_flag[0] != '0';
    const char* fused_tile_flag = std::getenv("PYQED_TDVP_FUSED_SITE_TILE");
    const long requested_fused_tile = fused_tile_flag ? std::strtol(fused_tile_flag,nullptr,10) : 0;
    if (sum_components && R == S && (!fused_site_flag || fused_site_flag[0] != '0')) {
        std::vector<std::size_t> owners;
        long double field_elements = 0, staged_elements = 0;
        ssize_t eligible = 0, temporary_rejected = 0, staging_rejected = 0;
        for (std::size_t i = 0; i < sum_components->size(); ++i) {
            const auto& component = (*sum_components)[i];
            const ssize_t K = component.spectator_size;
            if (!K || K >= A * B) continue;
            ++eligible;
            const bool common = std::any_of(owners.begin(),owners.end(),[&](std::size_t j) {
                const auto& previous = (*sum_components)[j];
                return previous.spectator_size == K
                    && std::equal(component.spectator_bra,component.spectator_bra + K * A,previous.spectator_bra)
                    && std::equal(component.spectator_ket,component.spectator_ket + K * B,previous.spectator_ket);
            });
            if (!common) {
                owners.push_back(i);
                field_elements += static_cast<long double>(K) * P * (Q * R + R * S + R * Q * S);
            }
            const ssize_t physical = component.diagonal ? P : P * Q;
            const long double output = static_cast<long double>(K) * physical * R * S;
            const long double fixed = 1.5L * K * component.M
                + static_cast<long double>(component.N) * (R * S + R) + output;
            const long double per_point = static_cast<long double>(component.M + 2 * K) * component.N
                + static_cast<long double>(K) * R * S;
            const long double capacity = (32ULL << 20) / sizeof(cdouble);
            if (component.M <= 0 || component.N <= 0 || fixed + per_point > capacity)
                ++temporary_rejected;
            else if (!requested_stream_site && (staged_elements + output) * sizeof(cdouble) > (64ULL << 20))
                ++staging_rejected;
            else {
                fuse_selected[i] = true;
                fused_physical_tiles[i] = std::min<ssize_t>(physical,(capacity - fixed) / per_point);
                if (requested_fused_tile > 0)
                    fused_physical_tiles[i] = std::min<ssize_t>(fused_physical_tiles[i],requested_fused_tile);
                staged_elements += output;
            }
        }
        // This bound includes every eligible frame, even those using ordinary groups.
        if (field_elements * sizeof(cdouble) > (128ULL << 20)) {
            std::fill(fuse_selected.begin(),fuse_selected.end(),false);
            fused_site_frame_rejected.fetch_add(eligible,std::memory_order_relaxed);
        } else {
            fused_site_temporary_rejected.fetch_add(temporary_rejected,std::memory_order_relaxed);
            fused_site_staging_rejected.fetch_add(staging_rejected,std::memory_order_relaxed);
        }
    }
    std::vector<std::vector<SiteRightGroup>> groups(static_cast<std::size_t>(P));
    std::vector<int> group_index(sum_components ? 0 : static_cast<std::size_t>(M) * P * Q, -1);
    const bool stream_site_fields = requested_stream_site
        && std::any_of(fuse_selected.begin(),fuse_selected.end(),[](char selected) { return selected; });
    struct SpectatorAction {
        const SiteSumComponentView* component;
        std::vector<cdouble> field, input, output, physical_field, general_field;
        std::vector<double> real_field, real_physical, real_general;
        std::vector<std::pair<ssize_t, ssize_t>> transitions;
        std::vector<ssize_t> physical_points;
        std::vector<ssize_t> general_points;
        ssize_t field_owner = -1;
    };
    std::vector<SpectatorAction> spectator_actions;
    std::size_t spectator_elements = 0;
    const char* accumulate_site_fields = std::getenv("PYQED_TDVP_ACCUMULATE_SITE_FIELDS");
    const bool direct_fields = !accumulate_site_fields || std::string(accumulate_site_fields) != "0";
    const char* index_site_flag = std::getenv("PYQED_TDVP_INDEX_SITE_FIELDS");
    const bool indexed_site_fields = index_site_flag && index_site_flag[0] == '1';
    double streamed_assembly_seconds = 0;
    auto assemble_component = [&](std::size_t component_index, ssize_t offset) {
        const auto& component = (*sum_components)[component_index];
        const ssize_t K = component.spectator_size;
        const std::size_t diagonal_elements = static_cast<std::size_t>(K) * P * Q * R;
        const std::size_t physical_elements = static_cast<std::size_t>(K) * P * R * S;
        bool general = fused_fields[component_index].general;
        for (ssize_t p = 0; p < P && !general; ++p)
            for (const auto& group : groups[p])
                if (!group.diagonal && group.q != p && group.m >= offset && group.m < offset + component.M) {
                    general = true; break;
                }
        const std::size_t general_elements = general ? static_cast<std::size_t>(K) * P * R * Q * S : 0;
        const std::size_t elements = diagonal_elements + physical_elements + general_elements;
        auto shared = std::find_if(spectator_actions.begin(), spectator_actions.end(), [&](const SpectatorAction& action) {
            return K && action.component->spectator_size == K
                && std::equal(component.spectator_bra, component.spectator_bra + K * A, action.component->spectator_bra)
                && std::equal(component.spectator_ket, component.spectator_ket + K * B, action.component->spectator_ket);
        });
        const bool common = shared != spectator_actions.end();
        const std::size_t extra = common ? (shared->general_field.empty() ? general_elements : 0) : elements;
        if (!K || K >= A * B || spectator_elements + extra > (128ULL << 20) / sizeof(cdouble)) {
            for (auto& physical : groups)
                for (auto& group : physical)
                    if (group.compact && group.m >= offset && group.m < offset + component.M) {
                        std::vector<cdouble> expanded(R * S, 0.0);
                        for (ssize_t r = 0; r < R; ++r) expanded[r * S + r] = group.right_eff[r];
                        group.right_eff.swap(expanded); group.compact = false;
                    }
            return;
        }
        const bool accumulate = common && direct_fields;
        SpectatorAction action{&component, std::vector<cdouble>(accumulate ? 0 : diagonal_elements, 0.0),
            std::vector<cdouble>(accumulate ? 0 : K * Q * S), std::vector<cdouble>(accumulate ? 0 : K * P * R),
            std::vector<cdouble>(accumulate ? 0 : physical_elements, 0.0),
            std::vector<cdouble>(accumulate ? 0 : general_elements, 0.0)};
        auto& fields = accumulate ? *shared : action;
        if (accumulate && general_elements && fields.general_field.empty())
            fields.general_field.resize(general_elements, 0.0);
        auto store_values = [&](const cdouble* values, ssize_t count, cdouble* target) {
            if (accumulate) for (ssize_t i = 0; i < count; ++i) target[i] += values[i];
            else std::copy_n(values, count, target);
        };
        bool active = false;
        bool batched = false;
        auto& fused = fused_fields[component_index];
        if (fused.valid) {
            const ssize_t physical = component.diagonal ? P : P * Q;
            const ssize_t columns = fused.diagonal ? R : R * S;
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t pq = 0; pq < physical; ++pq) {
                    const ssize_t p = component.diagonal ? pq : pq / Q;
                    const ssize_t q = component.diagonal ? pq : pq % Q;
                    const auto* source = fused.values.data() + (k * physical + pq) * columns;
                    if (fused.diagonal) store_values(source,R,
                        fields.field.data() + ((k * P + p) * Q + q) * R);
                    else if (p == q) store_values(source,R * S,
                        fields.physical_field.data() + (k * P + p) * R * S);
                    else if (general) for (ssize_t r = 0; r < R; ++r)
                        store_values(source + r * S,S,
                            fields.general_field.data() + ((k * P + p) * R + r) * Q * S + q * S);
                }
            active = fused.active;
            batched = true;
            std::vector<cdouble>().swap(fused.values);
        }
        const char* batch_field_flag = std::getenv("PYQED_TDVP_BATCH_SITE_FIELDS");
        if (!batched && batch_field_flag && batch_field_flag[0] == '1') {
            struct FieldTask {
                ssize_t p, q;
                std::vector<const SiteRightGroup*> selected;
            };
            std::map<std::vector<ssize_t>,std::vector<FieldTask>> batches;
            for (ssize_t p = 0; p < P; ++p) {
                std::vector<std::vector<const SiteRightGroup*>> selected(2 * Q);
                for (const auto& group : groups[p])
                    if (group.m >= offset && group.m < offset + component.M)
                        selected[(group.diagonal ? 0 : Q) + group.q].push_back(&group);
                for (ssize_t i = 0; i < 2 * Q; ++i) {
                    if (selected[i].empty()) continue;
                    std::vector<ssize_t> key{ i < Q ? 1 : 0 };
                    for (const auto* group : selected[i]) key.push_back(group->m);
                    batches[std::move(key)].push_back({p,i % Q,std::move(selected[i])});
                }
            }
            const long double budget = (32ULL << 20) / sizeof(cdouble);
            batched = std::all_of(batches.begin(),batches.end(),[&](const auto& batch) {
                const ssize_t count = batch.first.size() - 1;
                const ssize_t columns = batch.first.front() ? R : R * S;
                return columns <= INT_MAX && 1.5L * K * count
                    + static_cast<long double>(count + K) * columns <= budget;
            });
            if (batched) for (const auto& batch : batches) {
                const auto& tasks = batch.second;
                const bool diagonal = batch.first.front();
                const ssize_t count = batch.first.size() - 1;
                const ssize_t columns = diagonal ? R : R * S;
                const ssize_t tile = std::min<ssize_t>(64,std::min<ssize_t>(INT_MAX / columns,
                    (budget - 1.5L * K * count) / (static_cast<long double>(count + K) * columns)));
                std::vector<cdouble> coefficients(K * count);
                for (ssize_t k = 0; k < K; ++k)
                    for (ssize_t g = 0; g < count; ++g)
                        coefficients[k * count + g] = component.spectator_coefficients[
                            (tasks.front().selected[g]->m - offset) * K + k];
                for (ssize_t begin = 0; begin < static_cast<ssize_t>(tasks.size()); begin += tile) {
                    const ssize_t size = std::min<ssize_t>(tile,tasks.size() - begin);
                    const ssize_t width = size * columns;
                    std::vector<cdouble> right(count * width), values(K * width);
                    for (ssize_t g = 0; g < count; ++g)
                        for (ssize_t t = 0; t < size; ++t) {
                            const auto& group = *tasks[begin + t].selected[g];
                            auto* target = right.data() + g * width + t * columns;
                            if (diagonal) for (ssize_t r = 0; r < R; ++r)
                                target[r] = group.right_eff[group.compact ? r : r * S + r];
                            else std::copy_n(group.right_eff.data(),columns,target);
                        }
                    const std::vector<cdouble> empty;
                    environment_product(coefficients.data(),right.data(),values.data(),K,count,width,empty);
                    site_field_batch_products.fetch_add(1,std::memory_order_relaxed);
                    batched_site_field_tasks.fetch_add(size,std::memory_order_relaxed);
                    for (ssize_t k = 0; k < K; ++k)
                        for (ssize_t t = 0; t < size; ++t) {
                            const auto& task = tasks[begin + t];
                            const auto* source = values.data() + k * width + t * columns;
                            if (diagonal) store_values(source,R,fields.field.data() +
                                ((k * P + task.p) * Q + task.q) * R);
                            else if (task.p == task.q) store_values(source,R * S,
                                fields.physical_field.data() + (k * P + task.p) * R * S);
                            else for (ssize_t r = 0; r < R; ++r)
                                store_values(source + r * S,S,fields.general_field.data() +
                                    ((k * P + task.p) * R + r) * Q * S + task.q * S);
                        }
                }
            }
            if (batched) {
                active = !batches.empty();
                for (auto& physical : groups)
                    physical.erase(std::remove_if(physical.begin(),physical.end(),[&](const SiteRightGroup& group) {
                        return group.m >= offset && group.m < offset + component.M;
                    }),physical.end());
            }
        }
        if (!batched) for (ssize_t p = 0; p < P; ++p) {
            auto& physical = groups[p];
            const bool indexed = indexed_site_fields;
            using Selection = std::vector<const SiteRightGroup*>;
            std::vector<Selection> diagonal_by_q(indexed ? Q : 0), general_by_q(indexed ? Q : 0);
            if (indexed) for (const auto& group : physical)
                if (group.m >= offset && group.m < offset + component.M)
                    (group.diagonal ? diagonal_by_q : general_by_q)[group.q].push_back(&group);
            Selection selected_buffer;
            auto select = [&](bool diagonal, ssize_t q) -> const Selection& {
                if (indexed) return (diagonal ? diagonal_by_q : general_by_q)[q];
                selected_buffer.clear();
                for (const auto& group : physical)
                    if (group.diagonal == diagonal && group.q == q && group.m >= offset
                        && group.m < offset + component.M) selected_buffer.push_back(&group);
                return selected_buffer;
            };
            for (ssize_t q = 0; q < Q; ++q) {
                const auto& selected = select(true,q);
                if (selected.empty()) continue;
                active = true;
                const ssize_t count = selected.size();
                std::vector<cdouble> coefficients(K * count), right(count * R), values(K * R);
                for (ssize_t g = 0; g < count; ++g) {
                    const auto& group = *selected[g];
                    for (ssize_t k = 0; k < K; ++k)
                        coefficients[k * count + g] = component.spectator_coefficients[(group.m - offset) * K + k];
                    for (ssize_t r = 0; r < R; ++r) right[g * R + r] = group.right_eff[group.compact ? r : r * S + r];
                }
                const std::vector<cdouble> empty;
                environment_product(coefficients.data(), right.data(), values.data(), K, count, R, empty);
                for (ssize_t k = 0; k < K; ++k)
                    store_values(values.data() + k * R, R,
                        fields.field.data() + ((k * P + p) * Q + q) * R);
            }
            const auto& physical_selected = select(false,p);
            if (!physical_selected.empty()) {
                active = true;
                const ssize_t count = physical_selected.size();
                std::vector<cdouble> coefficients(K * count), right(count * R * S), values(K * R * S);
                for (ssize_t g = 0; g < count; ++g) {
                    const auto& group = *physical_selected[g];
                    for (ssize_t k = 0; k < K; ++k)
                        coefficients[k * count + g] = component.spectator_coefficients[(group.m - offset) * K + k];
                    std::copy_n(group.right_eff.data(), R * S, right.data() + g * R * S);
                }
                const std::vector<cdouble> empty;
                environment_product(coefficients.data(), right.data(), values.data(), K, count, R * S, empty);
                for (ssize_t k = 0; k < K; ++k)
                    store_values(values.data() + k * R * S, R * S,
                        fields.physical_field.data() + (k * P + p) * R * S);
            }
            for (ssize_t q = 0; q < Q; ++q) {
                if (q == p) continue;
                const auto& selected = select(false,q);
                if (selected.empty()) continue;
                active = true;
                const ssize_t count = selected.size();
                std::vector<cdouble> coefficients(K * count), right(count * R * S), values(K * R * S);
                for (ssize_t g = 0; g < count; ++g) {
                    const auto& group = *selected[g];
                    for (ssize_t k = 0; k < K; ++k)
                        coefficients[k * count + g] = component.spectator_coefficients[(group.m - offset) * K + k];
                    std::copy_n(group.right_eff.data(), R * S, right.data() + g * R * S);
                }
                const std::vector<cdouble> empty;
                environment_product(coefficients.data(), right.data(), values.data(), K, count, R * S, empty);
                for (ssize_t k = 0; k < K; ++k)
                    for (ssize_t r = 0; r < R; ++r)
                        store_values(values.data() + (k * R + r) * S, S,
                            fields.general_field.data() + ((k * P + p) * R + r) * Q * S + q * S);
            }
            physical.erase(std::remove_if(physical.begin(), physical.end(), [&](const SiteRightGroup& group) {
                return group.m >= offset && group.m < offset + component.M;
            }), physical.end());
        }
        if (active) {
            if (common) {
                if (!accumulate) {
                    for (std::size_t i = 0; i < diagonal_elements; ++i) shared->field[i] += action.field[i];
                    for (std::size_t i = 0; i < physical_elements; ++i) shared->physical_field[i] += action.physical_field[i];
                    if (!action.general_field.empty()) {
                        if (shared->general_field.empty()) shared->general_field = std::move(action.general_field);
                        else for (std::size_t i = 0; i < general_elements; ++i) shared->general_field[i] += action.general_field[i];
                    }
                }
                spectator_elements += extra;
            } else {
                spectator_elements += elements; spectator_actions.push_back(std::move(action));
            }
            site_spectator_components.fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto assemble = [&](std::size_t component_index, ssize_t offset) {
        const auto start = std::chrono::steady_clock::now();
        assemble_component(component_index,offset);
        if (stream_site_fields) streamed_assembly_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    };
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
        const std::size_t count = sum_components->size();
        const char* parallel_flag = std::getenv("PYQED_TDVP_PARALLEL_SITE_PREPARATION");
        if (!parallel_flag || std::string(parallel_flag) == "1")
            preparation_workers = std::max(1, std::min(workers, static_cast<int>(count)));
        const bool parallel = preparation_workers > 1;
        std::vector<ssize_t> offsets(count + 1, 0);
        for (std::size_t k = 0; k < count; ++k) offsets[k + 1] = offsets[k] + (*sum_components)[k].M;
        std::vector<std::array<double, 3>> times(count, {0,0,0});
        std::vector<std::vector<std::vector<SiteRightGroup>>> component_groups(
            parallel ? count : 0, std::vector<std::vector<SiteRightGroup>>(P));
        auto build_component = [&](std::size_t k) {
            const auto& component = (*sum_components)[k];
            const ssize_t m_offset = offsets[k];
            auto& target_groups = parallel ? component_groups[k] : groups;
            auto& coefficient_seconds = times[k][0];
            auto& product_seconds = times[k][1];
            auto& storage_seconds = times[k][2];
            const ssize_t RS = R * S;
            const char* compact_physical = std::getenv("PYQED_TDVP_COMPACT_SITE_PHYSICAL");
            const bool compact_rows = component.diagonal && compact_physical
                && std::string(compact_physical) == "1";
            const ssize_t physical_rows = compact_rows ? P : P * Q;
            if (compact_rows) compact_site_physical_components.fetch_add(1, std::memory_order_relaxed);
            std::vector<cdouble> right_matrix(static_cast<std::size_t>(component.N * RS));
            for (ssize_t n = 0; n < component.N; ++n)
                for (ssize_t r = 0; r < R; ++r)
                    std::copy_n(component.right + (r * component.N + n) * S, S,
                        right_matrix.data() + n * RS + r * S);
            const auto diagonal = diagonal_environment(right_matrix, component.N, R, S);
            const bool compact = component.spectator_size && component.spectator_size < A * B && !diagonal.empty();
            const char* compact_right = std::getenv("PYQED_TDVP_COMPACT_SITE_RIGHT");
            const bool direct_diagonal = compact && compact_right && std::string(compact_right) == "1";
            if (direct_diagonal) compact_site_right_components.fetch_add(1, std::memory_order_relaxed);
            if (fuse_selected[k]) {
                auto& field = fused_fields[k];
                const ssize_t K = component.spectator_size;
                const ssize_t physical = component.diagonal ? P : P * Q;
                const ssize_t tile = fused_physical_tiles[k];
                std::vector<cdouble> coefficients(K * component.M);
                for (ssize_t j = 0; j < K; ++j)
                    for (ssize_t m = 0; m < component.M; ++m)
                        coefficients[j * component.M + m] = component.spectator_coefficients[m * K + j];
                field.diagonal = !diagonal.empty();
                const ssize_t columns = field.diagonal ? R : RS;
                field.values.resize(K * physical * columns);
                ssize_t saved_groups = 0;
                for (ssize_t begin = 0; begin < physical; begin += tile) {
                    fused_site_tiles.fetch_add(1,std::memory_order_relaxed);
                    const auto fused_start = std::chrono::steady_clock::now();
                    const ssize_t size = std::min(tile,physical - begin);
                    std::vector<cdouble> screened(static_cast<std::size_t>(component.M) * component.N * size);
                    std::vector<char> active(static_cast<std::size_t>(component.M) * size,false);
                    for (ssize_t m = 0; m < component.M; ++m)
                        for (ssize_t n = 0; n < component.N; ++n)
                            for (ssize_t t = 0; t < size; ++t) {
                                const ssize_t pq = begin + t;
                                const ssize_t p = component.diagonal ? pq : pq / Q;
                                const ssize_t q = component.diagonal ? pq : pq % Q;
                                const auto value = component.coefficient(m,n,p,q,Q);
                                if (std::abs(value) <= w_cutoff) continue;
                                screened[(m * component.N + n) * size + t] = value;
                                active[m * size + t] = true;
                                field.active = true;
                                field.general = field.general || (diagonal.empty() && p != q);
                            }
                    const auto fused_coefficients_end = std::chrono::steady_clock::now();
                    std::vector<cdouble> projected(K * component.N * size), right_coefficients(K * size * component.N);
                    const std::vector<cdouble> empty;
                    environment_product(coefficients.data(),screened.data(),projected.data(),K,component.M,component.N * size,empty);
                    for (ssize_t j = 0; j < K; ++j)
                        for (ssize_t t = 0; t < size; ++t)
                            for (ssize_t n = 0; n < component.N; ++n)
                                right_coefficients[(j * size + t) * component.N + n] =
                                    projected[(j * component.N + n) * size + t];
                    std::vector<cdouble>().swap(screened);
                    std::vector<cdouble>().swap(projected);
                    std::vector<cdouble> values(K * size * columns);
                    environment_product(right_coefficients.data(),field.diagonal ? diagonal.data() : right_matrix.data(),
                        values.data(),K * size,component.N,columns,empty);
                    for (ssize_t j = 0; j < K; ++j)
                        std::copy_n(values.data() + j * size * columns,size * columns,
                            field.values.data() + (j * physical + begin) * columns);
                    saved_groups += std::count(active.begin(),active.end(),true);
                    if (profile_site) {
                        coefficient_seconds += std::chrono::duration<double>(fused_coefficients_end - fused_start).count();
                        product_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - fused_coefficients_end).count();
                    }
                }
                field.valid = true;
                fused_site_components.fetch_add(1,std::memory_order_relaxed);
                fused_site_saved_groups.fetch_add(saved_groups,std::memory_order_relaxed);
                return;
            }
            std::vector<cdouble> coefficients(static_cast<std::size_t>(physical_rows * component.N));
            const ssize_t result_columns = direct_diagonal ? R : RS;
            const std::vector<cdouble> empty_diagonal;
            std::vector<cdouble> effective(static_cast<std::size_t>(physical_rows * result_columns));
            const cdouble one(1.0, 0.0), zero(0.0, 0.0);
            for (ssize_t m = 0; m < component.M; ++m) {
                std::chrono::steady_clock::time_point coefficient_start, coefficient_end, product_end;
                if (profile_site) coefficient_start = std::chrono::steady_clock::now();
                std::fill(coefficients.begin(), coefficients.end(), zero);
                std::vector<bool> active(static_cast<std::size_t>(physical_rows), false);
                for (ssize_t n = 0; n < component.N; ++n) {
                    for (ssize_t p = 0; p < P; ++p) {
                        for (ssize_t q = compact_rows ? p : 0; q < (compact_rows ? p + 1 : Q); ++q) {
                            if (component.diagonal && p != q) continue;
                            const cdouble wval = component.coefficient(m, n, p, q, Q);
                            if (std::abs(wval) <= w_cutoff) {
                                continue;
                            }
                            const ssize_t pq = compact_rows ? p : p * Q + q;
                            coefficients[pq * component.N + n] = wval;
                            active[static_cast<std::size_t>(pq)] = true;
                        }
                    }
                }
                std::vector<ssize_t> transitions;
                for (ssize_t pq = 0; pq < physical_rows; ++pq) {
                    if (!active[static_cast<std::size_t>(pq)]) continue;
                    const ssize_t row = static_cast<ssize_t>(transitions.size());
                    if (row != pq) std::copy_n(coefficients.data() + pq * component.N,
                        component.N, coefficients.data() + row * component.N);
                    transitions.push_back(compact_rows ? pq * Q + pq : pq);
                }
                if (profile_site) {
                    coefficient_end = std::chrono::steady_clock::now();
                    coefficient_seconds += std::chrono::duration<double>(coefficient_end - coefficient_start).count();
                }
                if (transitions.empty()) continue;
                environment_product(coefficients.data(), direct_diagonal ? diagonal.data() : right_matrix.data(), effective.data(),
                                    transitions.size(), component.N, result_columns, direct_diagonal ? empty_diagonal : diagonal);
                if (direct_diagonal) diagonal_environment_products.fetch_add(1, std::memory_order_relaxed);
                if (profile_site) {
                    product_end = std::chrono::steady_clock::now();
                    product_seconds += std::chrono::duration<double>(product_end - coefficient_end).count();
                }
                for (std::size_t row = 0; row < transitions.size(); ++row) {
                    const ssize_t pq = transitions[row];
                    const cdouble* values = effective.data() + row * result_columns;
                    std::vector<cdouble> stored;
                    if (direct_diagonal) stored.assign(values, values + R);
                    else if (compact) {
                        stored.resize(R);
                        for (ssize_t r = 0; r < R; ++r) stored[r] = values[r * S + r];
                    } else stored.assign(values, values + RS);
                    target_groups[static_cast<std::size_t>(pq / Q)].push_back(
                        SiteRightGroup{m_offset + m, pq % Q, std::move(stored), !diagonal.empty(), compact});
                }
                if (profile_site) storage_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - product_end).count();
            }
        };
        if (stream_site_fields) {
            ssize_t offset = 0;
            const std::size_t window = std::min(preparation_workers,2);
            for (std::size_t begin = 0; begin < count; begin += window) {
                const std::size_t end = std::min(count,begin + window);
                std::exception_ptr failure;
                std::mutex failure_mutex;
                auto work = [&](std::size_t k) {
                    try { build_component(k); }
                    catch (...) {
                        std::lock_guard<std::mutex> guard(failure_mutex);
                        if (!failure) failure = std::current_exception();
                    }
                };
                if (!parallel) work(begin);
                else {
                    std::vector<std::thread> threads;
                    for (std::size_t k = begin; k < end; ++k) threads.emplace_back(work,k);
                    for (auto& thread : threads) thread.join();
                }
                if (failure) std::rethrow_exception(failure);
                long long staged_bytes = 0;
                for (std::size_t k = begin; k < end; ++k)
                    staged_bytes += fused_fields[k].values.size() * sizeof(cdouble);
                auto peak = peak_streamed_site_bytes.load(std::memory_order_relaxed);
                while (peak < staged_bytes && !peak_streamed_site_bytes.compare_exchange_weak(
                        peak,staged_bytes,std::memory_order_relaxed)) {}
                // Commit in component order; each output is bounded by 32 MiB.
                for (std::size_t k = begin; k < end; ++k) {
                    if (parallel)
                        for (ssize_t p = 0; p < P; ++p) {
                            for (auto& group : component_groups[k][p])
                                groups[p].push_back(std::move(group));
                            std::vector<SiteRightGroup>().swap(component_groups[k][p]);
                        }
                    assemble(k,offset);
                    offset += (*sum_components)[k].M;
                }
            }
        }
        else if (!parallel) for (std::size_t k = 0; k < count; ++k) build_component(k);
        else {
            std::atomic<std::size_t> next(0);
            std::exception_ptr failure;
            std::mutex failure_mutex;
            auto work = [&]() {
                try {
                    while (true) {
                        const auto k = next.fetch_add(1, std::memory_order_relaxed);
                        if (k >= count) break;
                        build_component(k);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> guard(failure_mutex);
                    if (!failure) failure = std::current_exception();
                }
            };
            std::vector<std::thread> threads;
            for (int worker = 0; worker < preparation_workers; ++worker) threads.emplace_back(work);
            for (auto& thread : threads) thread.join();
            if (failure) std::rethrow_exception(failure);
            for (std::size_t k = 0; k < count; ++k)
                for (ssize_t p = 0; p < P; ++p)
                    for (auto& group : component_groups[k][p]) groups[p].push_back(std::move(group));
        }
        for (const auto& time : times) {
            coefficient_seconds += time[0]; product_seconds += time[1]; storage_seconds += time[2];
        }
    }

    const auto groups_end = std::chrono::steady_clock::now();
    if (!stream_site_fields && sum_components && R == S) {
        ssize_t offset = 0;
        for (std::size_t k = 0; k < sum_components->size(); ++k) {
            assemble(k, offset);
            offset += (*sum_components)[k].M;
        }
    }
    site_spectator_groups.fetch_add(spectator_actions.size(), std::memory_order_relaxed);
    const auto field_assembly_end = std::chrono::steady_clock::now();
    for (auto& action : spectator_actions) {
        const ssize_t K = action.component->spectator_size;
        for (ssize_t p = 0; p < P; ++p) {
            for (ssize_t q = 0; q < Q; ++q) {
                bool active = false;
                for (ssize_t k = 0; k < K && !active; ++k)
                    for (ssize_t r = 0; r < R; ++r)
                        if (action.field[((k * P + p) * Q + q) * R + r] != cdouble(0.0, 0.0)) { active = true; break; }
                if (active) action.transitions.emplace_back(p, q);
            }
            bool active = false;
            for (ssize_t k = 0; k < K && !active; ++k)
                for (ssize_t rs = 0; rs < R * S; ++rs)
                    if (action.physical_field[(k * P + p) * R * S + rs] != cdouble(0.0, 0.0)) { active = true; break; }
            if (active) action.physical_points.push_back(p);
            active = false;
            if (!action.general_field.empty())
                for (ssize_t k = 0; k < K && !active; ++k)
                    for (ssize_t rqs = 0; rqs < R * Q * S; ++rqs)
                        if (action.general_field[(k * P + p) * R * Q * S + rqs] != cdouble(0.0, 0.0)) {
                            active = true; break;
                        }
            if (active) action.general_points.push_back(p);
        }
        auto retain_real = [](std::vector<cdouble>& complex, std::vector<double>& real) {
            if (!std::all_of(complex.begin(), complex.end(), [](cdouble value) { return value.imag() == 0.0; })) return;
            real.resize(complex.size());
            for (std::size_t i = 0; i < complex.size(); ++i) real[i] = complex[i].real();
            std::vector<cdouble>().swap(complex);
        };
        retain_real(action.field, action.real_field);
        retain_real(action.physical_field, action.real_physical);
        retain_real(action.general_field, action.real_general);
    }
    const auto field_layout_end = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < spectator_actions.size(); ++i) {
        auto& action = spectator_actions[i];
        for (std::size_t j = 0; j < i; ++j) {
            const auto& previous = spectator_actions[j];
            if (previous.field_owner >= 0 || action.component->spectator_size != previous.component->spectator_size
                || action.field != previous.field || action.physical_field != previous.physical_field
                || action.general_field != previous.general_field || action.real_field != previous.real_field
                || action.real_physical != previous.real_physical || action.real_general != previous.real_general) continue;
            action.field_owner = j;
            std::vector<cdouble>().swap(action.field);
            std::vector<cdouble>().swap(action.physical_field);
            std::vector<cdouble>().swap(action.general_field);
            std::vector<double>().swap(action.real_field);
            std::vector<double>().swap(action.real_physical);
            std::vector<double>().swap(action.real_general);
            break;
        }
    }
    std::size_t spectator_bytes = 0;
    for (const auto& action : spectator_actions)
        spectator_bytes += (action.field.size() + action.physical_field.size() + action.general_field.size()) * sizeof(cdouble)
            + (action.real_field.size() + action.real_physical.size() + action.real_general.size()) * sizeof(double);
    spectator_elements = (spectator_bytes + sizeof(cdouble) - 1) / sizeof(cdouble);
    const char* separate_site_budget = std::getenv("PYQED_TDVP_SEPARATE_SITE_BUDGET");
    const std::size_t selection_spectator_elements = !separate_site_budget
        || std::string(separate_site_budget) != "0" ? 0 : spectator_elements;
    const auto spectators_end = std::chrono::steady_clock::now();
    auto residual_copy_end = spectators_end, residual_aggregation_end = spectators_end;
    std::array<double,4> unit_preparation_seconds{};
    auto site_phases = [&](std::chrono::steady_clock::time_point solve) {
        return std::vector<std::pair<std::string, double>>{
            {"effective_groups", std::chrono::duration<double>(groups_end - contraction_start).count() - streamed_assembly_seconds},
            {"spectator_fields", std::chrono::duration<double>(spectators_end - groups_end).count() + streamed_assembly_seconds},
            {"residual_preparation", std::chrono::duration<double>(solve - spectators_end).count()}};
    };
    auto site_preparation = [&](std::chrono::steady_clock::time_point solve) {
        return std::vector<std::pair<std::string, double>>{
            {"coefficient_rows", coefficient_seconds}, {"right_products", product_seconds},
            {"group_storage", storage_seconds},
            {"spectator_assembly", std::chrono::duration<double>(field_assembly_end - groups_end).count() + streamed_assembly_seconds},
            {"spectator_layout", std::chrono::duration<double>(field_layout_end - field_assembly_end).count()},
            {"spectator_sharing", std::chrono::duration<double>(spectators_end - field_layout_end).count()},
            {"residual_copy", std::chrono::duration<double>(residual_copy_end - spectators_end).count()},
            {"residual_aggregation", std::chrono::duration<double>(residual_aggregation_end - residual_copy_end).count()},
            {"residual_action_setup", std::chrono::duration<double>(solve - residual_aggregation_end).count()},
            {"unit_selection", unit_preparation_seconds[0]},
            {"unit_packing", unit_preparation_seconds[1]},
            {"unit_products", unit_preparation_seconds[2]},
            {"unit_storage", unit_preparation_seconds[3]}};
    };
    auto add_spectators = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        const cdouble one(1.0, 0.0), zero(0.0, 0.0);
        for (auto& action : spectator_actions) {
            const auto& component = *action.component;
            const auto& fields = action.field_owner < 0 ? action : spectator_actions[action.field_owner];
            const ssize_t K = component.spectator_size;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                K, Q * S, B, &one, component.spectator_ket, B, local.data(), Q * S,
                &zero, action.input.data(), Q * S);
            std::fill(action.output.begin(), action.output.end(), zero);
            for (ssize_t k = 0; k < K; ++k)
                for (const auto& transition : action.transitions) {
                    const ssize_t p = transition.first, q = transition.second;
                        for (ssize_t r = 0; r < R; ++r)
                            if (!fields.real_field.empty())
                                action.output[(k * P + p) * R + r] += fields.real_field[((k * P + p) * Q + q) * R + r]
                                    * action.input[(k * Q + q) * S + r];
                            else action.output[(k * P + p) * R + r] += fields.field[((k * P + p) * Q + q) * R + r]
                                    * action.input[(k * Q + q) * S + r];
                }
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t p : action.physical_points)
                    if (!fields.real_physical.empty())
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            R, 2, S, 1.0, fields.real_physical.data() + (k * P + p) * R * S, S,
                            reinterpret_cast<const double*>(action.input.data() + (k * Q + p) * S), 2,
                            1.0, reinterpret_cast<double*>(action.output.data() + (k * P + p) * R), 2);
                    else cblas_zgemv(CblasRowMajor, CblasNoTrans, R, S, &one,
                        fields.physical_field.data() + (k * P + p) * R * S, S,
                        action.input.data() + (k * Q + p) * S, 1,
                        &one, action.output.data() + (k * P + p) * R, 1);
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t p : action.general_points)
                    if (!fields.real_general.empty())
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            R, 2, Q * S, 1.0, fields.real_general.data() + (k * P + p) * R * Q * S, Q * S,
                            reinterpret_cast<const double*>(action.input.data() + k * Q * S), 2,
                            1.0, reinterpret_cast<double*>(action.output.data() + (k * P + p) * R), 2);
                    else cblas_zgemv(CblasRowMajor, CblasNoTrans, R, Q * S, &one,
                        fields.general_field.data() + (k * P + p) * R * Q * S, Q * S,
                        action.input.data() + k * Q * S, 1, &one, action.output.data() + (k * P + p) * R, 1);
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A, P * R, K, &one, component.spectator_bra, A, action.output.data(), P * R,
                &one, out.data(), P * R);
        }
    };
    std::vector<ssize_t> residual_remap;
    ssize_t residual_m = M;
    const char* direct_residual = std::getenv("PYQED_TDVP_DIRECT_RESIDUAL_LEFT");
    if ((!direct_residual || std::string(direct_residual) != "0") && !spectator_actions.empty()) {
        residual_remap.assign(M, -1);
        for (const auto& physical : groups)
            for (const auto& group : physical) residual_remap[group.m] = 0;
        residual_m = 0;
        for (auto& m : residual_remap) if (m == 0) m = residual_m++;
        residual_left_saved_elements.fetch_add((M - residual_m) * A * B, std::memory_order_relaxed);
    }
    std::vector<cdouble> left_stack(static_cast<std::size_t>(residual_m) * A * B, 0.0);
    if (sum_components == nullptr) {
    for (ssize_t m = 0; m < M; ++m) {
        const ssize_t target = residual_remap.empty() ? m : residual_remap[m];
        if (target < 0) continue;
        for (ssize_t a = 0; a < A; ++a) {
            for (ssize_t b = 0; b < B; ++b) {
                left_stack[(target * A + a) * B + b] = left_ptr[(a * M + m) * B + b];
            }
        }
    }
    } else {
        ssize_t m_offset = 0;
        for (const SiteSumComponentView& component : *sum_components) {
            for (ssize_t m = 0; m < component.M; ++m) {
                const ssize_t target = residual_remap.empty() ? m_offset + m : residual_remap[m_offset + m];
                if (target < 0) continue;
                for (ssize_t a = 0; a < A; ++a) {
                    for (ssize_t b = 0; b < B; ++b) {
                        left_stack[(target * A + a) * B + b] =
                            component.left[(a * component.M + m) * B + b];
                    }
                }
            }
            m_offset += component.M;
        }
    }

    const std::size_t matrix_units = static_cast<std::size_t>(A) * B;
    auto compact_left_channels = [&]() {
        std::vector<ssize_t> remap(M, -1);
        for (const auto& physical : groups)
            for (const auto& group : physical) remap[group.m] = 0;
        std::vector<cdouble> active_left;
        ssize_t active_m = 0;
        for (ssize_t m = 0; m < M; ++m) if (remap[m] == 0) {
            remap[m] = active_m++;
            const auto* values = left_stack.data() + m * matrix_units;
            active_left.insert(active_left.end(), values, values + matrix_units);
        }
        for (auto& physical : groups)
            for (auto& group : physical) group.m = remap[group.m];
        left_stack.swap(active_left);
        M = active_m;
    };
    if (!residual_remap.empty()) {
        for (auto& physical : groups)
            for (auto& group : physical) group.m = residual_remap[group.m];
        M = residual_m;
    } else if (!spectator_actions.empty()) compact_left_channels();
    residual_copy_end = std::chrono::steady_clock::now();
    std::size_t total_groups = 0;
    for (const auto& physical_groups : groups) total_groups += physical_groups.size();
    const bool expanded_state = std::getenv("PYQED_TDVP_EXPANDED_SITE_STATE");
    std::vector<bool> diagonal_transition(static_cast<std::size_t>(P) * Q,
        !std::getenv("PYQED_TDVP_EXPANDED_SITE_STATE")
        && !std::getenv("PYQED_TDVP_EXPANDED_SITE_AGGREGATION"));
    std::vector<bool> active_transition(static_cast<std::size_t>(P) * Q, false);
    for (ssize_t p = 0; p < P; ++p)
        for (const auto& group : groups[p]) {
            active_transition[p * Q + group.q] = true;
            if (!group.diagonal) diagonal_transition[p * Q + group.q] = false;
        }
    std::size_t proposed_elements = 0;
    for (ssize_t pq = 0; pq < P * Q; ++pq)
        if (active_transition[pq]) proposed_elements += matrix_units * R * (diagonal_transition[pq] ? 1 : S);
    if (expanded_state) proposed_elements = matrix_units * P * Q * R * S;
    bool state_units = false;
    if (!std::getenv("PYQED_TDVP_PARTIAL_SITE_UNITS") && M > (expanded_state ? 4 : 1) * A * B
        && static_cast<long double>(total_groups) * R * S > 2.0L * proposed_elements
        && proposed_elements <= (128ULL << 20) / sizeof(cdouble) - selection_spectator_elements) {
        // Exactly factor L_m into state-bond matrix units and aggregate its
        // right factors. This removes redundant virtual channels without SVD.
        auto unit_tick = profile_site ? std::chrono::steady_clock::now() : spectators_end;
        auto unit_phase = [&](int phase) {
            if (!profile_site) return;
            const auto now = std::chrono::steady_clock::now();
            unit_preparation_seconds[phase] += std::chrono::duration<double>(now - unit_tick).count();
            unit_tick = now;
        };
        const char* index_units_flag = std::getenv("PYQED_TDVP_INDEX_UNIT_GROUPS");
        const bool index_units = index_units_flag && index_units_flag[0] == '1';
        for (ssize_t p = 0; p < P; ++p) {
            auto& physical_groups = groups[static_cast<std::size_t>(p)];
            std::vector<SiteRightGroup> compressed;
            compressed.reserve(matrix_units * Q);
            std::vector<std::vector<const SiteRightGroup*>> indexed;
            if (index_units) {
                indexed.resize(Q);
                for (const auto& group : physical_groups) indexed[group.q].push_back(&group);
            }
            for (ssize_t q = 0; q < Q; ++q) {
                std::vector<const SiteRightGroup*> selected;
                if (index_units) selected.swap(indexed[q]);
                else for (const auto& group : physical_groups) if (group.q == q) selected.push_back(&group);
                if (selected.empty()) continue;
                const bool compact = diagonal_transition[p * Q + q];
                std::vector<const SiteRightGroup*> diagonal_selected;
                if (!std::getenv("PYQED_TDVP_EXPANDED_SITE_AGGREGATION")) {
                    for (const auto* group : selected)
                        if (group->diagonal) diagonal_selected.push_back(group);
                    selected.erase(std::remove_if(selected.begin(), selected.end(),
                        [](const SiteRightGroup* group) { return group->diagonal; }), selected.end());
                }
                const ssize_t count = static_cast<ssize_t>(selected.size());
                unit_phase(0);
                std::vector<cdouble> coefficients(matrix_units * count);
                std::vector<cdouble> right(static_cast<std::size_t>(count) * R * S);
                for (ssize_t g = 0; g < count; ++g) {
                    const auto& group = *selected[static_cast<std::size_t>(g)];
                    for (std::size_t unit = 0; unit < matrix_units; ++unit)
                        coefficients[unit * count + g] = left_stack[group.m * matrix_units + unit];
                    std::copy(group.right_eff.begin(), group.right_eff.end(), right.data() + g * R * S);
                }
                std::vector<cdouble> effective(compact ? 0 : matrix_units * R * S);
                std::vector<cdouble> diagonal_effective;
                const cdouble one(1.0, 0.0), zero(0.0, 0.0);
                unit_phase(1);
                if (count) cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(matrix_units), static_cast<int>(R*S), static_cast<int>(count),
                    &one, coefficients.data(), static_cast<int>(count), right.data(), static_cast<int>(R*S),
                    &zero, effective.data(), static_cast<int>(R*S));
                unit_phase(2);
                if (!diagonal_selected.empty()) {
                    diagonal_site_aggregations.fetch_add(1, std::memory_order_relaxed);
                    const ssize_t diagonal_count = diagonal_selected.size();
                    std::vector<cdouble> diagonal_coefficients(matrix_units * diagonal_count);
                    std::vector<cdouble> diagonal_right(diagonal_count * R);
                    for (ssize_t g = 0; g < diagonal_count; ++g) {
                        const auto& group = *diagonal_selected[g];
                        for (std::size_t unit = 0; unit < matrix_units; ++unit)
                            diagonal_coefficients[unit * diagonal_count + g] =
                                left_stack[group.m * matrix_units + unit];
                        for (ssize_t r = 0; r < R; ++r)
                            diagonal_right[g * R + r] = group.right_eff[r * S + r];
                    }
                    diagonal_effective.resize(matrix_units * R);
                    unit_phase(1);
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(matrix_units), static_cast<int>(R), static_cast<int>(diagonal_count),
                        &one, diagonal_coefficients.data(), static_cast<int>(diagonal_count),
                        diagonal_right.data(), static_cast<int>(R), &zero,
                        diagonal_effective.data(), static_cast<int>(R));
                    unit_phase(2);
                    if (!compact) for (std::size_t unit = 0; unit < matrix_units; ++unit)
                        for (ssize_t r = 0; r < R; ++r)
                            effective[unit * R * S + r * S + r] += diagonal_effective[unit * R + r];
                }
                for (std::size_t unit = 0; unit < matrix_units; ++unit) {
                    const ssize_t columns = compact ? R : R * S;
                    const cdouble* values = (compact ? diagonal_effective.data() : effective.data()) + unit * columns;
                    compressed.push_back({static_cast<ssize_t>(unit), q,
                        std::vector<cdouble>(values, values + columns), compact, compact});
                }
                if (compact) diagonal_unit_factors.fetch_add(matrix_units, std::memory_order_relaxed);
                unit_phase(3);
            }
            physical_groups.swap(compressed);
        }
        compressed_site_channels.fetch_add(M - static_cast<ssize_t>(matrix_units), std::memory_order_relaxed);
        M = static_cast<ssize_t>(matrix_units);
        state_units = true;
        std::vector<cdouble>().swap(left_stack);
    }
    residual_aggregation_end = std::chrono::steady_clock::now();
    if (state_units) {
        struct UnitAction {
            std::vector<ssize_t> inputs;
            std::vector<cdouble> right;
            std::vector<ssize_t> diagonal_inputs;
            std::vector<cdouble> diagonal_right;
        };
        std::vector<UnitAction> actions(static_cast<std::size_t>(P) * A);
        std::size_t max_input = 0;
        for (ssize_t p = 0; p < P; ++p)
            for (ssize_t a = 0; a < A; ++a) {
                auto& action = actions[p * A + a];
                std::vector<SiteRightGroup*> selected;
                for (auto& group : groups[p]) if (group.m / B == a) {
                    if (group.compact) {
                        action.diagonal_inputs.push_back((group.m % B) * Q + group.q);
                        action.diagonal_right.insert(action.diagonal_right.end(), group.right_eff.begin(), group.right_eff.end());
                        std::vector<cdouble>().swap(group.right_eff);
                        continue;
                    }
                    selected.push_back(&group);
                    action.inputs.push_back((group.m % B) * Q + group.q);
                }
                const std::size_t columns = selected.size() * S;
                max_input = std::max(max_input, columns);
                action.right.resize(static_cast<std::size_t>(R) * columns);
                for (std::size_t g = 0; g < selected.size(); ++g) {
                    auto& group = *selected[g];
                    for (ssize_t r = 0; r < R; ++r)
                        std::copy_n(group.right_eff.data() + r * S, S,
                            action.right.data() + r * columns + g * S);
                    std::vector<cdouble>().swap(group.right_eff);
                }
            }
        matrix_unit_site_calls.fetch_add(1, std::memory_order_relaxed);
        const int threads = std::max(1, std::min(workers, static_cast<int>(P)));
        std::vector<std::vector<cdouble>> inputs(threads, std::vector<cdouble>(max_input));
        const cdouble alpha(1.0, 0.0), beta(0.0, 0.0);
        ActionWorkers action_workers(threads);
        auto action = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            out.assign(static_cast<std::size_t>(A) * P * R, cdouble(0.0, 0.0));
            auto apply = [&](ssize_t begin, ssize_t end, int worker) {
                auto& input = inputs[worker];
                for (ssize_t p = begin; p < end; ++p)
                    for (ssize_t a = 0; a < A; ++a) {
                        const auto& factor = actions[p * A + a];
                        if (!factor.inputs.empty()) {
                            for (std::size_t g = 0; g < factor.inputs.size(); ++g)
                                std::copy_n(local.data() + factor.inputs[g] * S, S, input.data() + g * S);
                            cblas_zgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(R),
                                static_cast<int>(factor.inputs.size() * S), &alpha,
                                factor.right.data(), static_cast<int>(factor.inputs.size() * S),
                                input.data(), 1, &beta, out.data() + (a * P + p) * R, 1);
                        }
                        cdouble* output = out.data() + (a * P + p) * R;
                        for (std::size_t g = 0; g < factor.diagonal_inputs.size(); ++g) {
                            const auto* source = local.data() + factor.diagonal_inputs[g] * S;
                            const auto* values = factor.diagonal_right.data() + g * R;
                            for (ssize_t r = 0; r < R; ++r) output[r] += values[r] * source[r];
                        }
                    }
            };
            if (threads == 1) apply(0, P, 0);
            else if (action_workers.active()) action_workers.run([&](int worker) {
                apply(P * worker / threads, P * (worker + 1) / threads, worker);
            });
            else {
                std::vector<std::thread> pool;
                for (int worker = 0; worker < threads; ++worker)
                    pool.emplace_back([&, worker]() { apply(P * worker / threads,
                        P * (worker + 1) / threads, worker); });
                for (auto& thread : pool) thread.join();
            }
        };
        const auto solve_start = std::chrono::steady_clock::now();
        auto summed_action = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
            action(local, out); add_spectators(local, out);
        };
        auto result = lanczos_apply(vec, dt, krylov_dim, tol, summed_action, imaginary_time);
        record_contraction(false, {B, Q, S}, M, contraction_start, solve_start, site_phases(solve_start), {}, site_preparation(solve_start), preparation_workers);
        return result;
    }
    struct DiagonalUnitAction {
        std::vector<ssize_t> inputs;
        std::vector<cdouble> right;
    };
    std::vector<DiagonalUnitAction> diagonal_actions(static_cast<std::size_t>(P) * A);
    std::vector<std::vector<cdouble>> physical_actions(static_cast<std::size_t>(P) * A);
    std::size_t compact_elements = selection_spectator_elements;
    if (!expanded_state && !std::getenv("PYQED_TDVP_EXPANDED_SITE_AGGREGATION")) {
        std::size_t diagonal_groups = 0, transitions = 0;
        for (ssize_t p = 0; p < P; ++p)
            for (ssize_t q = 0; q < Q; ++q) {
                std::size_t count = 0;
                for (const auto& group : groups[p]) if (group.q == q && group.diagonal) ++count;
                diagonal_groups += count;
                if (count) ++transitions;
            }
        const std::size_t elements = matrix_units * transitions * R;
        if (elements && elements <= (128ULL << 20) / sizeof(cdouble) - selection_spectator_elements
            && static_cast<long double>(diagonal_groups) * R * S > 2.0L * elements) {
            partial_site_unit_calls.fetch_add(1, std::memory_order_relaxed);
            compact_elements += elements;
            const cdouble one(1.0, 0.0), zero(0.0, 0.0);
            for (ssize_t p = 0; p < P; ++p) {
                for (ssize_t q = 0; q < Q; ++q) {
                    std::vector<const SiteRightGroup*> selected;
                    for (const auto& group : groups[p])
                        if (group.q == q && group.diagonal) selected.push_back(&group);
                    const ssize_t count = selected.size();
                    if (!count) continue;
                    std::vector<cdouble> coefficients(matrix_units * count), right(count * R);
                    for (ssize_t g = 0; g < count; ++g) {
                        const auto& group = *selected[g];
                        for (std::size_t unit = 0; unit < matrix_units; ++unit)
                            coefficients[unit * count + g] = left_stack[group.m * matrix_units + unit];
                        for (ssize_t r = 0; r < R; ++r) right[g * R + r] = group.right_eff[r * S + r];
                    }
                    std::vector<cdouble> effective(matrix_units * R);
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(matrix_units), static_cast<int>(R), static_cast<int>(count),
                        &one, coefficients.data(), static_cast<int>(count), right.data(), static_cast<int>(R),
                        &zero, effective.data(), static_cast<int>(R));
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t b = 0; b < B; ++b) {
                            auto& action = diagonal_actions[p * A + a];
                            action.inputs.push_back(b * Q + q);
                            const auto* values = effective.data() + (a * B + b) * R;
                            action.right.insert(action.right.end(), values, values + R);
                        }
                    diagonal_unit_factors.fetch_add(matrix_units, std::memory_order_relaxed);
                }
                auto& physical = groups[p];
                physical.erase(std::remove_if(physical.begin(), physical.end(),
                    [](const SiteRightGroup& group) { return group.diagonal; }), physical.end());
            }
            compact_left_channels();
        }
    }
    if (!expanded_state && !std::getenv("PYQED_TDVP_SEPARATE_PHYSICAL_UNITS")) {
        std::vector<bool> physical_only(M, true), active(M, false);
        for (ssize_t p = 0; p < P; ++p)
            for (const auto& group : groups[p]) {
                active[group.m] = true;
                if (group.q != p) physical_only[group.m] = false;
            }
        ssize_t virtuals = 0;
        for (ssize_t m = 0; m < M; ++m) if (active[m] && physical_only[m]) ++virtuals;
        std::size_t selected_groups = 0, points = 0;
        for (ssize_t p = 0; p < P; ++p) {
            std::size_t count = 0;
            for (const auto& group : groups[p]) if (physical_only[group.m]) ++count;
            selected_groups += count;
            if (count) ++points;
        }
        const std::size_t elements = matrix_units * points * R * S;
        const long double previous_cost = static_cast<long double>(virtuals) * A * B * Q * S
            + static_cast<long double>(selected_groups) * A * R * S;
        if (elements && elements <= 128 * 1024 * 1024 / sizeof(cdouble) - compact_elements
            && previous_cost > 2.0L * elements) {
            physical_site_unit_calls.fetch_add(1, std::memory_order_relaxed);
            const cdouble one(1.0, 0.0), zero(0.0, 0.0);
            for (ssize_t p = 0; p < P; ++p) {
                std::vector<const SiteRightGroup*> selected;
                for (const auto& group : groups[p]) if (physical_only[group.m]) selected.push_back(&group);
                const ssize_t count = selected.size();
                if (!count) continue;
                std::vector<cdouble> coefficients(matrix_units * count), right(count * R * S);
                for (ssize_t g = 0; g < count; ++g) {
                    const auto& group = *selected[g];
                    for (std::size_t unit = 0; unit < matrix_units; ++unit)
                        coefficients[unit * count + g] = left_stack[group.m * matrix_units + unit];
                    std::copy(group.right_eff.begin(), group.right_eff.end(), right.data() + g * R * S);
                }
                std::vector<cdouble> effective(matrix_units * R * S);
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(matrix_units), static_cast<int>(R * S), static_cast<int>(count),
                    &one, coefficients.data(), static_cast<int>(count), right.data(), static_cast<int>(R * S),
                    &zero, effective.data(), static_cast<int>(R * S));
                for (ssize_t a = 0; a < A; ++a) {
                    auto& action = physical_actions[p * A + a];
                    action.resize(static_cast<std::size_t>(R) * B * S);
                    for (ssize_t r = 0; r < R; ++r)
                        for (ssize_t b = 0; b < B; ++b)
                            std::copy_n(effective.data() + ((a * B + b) * R + r) * S, S,
                                action.data() + (r * B + b) * S);
                }
                auto& physical = groups[p];
                physical.erase(std::remove_if(physical.begin(), physical.end(),
                    [&](const SiteRightGroup& group) { return physical_only[group.m]; }), physical.end());
            }
            compact_left_channels();
        }
    }
    auto add_diagonal = [&](ssize_t p, const std::vector<cdouble>& local, std::vector<cdouble>& out, cdouble* input_scratch) {
        if (!physical_actions[p * A].empty())
            for (ssize_t b = 0; b < B; ++b)
                std::copy_n(local.data() + (b * Q + p) * S, S, input_scratch + b * S);
        const cdouble one(1.0, 0.0);
        for (ssize_t a = 0; a < A; ++a) {
            const auto& action = diagonal_actions[p * A + a];
            auto* output = out.data() + (a * P + p) * R;
            for (std::size_t g = 0; g < action.inputs.size(); ++g) {
                const auto* input = local.data() + action.inputs[g] * S;
                const auto* values = action.right.data() + g * R;
                for (ssize_t r = 0; r < R; ++r) output[r] += values[r] * input[r];
            }
            const auto& physical = physical_actions[p * A + a];
            if (!physical.empty()) cblas_zgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(R), static_cast<int>(B * S), &one,
                physical.data(), static_cast<int>(B * S), input_scratch, 1, &one, output, 1);
        }
    };
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
    std::vector<std::vector<cdouble>> physical_inputs(p_threads,
        std::vector<cdouble>(static_cast<std::size_t>(B) * S));
    ActionWorkers action_workers(M ? std::max(q_threads, p_threads) : 1);

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        out.assign(dim, cdouble(0.0, 0.0));
        if (M == 0) {
            for (ssize_t p = 0; p < P; ++p) add_diagonal(p, local, out, physical_inputs[0].data());
            return;
        }
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
        } else if (action_workers.active()) {
            action_workers.run([&](int worker) {
                if (worker < q_threads)
                    apply_q_range(Q * worker / q_threads, Q * (worker + 1) / q_threads);
            });
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
                add_diagonal(p, local, out, physical_inputs[worker].data());
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
            add_diagonal(p, local, out, physical_inputs[worker].data());
        }
        };
        if (p_threads == 1) {
            apply_p_range(0, P, 0);
        } else if (action_workers.active()) {
            action_workers.run([&](int worker) {
                if (worker < p_threads)
                    apply_p_range(P * worker / p_threads, P * (worker + 1) / p_threads, worker);
            });
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

    const auto solve_start = std::chrono::steady_clock::now();
    auto summed_action = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        matvec(local, out); add_spectators(local, out);
    };
    auto result = lanczos_apply(vec, dt, krylov_dim, tol, summed_action, imaginary_time);
    record_contraction(false, {B, Q, S}, M, contraction_start, solve_start, site_phases(solve_start), {}, site_preparation(solve_start), preparation_workers);
    return result;
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
    const auto contraction_start = std::chrono::steady_clock::now();
    double left_seconds = 0., residual_seconds = 0., field_seconds = 0.;
    struct DiagonalField {
        std::size_t component;
        std::vector<cdouble> values, input, output;
        std::vector<double> real_values;
    };
    std::vector<DiagonalField> fields;
    std::vector<bool> factored(components.size(), false);
    const std::size_t field_budget = std::min<std::size_t>(
        (64ULL << 20) / sizeof(cdouble), max_workspace_elements / 2);
    std::size_t retained = 0;
    const cdouble one(1., 0.), zero(0., 0.);
    // Contract the nuclear frame once; electronic bra/ket and right banks stay separate.
    struct Projection {
        std::vector<cdouble> values;
        std::vector<double> real_values;
    };
    struct GeneralField {
        std::size_t component, projection;
        std::vector<cdouble> right, input, output;
    };
    struct Candidate {
        std::vector<std::size_t> channels;
        std::size_t elements;
        double benefit;
        bool real;
    };
    const std::size_t scratch_budget = std::min<std::size_t>(
        (32ULL << 20) / sizeof(cdouble), max_workspace_elements);
    auto is_real = [](const cdouble* values, std::size_t count) {
        return std::all_of(values, values + count, [](cdouble value) { return value.imag() == 0.; });
    };
    std::vector<bool> diagonal_right(components.size(), true), grouped(components.size(), false);
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto& x = components[c];
        for (ssize_t r = 0; r < S && diagonal_right[c]; ++r)
            for (ssize_t n = 0; n < x.N && diagonal_right[c]; ++n)
                for (ssize_t s = 0; s < S; ++s)
                    if (r != s && x.right[(r * x.N + n) * S + s] != cdouble(0.)) {
                        diagonal_right[c] = false;
                        break;
                    }
    }
    auto equal_projection = [&](std::size_t a, std::size_t b) {
        const auto& x = components[a];
        const auto& y = components[b];
        if (x.M != y.M || x.N != y.N || x.diagonal != y.diagonal ||
            x.spectator_size != y.spectator_size) return false;
        const auto count = static_cast<std::size_t>(x.M) * x.N * Q * (x.diagonal ? 1 : Q);
        return (x.W == y.W || std::equal(x.W, x.W + count, y.W)) &&
            (x.spectator_coefficients == y.spectator_coefficients ||
             std::equal(x.spectator_coefficients, x.spectator_coefficients + x.M * x.spectator_size,
                        y.spectator_coefficients));
    };
    std::vector<Candidate> candidates;
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto& x = components[c];
        const ssize_t K = x.spectator_size;
        if (grouped[c] || diagonal_right[c] || !K || K >= B * B) continue;
        Candidate candidate{{c}, 0, 0., false};
        grouped[c] = true;
        for (std::size_t d = c + 1; d < components.size(); ++d)
            if (!grouped[d] && !diagonal_right[d] && equal_projection(c, d)) {
                candidate.channels.push_back(d);
                grouped[d] = true;
            }
        const ssize_t physical = x.diagonal ? Q : Q * Q;
        candidate.real = is_real(x.W, static_cast<std::size_t>(x.M) * x.N * physical) &&
                         is_real(x.spectator_coefficients, x.M * K);
        const std::size_t projected_size = static_cast<std::size_t>(K) * x.N * physical;
        candidate.elements = (candidate.real ? (projected_size + 1) / 2 : projected_size) +
            candidate.channels.size() * (x.N * S * S + 2 * K * Q * S);
        candidate.benefit = static_cast<double>(candidate.channels.size()) * x.M * x.N * physical * S * S;
        candidates.push_back(std::move(candidate));
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.benefit > b.benefit;
    });
    std::vector<Projection> projections;
    std::vector<GeneralField> general_fields;
    ssize_t largest_n = 0;
    int field_workers = std::max(1, workers);
    for (const auto& candidate : candidates) {
        const auto c = candidate.channels.front();
        const auto& x = components[c];
        const ssize_t K = x.spectator_size, physical = x.diagonal ? Q : Q * Q;
        const std::size_t fixed = K * x.M, per_point = (x.M + K) * x.N;
        const std::size_t action_scratch = static_cast<std::size_t>(2) * x.N * Q * S;
        if (candidate.elements > field_budget - retained || fixed >= scratch_budget ||
            per_point > scratch_budget - fixed || action_scratch > scratch_budget) continue;
        field_workers = std::min<int>(field_workers, scratch_budget / action_scratch);
        largest_n = std::max(largest_n, x.N);
        Projection projection;
        if (candidate.real) projection.real_values.resize(static_cast<std::size_t>(K) * x.N * physical);
        else projection.values.resize(static_cast<std::size_t>(K) * x.N * physical);
        const ssize_t tile = std::min<std::size_t>(physical, (scratch_budget - fixed) / per_point);
        std::vector<cdouble> coefficients(K * x.M);
        for (ssize_t k = 0; k < K; ++k)
            for (ssize_t m = 0; m < x.M; ++m)
                coefficients[k * x.M + m] = x.spectator_coefficients[m * K + k];
        for (ssize_t begin = 0; begin < physical; begin += tile) {
            const ssize_t width = std::min(tile, physical - begin);
            std::vector<cdouble> screened(x.M * x.N * width), projected(K * x.N * width);
            for (ssize_t m = 0; m < x.M; ++m)
                for (ssize_t n = 0; n < x.N; ++n)
                    for (ssize_t t = 0; t < width; ++t) {
                        const ssize_t pq = begin + t;
                        const auto value = x.coefficient(m, n, x.diagonal ? pq : pq / Q,
                                                        x.diagonal ? pq : pq % Q, Q);
                        if (std::abs(value) > w_cutoff) screened[(m * x.N + n) * width + t] = value;
                    }
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, x.N * width, x.M,
                &one, coefficients.data(), x.M, screened.data(), x.N * width,
                &zero, projected.data(), x.N * width);
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t n = 0; n < x.N; ++n) {
                    const auto offset = (k * x.N + n) * physical + begin;
                    const auto source = projected.data() + (k * x.N + n) * width;
                    if (candidate.real)
                        for (ssize_t t = 0; t < width; ++t) projection.real_values[offset + t] = source[t].real();
                    else std::copy_n(source, width, projection.values.data() + offset);
                }
        }
        const auto index = projections.size();
        projections.push_back(std::move(projection));
        for (const auto d : candidate.channels) {
            const auto& channel = components[d];
            GeneralField field{d, index, std::vector<cdouble>(x.N * S * S),
                               std::vector<cdouble>(K * Q * S), std::vector<cdouble>(K * Q * S)};
            for (ssize_t n = 0; n < x.N; ++n)
                for (ssize_t s = 0; s < S; ++s)
                    for (ssize_t r = 0; r < S; ++r)
                        field.right[(n * S + s) * S + r] = channel.right[(r * x.N + n) * S + s];
            general_fields.push_back(std::move(field));
            factored[d] = true;
        }
        retained += candidate.elements;
        projected_site_components.fetch_add(candidate.channels.size(), std::memory_order_relaxed);
        shared_site_projections.fetch_add(candidate.channels.size() - 1, std::memory_order_relaxed);
        if (candidate.real) real_site_projections.fetch_add(1, std::memory_order_relaxed);
    }
    const std::size_t work_size = static_cast<std::size_t>(largest_n) * Q * S;
    std::vector<std::vector<cdouble>> field_work(field_workers), field_ordered(field_workers);
    ActionWorkers field_team(field_workers);
    auto add_general_fields = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        for (auto& field : general_fields) {
            const auto& x = components[field.component];
            const auto& values = projections[field.projection].values;
            const auto& real_values = projections[field.projection].real_values;
            const bool real = !real_values.empty();
            const ssize_t K = x.spectator_size;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, Q * S, B,
                &one, x.spectator_ket, B, local.data(), Q * S, &zero, field.input.data(), Q * S);
            auto action = [&](int worker) {
                auto& z = field_work[worker];
                auto& ordered = field_ordered[worker];
                for (ssize_t k = worker; k < K; k += field_workers) {
                    if (x.diagonal) {
                        for (ssize_t n = 0; n < x.N; ++n)
                            for (ssize_t p = 0; p < Q; ++p)
                                for (ssize_t s = 0; s < S; ++s)
                                    z[(n * Q + p) * S + s] = (real ? real_values[(k * x.N + n) * Q + p] :
                                        values[(k * x.N + n) * Q + p]) *
                                        field.input[(k * Q + p) * S + s];
                    } else if (real) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.N * Q, 2 * S, Q,
                            1., real_values.data() + k * x.N * Q * Q, Q,
                            reinterpret_cast<const double*>(field.input.data() + k * Q * S), 2 * S,
                            0., reinterpret_cast<double*>(z.data()), 2 * S);
                    } else {
                        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x.N * Q, S, Q,
                            &one, values.data() + k * x.N * Q * Q, Q,
                            field.input.data() + k * Q * S, S, &zero, z.data(), S);
                    }
                    for (ssize_t p = 0; p < Q; ++p)
                        for (ssize_t n = 0; n < x.N; ++n)
                            std::copy_n(z.data() + (n * Q + p) * S, S, ordered.data() + (p * x.N + n) * S);
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Q, S, x.N * S,
                        &one, ordered.data(), x.N * S, field.right.data(), S,
                        &zero, field.output.data() + k * Q * S, S);
                }
            };
            if (field_team.active()) field_team.run(action);
            else {
                std::vector<std::thread> threads;
                for (int worker = 1; worker < field_workers; ++worker) threads.emplace_back(action, worker);
                action(0);
                for (auto& thread : threads) thread.join();
            }
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, B, Q * S, K,
                &one, x.spectator_bra, B, field.output.data(), Q * S, &one, out.data(), Q * S);
        }
    };
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto& component = components[c];
        if (factored[c]) continue;
        const ssize_t K = component.spectator_size;
        if (!K || K >= B * B) continue;
        const ssize_t physical = component.diagonal ? Q : Q * Q;
        const std::size_t elements = static_cast<std::size_t>(K) * physical * S;
        const std::size_t buffers = static_cast<std::size_t>(2) * K * Q * S;
        if (!diagonal_right[c]) continue;
        const bool real = is_real(component.W, static_cast<std::size_t>(component.M) * component.N * physical) &&
            is_real(component.spectator_coefficients, component.M * K) &&
            is_real(component.right, static_cast<std::size_t>(S) * component.N * S);
        const std::size_t stored_elements = real ? (elements + 1) / 2 : elements;
        if (stored_elements + buffers > field_budget - retained) continue;
        const std::size_t fixed = K * component.M + component.N * S;
        const std::size_t per_point = component.M * component.N + 2 * K * component.N + K * S;
        const std::size_t scratch_budget = std::min<std::size_t>(
            (32ULL << 20) / sizeof(cdouble), max_workspace_elements);
        if (fixed >= scratch_budget || per_point > scratch_budget - fixed) continue;
        const ssize_t tile = std::min<std::size_t>(physical, (scratch_budget - fixed) / per_point);
        std::vector<cdouble> coefficients(K * component.M), right(component.N * S);
        for (ssize_t k = 0; k < K; ++k)
            for (ssize_t m = 0; m < component.M; ++m)
                coefficients[k * component.M + m] = component.spectator_coefficients[m * K + k];
        for (ssize_t n = 0; n < component.N; ++n)
            for (ssize_t r = 0; r < S; ++r)
                right[n * S + r] = component.right[(r * component.N + n) * S + r];
        DiagonalField field{c, std::vector<cdouble>(real ? 0 : elements),
            std::vector<cdouble>(K * Q * S), std::vector<cdouble>(K * Q * S),
            std::vector<double>(real ? elements : 0)};
        for (ssize_t begin = 0; begin < physical; begin += tile) {
            const ssize_t width = std::min(tile, physical - begin);
            std::vector<cdouble> screened(component.M * component.N * width);
            for (ssize_t m = 0; m < component.M; ++m)
                for (ssize_t n = 0; n < component.N; ++n)
                    for (ssize_t t = 0; t < width; ++t) {
                        const ssize_t pq = begin + t;
                        const auto value = component.coefficient(m, n,
                            component.diagonal ? pq : pq / Q, component.diagonal ? pq : pq % Q, Q);
                        if (std::abs(value) > w_cutoff)
                            screened[(m * component.N + n) * width + t] = value;
                    }
            std::vector<cdouble> projected(K * component.N * width), ordered(K * width * component.N);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                K, component.N * width, component.M, &one, coefficients.data(), component.M,
                screened.data(), component.N * width, &zero, projected.data(), component.N * width);
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t t = 0; t < width; ++t)
                    for (ssize_t n = 0; n < component.N; ++n)
                        ordered[(k * width + t) * component.N + n] = projected[(k * component.N + n) * width + t];
            std::vector<cdouble>().swap(screened);
            std::vector<cdouble>().swap(projected);
            std::vector<cdouble> values(K * width * S);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                K * width, S, component.N, &one, ordered.data(), component.N,
                right.data(), S, &zero, values.data(), S);
            for (ssize_t k = 0; k < K; ++k) {
                const auto offset = (k * physical + begin) * S;
                const auto source = values.data() + k * width * S;
                if (real)
                    for (ssize_t t = 0; t < width * S; ++t) field.real_values[offset + t] = source[t].real();
                else std::copy_n(source, width * S, field.values.data() + offset);
            }
        }
        retained += stored_elements + buffers;
        factored[c] = true;
        fields.push_back(std::move(field));
        site_spectator_components.fetch_add(1, std::memory_order_relaxed);
    }
    for (int worker = 0; worker < field_workers; ++worker) {
        field_work[worker].resize(work_size);
        field_ordered[worker].resize(work_size);
    }
    const auto retained_bytes = static_cast<long long>(retained * sizeof(cdouble));
    auto previous_peak = peak_projected_site_bytes.load(std::memory_order_relaxed);
    while (previous_peak < retained_bytes && !peak_projected_site_bytes.compare_exchange_weak(
        previous_peak, retained_bytes, std::memory_order_relaxed)) {}
    auto add_fields = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        add_general_fields(local, out);
        for (auto& field : fields) {
            const auto& component = components[field.component];
            const ssize_t K = component.spectator_size;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, Q * S, B,
                &one, component.spectator_ket, B, local.data(), Q * S,
                &zero, field.input.data(), Q * S);
            std::fill(field.output.begin(), field.output.end(), cdouble(0.));
            for (ssize_t k = 0; k < K; ++k)
                for (ssize_t p = 0; p < Q; ++p)
                    for (ssize_t q = component.diagonal ? p : 0;
                         q < (component.diagonal ? p + 1 : Q); ++q) {
                        const ssize_t pq = component.diagonal ? p : p * Q + q;
                        const ssize_t physical = component.diagonal ? Q : Q * Q;
                        for (ssize_t r = 0; r < S; ++r)
                            field.output[(k * Q + p) * S + r] +=
                                (field.real_values.empty() ? field.values[(k * physical + pq) * S + r] :
                                 field.real_values[(k * physical + pq) * S + r]) * field.input[(k * Q + q) * S + r];
                    }
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, B, Q * S, K,
                &one, component.spectator_bra, B, field.output.data(), Q * S,
                &one, out.data(), Q * S);
        }
    };
    // Linearity permits local left sums only when both other factors match exactly.
    std::vector<std::vector<std::size_t>> residual_members(components.size());
    for (std::size_t c = 0; c < components.size(); ++c) {
        if (factored[c]) continue;
        const auto& x = components[c];
        std::size_t representative = c;
        for (std::size_t r = 0; r < c; ++r) {
            if (residual_members[r].empty()) continue;
            const auto& y = components[r];
            if (x.M != y.M || x.N != y.N || x.diagonal != y.diagonal) continue;
            const auto count = static_cast<std::size_t>(x.M) * x.N * Q * (x.diagonal ? 1 : Q);
            if ((x.W == y.W || std::equal(x.W, x.W + count, y.W)) &&
                (x.right == y.right || std::equal(x.right, x.right + S * x.N * S, y.right))) {
                representative = r;
                shared_site_right_components.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        residual_members[representative].push_back(c);
    }
    std::vector<std::vector<SiteSumGroupKey>> groups(
        static_cast<std::size_t>(Q)
    );
    ssize_t m_offset = 0;
    std::size_t max_groups = 0;
    for (std::size_t k = 0; k < components.size(); ++k) {
        const SiteSumComponentView& component = components[k];
        if (residual_members[k].empty()) continue;
        for (ssize_t m = 0; m < component.M; ++m) {
            for (ssize_t p = 0; p < Q; ++p) {
                for (ssize_t q = 0; q < Q; ++q) {
                    bool active = false;
                    for (ssize_t n = 0; n < component.N; ++n) {
                        if (
                            std::abs(component.coefficient(m, n, p, q, Q)) > w_cutoff
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

    total_m = m_offset;
    std::vector<cdouble> left_stack(
        static_cast<std::size_t>(total_m) * B * B,
        0.0
    );
    m_offset = 0;
    for (std::size_t k = 0; k < components.size(); ++k) {
        if (residual_members[k].empty()) continue;
        const auto& component = components[k];
        for (ssize_t m = 0; m < component.M; ++m) {
            for (ssize_t a = 0; a < B; ++a) {
                for (ssize_t b = 0; b < B; ++b) {
                    for (const auto c : residual_members[k])
                        left_stack[((m_offset + m) * B + a) * B + b] +=
                            components[c].left[(a * component.M + m) * B + b];
                }
            }
        }
        m_offset += component.M;
    }

    const int q_threads = std::max(1, std::min(workers, static_cast<int>(Q)));
    const int p_threads = std::max(1, std::min(workers, static_cast<int>(Q)));
    ssize_t maximum_n = 1;
    std::size_t maximum_mq = 0, general_mq = 0, diagonal_mq = 0;
    for (std::size_t c = 0; c < components.size(); ++c) {
        if (residual_members[c].empty()) continue;
        const auto& x = components[c];
        maximum_n = std::max(maximum_n, x.N);
    }
    for (const auto& local_groups : groups) {
        for (std::size_t first = 0; first < local_groups.size();) {
            auto last = first + 1;
            while (last < local_groups.size() && local_groups[last].component == local_groups[first].component) ++last;
            maximum_mq = std::max(maximum_mq, last - first);
            auto& width = diagonal_right[local_groups[first].component] ? diagonal_mq : general_mq;
            width = std::max(width, last - first);
            first = last;
        }
    }
    const std::size_t frame_scratch = 2 * work_size * field_workers + q_threads * B * S;
    const std::size_t stream_budget = scratch_budget > frame_scratch ?
        (scratch_budget - frame_scratch) / p_threads : 0;
    std::size_t stream_rows = maximum_mq;
    // Tile active input rows too: increasing state ranks must not force the
    // entire action back to the more expensive right-environment projection.
    if (general_mq && stream_budget > static_cast<std::size_t>(B * S)) {
        const std::size_t n_target = std::min<ssize_t>(maximum_n, 64);
        const std::size_t fixed = B * S * (1 + 2 * n_target);
        if (stream_budget > fixed) {
            const std::size_t rows = (stream_budget - fixed) / (B * S + n_target);
            if (rows >= 64) stream_rows = std::min(stream_rows, rows);
        }
    }
    const std::size_t stream_left_elements = std::max(
        std::min(general_mq, stream_rows) * B * S,
        std::min(diagonal_mq, stream_rows) * S);
    const std::size_t stream_fixed = stream_left_elements + B * S;
    const std::size_t stream_per_n = stream_rows + 2 * B * S;
    const ssize_t stream_tile = stream_budget > stream_fixed ? std::min<std::size_t>(
        maximum_n, (stream_budget - stream_fixed) / stream_per_n) : 0;
    std::vector<bool> real_coefficients(components.size(), false);
    bool real_residual = true;
    for (std::size_t k = 0; k < components.size(); ++k) {
        if (residual_members[k].empty()) continue;
        const auto& x = components[k];
        real_coefficients[k] = is_real(x.W, static_cast<std::size_t>(x.M) * x.N * Q * (x.diagonal ? 1 : Q));
        real_residual = real_residual && real_coefficients[k];
    }
    // Real coefficient products use real BLAS on interleaved complex states;
    // compare that cost with complex right-environment projection.
    const char* contraction_order = std::getenv("PYQED_TDVP_SITE_CONTRACTION");
    const bool prefer_stream = contraction_order && std::string(contraction_order) == "stream";
    const bool stream_right = total_m && (prefer_stream || B < (real_residual ? 2*S : S) || general_mq == 0) && stream_tile > 0;
    if (stream_right) {
        streamed_site_right_calls.fetch_add(1, std::memory_order_relaxed);
        const auto bytes = static_cast<long long>((frame_scratch +
            p_threads * (stream_fixed + stream_tile * stream_per_n)) * sizeof(cdouble));
        auto peak = peak_streamed_right_scratch_bytes.load(std::memory_order_relaxed);
        while (peak < bytes && !peak_streamed_right_scratch_bytes.compare_exchange_weak(
            peak, bytes, std::memory_order_relaxed)) {}
    }
    std::vector<std::vector<cdouble>> right_banks(components.size());
    for (std::size_t k = 0; k < components.size(); ++k) {
        const auto& component = components[k];
        if (stream_right && !residual_members[k].empty()) {
            if (real_coefficients[k]) real_streamed_site_components.fetch_add(1, std::memory_order_relaxed);
        }
        if (residual_members[k].empty() || (stream_right && !diagonal_right[k])) continue;
        auto& bank = right_banks[k];
        if (diagonal_right[k]) {
            compact_residual_right_components.fetch_add(1, std::memory_order_relaxed);
            bank.resize(component.N * S);
            for (ssize_t n = 0; n < component.N; ++n)
                for (ssize_t r = 0; r < S; ++r)
                    bank[n * S + r] = component.right[(r * component.N + n) * S + r];
            continue;
        }
        bank.resize(static_cast<std::size_t>(component.N) * S * S);
        for (ssize_t n = 0; n < component.N; ++n)
            for (ssize_t r = 0; r < S; ++r)
                std::copy_n(component.right + (r * component.N + n) * S, S,
                            bank.data() + (n * S + r) * S);
    }
    const std::size_t elements_per_group =
        static_cast<std::size_t>(S) * static_cast<std::size_t>(B + 2*S) + maximum_n;
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
    // Screen and pack immutable coefficient rows once per local solve. The
    // cache has a separate bounded allowance equal to the scratch budget.
    std::vector<std::vector<cdouble>> coefficient_cache(Q);
    const char* cache_right_env = std::getenv("PYQED_TDVP_CACHE_RIGHT");
    const bool cache_right = !cache_right_env || std::string(cache_right_env) != "0";
    const char* real_right_env = std::getenv("PYQED_TDVP_REAL_RIGHT");
    const bool real_right = real_right_env && std::string(real_right_env) == "1";
    std::vector<std::vector<cdouble>> projected_right_cache(Q);
    std::vector<std::vector<std::size_t>> projected_right_offsets(Q);
    std::vector<std::vector<std::size_t>> coefficient_offsets(Q);
    std::size_t cached_elements = 0;
    for (ssize_t p = 0; p < Q; ++p) {
        auto& offsets = coefficient_offsets[p];
        offsets.assign(groups[p].size(), std::numeric_limits<std::size_t>::max());
        for (std::size_t j = 0; j < groups[p].size(); ++j) {
            const auto& group = groups[p][j];
            const auto& x = components[group.component];
            if ((!stream_right && cache_right) || static_cast<std::size_t>(x.N) > max_workspace_elements - cached_elements) continue;
            offsets[j] = coefficient_cache[p].size();
            for (ssize_t n = 0; n < x.N; ++n) {
                const auto value = x.coefficient(group.m, n, p, group.q, Q);
                coefficient_cache[p].push_back(std::abs(value) > w_cutoff ? value : cdouble(0.));
            }
            cached_elements += x.N;
        }
    }
    // The projected right environment is independent of the Krylov vector.
    // Reuse whole contraction blocks under the same cache allowance, rather
    // than caching their coefficients and repeating the matrix product.
    // Match cache ownership to the static p-range partition: filling only
    // the first p values otherwise leaves the last worker entirely uncached.
    const char* balanced_cache_env = std::getenv("PYQED_TDVP_BALANCED_CACHE");
    const bool balanced_cache = balanced_cache_env && std::string(balanced_cache_env) == "1";
    std::vector<std::size_t> worker_cache_elements(p_threads, 0);
    for (ssize_t p = 0; p < Q; ++p) {
        auto& offsets = projected_right_offsets[p];
        offsets.assign(groups[p].size(), std::numeric_limits<std::size_t>::max());
        if (stream_right || !cache_right) continue;
        const int owner = std::min(p_threads-1, static_cast<int>(((p+1)*p_threads-1)/Q));
        const std::size_t owner_budget = max_workspace_elements/p_threads
            + (static_cast<std::size_t>(owner) < max_workspace_elements%p_threads ? 1 : 0);
        for (std::size_t block = 0; block < groups[p].size(); block += block_groups) {
            const auto stop = std::min(groups[p].size(), block + block_groups);
            for (auto begin = block; begin < stop;) {
                const auto c = groups[p][begin].component;
                auto end = begin + 1;
                while (end < stop && groups[p][end].component == c) ++end;
                const std::size_t columns = diagonal_right[c] ? S : S*S;
                const auto elements = (end-begin)*columns;
                if (elements <= max_workspace_elements-cached_elements &&
                    (!balanced_cache || elements <= owner_budget-worker_cache_elements[owner])) {
                    offsets[begin] = projected_right_cache[p].size();
                    projected_right_cache[p].resize(offsets[begin]+elements);
                    cached_elements += elements;
                    worker_cache_elements[owner] += elements;
                }
                begin = end;
            }
        }
    }
    bool projected_right_ready = false;
    auto coefficient_at = [&](ssize_t p, std::size_t j, ssize_t n) {
        const auto offset = coefficient_offsets[p][j];
        if (offset != std::numeric_limits<std::size_t>::max()) return coefficient_cache[p][offset+n];
        const auto& g = groups[p][j];
        const auto value = components[g.component].coefficient(g.m,n,p,g.q,Q);
        return std::abs(value) > w_cutoff ? value : cdouble(0.);
    };
    ActionWorkers residual_workers(p_threads, true);
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
            stream_right ? stream_left_elements : static_cast<std::size_t>(B) * block_groups * S,
            0.0
        )
    );
    std::vector<std::vector<cdouble>> right_concat_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(
            stream_right ? stream_tile * B * S : static_cast<std::size_t>(S) * block_groups * S,
            0.0
        )
    );
    std::vector<std::vector<cdouble>> out_p_scratch(
        static_cast<std::size_t>(p_threads),
        std::vector<cdouble>(static_cast<std::size_t>(B) * S, 0.0)
    );
    std::vector<std::vector<cdouble>> coefficient_scratch(p_threads,
        std::vector<cdouble>(stream_right ? stream_tile * stream_rows : block_groups * maximum_n));
    std::vector<std::vector<cdouble>> effective_scratch(p_threads,
        std::vector<cdouble>(stream_right ? stream_tile * B * S : block_groups * S * S));
    const cdouble alpha_blas(1.0, 0.0);
    const cdouble beta_zero(0.0, 0.0);
    const cdouble beta_one(1.0, 0.0);
    const auto solve_start = std::chrono::steady_clock::now();

    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        const auto left_start = std::chrono::steady_clock::now();
        out.assign(dim, cdouble(0.0, 0.0));
        if (!total_m) { add_fields(local, out); return; }
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
        } else if (residual_workers.active()) {
            residual_workers.run([&](int worker) {
                apply_q_range(Q * worker / q_threads, Q * (worker+1) / q_threads, worker);
            });
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

        const auto residual_start = std::chrono::steady_clock::now();
        left_seconds += std::chrono::duration<double>(residual_start - left_start).count();
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
                if (stream_right) {
                    auto& coefficients = coefficient_scratch[worker];
                    auto& effective = effective_scratch[worker];
                    for (std::size_t first_group = 0; first_group < local_groups.size();) {
                        const auto c = local_groups[first_group].component;
                        const auto& x = components[c];
                        auto last_group = first_group + 1;
                        while (last_group < local_groups.size() && local_groups[last_group].component == c) ++last_group;
                        if (!diagonal_right[c]) {
                            // Sum input-row tiles before applying the common right environment.
                            // Its large contraction is independent of the row-tile partition.
                            for (ssize_t start = 0; start < x.N; start += stream_tile) {
                                const ssize_t width = std::min(stream_tile, x.N - start);
                                for (std::size_t block = first_group; block < last_group; block += stream_rows) {
                                    const ssize_t mq = std::min<std::size_t>(stream_rows, last_group - block);
                                    for (ssize_t t = 0; t < mq; ++t) {
                                        const auto& group = local_groups[block + t];
                                        std::copy_n(left_times_local.data() +
                                            (group.q * total_m + group.global_m) * B * S, B * S,
                                            left_concat.data() + t * B * S);
                                    }
                                    for (ssize_t n = 0; n < width; ++n)
                                        for (ssize_t t = 0; t < mq; ++t) {
                                            const auto& group = local_groups[block + t];
                                            const auto value = coefficient_at(p, block+t, start+n);
                                            if (real_coefficients[c])
                                                reinterpret_cast<double*>(coefficients.data())[n * mq + t] =
                                                    std::abs(value) > w_cutoff ? value.real() : 0.;
                                            else coefficients[n * mq + t] =
                                                std::abs(value) > w_cutoff ? value : cdouble(0.);
                                        }
                                    if (real_coefficients[c])
                                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, 2*B*S, mq,
                                            1., reinterpret_cast<const double*>(coefficients.data()), mq,
                                            reinterpret_cast<const double*>(left_concat.data()), 2*B*S,
                                            block == first_group ? 0. : 1., reinterpret_cast<double*>(effective.data()), 2*B*S);
                                    else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, B*S, mq,
                                        &alpha_blas, coefficients.data(), mq, left_concat.data(), B*S,
                                        block == first_group ? &beta_zero : &beta_one, effective.data(), B*S);
                                }
                                for (ssize_t n = 0; n < width; ++n)
                                    for (ssize_t s = 0; s < S; ++s)
                                        for (ssize_t a = 0; a < B; ++a)
                                            right_concat[(n * S + s) * B + a] = effective[(n * B + a) * S + s];
                                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, B, width*S,
                                    &alpha_blas, x.right + start*S, x.N*S, right_concat.data(), B,
                                    &beta_one, out_p.data(), B);
                            }
                            first_group = last_group;
                            continue;
                        }
                        last_group = std::min(last_group, first_group + stream_rows);
                        const ssize_t mq = last_group - first_group;
                        for (ssize_t start = 0; start < x.N; start += stream_tile) {
                            const ssize_t width = std::min(stream_tile, x.N - start);
                            for (ssize_t n = 0; n < width; ++n)
                                for (ssize_t t = 0; t < mq; ++t) {
                                    const auto& group = local_groups[first_group + t];
                                    const auto value = coefficient_at(p, first_group+t, start+n);
                                    if (real_coefficients[c])
                                        reinterpret_cast<double*>(coefficients.data())[n * mq + t] =
                                            std::abs(value) > w_cutoff ? value.real() : 0.;
                                    else coefficients[n * mq + t] = std::abs(value) > w_cutoff ? value : cdouble(0.);
                                }
                            if (real_coefficients[c])
                                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, mq, 2*S, width,
                                        1., reinterpret_cast<const double*>(coefficients.data()), mq,
                                        reinterpret_cast<const double*>(right_banks[c].data() + start * S), 2*S,
                                        start == 0 ? 0. : 1., reinterpret_cast<double*>(left_concat.data()), 2*S);
                                else cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, mq, S, width,
                                    &alpha_blas, coefficients.data(), mq, right_banks[c].data() + start * S, S,
                                    start == 0 ? &beta_zero : &beta_one, left_concat.data(), S);
                        }
                        if (diagonal_right[c]) {
                            for (ssize_t t = 0; t < mq; ++t) {
                                const auto& group = local_groups[first_group + t];
                                const auto* input = left_times_local.data() +
                                    (group.q * total_m + group.global_m) * B * S;
                                for (ssize_t a = 0; a < B; ++a)
                                    for (ssize_t r = 0; r < S; ++r)
                                        out_p[r * B + a] += left_concat[t * S + r] * input[a * S + r];
                            }
                        }
                        first_group = last_group;
                    }
                    for (ssize_t a = 0; a < B; ++a)
                        for (ssize_t r = 0; r < S; ++r)
                            out[(a * Q + p) * S + r] = out_p[r * B + a];
                    continue;
                }
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
                    }
                    // Group consecutive queries from one component. Its right
                    // environment is packed once per solve, not per Krylov vector.
                    auto& coefficients = coefficient_scratch[static_cast<std::size_t>(worker)];
                    auto& effective = effective_scratch[static_cast<std::size_t>(worker)];
                    for (std::size_t begin = 0; begin < count;) {
                        const auto component_index = local_groups[block_begin + begin].component;
                        const auto& component = components[component_index];
                        std::size_t end = begin + 1;
                        while (end < count && local_groups[block_begin + end].component == component_index) ++end;
                        const auto cache_offset = projected_right_offsets[p][block_begin+begin];
                        const bool cacheable = cache_offset != std::numeric_limits<std::size_t>::max();
                        const ssize_t columns = diagonal_right[component_index] ? S : S * S;
                        auto* projected = cacheable ? projected_right_cache[p].data()+cache_offset : effective.data();
                        if (!cacheable || !projected_right_ready) {
                        const bool real_product = real_right && real_coefficients[component_index];
                        for (std::size_t j = begin; j < end; ++j) {
                            const auto& group = local_groups[block_begin + j];
                            for (ssize_t n = 0; n < component.N; ++n) {
                                const cdouble value = coefficient_at(p, block_begin+j, n);
                                if (real_product)
                                    reinterpret_cast<double*>(coefficients.data())[(j-begin) * component.N + n] =
                                        std::abs(value) > w_cutoff ? value.real() : 0.;
                                else coefficients[(j-begin) * component.N + n] = std::abs(value) > w_cutoff ? value : cdouble(0.0);
                            }
                        }
                        // Exactly real coefficients multiply interleaved real/imaginary
                        // environment columns; neither component is discarded.
                        if (real_product)
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                static_cast<int>(end-begin), static_cast<int>(2*columns), static_cast<int>(component.N),
                                1., reinterpret_cast<const double*>(coefficients.data()), static_cast<int>(component.N),
                                reinterpret_cast<const double*>(right_banks[component_index].data()), static_cast<int>(2*columns),
                                0., reinterpret_cast<double*>(projected), static_cast<int>(2*columns));
                        else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(end-begin), static_cast<int>(columns), static_cast<int>(component.N),
                            &alpha_blas, coefficients.data(), static_cast<int>(component.N),
                            right_banks[component_index].data(), static_cast<int>(columns),
                            &beta_zero, projected, static_cast<int>(columns));
                        }
                        for (std::size_t j = begin; j < end; ++j)
                            for (ssize_t r = 0; r < S; ++r) {
                                auto* target = right_concat.data() + r * kdim + j * S;
                                if (diagonal_right[component_index]) {
                                    std::fill_n(target, S, cdouble(0.));
                                    target[r] = projected[(j-begin) * S + r];
                                } else std::copy_n(projected + ((j-begin)*S+r)*S, S, target);
                            }
                        begin = end;
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
        } else if (residual_workers.active()) {
            residual_workers.run([&](int worker) {
                apply_p_range(Q * worker / p_threads, Q * (worker+1) / p_threads, worker);
            });
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
        const auto field_start = std::chrono::steady_clock::now();
        projected_right_ready = true;
        residual_seconds += std::chrono::duration<double>(field_start - residual_start).count();
        add_fields(local, out);
        field_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - field_start).count();
    };

    auto result = lanczos_apply(vec, dt, krylov_dim, tol, matvec, imaginary_time);
    record_contraction(false, {B,Q,S}, total_m, contraction_start, solve_start,
        {}, {{"left_state", left_seconds}, {"residual_action", residual_seconds},
             {"projected_fields", field_seconds}}, {}, workers);
    return result;
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
    bool imaginary_time = false,
    const std::vector<std::vector<CArray>>& spectator_factors = {}
) {
    const std::size_t count = lefts.size();
    if (!spectator_factors.empty() && spectator_factors.size() != count)
        throw std::runtime_error("site spectator count must match components");
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
            left.ndim() != 3 || (W.ndim() != 3 && W.ndim() != 4) || right.ndim() != 3
            || left.shape(0) != B || left.shape(2) != B
            || W.shape(0) != left.shape(1)
            || W.shape(2) != Q || (W.ndim() == 4 && W.shape(3) != Q)
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
                operators[k].ndim() == 3,
            }
        );
        if (operators[k].ndim() == 3) compact_site_cores.fetch_add(1, std::memory_order_relaxed);
        if (!spectator_factors.empty() && !spectator_factors[k].empty()) {
            const auto& frame = spectator_factors[k];
            if (frame.size() != 3 || frame[0].ndim() != 2 || frame[1].ndim() != 2 || frame[2].ndim() != 2
                || frame[0].shape(1) != B || frame[1].shape(1) != B
                || frame[0].shape(0) != frame[1].shape(0) || frame[2].shape(1) != frame[0].shape(0)
                || frame[2].shape(0) != lefts[k].shape(1))
                throw std::runtime_error("site spectator shapes require bra(K,B), ket(K,B), coefficients(M,K)");
            components.back().spectator_bra = frame[0].data();
            components.back().spectator_ket = frame[1].data();
            components.back().spectator_coefficients = frame[2].data();
            components.back().spectator_size = frame[0].shape(0);
        }
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
                        if (component.diagonal && p != q) continue;
                        for (ssize_t n = 0; n < component.N; ++n) {
                            if (
                                std::abs(component.coefficient(m, n, p, q, Q)) > w_cutoff
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
                        ] = W.ndim() == 3 ? (p == q ? W.data()[(m * N + n) * Q + p] : cdouble(0.0, 0.0))
                                          : W.data()[((m * N + n) * Q + p) * Q + q];
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
    std::vector<double> real_blocks;
    std::vector<double> split_blocks;
};

struct TwoSiteSumComponentView {
    const cdouble* left;
    const cdouble* first;
    const cdouble* second;
    const cdouble* right;
    ssize_t M;
    ssize_t N;
    ssize_t O;
    const cdouble* prefix_bra = nullptr;
    const cdouble* prefix_ket = nullptr;
    const cdouble* spectator_bra = nullptr;
    const cdouble* spectator_ket = nullptr;
    const cdouble* spectator_coefficients = nullptr;
    ssize_t spectator_size = 0;
    bool first_diagonal = false;
    bool second_diagonal = false;
    cdouble first_coefficient(ssize_t m, ssize_t n, ssize_t p, ssize_t q, ssize_t physical) const {
        return first_diagonal ? (p == q ? first[(m * N + n) * physical + p] : cdouble(0.0, 0.0))
                              : first[((m * N + n) * physical + p) * physical + q];
    }
    cdouble second_coefficient(ssize_t n, ssize_t o, ssize_t r, ssize_t s, ssize_t physical) const {
        return second_diagonal ? (r == s ? second[(n * O + o) * physical + r] : cdouble(0.0, 0.0))
                               : second[((n * O + o) * physical + r) * physical + s];
    }
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
    const std::vector<TwoSiteSumComponentView>* sum_components = nullptr,
    const std::function<std::vector<cdouble>(const std::vector<cdouble>&)>& midpoint = {}
) {
    const auto contraction_start = std::chrono::steady_clock::now();
    std::vector<TwoSitePhysicalGroup> left_groups(
        static_cast<std::size_t>(N) * Q
    );
    const char* point_batch_mode = std::getenv("PYQED_TDVP_POINT_DIAGONAL_BATCH");
    const bool point_batch = point_batch_mode && point_batch_mode[0] == '1';
    std::vector<TwoSitePhysicalGroup> right_groups(
        static_cast<std::size_t>(N) * S
    );
    std::vector<std::pair<std::string, double>> phases;
    auto phase_start = contraction_start;
    const char* profile_phases = std::getenv("PYQED_TDVP_PROFILE_CONTRACTIONS");
    auto mark_phase = [&](const char* name) {
        if (!profile_phases || profile_phases[0] != '1') return;
        const auto now = std::chrono::steady_clock::now();
        phases.emplace_back(name, std::chrono::duration<double>(now - phase_start).count());
        phase_start = now;
    };

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

    std::vector<std::vector<cdouble>> left_environments(sum_components->size());
    std::vector<std::vector<cdouble>> right_environments(sum_components->size());
    std::vector<std::vector<cdouble>> right_diagonals(sum_components->size());
    for (std::size_t k = 0; k < sum_components->size(); ++k) {
        const auto& component = (*sum_components)[k];
        auto& left = left_environments[k];
        auto& right = right_environments[k];
        left.resize(static_cast<std::size_t>(component.M) * A * B);
        right.resize(static_cast<std::size_t>(component.O) * C * D);
        for (ssize_t m = 0; m < component.M; ++m)
            for (ssize_t a = 0; a < A; ++a)
                std::copy_n(component.left + (a * component.M + m) * B, B,
                    left.data() + (m * A + a) * B);
        for (ssize_t o = 0; o < component.O; ++o)
            for (ssize_t c = 0; c < C; ++c)
                std::copy_n(component.right + (c * component.O + o) * D, D,
                    right.data() + (o * C + c) * D);
        right_diagonals[k] = diagonal_environment(right, component.O, C, D);
    }
    mark_phase("environment_layout");
    std::vector<std::vector<cdouble>> packed_left(static_cast<std::size_t>(N));
    std::vector<std::vector<cdouble>> packed_right(static_cast<std::size_t>(N));
    std::vector<std::vector<cdouble>> early_right_values(static_cast<std::size_t>(N));
    std::vector<std::vector<cdouble>> spectator_fields(N);
    std::vector<char> diagonal_spectator_fields(N, false);
    std::atomic<std::size_t> sparse_field_elements(0);
    const char* sparse_frame_flag = std::getenv("PYQED_TDVP_SPARSE_SPECTATORS");
    const bool sparse_frames = !sparse_frame_flag || std::string(sparse_frame_flag) == "1";
    const char* spectator_preparation_flag = std::getenv("PYQED_TDVP_COMPACT_SPECTATOR_PREPARATION");
    const bool compact_spectator_preparation = !spectator_preparation_flag || spectator_preparation_flag[0] != '0';
    const bool direct_packing = !std::getenv("PYQED_TDVP_STAGED_PAIR_PACKING");
    const char* screening_mode = std::getenv("PYQED_TDVP_SHORT_SCREENING");
    const bool short_screening = !screening_mode || screening_mode[0] == '1';
    const char* preparation_mode = std::getenv("PYQED_TDVP_REUSE_PREPARATION");
    const bool reuse_preparation = preparation_mode && preparation_mode[0] == '1';
    struct GroupWorkspace {
        std::vector<std::pair<ssize_t, ssize_t>> transitions;
        std::vector<cdouble> coefficients, effective, conditional;
        std::vector<std::size_t> active_rows;
    };
    auto build_groups_for_n = [&](ssize_t global_n, GroupWorkspace& workspace) {
        const TwoSiteSumComponentView& component =
            *component_for_n[static_cast<std::size_t>(global_n)];
        const ssize_t n = local_n[static_cast<std::size_t>(global_n)];
        const std::size_t component_index = &component - sum_components->data();
        auto build_side = [&](bool first) {
            const ssize_t input = first ? Q : S, output = first ? P : R;
            const ssize_t bond = first ? component.M : component.O;
            const ssize_t block = first ? A * B : C * D;
            GroupWorkspace temporary;
            auto& scratch = reuse_preparation ? workspace : temporary;
            auto& transitions = scratch.transitions;
            auto& coefficients = scratch.coefficients;
            transitions.clear();
            coefficients.clear();
            const bool physical_diagonal = first ? component.first_diagonal : component.second_diagonal;
            const ssize_t physical_rows = physical_diagonal ? std::min(input, output) : input * output;
            transitions.reserve(static_cast<std::size_t>(physical_rows));
            coefficients.reserve(static_cast<std::size_t>(physical_rows * bond));
            for (ssize_t q = 0; q < input; ++q)
                for (ssize_t p = 0; p < output; ++p) {
                    if ((first ? component.first_diagonal : component.second_diagonal) && p != q) continue;
                    const std::size_t begin = coefficients.size();
                    bool active = false;
                    for (ssize_t b = 0; b < bond; ++b) {
                        const cdouble w = first
                            ? component.first_coefficient(b, n, p, q, Q)
                            : component.second_coefficient(n, b, p, q, S);
                        coefficients.push_back(w);
                        active = active || w != cdouble(0.0, 0.0);
                    }
                    if (active) transitions.emplace_back(p, q);
                    else coefficients.resize(begin);
                }
            if (transitions.empty()) return;
            auto& effective = scratch.effective;
            const char* compact_right_flag = std::getenv("PYQED_TDVP_COMPACT_PAIR_RIGHT");
            const bool compact_right = !first && C == D && C > 1
                && !right_diagonals[component_index].empty()
                && compact_right_flag && compact_right_flag[0] == '1';
            const ssize_t effective_block = compact_right ? C : block;
            effective.resize(transitions.size() * effective_block);
            const auto& environment = first ? left_environments[component_index] : right_environments[component_index];
            const std::vector<cdouble> empty;
            environment_product(coefficients.data(), compact_right ? right_diagonals[component_index].data() : environment.data(), effective.data(),
                                transitions.size(), bond, effective_block, first || compact_right ? empty : right_diagonals[component_index]);
            if (compact_right) {
                diagonal_environment_products.fetch_add(1,std::memory_order_relaxed);
                compact_pair_right_products.fetch_add(1,std::memory_order_relaxed);
            }
            auto& active_rows = scratch.active_rows;
            active_rows.clear();
            long long inspected = 0;
            for (std::size_t row = 0; row < transitions.size(); ++row) {
                const cdouble* values = effective.data() + row * effective_block;
                bool active;
                if (short_screening) {
                    active = cutoff < 0.0 || std::isnan(cutoff);
                    for (ssize_t i = 0; i < effective_block && !active; ++i) {
                        ++inspected;
                        active = std::abs(values[i]) > cutoff;
                    }
                } else {
                    double max_abs = 0.0;
                    for (ssize_t i = 0; i < effective_block; ++i) max_abs = std::max(max_abs, std::abs(values[i]));
                    inspected += effective_block;
                    active = !(max_abs <= cutoff);
                }
                if (!active) continue;
                active_rows.push_back(row);
            }
            screened_environment_values.fetch_add(inspected, std::memory_order_relaxed);
            const bool small_prefix = static_cast<std::size_t>(P * A * Q * B) <= 64
                && static_cast<std::size_t>(R * C * S * D) <= 1024 * 1024;
            const bool pack = input > 1 && (first
                ? (active_rows.size() * 2 >= static_cast<std::size_t>(P * Q)
                   || static_cast<std::size_t>(P * A * Q * B) <= 64)
                : (active_rows.size() * 2 >= static_cast<std::size_t>(R * S) || small_prefix));
            const char* direct_right_flag = std::getenv("PYQED_TDVP_DIRECT_PAIR_RIGHT");
            if (compact_right && N > 128 && direct_right_flag && direct_right_flag[0] == '1'
                && !std::getenv("PYQED_TDVP_DENSE_RIGHT_STATE")
                && std::any_of(active_rows.begin(),active_rows.end(),[&](std::size_t row) {
                    if (transitions[row].first == transitions[row].second) return false;
                    const auto* values = effective.data() + row * effective_block;
                    return std::any_of(values,values + C,[](cdouble z) { return z != cdouble(0.,0.); });
                })) {
                auto& values = early_right_values[global_n];
                values.resize(static_cast<std::size_t>(R) * S * C,0.);
                for (std::size_t row : active_rows)
                    std::copy_n(effective.data() + row * C,C,
                        values.data() + (transitions[row].first * S + transitions[row].second) * C);
                direct_pair_right_factors.fetch_add(1,std::memory_order_relaxed);
                return;
            }
            if (first && sparse_frames && !pack && component.spectator_size
                && component.spectator_size < A * B && P == Q && !active_rows.empty()
                && std::all_of(active_rows.begin(), active_rows.end(), [&](std::size_t row) {
                    return transitions[row].first == transitions[row].second;
                })) {
                const ssize_t K = component.spectator_size;
                const std::size_t elements = static_cast<std::size_t>(K) * P;
                const std::size_t limit = (128ULL << 20) / sizeof(cdouble);
        const char* real_field_mode = std::getenv("PYQED_TDVP_REAL_PREFIX_FIELDS");
        const bool real_products = !real_field_mode || real_field_mode[0] == '1';
                const auto previous = sparse_field_elements.fetch_add(elements, std::memory_order_relaxed);
                if (elements <= limit && previous <= limit - elements) {
                    auto& conditional = scratch.conditional;
                    conditional.resize(transitions.size() * K);
                    environment_product(coefficients.data(), component.spectator_coefficients,
                        conditional.data(), transitions.size(), bond, K, empty);
                    auto& field = spectator_fields[global_n];
                    field.resize(elements, 0.0);
                    for (std::size_t row : active_rows)
                        for (ssize_t k = 0; k < K; ++k)
                            field[k * P + transitions[row].first] = conditional[row * K + k];
                    diagonal_spectator_fields[global_n] = true;
                    sparse_spectator_channels.fetch_add(1, std::memory_order_relaxed);
                    compact_spectator_channels.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                sparse_field_elements.fetch_sub(elements, std::memory_order_relaxed);
            }
            if (direct_packing && pack) {
                if (first && component.spectator_size && compact_spectator_preparation
                    && static_cast<std::size_t>(P * A * Q * B) > 64) {
                    const ssize_t K = component.spectator_size;
                    auto& conditional = scratch.conditional;
                    conditional.resize(transitions.size() * K);
                    environment_product(coefficients.data(), component.spectator_coefficients,
                        conditional.data(), transitions.size(), bond, K, empty);
                    auto& field = spectator_fields[global_n];
                    field.resize(static_cast<std::size_t>(K) * P * Q, 0.0);
                    for (std::size_t row : active_rows)
                        for (ssize_t k = 0; k < K; ++k)
                            field[(k * P + transitions[row].first) * Q + transitions[row].second]
                                = conditional[row * K + k];
                    direct_pair_factors.fetch_add(1, std::memory_order_relaxed);
                    compact_spectator_channels.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                auto& packed = first ? packed_left[global_n] : packed_right[global_n];
                const ssize_t rows = first ? A : C, columns = first ? B : D;
                packed.resize(static_cast<std::size_t>(output) * rows * input * columns, 0.0);
                for (std::size_t row : active_rows)
                    for (ssize_t a = 0; a < rows; ++a) {
                        const ssize_t offset = ((transitions[row].first * rows + a) * input
                                                + transitions[row].second) * columns;
                        if (compact_right) packed[offset + a] = effective[row * effective_block + a];
                        else std::copy_n(effective.data() + row * block + a * columns, columns,
                                         packed.data() + offset);
                    }
                direct_pair_factors.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            for (std::size_t row : active_rows) {
                const cdouble* values = effective.data() + row * effective_block;
                auto& group = first ? left_groups[static_cast<std::size_t>(global_n) * Q + transitions[row].second]
                                    : right_groups[static_cast<std::size_t>(global_n) * S + transitions[row].second];
                group.outputs.push_back(transitions[row].first);
                if (compact_right) {
                    const std::size_t begin = group.blocks.size();
                    group.blocks.resize(begin + block,0.);
                    for (ssize_t c = 0; c < C; ++c) group.blocks[begin + c * D + c] = values[c];
                } else group.blocks.insert(group.blocks.end(), values, values + block);
            }
        };
        build_side(true);
        build_side(false);
    };

    const int setup_threads = std::max(1, std::min(workers, static_cast<int>(N)));
    if (setup_threads == 1) {
        GroupWorkspace workspace;
        for (ssize_t n = 0; n < N; ++n) {
            build_groups_for_n(n, workspace);
        }
    } else {
        std::atomic<ssize_t> next_n(0);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(setup_threads));
        for (int worker = 0; worker < setup_threads; ++worker) {
            threads.emplace_back([&]() {
                GroupWorkspace workspace;
                while (true) {
                    const ssize_t n = next_n.fetch_add(1, std::memory_order_relaxed);
                    if (n >= N) {
                        return;
                    }
                    build_groups_for_n(n, workspace);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    mark_phase("effective_groups");
    // Dense physical transitions benefit from one BLAS call rather than Q
    // small calls and scatter-adds. Keep diagonal/sparse transitions sparse.
    for (ssize_t n = 0; n < N; ++n) {
        if (!packed_left[n].empty()) continue;
        std::size_t outputs = 0;
        for (ssize_t q = 0; q < Q; ++q)
            outputs += left_groups[static_cast<std::size_t>(n) * Q + q].outputs.size();
        if ((outputs * 2 < static_cast<std::size_t>(P * Q)
             && static_cast<std::size_t>(P * A * Q * B) > 64) || Q <= 1) continue;
        auto& packed = packed_left[static_cast<std::size_t>(n)];
        packed.resize(static_cast<std::size_t>(P * A * Q * B), 0.0);
        for (ssize_t q = 0; q < Q; ++q) {
            auto& group = left_groups[static_cast<std::size_t>(n) * Q + q];
            for (std::size_t lp = 0; lp < group.outputs.size(); ++lp) {
                const ssize_t p = group.outputs[lp];
                for (ssize_t a = 0; a < A; ++a)
                    std::copy_n(group.blocks.data() + (lp * A + a) * B, B,
                        packed.data() + ((p * A + a) * Q + q) * B);
            }
            std::vector<cdouble>().swap(group.blocks);
            std::vector<ssize_t>().swap(group.outputs);
        }
    }
    bool has_packed_right = false;
    for (ssize_t n = 0; n < N; ++n) {
        if (!early_right_values[n].empty()) continue;
        if (!packed_right[n].empty()) { has_packed_right = true; continue; }
        std::size_t outputs = 0;
        for (ssize_t s = 0; s < S; ++s)
            outputs += right_groups[static_cast<std::size_t>(n) * S + s].outputs.size();
        const bool small_prefix = static_cast<std::size_t>(P * A * Q * B) <= 64
            && static_cast<std::size_t>(R * C * S * D) <= 1024 * 1024;
        if ((outputs * 2 < static_cast<std::size_t>(R * S) && !small_prefix) || S <= 1) continue;
        has_packed_right = true;
        auto& packed = packed_right[static_cast<std::size_t>(n)];
        packed.resize(static_cast<std::size_t>(R * C * S * D), 0.0);
        for (ssize_t s = 0; s < S; ++s) {
            auto& group = right_groups[static_cast<std::size_t>(n) * S + s];
            for (std::size_t rr = 0; rr < group.outputs.size(); ++rr) {
                const ssize_t r = group.outputs[rr];
                for (ssize_t c = 0; c < C; ++c)
                    std::copy_n(group.blocks.data() + (rr * C + c) * D, D,
                        packed.data() + ((r * C + c) * S + s) * D);
            }
            std::vector<cdouble>().swap(group.blocks);
            std::vector<ssize_t>().swap(group.outputs);
        }
    }
    mark_phase("packing");
    std::vector<char> spectator_selected(N, false);
    for (ssize_t n = 0; n < N; ++n)
        spectator_selected[n] = component_for_n[n]->spectator_size
            && (!packed_left[n].empty() || !spectator_fields[n].empty());
    // Matrix-unit electronic prefixes recur across nuclear channel terms.
    // Factor an exactly common left action before applying it to the state.
    if (static_cast<std::size_t>(P * A * Q * B) <= 64 && N <= 128) {
        std::vector<ssize_t> representatives;
        for (ssize_t n = 0; n < N; ++n) {
            auto& left = packed_left[static_cast<std::size_t>(n)];
            auto& right = packed_right[static_cast<std::size_t>(n)];
            if (left.empty() || right.empty()) continue;
            bool merged = false;
            for (const ssize_t previous : representatives) {
                if (left != packed_left[static_cast<std::size_t>(previous)]) continue;
                shared_pair_channels.fetch_add(1, std::memory_order_relaxed);
                auto& target = packed_right[static_cast<std::size_t>(previous)];
                for (std::size_t i = 0; i < right.size(); ++i) target[i] += right[i];
                std::vector<cdouble>().swap(left);
                std::vector<cdouble>().swap(right);
                merged = true;
                break;
            }
            if (!merged) representatives.push_back(n);
        }
    }
    std::vector<cdouble> diagonal_factors;
    if (P == Q && R == S
        && static_cast<std::size_t>(P) * R * A * B * C * D
            <= (32ULL << 20) / sizeof(cdouble)) {
        std::vector<ssize_t> diagonal;
        for (ssize_t n = 0; n < N; ++n) {
            if (spectator_selected[n]) continue;
            bool active = true;
            auto diagonal_side = [&](bool first) {
                if (!first && !early_right_values[n].empty()) return false;
                const auto& packed = first ? packed_left[n] : packed_right[n];
                const ssize_t physical = first ? Q : S;
                const ssize_t rows = first ? A : C, columns = first ? B : D;
                if (!packed.empty()) {
                    for (ssize_t p = 0; p < physical; ++p)
                        for (ssize_t q = 0; q < physical; ++q) if (p != q)
                            for (ssize_t a = 0; a < rows; ++a)
                                for (ssize_t b = 0; b < columns; ++b)
                                    if (packed[((p * rows + a) * physical + q) * columns + b]
                                        != cdouble(0.0, 0.0)) return false;
                } else for (ssize_t q = 0; q < physical; ++q) {
                    const auto& group = first ? left_groups[n * Q + q] : right_groups[n * S + q];
                    for (ssize_t p : group.outputs) if (p != q) return false;
                }
                return true;
            };
            active = diagonal_side(true) && diagonal_side(false);
            if (active) diagonal.push_back(n);
        }
        const ssize_t count = static_cast<ssize_t>(diagonal.size());
        const bool compact_boundary = point_batch && C == 1 && D == 1;
        const ssize_t threshold = compact_boundary ? 4 * S : 4 * A * B;
        const long double preparation_bytes = sizeof(cdouble) * static_cast<long double>(count)
            * (P * A * B + R * C * D);
        if (count > threshold && (!compact_boundary || preparation_bytes <= (128ULL << 20))) {
            diagonal_pair_channels.fetch_add(count, std::memory_order_relaxed);
            if (compact_boundary) compact_boundary_diagonal_channels.fetch_add(count, std::memory_order_relaxed);
            // Physical projectors stay diagonal. Expand only the small left
            // state-bond matrices in E_ab and aggregate their right factors.
            diagonal_factors.resize(static_cast<std::size_t>(P) * R * A * B * C * D);
            std::vector<cdouble> left(static_cast<std::size_t>(P) * A * B * count, 0.0);
            std::vector<cdouble> right(static_cast<std::size_t>(R) * count * C * D, 0.0);
            for (ssize_t k = 0; k < count; ++k) {
                const ssize_t n = diagonal[k];
                for (ssize_t p = 0; p < P; ++p) {
                    const auto& packed = packed_left[n];
                    const auto& group = left_groups[n * Q + p];
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t b = 0; b < B; ++b)
                            left[((p * A + a) * B + b) * count + k] = !packed.empty()
                                ? packed[((p * A + a) * Q + p) * B + b]
                                : (group.blocks.empty() ? cdouble(0.0, 0.0) : group.blocks[a * B + b]);
                }
                for (ssize_t r = 0; r < R; ++r) {
                    const auto& packed = packed_right[n];
                    const auto& group = right_groups[n * S + r];
                    for (ssize_t c = 0; c < C; ++c)
                        for (ssize_t d = 0; d < D; ++d)
                            right[((r * count + k) * C + c) * D + d] = !packed.empty()
                                ? packed[((r * C + c) * S + r) * D + d]
                                : (group.blocks.empty() ? cdouble(0.0, 0.0) : group.blocks[c * D + d]);
                }
                std::vector<cdouble>().swap(packed_left[n]);
                std::vector<cdouble>().swap(packed_right[n]);
                for (ssize_t q = 0; q < Q; ++q) left_groups[n * Q + q] = {};
                for (ssize_t s = 0; s < S; ++s) right_groups[n * S + s] = {};
            }
            std::vector<cdouble> factor(static_cast<std::size_t>(A) * B * C * D);
            const cdouble alpha(1.0, 0.0), beta(0.0, 0.0);
            for (ssize_t p = 0; p < P; ++p)
                for (ssize_t r = 0; r < R; ++r) {
                    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(A * B), static_cast<int>(C * D), static_cast<int>(count),
                        &alpha, left.data() + p * A * B * count, static_cast<int>(count),
                        right.data() + r * count * C * D, static_cast<int>(C * D),
                        &beta, factor.data(), static_cast<int>(C * D));
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t b = 0; b < B; ++b)
                            for (ssize_t c = 0; c < C; ++c)
                                for (ssize_t d = 0; d < D; ++d)
                                    diagonal_factors[((((p * R + r) * A + a) * C + c) * B + b) * D + d]
                                        = factor[((a * B + b) * C + c) * D + d];
                }
        }
    }
    if (C == 1 && D == 1 && R == S) {
        std::vector<ssize_t> diagonal;
        for (ssize_t n = 0; n < N; ++n) {
            if (component_for_n[n]->prefix_bra || spectator_selected[n]) continue;
            bool left_active = !packed_left[n].empty();
            for (ssize_t q = 0; q < Q && !left_active; ++q)
                left_active = !left_groups[n * Q + q].outputs.empty();
            if (!left_active) continue;
            bool is_diagonal = true;
            const auto& packed = packed_right[n];
            if (!packed.empty()) {
                for (ssize_t r = 0; r < R; ++r)
                    for (ssize_t s = 0; s < S; ++s)
                        if (r != s && packed[r * S + s] != cdouble(0.0, 0.0))
                            is_diagonal = false;
            } else for (ssize_t s = 0; s < S; ++s)
                for (ssize_t r : right_groups[n * S + s].outputs)
                    if (r != s) is_diagonal = false;
            if (is_diagonal && point_batch && P == Q && packed_left[n].empty()) {
                bool left_diagonal = true;
                for (ssize_t q = 0; q < Q && left_diagonal; ++q)
                    for (ssize_t p : left_groups[n * Q + q].outputs)
                        if (p != q) { left_diagonal = false; break; }
                if (left_diagonal) continue;
            }
            if (is_diagonal) diagonal.push_back(n);
        }
        const ssize_t count = static_cast<ssize_t>(diagonal.size());
        const ssize_t left_size = P * A * Q * B;
        if (count > 4 * S && static_cast<std::size_t>(S) * left_size
            <= (32ULL << 20) / sizeof(cdouble)) {
            compressed_boundary_channels.fetch_add(count - S, std::memory_order_relaxed);
            // Only diagonal right factors are expanded in E_ss. The remaining
            // channels stay factorized in their existing physical groups.
            std::vector<cdouble> coefficients(static_cast<std::size_t>(S) * count, 0.0);
            for (ssize_t k = 0; k < count; ++k) {
                const ssize_t n = diagonal[k];
                for (ssize_t s = 0; s < S; ++s) {
                    const auto& packed = packed_right[n];
                    const auto& group = right_groups[n * S + s];
                    coefficients[s * count + k] = !packed.empty() ? packed[s * S + s]
                        : (group.blocks.empty() ? cdouble(0.0, 0.0) : group.blocks[0]);
                }
            }
            std::vector<cdouble> combined(static_cast<std::size_t>(S) * left_size, 0.0);
            constexpr ssize_t batch = 64;
            std::vector<cdouble> factors(static_cast<std::size_t>(batch) * left_size);
            const cdouble alpha(1.0, 0.0);
            for (ssize_t begin = 0; begin < count; begin += batch) {
                const ssize_t size = std::min(batch, count - begin);
                std::fill(factors.begin(), factors.end(), cdouble(0.0, 0.0));
                for (ssize_t k = 0; k < size; ++k) {
                    const ssize_t n = diagonal[begin + k];
                    cdouble* target = factors.data() + k * left_size;
                    const auto& packed = packed_left[n];
                    if (!packed.empty()) std::copy(packed.begin(), packed.end(), target);
                    else for (ssize_t q = 0; q < Q; ++q) {
                        const auto& group = left_groups[n * Q + q];
                        for (std::size_t j = 0; j < group.outputs.size(); ++j)
                            for (ssize_t a = 0; a < A; ++a)
                                std::copy_n(group.blocks.data() + (j * A + a) * B, B,
                                    target + ((group.outputs[j] * A + a) * Q + q) * B);
                    }
                    std::vector<cdouble>().swap(packed_left[n]);
                    std::vector<cdouble>().swap(packed_right[n]);
                    for (ssize_t q = 0; q < Q; ++q) left_groups[n * Q + q] = {};
                    for (ssize_t s = 0; s < S; ++s) right_groups[n * S + s] = {};
                }
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(S), static_cast<int>(left_size), static_cast<int>(size),
                    &alpha, coefficients.data() + begin, static_cast<int>(count),
                    factors.data(), static_cast<int>(left_size), &alpha,
                    combined.data(), static_cast<int>(left_size));
            }
            for (ssize_t s = 0; s < S; ++s) {
                const ssize_t n = diagonal[s];
                packed_left[n].assign(combined.data() + s * left_size,
                                      combined.data() + (s + 1) * left_size);
                auto& group = right_groups[n * S + s];
                group.outputs.push_back(s);
                group.blocks.push_back(cdouble(1.0, 0.0));
            }
        }
    }
    std::vector<ssize_t> prefix_component(N, -1);
    std::vector<std::vector<cdouble>> prefix_physical(N);
    for (ssize_t n = 0; n < N; ++n) {
        const auto& component = *component_for_n[n];
        if (!component.prefix_bra || component.M != 1) continue;
        auto& physical = prefix_physical[n];
        physical.assign(static_cast<std::size_t>(P) * Q, 0.0);
        const auto& packed = packed_left[n];
        if (!packed.empty()) {
            for (ssize_t p = 0; p < P; ++p)
                for (ssize_t q = 0; q < Q; ++q) {
                    bool active = false;
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t b = 0; b < B; ++b)
                            active = active || packed[((p * A + a) * Q + q) * B + b] != cdouble(0.0, 0.0);
                    if (active) physical[p * Q + q] = component.first_coefficient(0, local_n[n], p, q, Q);
                }
        } else for (ssize_t q = 0; q < Q; ++q)
            for (ssize_t p : left_groups[n * Q + q].outputs)
                physical[p * Q + q] = component.first_coefficient(0, local_n[n], p, q, Q);
        prefix_component[n] = &component - sum_components->data();
    }
    ssize_t remaining = 0;
    for (ssize_t n = 0; n < N; ++n) {
        bool left_active = !packed_left[n].empty() || !spectator_fields[n].empty(), right_active = !packed_right[n].empty() || !early_right_values[n].empty();
        for (ssize_t q = 0; q < Q && !left_active; ++q)
            left_active = !left_groups[n * Q + q].outputs.empty();
        for (ssize_t s = 0; s < S && !right_active; ++s)
            right_active = !right_groups[n * S + s].outputs.empty();
        if (!left_active || !right_active) continue;
        if (remaining != n) {
            component_for_n[remaining] = component_for_n[n];
            local_n[remaining] = local_n[n];
            spectator_selected[remaining] = spectator_selected[n];
            spectator_fields[remaining] = std::move(spectator_fields[n]);
            diagonal_spectator_fields[remaining] = diagonal_spectator_fields[n];
            prefix_component[remaining] = prefix_component[n];
            prefix_physical[remaining] = std::move(prefix_physical[n]);
            packed_left[remaining] = std::move(packed_left[n]);
            packed_right[remaining] = std::move(packed_right[n]);
            early_right_values[remaining] = std::move(early_right_values[n]);
            for (ssize_t q = 0; q < Q; ++q)
                left_groups[remaining * Q + q] = std::move(left_groups[n * Q + q]);
            for (ssize_t s = 0; s < S; ++s)
                right_groups[remaining * S + s] = std::move(right_groups[n * S + s]);
        }
        ++remaining;
    }
    N = remaining;
    spectator_fields.resize(N);
    diagonal_spectator_fields.resize(N);
    prefix_component.resize(N);
    prefix_physical.resize(N);
    for (ssize_t n = 0; n < N; ++n)
        if (prefix_component[n] >= 0)
            factorized_prefix_channels.fetch_add(1, std::memory_order_relaxed);
    packed_left.resize(N);
    packed_right.resize(N);
    early_right_values.resize(N);
    left_groups.resize(N * Q);
    right_groups.resize(N * S);
    mark_phase("aggregation");
    auto real_values = [](const std::vector<cdouble>& values) {
        std::vector<double> real;
        if (values.empty() || !std::all_of(values.begin(), values.end(),
                [](cdouble z) { return z.imag() == 0.0; })) return real;
        real.reserve(values.size());
        for (cdouble z : values) real.push_back(z.real());
        return real;
    };
    std::vector<std::vector<double>> real_right(N), real_physical(N);
    std::vector<std::vector<double>> split_right(N);
    std::vector<std::vector<double>> split_left(N);
    std::vector<std::vector<cdouble>> prefix_diagonal(N);
    struct DiagonalRight {
        std::vector<cdouble> values;
        std::vector<std::pair<ssize_t, ssize_t>> transitions;
    };
    std::vector<DiagonalRight> diagonal_right(N);
    const bool split_sparse = !std::getenv("PYQED_TDVP_COMPLEX_SPARSE_GROUPS");
    auto split_group = [](TwoSitePhysicalGroup& group) {
        const std::size_t count = group.blocks.size();
        const bool real = std::all_of(group.blocks.begin(), group.blocks.end(),
            [](cdouble value) { return value.imag() == 0; });
        group.split_blocks.resize(count * (real ? 1 : 2));
        for (std::size_t i = 0; i < count; ++i) {
            group.split_blocks[i] = group.blocks[i].real();
            if (!real) group.split_blocks[count + i] = group.blocks[i].imag();
        }
        std::vector<cdouble>().swap(group.blocks);
    };
    bool has_diagonal_right = false;
    std::vector<std::vector<double>> real_spectator_fields(N);
    std::vector<char> spectator_components(sum_components->size(), false);
    for (ssize_t n = 0; n < N; ++n) {
        const auto& component = *component_for_n[n];
        if (spectator_selected[n]) {
                const auto& packed = packed_left[n];
                if (packed.empty() && spectator_fields[n].empty()) continue;
                auto& field = spectator_fields[n];
                spectator_components[&component - sum_components->data()] = true;
                spectator_frame_channels.fetch_add(1, std::memory_order_relaxed);
                if (field.empty()) {
                field.resize(component.spectator_size * P * Q, 0.0);
                for (ssize_t p = 0; p < P; ++p)
                    for (ssize_t q = 0; q < Q; ++q) {
                        bool active = false;
                        for (ssize_t a = 0; a < A && !active; ++a)
                            for (ssize_t b = 0; b < B; ++b)
                                if (packed[((p * A + a) * Q + q) * B + b] != cdouble(0.0, 0.0)) {
                                    active = true; break;
                                }
                        if (!active) continue;
                        for (ssize_t k = 0; k < component.spectator_size; ++k)
                            for (ssize_t m = 0; m < component.M; ++m)
                                field[(k * P + p) * Q + q] +=
                                    component.spectator_coefficients[m * component.spectator_size + k]
                                    * component.first_coefficient(m, local_n[n], p, q, P);
                    }
                }
                std::vector<cdouble>().swap(packed_left[n]);
                real_spectator_fields[n] = real_values(field);
        }
    }
    for (ssize_t n = 0; n < N; ++n) {
        if (!early_right_values[n].empty()) {
            auto& factor = diagonal_right[n];
            factor.values = std::move(early_right_values[n]);
            for (ssize_t r = 0; r < R; ++r)
                for (ssize_t s = 0; s < S; ++s) {
                    const auto* values = factor.values.data() + (r * S + s) * C;
                    if (std::any_of(values,values + C,[](cdouble z) { return z != cdouble(0.,0.); }))
                        factor.transitions.emplace_back(r,s);
                }
            has_diagonal_right = true;
            diagonal_right_states.fetch_add(1,std::memory_order_relaxed);
        } else if (C == D && C > 1 && !std::getenv("PYQED_TDVP_DENSE_RIGHT_STATE")) {
            bool diagonal = true, active = false;
            const auto& packed = packed_right[n];
            auto check = [&](const cdouble* block) {
                active = true;
                for (ssize_t c = 0; c < C && diagonal; ++c)
                    for (ssize_t d = 0; d < D; ++d)
                        if (c != d && block[c * D + d] != cdouble(0.0, 0.0)) { diagonal = false; break; }
            };
            if (!packed.empty()) {
                active = true;
                for (ssize_t r = 0; r < R && diagonal; ++r)
                    for (ssize_t s = 0; s < S && diagonal; ++s)
                        for (ssize_t c = 0; c < C && diagonal; ++c)
                            for (ssize_t d = 0; d < D; ++d)
                                if (c != d && packed[(r * C + c) * S * D + s * D + d] != cdouble(0.0, 0.0)) {
                                    diagonal = false; break;
                                }
            } else for (ssize_t s = 0; s < S && diagonal; ++s) {
                const auto& group = right_groups[n * S + s];
                for (std::size_t j = 0; j < group.outputs.size() && diagonal; ++j)
                    check(group.blocks.data() + j * C * D);
            }
            if (diagonal && active) {
                auto& factor = diagonal_right[n];
                factor.values.resize(static_cast<std::size_t>(R) * S * C, 0.0);
                if (!packed.empty()) {
                    for (ssize_t r = 0; r < R; ++r)
                        for (ssize_t s = 0; s < S; ++s)
                            for (ssize_t c = 0; c < C; ++c)
                                factor.values[(r * S + s) * C + c] = packed[(r * C + c) * S * D + s * D + c];
                } else for (ssize_t s = 0; s < S; ++s) {
                    const auto& group = right_groups[n * S + s];
                    for (std::size_t j = 0; j < group.outputs.size(); ++j)
                        for (ssize_t c = 0; c < C; ++c)
                            factor.values[(group.outputs[j] * S + s) * C + c] = group.blocks[(j * C + c) * D + c];
                }
                for (ssize_t r = 0; r < R; ++r)
                    for (ssize_t s = 0; s < S; ++s) {
                        const auto* values = factor.values.data() + (r * S + s) * C;
                        if (std::any_of(values, values + C, [](cdouble z) { return z != cdouble(0.0, 0.0); }))
                            factor.transitions.emplace_back(r, s);
                    }
                std::vector<cdouble>().swap(packed_right[n]);
                for (ssize_t s = 0; s < S; ++s) right_groups[n * S + s] = {};
                has_diagonal_right = true;
                diagonal_right_states.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (prefix_component[n] >= 0 && P == Q
            && !std::getenv("PYQED_TDVP_DENSE_PREFIX_DIAGONAL")) {
            bool diagonal = true;
            for (ssize_t p = 0; p < P && diagonal; ++p)
                for (ssize_t q = 0; q < Q; ++q)
                    if (p != q && prefix_physical[n][p * Q + q] != cdouble(0.0, 0.0)) {
                        diagonal = false;
                        break;
                    }
            if (diagonal) {
                prefix_diagonal[n].resize(P);
                for (ssize_t p = 0; p < P; ++p)
                    prefix_diagonal[n][p] = prefix_physical[n][p * Q + p];
                diagonal_prefix_factors.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (prefix_component[n] < 0 && packed_left[n].size() >= 16 && S * D >= 4
            && !std::getenv("PYQED_TDVP_COMPLEX_LEFT_BLAS")) {
            const auto count = packed_left[n].size();
            const bool real = std::all_of(packed_left[n].begin(), packed_left[n].end(),
                                         [](cdouble value) { return value.imag() == 0.0; });
            auto& parts = split_left[n];
            parts.resize(real ? count : 2 * count);
            for (std::size_t i = 0; i < count; ++i) {
                parts[i] = packed_left[n][i].real();
                if (!real) parts[count + i] = packed_left[n][i].imag();
            }
            std::vector<cdouble>().swap(packed_left[n]);
            split_left_pair_factors.fetch_add(1, std::memory_order_relaxed);
        }
        real_physical[n] = real_values(prefix_physical[n]);
        if (!real_physical[n].empty())
            real_pair_factors.fetch_add(1, std::memory_order_relaxed);
        real_right[n] = real_values(packed_right[n]);
        if (!real_right[n].empty()) {
            std::vector<cdouble>().swap(packed_right[n]);
            real_pair_factors.fetch_add(1, std::memory_order_relaxed);
        } else if (packed_right[n].size() >= 64 && P >= 2
                   && !std::getenv("PYQED_TDVP_COMPLEX_BLAS")) {
            const auto count = packed_right[n].size();
            auto& parts = split_right[n];
            parts.resize(2 * count);
            for (std::size_t i = 0; i < count; ++i) {
                parts[i] = packed_right[n][i].real();
                parts[count + i] = packed_right[n][i].imag();
            }
            std::vector<cdouble>().swap(packed_right[n]);
            split_complex_pair_factors.fetch_add(1, std::memory_order_relaxed);
        }
        for (ssize_t s = 0; s < S; ++s) {
            auto& group = right_groups[n * S + s];
            group.real_blocks = real_values(group.blocks);
            if (!group.real_blocks.empty()) {
                std::vector<cdouble>().swap(group.blocks);
                real_pair_factors.fetch_add(1, std::memory_order_relaxed);
            } else if (split_sparse && group.blocks.size() >= 64 && D >= 8
                       && (prefix_component[n] >= 0 ? P : A * P) >= 8) {
                split_group(group);
                split_sparse_right_factors.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (split_sparse && prefix_component[n] < 0 && S * D >= 8)
            for (ssize_t q = 0; q < Q; ++q) {
                auto& group = left_groups[n * Q + q];
                if (group.blocks.size() < 64) continue;
                split_group(group);
                split_sparse_left_factors.fetch_add(1, std::memory_order_relaxed);
            }
    }
    std::size_t max_left_rows = 0;
    std::size_t max_right_rows = has_packed_right ? static_cast<std::size_t>(R * C) : 0;
    if (has_diagonal_right) max_right_rows = R * C;
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
    mark_phase("action_factors");
    const ssize_t AP = A * P;
    const ssize_t SD = S * D;
    const ssize_t PRC = P * R * C;
    const char* shared_frames_flag = std::getenv("PYQED_TDVP_SHARED_PAIR_FRAMES");
    const bool shared_frames = shared_frames_flag && shared_frames_flag[0] == '1';
    std::vector<std::size_t> spectator_offsets(sum_components->size() + 1, 0);
    std::vector<std::size_t> ket_owner(sum_components->size()), bra_owner(sum_components->size());
    std::size_t spectator_storage = 0;
    for (std::size_t k = 0; k < sum_components->size(); ++k) {
        ket_owner[k] = bra_owner[k] = k;
        const auto& component = (*sum_components)[k];
        if (!spectator_components[k]) continue;
        if (shared_frames) for (std::size_t j = 0; j < k; ++j) {
            const auto& previous = (*sum_components)[j];
            if (!spectator_components[j] || previous.spectator_size != component.spectator_size) continue;
            if (ket_owner[k] == k && std::equal(component.spectator_ket,
                    component.spectator_ket + component.spectator_size * B, previous.spectator_ket))
                ket_owner[k] = ket_owner[j];
            if (bra_owner[k] == k && std::equal(component.spectator_bra,
                    component.spectator_bra + component.spectator_size * A, previous.spectator_bra))
                bra_owner[k] = bra_owner[j];
        }
        if (ket_owner[k] != k) shared_pair_ket_frames.fetch_add(1, std::memory_order_relaxed);
        if (bra_owner[k] != k) {
            spectator_offsets[k] = spectator_offsets[bra_owner[k]];
            shared_pair_bra_frames.fetch_add(1, std::memory_order_relaxed);
        } else {
            spectator_offsets[k] = spectator_storage;
            spectator_storage += component.spectator_size * static_cast<std::size_t>(PRC);
        }
    }
    spectator_offsets.back() = spectator_storage;
    const bool defer_spectator = !std::getenv("PYQED_TDVP_SEPARATE_SPECTATOR_OUTPUT")
        && spectator_offsets.back() * sizeof(cdouble) <= (8ULL << 20);
    struct PrefixBatch {
        ssize_t component;
        std::vector<ssize_t> channels;
        std::vector<double> real;
        std::vector<cdouble> complex;
        std::vector<cdouble> diagonal;
        std::vector<cdouble> right_diagonal;
        std::vector<double> right_parts;
    };
    std::vector<PrefixBatch> prefix_batches;
    std::vector<ssize_t> channel_batch(N, -1);
    std::size_t batch_workspace = 0;
    const char* batch_prefix = std::getenv("PYQED_TDVP_BATCH_PREFIX");
    const char* diagonal_batch = std::getenv("PYQED_TDVP_BATCH_DIAGONAL_RIGHT");
    const bool batch_diagonal_right = !diagonal_batch || diagonal_batch[0] != '0';
    if (!batch_prefix || batch_prefix[0] != '0') {
        const char* wide_prefix = std::getenv("PYQED_TDVP_WIDE_PREFIX_BATCH");
        const std::size_t channel_limit = wide_prefix && wide_prefix[0] == '1' ? 64 : 16;
        const std::size_t limit = std::min<std::size_t>(channel_limit, (8ULL << 20) / sizeof(cdouble) / P / (SD + (batch_diagonal_right ? S : 0)));
        if (limit >= 4) for (ssize_t component = 0; component < static_cast<ssize_t>(sum_components->size()); ++component)
            for (bool real : {false, true})
                for (int left_diagonal = 0; left_diagonal < (batch_diagonal_right ? 2 : 1); ++left_diagonal)
                    for (int diagonal_group = 0; diagonal_group < (batch_diagonal_right ? 2 : 1); ++diagonal_group) {
                std::vector<ssize_t> channels;
                for (ssize_t n = 0; n < N; ++n)
                    if (prefix_component[n] == component && prefix_diagonal[n].empty() == (left_diagonal == 0)
                        && (!left_diagonal || !diagonal_right[n].values.empty())
                        && (!batch_diagonal_right || (!diagonal_right[n].values.empty()) == (diagonal_group == 1))
                        && (!real_physical[n].empty()) == real) channels.push_back(n);
                for (std::size_t start = 0; start + 4 <= channels.size(); start += limit) {
                    const std::size_t count = std::min(limit, channels.size() - start);
                    PrefixBatch batch;
                    batch.component = component;
                    for (std::size_t i = 0; i < count; ++i) {
                        const ssize_t n = channels[start + i];
                        batch.channels.push_back(n);
                        channel_batch[n] = prefix_batches.size();
                        if (left_diagonal) batch.diagonal.insert(batch.diagonal.end(), prefix_diagonal[n].begin(), prefix_diagonal[n].end());
                        else if (real) batch.real.insert(batch.real.end(), real_physical[n].begin(), real_physical[n].end());
                        else batch.complex.insert(batch.complex.end(), prefix_physical[n].begin(), prefix_physical[n].end());
                    }
                    if (batch_diagonal_right && diagonal_group == 1) {
                        batch.right_diagonal.resize(static_cast<std::size_t>(C * R * S) * count);
                        for (ssize_t c = 0; c < C; ++c)
                            for (ssize_t r = 0; r < R; ++r)
                                for (std::size_t i = 0; i < count; ++i)
                                    for (ssize_t s = 0; s < S; ++s)
                                        batch.right_diagonal[((c * R + r) * count + i) * S + s] =
                                            diagonal_right[batch.channels[i]].values[(r * S + s) * C + c];
                        if (!std::getenv("PYQED_TDVP_COMPLEX_DIAGONAL_BATCH")) {
                            const std::size_t size = batch.right_diagonal.size();
                            const bool real = std::all_of(batch.right_diagonal.begin(), batch.right_diagonal.end(),
                                [](cdouble value) { return value.imag() == 0.0; });
                            batch.right_parts.resize((real ? 1 : 2) * size);
                            for (std::size_t i = 0; i < size; ++i) {
                                batch.right_parts[i] = batch.right_diagonal[i].real();
                                if (!real) batch.right_parts[size + i] = batch.right_diagonal[i].imag();
                            }
                            std::vector<cdouble>().swap(batch.right_diagonal);
                        }
                    }
                    batch_workspace = std::max(batch_workspace, count * static_cast<std::size_t>(P * SD));
                    batched_prefix_channels.fetch_add(count, std::memory_order_relaxed);
                    prefix_batches.push_back(std::move(batch));
                }
            }
    }
    std::vector<const cdouble*> output_bras;
    std::vector<ssize_t> output_group(sum_components->size(), -1);
    if (A > 1 && !std::getenv("PYQED_TDVP_EXPANDED_PREFIX_OUTPUT")) {
        for (ssize_t n = 0; n < N; ++n) {
            const ssize_t component = prefix_component[n];
            if (component < 0 || output_group[component] >= 0) continue;
            const cdouble* bra = (*sum_components)[component].prefix_bra;
            auto found = std::find_if(output_bras.begin(), output_bras.end(),
                [&](const cdouble* candidate) { return std::equal(bra, bra + A, candidate); });
            output_group[component] = found - output_bras.begin();
            if (found == output_bras.end()) output_bras.push_back(bra);
        }
        if (output_bras.size() * static_cast<std::size_t>(PRC) > (8ULL << 20) / sizeof(cdouble)) {
            output_bras.clear();
            std::fill(output_group.begin(), output_group.end(), -1);
        } else for (ssize_t n = 0; n < N; ++n)
            if (prefix_component[n] >= 0) shared_prefix_outputs.fetch_add(1, std::memory_order_relaxed);
    }
    for (const auto& batch : prefix_batches)
        if ((!batch.right_diagonal.empty() || !batch.right_parts.empty()) && output_group[batch.component] >= 0)
            batched_diagonal_channels.fetch_add(batch.channels.size(), std::memory_order_relaxed);
    struct PrefixField {
        ssize_t component;
        std::vector<ssize_t> channels;
        std::vector<cdouble> values;
        bool state_diagonal = true, physical_diagonal = false;
    };
    std::vector<PrefixField> prefix_fields;
    std::vector<ssize_t> channel_field(N, -1);
    const char* prefix_field_mode = std::getenv("PYQED_TDVP_PREFIX_FIELDS");
    if (prefix_field_mode && prefix_field_mode[0] == '1' && P == Q && C == D) {
        std::size_t field_elements = 0;
        for (ssize_t component = 0; component < static_cast<ssize_t>(sum_components->size()); ++component) {
            if (output_group[component] < 0) continue;
            PrefixField field;
            field.component = component;
            for (ssize_t n = 0; n < N; ++n)
                if (prefix_component[n] == component && !prefix_diagonal[n].empty()
                        && !diagonal_right[n].values.empty()) field.channels.push_back(n);
            const ssize_t count = field.channels.size();
            const std::size_t size = static_cast<std::size_t>(P) * C * R * S;
            const long double proposed = field_elements + static_cast<long double>(size)
                + static_cast<long double>(count) * (P + C * R * S);
            if (count < 4 || proposed * sizeof(cdouble) > (128ULL << 20)) continue;
            std::vector<cdouble> coefficients(static_cast<std::size_t>(P) * count);
            std::vector<cdouble> right(static_cast<std::size_t>(count) * C * R * S);
            for (ssize_t i = 0; i < count; ++i) {
                const ssize_t n = field.channels[i];
                for (ssize_t p = 0; p < P; ++p) coefficients[p * count + i] = prefix_diagonal[n][p];
                for (ssize_t c = 0; c < C; ++c)
                    for (ssize_t r = 0; r < R; ++r)
                        for (ssize_t s = 0; s < S; ++s)
                            right[((i * C + c) * R + r) * S + s] = diagonal_right[n].values[(r * S + s) * C + c];
                channel_field[n] = prefix_fields.size();
            }
            field.values.resize(size);
            const std::vector<cdouble> no_diagonal;
            environment_product(coefficients.data(), right.data(), field.values.data(),
                P, count, C * R * S, no_diagonal);
            field_elements += size;
            prefix_field_channels.fetch_add(count, std::memory_order_relaxed);
            prefix_fields.push_back(std::move(field));
        }
    }
    const char* coordinate_field_mode = std::getenv("PYQED_TDVP_COORDINATE_PREFIX_FIELDS");
    if (coordinate_field_mode && coordinate_field_mode[0] == '1' && P == Q) {
        auto physically_diagonal = [&](ssize_t n) {
            if (R != S) return false;
            if (!diagonal_right[n].values.empty()) {
                for (ssize_t r = 0; r < R; ++r)
                    for (ssize_t s = 0; s < S; ++s) if (r != s)
                        for (ssize_t c = 0; c < C; ++c)
                            if (diagonal_right[n].values[(r * S + s) * C + c] != cdouble(0.,0.)) return false;
            } else if (!packed_right[n].empty() || !real_right[n].empty() || !split_right[n].empty()) {
                for (ssize_t r = 0; r < R; ++r)
                    for (ssize_t s = 0; s < S; ++s) if (r != s)
                        for (ssize_t c = 0; c < C; ++c)
                            for (ssize_t d = 0; d < D; ++d) {
                                const ssize_t i = ((r * C + c) * S + s) * D + d;
                                if ((!packed_right[n].empty() && packed_right[n][i] != cdouble(0.,0.))
                                    || (!real_right[n].empty() && real_right[n][i] != 0.)
                                    || (!split_right[n].empty() && (split_right[n][i] != 0. || split_right[n][R * C * S * D + i] != 0.))) return false;
                            }
            } else for (ssize_t s = 0; s < S; ++s)
                for (ssize_t r : right_groups[n * S + s].outputs) if (r != s) return false;
            return true;
        };
        std::vector<PrefixField> candidates;
        for (ssize_t n = 0; n < N; ++n) {
            const ssize_t component = prefix_component[n];
            if (component < 0 || output_group[component] < 0 || channel_field[n] >= 0 || prefix_diagonal[n].empty()) continue;
            const bool point = physically_diagonal(n);
            auto found = std::find_if(candidates.begin(), candidates.end(), [&](const PrefixField& field) {
                return field.physical_diagonal == point && output_group[field.component] == output_group[component]
                    && std::equal((*sum_components)[component].prefix_ket, (*sum_components)[component].prefix_ket + B,
                                  (*sum_components)[field.component].prefix_ket);
            });
            if (found == candidates.end()) {
                candidates.push_back({component, {}, {}, false, point});
                found = candidates.end() - 1;
            }
            found->channels.push_back(n);
        }
        std::size_t resident = 0;
        for (const auto& field : prefix_fields) resident += field.values.size();
        const std::size_t limit = (128ULL << 20) / sizeof(cdouble);
        auto group_value = [](const TwoSitePhysicalGroup& group, ssize_t i) {
            if (!group.blocks.empty()) return group.blocks[i];
            if (!group.real_blocks.empty()) return cdouble(group.real_blocks[i],0.);
            return cdouble(group.split_blocks[i],group.split_blocks[group.split_blocks.size() / 2 + i]);
        };
        // Compact coordinate-diagonal fields get first use of the shared cap.
        for (bool point : {true, false}) for (auto& field : candidates) {
            if (field.physical_diagonal != point || field.channels.size() < 4) continue;
            const std::size_t stride = static_cast<std::size_t>(R) * C * (point ? D : S * D);
            const std::size_t size = static_cast<std::size_t>(P) * stride;
            if (resident >= limit || size >= limit - resident) continue;
            const std::size_t tile = std::min<std::size_t>({16,field.channels.size(),(limit - resident - size) / (2 * P + stride)});
            if (!tile) continue;
            field.values.resize(size,0.);
            std::vector<cdouble> coefficients(P * tile), right(tile * stride);
            for (std::size_t begin = 0; begin < field.channels.size(); begin += tile) {
                const ssize_t count = std::min<std::size_t>(tile,field.channels.size() - begin);
                std::fill(right.begin(),right.end(),cdouble(0.,0.));
                for (ssize_t k = 0; k < count; ++k) {
                    const ssize_t n = field.channels[begin + k];
                    for (ssize_t p = 0; p < P; ++p) coefficients[p * count + k] = prefix_diagonal[n][p];
                    auto store = [&](ssize_t r, ssize_t c, ssize_t s, ssize_t d, cdouble value) {
                        if (point && r != s) return;
                        const std::size_t i = point ? (r * C + c) * D + d : ((r * C + c) * S + s) * D + d;
                        right[k * stride + i] = value;
                    };
                    if (!diagonal_right[n].values.empty()) {
                        for (ssize_t r = 0; r < R; ++r)
                            for (ssize_t s = 0; s < S; ++s)
                                for (ssize_t c = 0; c < C; ++c) store(r,c,s,c,diagonal_right[n].values[(r * S + s) * C + c]);
                    } else if (!packed_right[n].empty() || !real_right[n].empty() || !split_right[n].empty()) {
                        for (ssize_t r = 0; r < R; ++r)
                            for (ssize_t c = 0; c < C; ++c)
                                for (ssize_t s = 0; s < S; ++s)
                                    for (ssize_t d = 0; d < D; ++d) {
                                        const ssize_t i = ((r * C + c) * S + s) * D + d;
                                        const auto value = !packed_right[n].empty() ? packed_right[n][i]
                                            : (!real_right[n].empty() ? cdouble(real_right[n][i],0.)
                                               : cdouble(split_right[n][i],split_right[n][R * C * S * D + i]));
                                        store(r,c,s,d,value);
                                    }
                    } else for (ssize_t s = 0; s < S; ++s) {
                        const auto& group = right_groups[n * S + s];
                        for (ssize_t j = 0; j < static_cast<ssize_t>(group.outputs.size()); ++j)
                            for (ssize_t c = 0; c < C; ++c)
                                for (ssize_t d = 0; d < D; ++d) store(group.outputs[j],c,s,d,group_value(group,(j * C + c) * D + d));
                    }
                }
                const cdouble one(1.,0.);
                const char* real_field_mode = std::getenv("PYQED_TDVP_REAL_PREFIX_FIELDS");
                const bool real_products = !real_field_mode || real_field_mode[0] == '1';
                const auto real = real_products ? real_values(coefficients) : std::vector<double>();
                if (!real.empty()) {
                    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,P,2 * stride,count,
                        1.,real.data(),count,reinterpret_cast<const double*>(right.data()),2 * stride,
                        1.,reinterpret_cast<double*>(field.values.data()),2 * stride);
                    coordinate_prefix_real_products.fetch_add(1,std::memory_order_relaxed);
                } else cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,P,stride,count,
                        &one,coefficients.data(),count,right.data(),stride,&one,field.values.data(),stride);
            }
            resident += size;
            for (ssize_t n : field.channels) channel_field[n] = prefix_fields.size();
            coordinate_prefix_field_channels.fetch_add(field.channels.size(),std::memory_order_relaxed);
            if (point) point_prefix_field_channels.fetch_add(field.channels.size(),std::memory_order_relaxed);
            prefix_fields.push_back(std::move(field));
        }
    }
    if (!prefix_fields.empty()) for (auto& batch : prefix_batches) {
            if (std::any_of(batch.channels.begin(), batch.channels.end(),
                    [&](ssize_t n) { return channel_field[n] >= 0; })) {
                for (ssize_t n : batch.channels) if (channel_field[n] < 0) channel_batch[n] = -1;
                std::vector<double>().swap(batch.real);
                std::vector<double>().swap(batch.right_parts);
                std::vector<cdouble>().swap(batch.complex);
                std::vector<cdouble>().swap(batch.diagonal);
                std::vector<cdouble>().swap(batch.right_diagonal);
            }
    }
    // Regroup diagonal-left channels without composing a local Hamiltonian:
    // L_p theta_p, followed by the bank of original right factors.
    std::vector<ssize_t> diagonal_left_batch;
    std::vector<char> diagonal_left_selected(N, false);
    std::vector<cdouble> diagonal_left_bank, diagonal_right_bank;
    std::vector<double> diagonal_left_parts, diagonal_right_parts;
    std::vector<cdouble> point_right_weights;
    const bool split_diagonal_batch = !std::getenv("PYQED_TDVP_COMPLEX_DIAGONAL_LEFT_BATCH");
    const char* diagonal_batch_mode = std::getenv("PYQED_TDVP_DIAGONAL_LEFT_BATCH");
    if (((diagonal_batch_mode && std::string(diagonal_batch_mode) == "1") || point_batch) && P == Q) {
        for (ssize_t n = 0; n < N; ++n) {
            if (prefix_component[n] >= 0 || spectator_selected[n]
                || !packed_left[n].empty() || !split_left[n].empty()) continue;
            bool active = false, diagonal = true;
            for (ssize_t q = 0; q < Q && diagonal; ++q) {
                const auto& group = left_groups[n * Q + q];
                active = active || !group.outputs.empty();
                for (ssize_t p : group.outputs) if (p != q) { diagonal = false; break; }
            }
            if (!active || !diagonal) continue;
            if (point_batch) {
                if (C != 1 || D != 1 || R != S) continue;
                std::vector<cdouble> weights(S);
                bool point_diagonal = true;
                auto accept = [&](ssize_t r, ssize_t s, cdouble value) {
                    if (r == s) weights[s] = value;
                    else if (value != cdouble(0.0, 0.0)) point_diagonal = false;
                };
                if (!packed_right[n].empty() || !real_right[n].empty() || !split_right[n].empty()) {
                    for (ssize_t r = 0; r < R && point_diagonal; ++r)
                        for (ssize_t s = 0; s < S && point_diagonal; ++s) {
                            const ssize_t i = r * S + s;
                            accept(r, s, !real_right[n].empty() ? cdouble(real_right[n][i], 0)
                                : !split_right[n].empty() ? cdouble(split_right[n][i], split_right[n][R * S + i])
                                : packed_right[n][i]);
                        }
                } else for (ssize_t s = 0; s < S && point_diagonal; ++s) {
                    const auto& group = right_groups[n * S + s];
                    for (std::size_t j = 0; j < group.outputs.size(); ++j)
                        accept(group.outputs[j], s, !group.real_blocks.empty() ? cdouble(group.real_blocks[j], 0)
                            : !group.split_blocks.empty() ? cdouble(group.split_blocks[j], group.split_blocks[group.outputs.size() + j])
                            : group.blocks[j]);
                }
                if (!point_diagonal) continue;
                const auto proposed = static_cast<long double>(diagonal_left_batch.size() + 1);
                if (sizeof(cdouble) * (proposed * (P * A * B + S)
                        + workers * (2 * std::min<long double>(proposed, 32) * A * S + A * S + B * S))
                        > (128ULL << 20)) break;
                point_right_weights.insert(point_right_weights.end(), weights.begin(), weights.end());
            }
            diagonal_left_batch.push_back(n);
        }
        const ssize_t count = diagonal_left_batch.size();
        const long double bytes = sizeof(cdouble) * (
            static_cast<long double>(P) * count * A * B
            + (point_batch ? static_cast<long double>(count) * S : static_cast<long double>(R) * C * count * SD)
            + static_cast<long double>(workers) * (2 * std::min<ssize_t>(count, 32) * A * SD + A * R * C + B * SD));
        if (count < 8 || bytes > (128ULL << 20)) {
            diagonal_left_batch.clear();
            std::vector<cdouble>().swap(point_right_weights);
        }
        else {
            diagonal_left_batched_channels.fetch_add(count, std::memory_order_relaxed);
            if (point_batch) point_diagonal_batched_channels.fetch_add(count, std::memory_order_relaxed);
            const std::size_t left_size = static_cast<std::size_t>(P) * count * A * B;
            const std::size_t right_size = point_batch ? 0 : static_cast<std::size_t>(R) * C * count * SD;
            if (split_diagonal_batch) {
                diagonal_left_parts.resize(2 * left_size);
                diagonal_right_parts.resize(2 * right_size);
            } else {
                diagonal_left_bank.resize(left_size);
                diagonal_right_bank.resize(right_size);
            }
            auto store = [&](std::vector<cdouble>& values, std::vector<double>& parts,
                             std::size_t size, std::size_t i, cdouble value) {
                if (split_diagonal_batch) { parts[i] = value.real(); parts[size + i] = value.imag(); }
                else values[i] = value;
            };
            for (ssize_t k = 0; k < count; ++k) {
                const ssize_t n = diagonal_left_batch[k];
                diagonal_left_selected[n] = true;
                for (ssize_t p = 0; p < P; ++p) {
                    const auto& group = left_groups[n * Q + p];
                    if (group.outputs.empty()) continue;
                    for (ssize_t ab = 0; ab < A * B; ++ab) {
                        cdouble value;
                        if (!group.split_blocks.empty()) value = cdouble(group.split_blocks[ab],
                            group.split_blocks.size() > static_cast<std::size_t>(A * B)
                                ? group.split_blocks[A * B + ab] : 0.0);
                        else value = group.blocks[ab];
                        store(diagonal_left_bank, diagonal_left_parts, left_size,
                            (p * count + k) * A * B + ab, value);
                    }
                }
                if (point_batch) continue;
                auto put = [&](ssize_t rc, ssize_t sd, cdouble value) {
                    store(diagonal_right_bank, diagonal_right_parts, right_size,
                        (rc * count + k) * SD + sd, value);
                };
                if (!packed_right[n].empty() || !real_right[n].empty() || !split_right[n].empty()) {
                    for (ssize_t rc = 0; rc < R * C; ++rc)
                        for (ssize_t sd = 0; sd < SD; ++sd) {
                            const ssize_t i = rc * SD + sd;
                            const cdouble value = !real_right[n].empty() ? cdouble(real_right[n][i], 0)
                                : !split_right[n].empty() ? cdouble(split_right[n][i], split_right[n][R * C * SD + i])
                                : packed_right[n][i];
                            put(rc, sd, value);
                        }
                } else if (!diagonal_right[n].values.empty()) {
                    for (const auto& transition : diagonal_right[n].transitions)
                        for (ssize_t c = 0; c < C; ++c)
                            put(transition.first * C + c, transition.second * D + c,
                                diagonal_right[n].values[(transition.first * S + transition.second) * C + c]);
                } else for (ssize_t s = 0; s < S; ++s) {
                    const auto& group = right_groups[n * S + s];
                    const ssize_t size = group.outputs.size() * C * D;
                    for (ssize_t rr = 0; rr < static_cast<ssize_t>(group.outputs.size()); ++rr)
                        for (ssize_t c = 0; c < C; ++c)
                            for (ssize_t d = 0; d < D; ++d) {
                                const ssize_t i = (rr * C + c) * D + d;
                                const cdouble value = !group.real_blocks.empty() ? cdouble(group.real_blocks[i], 0)
                                    : !group.split_blocks.empty() ? cdouble(group.split_blocks[i], group.split_blocks[size + i])
                                    : group.blocks[i];
                                put(group.outputs[rr] * C + c, s * D + d, value);
                            }
                }
            }
        }
    }
    std::vector<ssize_t> diagonal_field_channels;
    std::vector<char> diagonal_field_selected(N, false);
    std::vector<ssize_t> diagonal_field_tasks(N, -1);
    ssize_t diagonal_field_task_count = 1;
    std::vector<cdouble> diagonal_left_field;
    const char* diagonal_field_mode = std::getenv("PYQED_TDVP_DIAGONAL_LEFT_FIELDS");
    if ((!diagonal_field_mode || diagonal_field_mode[0] == '1') && C == 1 && D == 1 && P == Q) {
        for (ssize_t n = 0; n < N; ++n) {
            if (diagonal_left_selected[n] || prefix_component[n] >= 0 || !spectator_fields[n].empty()
                    || !packed_left[n].empty() || !split_left[n].empty()) continue;
            bool active = false, diagonal = true;
            for (ssize_t q = 0; q < Q && diagonal; ++q) {
                const auto& group = left_groups[n * Q + q];
                active = active || !group.outputs.empty();
                for (ssize_t p : group.outputs) if (p != q) { diagonal = false; break; }
            }
            if (active && diagonal) diagonal_field_channels.push_back(n);
        }
        const ssize_t count = diagonal_field_channels.size();
        diagonal_left_field_candidates.fetch_add(count,std::memory_order_relaxed);
        const std::size_t size = static_cast<std::size_t>(P) * A * B * R * S;
        const long double temporary = 2.L * size + static_cast<long double>(count) * (P * A * B + R * S)
            + static_cast<long double>(std::max(1, std::min(workers, static_cast<int>(N)))) * A * R;
        if (count >= 4 && temporary * sizeof(cdouble) <= (128ULL << 20)) {
            std::vector<cdouble> coefficients(static_cast<std::size_t>(P) * A * B * count);
            std::vector<cdouble> right(static_cast<std::size_t>(count) * R * S);
            auto value = [](const TwoSitePhysicalGroup& group, ssize_t i, ssize_t size) {
                if (!group.real_blocks.empty()) return cdouble(group.real_blocks[i], 0.);
                if (!group.split_blocks.empty()) return cdouble(group.split_blocks[i],
                    group.split_blocks.size() > static_cast<std::size_t>(size) ? group.split_blocks[size + i] : 0.);
                return group.blocks[i];
            };
            for (ssize_t k = 0; k < count; ++k) {
                const ssize_t n = diagonal_field_channels[k];
                for (ssize_t p = 0; p < P; ++p) {
                    const auto& group = left_groups[n * Q + p];
                    if (group.outputs.empty()) continue;
                    for (ssize_t ab = 0; ab < A * B; ++ab)
                        coefficients[(p * A * B + ab) * count + k] = value(group, ab, A * B);
                }
                for (ssize_t r = 0; r < R; ++r)
                    for (ssize_t s = 0; s < S; ++s) {
                        const ssize_t i = r * S + s;
                        if (!packed_right[n].empty()) right[k * R * S + i] = packed_right[n][i];
                        else if (!real_right[n].empty()) right[k * R * S + i] = real_right[n][i];
                        else if (!split_right[n].empty()) right[k * R * S + i] = cdouble(split_right[n][i], split_right[n][R * S + i]);
                        else if (!diagonal_right[n].values.empty()) right[k * R * S + i] = diagonal_right[n].values[i];
                    }
                if (packed_right[n].empty() && real_right[n].empty() && split_right[n].empty() && diagonal_right[n].values.empty())
                    for (ssize_t s = 0; s < S; ++s) {
                        const auto& group = right_groups[n * S + s];
                        for (ssize_t i = 0; i < static_cast<ssize_t>(group.outputs.size()); ++i)
                            right[(k * R + group.outputs[i]) * S + s] = value(group, i, group.outputs.size());
                    }
                diagonal_field_selected[n] = true;
            }
            std::vector<cdouble> product(size);
            const std::vector<cdouble> empty;
            environment_product(coefficients.data(), right.data(), product.data(), P * A * B, count, R * S, empty);
            diagonal_left_field.resize(size);
            for (ssize_t p = 0; p < P; ++p)
                for (ssize_t a = 0; a < A; ++a)
                    for (ssize_t b = 0; b < B; ++b)
                        for (ssize_t r = 0; r < R; ++r)
                            for (ssize_t s = 0; s < S; ++s)
                                diagonal_left_field[(p * A * R + a * R + r) * B * S + b * S + s] =
                                    product[((p * A + a) * B + b) * R * S + r * S + s];
            diagonal_left_field_channels.fetch_add(count, std::memory_order_relaxed);
        } else {
            if (count >= 4 && temporary * sizeof(cdouble) > (128ULL << 20))
                diagonal_left_field_budget_rejected.fetch_add(count,std::memory_order_relaxed);
            diagonal_field_channels.clear();
        }
    }
    std::vector<double> coordinate_field_parts;
    const char* parallel_coordinate_flag = std::getenv("PYQED_TDVP_PARALLEL_COORDINATE_FIELDS");
    if (!diagonal_field_channels.empty()) {
        if (parallel_coordinate_flag && parallel_coordinate_flag[0] == '1')
            diagonal_field_task_count = std::min<ssize_t>(P,diagonal_field_channels.size());
        for (ssize_t k = 0; k < diagonal_field_task_count; ++k)
            diagonal_field_tasks[diagonal_field_channels[k]] = k;
    }
    bool real_coordinate_field = false;
    const char* coordinate_action_flag = std::getenv("PYQED_TDVP_REAL_COORDINATE_ACTION");
    if (!diagonal_left_field.empty() && coordinate_action_flag && coordinate_action_flag[0] == '1') {
        real_coordinate_field = std::all_of(diagonal_left_field.begin(),diagonal_left_field.end(),
            [](cdouble z) { return z.imag() == 0.; });
        const std::size_t size = diagonal_left_field.size();
        const long double bytes = static_cast<long double>(size) * sizeof(double) * (real_coordinate_field ? 3 : 6);
        if (bytes <= (128ULL << 20)) {
            coordinate_field_parts.resize((real_coordinate_field ? 1 : 4) * size);
            if (real_coordinate_field) {
                for (std::size_t i = 0; i < size; ++i) coordinate_field_parts[i] = diagonal_left_field[i].real();
                real_coordinate_fields.fetch_add(1,std::memory_order_relaxed);
            } else {
                const ssize_t rows = A * R, columns = B * S;
                for (ssize_t p = 0; p < P; ++p)
                    for (ssize_t row = 0; row < rows; ++row)
                        for (ssize_t column = 0; column < columns; ++column) {
                            const auto z = diagonal_left_field[(p * rows + row) * columns + column];
                            const ssize_t i = p * 4 * rows * columns + 2 * column * 2 * rows + 2 * row;
                            coordinate_field_parts[i] = z.real();
                            coordinate_field_parts[i + 1] = z.imag();
                            coordinate_field_parts[i + 2 * rows] = -z.imag();
                            coordinate_field_parts[i + 2 * rows + 1] = z.real();
                        }
                interleaved_coordinate_fields.fetch_add(1,std::memory_order_relaxed);
            }
            std::vector<cdouble>().swap(diagonal_left_field);
        }
    }
    std::vector<std::shared_ptr<const std::vector<double>>> interleaved_right(N);
    const char* interleaved_mode = std::getenv("PYQED_TDVP_INTERLEAVED_RIGHT");
    if (!interleaved_mode || interleaved_mode[0] == '1') {
        std::size_t elements = 0;
        for (ssize_t n = 0; n < N; ++n) {
            const ssize_t rows = prefix_component[n] >= 0 ? P : AP;
            if (diagonal_left_selected[n] || diagonal_field_selected[n] || channel_field[n] >= 0
                    || split_right[n].empty() || rows < 8 || rows < R * C) continue;
            const std::size_t size = static_cast<std::size_t>(R) * C * SD;
            if (elements + 4 * size > (128ULL << 20) / sizeof(double)) continue;
            const char* cache_mode = std::getenv("PYQED_TDVP_CACHE_RIGHT_BLOCKS");
            const bool reusable = C == 1 && D == 1 && (!cache_mode || cache_mode[0] != '0');
            interleaved_right[n] = interleaved_right_block(split_right[n], R * C, SD, reusable);
            elements += interleaved_right[n]->size();
            std::vector<double>().swap(split_right[n]);
            interleaved_right_factors.fetch_add(1, std::memory_order_relaxed);
        }
    }
    struct PackedLeftBatch {
        std::vector<ssize_t> channels;
        std::vector<double> parts;
    };
    std::vector<PackedLeftBatch> packed_left_batches;
    std::vector<ssize_t> packed_left_batch_index(N, -1);
    std::size_t packed_left_workspace = 0;
    std::size_t packed_left_tile_elements = 0;
    const char* stream_left_mode = std::getenv("PYQED_TDVP_STREAM_LEFT_BANKS");
    const bool stream_left_banks = !stream_left_mode || stream_left_mode[0] == '1';
    const char* fused_left_mode = std::getenv("PYQED_TDVP_FUSED_LEFT_PRODUCTS");
    const bool fused_left_products = fused_left_mode && fused_left_mode[0] == '1';
    const char* packed_batch_mode = std::getenv("PYQED_TDVP_PACKED_LEFT_BATCH");
    if (!packed_batch_mode || packed_batch_mode[0] == '1') {
        std::vector<ssize_t> eligible;
        const std::size_t size = static_cast<std::size_t>(AP) * Q * B;
        for (ssize_t n = 0; n < N; ++n)
            if (!diagonal_left_selected[n] && prefix_component[n] < 0
                    && spectator_fields[n].empty() && split_left[n].size() == 2 * size)
                eligible.push_back(n);
        std::size_t bank_elements = 0;
        const char* fixed_tiles_mode = std::getenv("PYQED_TDVP_FIXED_LEFT_TILES");
        const bool fixed_tiles = !fixed_tiles_mode || fixed_tiles_mode[0] != '0';
        const std::size_t bank_limit = (128ULL << 20) / sizeof(double)
            / (stream_left_banks ? std::max(1, std::min(workers, static_cast<int>(N))) : 1);
        const std::size_t output_limit = (16ULL << 20) / (static_cast<std::size_t>(AP) * SD * sizeof(cdouble));
        for (std::size_t start = 0; start < eligible.size();) {
            std::size_t count = std::min<std::size_t>(16, eligible.size() - start);
            if (!fixed_tiles) {
                count = std::min(count, (bank_limit - bank_elements) / (2 * size));
                count = std::min(count, output_limit);
                if (count < 4) break;
            }
            const std::size_t elements = 2 * count * size;
            if (count >= 4) {
                if (bank_elements + elements > bank_limit)
                    packed_left_bank_rejected.fetch_add(count, std::memory_order_relaxed);
                if (count > output_limit)
                    packed_left_product_rejected.fetch_add(count, std::memory_order_relaxed);
            }
            if (count < 4 || bank_elements + elements > bank_limit
                    || count * static_cast<std::size_t>(AP) * SD * sizeof(cdouble) > (16ULL << 20)) {
                start += count;
                continue;
            }
            PackedLeftBatch batch;
            if (!stream_left_banks) batch.parts.resize(elements);
            for (std::size_t i = 0; i < count; ++i) {
                const ssize_t n = eligible[start + i];
                batch.channels.push_back(n);
                packed_left_batch_index[n] = packed_left_batches.size();
                if (!stream_left_banks) {
                    std::copy_n(split_left[n].data(), size, batch.parts.data() + i * size);
                    std::copy_n(split_left[n].data() + size, size, batch.parts.data() + count * size + i * size);
                    std::vector<double>().swap(split_left[n]);
                }
            }
            if (!stream_left_banks) bank_elements += elements;
            else packed_left_tile_elements = std::max(packed_left_tile_elements, elements);
            packed_left_workspace = std::max(packed_left_workspace, count * static_cast<std::size_t>(AP) * SD);
            packed_left_batch_channels.fetch_add(count, std::memory_order_relaxed);
            packed_left_batches.push_back(std::move(batch));
            start += count;
        }
    }
    std::vector<cdouble> x_by_q(
        static_cast<std::size_t>(Q) * B * SD,
        0.0
    );
    struct Scratch {
        std::vector<cdouble> intermediate;
        std::vector<cdouble> left_contribution;
        std::vector<cdouble> intermediate_s;
        std::vector<cdouble> right_contribution;
        std::vector<cdouble> imaginary_contribution;
        std::vector<cdouble> imaginary_intermediate;
        std::vector<cdouble> out;
        std::vector<cdouble> prefix_out;
        std::vector<cdouble> batch_out;
        std::vector<cdouble> packed_batch_products;
        std::vector<double> packed_left_tile;
        std::vector<cdouble> coordinate_field_output;
        std::vector<cdouble> batch_right_input;
        std::vector<cdouble> spectator_output;
        std::vector<cdouble> spectator_sum;
        std::vector<cdouble> diagonal_left_output, diagonal_right_input, diagonal_batch_output;
        std::vector<cdouble> diagonal_rotated_input;
        double left_seconds = 0.0, right_seconds = 0.0;
        double left_layout_seconds[4] = {}, right_layout_seconds[5] = {};

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
            imaginary_contribution(right_size, 0.0),
            imaginary_intermediate(intermediate_size, 0.0),
            out(out_size, 0.0) {}
    };
    const int thread_count = std::max(
        1,
        std::min(workers, static_cast<int>(N))
    );
    std::vector<Scratch> scratches;
    ssize_t frame_rows = A;
    for (const auto& component : *sum_components)
        frame_rows = std::max(frame_rows, component.spectator_size);
    const ssize_t action_rows = frame_rows * P;
    ssize_t diagonal_left_channels = 0;
    if (profile_phases && profile_phases[0] == '1') {
        for (ssize_t n = 0; n < N; ++n) {
            if (prefix_component[n] >= 0 || spectator_selected[n] || !packed_left[n].empty() || !split_left[n].empty()) continue;
            bool active = false, diagonal = true;
            for (ssize_t q = 0; q < Q && diagonal; ++q) {
                const auto& group = left_groups[n * Q + q];
                active = active || !group.outputs.empty();
                for (ssize_t p : group.outputs) if (p != q) { diagonal = false; break; }
            }
            if (active && diagonal) {
                ++diagonal_left_channels;
                if (C > 1 && C == D && !diagonal_right[n].values.empty())
                    diagonal_left_state_diagonal_channels.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    diagonal_left_sparse_channels.fetch_add(diagonal_left_channels, std::memory_order_relaxed);
    const char* full_pair_clear = std::getenv("PYQED_TDVP_FULL_PAIR_CLEAR");
    const bool clear_workspace = !full_pair_clear || std::string(full_pair_clear) != "0";
    scratches.reserve(static_cast<std::size_t>(thread_count));
    for (int worker = 0; worker < thread_count; ++worker) {
        scratches.emplace_back(
            static_cast<std::size_t>(action_rows) * SD,
            max_left_rows * static_cast<std::size_t>(SD),
            static_cast<std::size_t>(action_rows) * SD,
            static_cast<std::size_t>(action_rows) * max_right_rows,
            dim
        );
        scratches.back().prefix_out.resize(output_bras.size() * static_cast<std::size_t>(PRC));
        scratches.back().batch_out.resize(batch_workspace);
        scratches.back().packed_batch_products.resize(2 * packed_left_workspace);
        scratches.back().packed_left_tile.resize(packed_left_tile_elements);
        if (!diagonal_left_field.empty() || !coordinate_field_parts.empty()) scratches.back().coordinate_field_output.resize(A * R);
        if (defer_spectator) scratches.back().spectator_sum.resize(spectator_offsets.back());
        if (std::any_of(spectator_components.begin(), spectator_components.end(),
                        [](char selected) { return selected; }))
            scratches.back().spectator_output.resize(static_cast<std::size_t>(action_rows) * SD);
        if (batch_diagonal_right) scratches.back().batch_right_input.resize(batch_workspace / D);
        if (!diagonal_left_batch.empty()) {
            const std::size_t tile = std::min<std::size_t>(diagonal_left_batch.size(), 32);
            scratches.back().diagonal_left_output.resize(tile * A * SD);
            if (!point_batch) {
                scratches.back().diagonal_right_input.resize(tile * A * SD);
                scratches.back().diagonal_batch_output.resize(A * R * C);
            }
            if (split_diagonal_batch) scratches.back().diagonal_rotated_input.resize(B * SD);
        }
    }
    const cdouble one(1.0, 0.0);
    const cdouble zero(0.0, 0.0);
    std::vector<std::vector<cdouble>> prefix_inputs(sum_components->size());
    std::vector<ssize_t> prefix_input_owner(sum_components->size());
    const char* shared_prefix_flag = std::getenv("PYQED_TDVP_SHARED_PREFIX_INPUTS");
    for (std::size_t k = 0; k < sum_components->size(); ++k) {
        prefix_input_owner[k] = k;
        const auto& component = (*sum_components)[k];
        if (!component.prefix_bra || !shared_prefix_flag || shared_prefix_flag[0] != '1') continue;
        for (std::size_t j = 0; j < k; ++j) {
            const auto& previous = (*sum_components)[j];
            if (previous.prefix_bra && std::equal(component.prefix_ket,component.prefix_ket + B,previous.prefix_ket)) {
                prefix_input_owner[k] = prefix_input_owner[j];
                shared_prefix_input_components.fetch_add(1,std::memory_order_relaxed);
                break;
            }
        }
    }
    std::vector<std::vector<cdouble>> spectator_inputs(sum_components->size());
    for (std::size_t k = 0; k < sum_components->size(); ++k)
        if ((*sum_components)[k].prefix_bra && prefix_input_owner[k] == static_cast<ssize_t>(k))
            prefix_inputs[k].resize(static_cast<std::size_t>(Q) * SD);
    auto prefix_input = [&](ssize_t k) -> const std::vector<cdouble>& {
        return prefix_inputs[prefix_input_owner[k]];
    };
    for (std::size_t k = 0; k < sum_components->size(); ++k)
        if (spectator_components[k] && ket_owner[k] == k)
            spectator_inputs[k].resize((*sum_components)[k].spectator_size * Q * SD);

    ActionWorkers action_workers(thread_count);
    auto matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        for (std::size_t k = 0; k < spectator_inputs.size(); ++k) {
            auto& input = spectator_inputs[k];
            if (input.empty()) continue;
            const auto& component = (*sum_components)[k];
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                component.spectator_size, Q * SD, B, &one,
                component.spectator_ket, B, local.data(), Q * SD,
                &zero, input.data(), Q * SD);
        }
        for (std::size_t k = 0; k < prefix_inputs.size(); ++k) {
            auto& input = prefix_inputs[k];
            if (input.empty()) continue;
            std::fill(input.begin(), input.end(), cdouble(0.0, 0.0));
            const auto* ket = (*sum_components)[k].prefix_ket;
            for (ssize_t b = 0; b < B; ++b)
                for (ssize_t q = 0; q < Q; ++q)
                    for (ssize_t sd = 0; sd < SD; ++sd)
                        input[q * SD + sd] += ket[b] * local[(b * Q + q) * SD + sd];
        }
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
            std::fill(scratch.prefix_out.begin(), scratch.prefix_out.end(), cdouble(0.0, 0.0));
            std::fill(scratch.spectator_sum.begin(), scratch.spectator_sum.end(), cdouble(0.0, 0.0));
        }
        auto apply_range = [&](ssize_t begin, ssize_t end, Scratch& scratch) {
        for (ssize_t work = begin; work < end; ++work) {
            if (diagonal_field_selected[work]) {
                const ssize_t task = diagonal_field_tasks[work];
                if (task < 0) continue;
                const auto start = std::chrono::steady_clock::now();
                for (ssize_t p = task; p < P; p += diagonal_field_task_count) {
                    if (!coordinate_field_parts.empty()) {
                        if (real_coordinate_field) cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                            A * R,2,B * S,1.,coordinate_field_parts.data() + p * A * R * B * S,B * S,
                            reinterpret_cast<const double*>(x_by_q.data() + p * B * S),2,0.,
                            reinterpret_cast<double*>(scratch.coordinate_field_output.data()),2);
                        else cblas_dgemv(CblasColMajor,CblasNoTrans,2 * A * R,2 * B * S,
                            1.,coordinate_field_parts.data() + p * 4 * A * R * B * S,2 * A * R,
                            reinterpret_cast<const double*>(x_by_q.data() + p * B * S),1,0.,
                            reinterpret_cast<double*>(scratch.coordinate_field_output.data()),1);
                    } else cblas_zgemv(CblasRowMajor, CblasNoTrans, A * R, B * S, &one,
                        diagonal_left_field.data() + p * A * R * B * S, B * S,
                        x_by_q.data() + p * B * S, 1, &zero, scratch.coordinate_field_output.data(), 1);
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t r = 0; r < R; ++r)
                            scratch.out[(a * P + p) * R + r] += scratch.coordinate_field_output[a * R + r];
                }
                if (profile_phases && profile_phases[0] == '1') {
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    scratch.left_seconds += elapsed;
                    scratch.left_layout_seconds[3] += elapsed;
                }
                continue;
            }
            if (diagonal_left_selected[work]) continue;
            if (channel_field[work] >= 0) {
                const auto& field = prefix_fields[channel_field[work]];
                if (field.channels.front() != work) continue;
                const auto start = std::chrono::steady_clock::now();
                cdouble* target = scratch.prefix_out.data() + output_group[field.component] * PRC;
                if (!field.state_diagonal) {
                    for (ssize_t p = 0; p < P; ++p) {
                        if (field.physical_diagonal) for (ssize_t r = 0; r < R; ++r) {
                            cblas_zgemv(CblasRowMajor,CblasNoTrans,C,D,&one,
                                field.values.data() + (p * R + r) * C * D,D,
                                prefix_input(field.component).data() + p * SD + r * D,1,&zero,scratch.right_contribution.data(),1);
                            for (ssize_t c = 0; c < C; ++c) target[(r * C + c) * P + p] += scratch.right_contribution[c];
                        } else {
                            cblas_zgemv(CblasRowMajor,CblasNoTrans,R * C,SD,&one,
                                field.values.data() + p * R * C * SD,SD,
                                prefix_input(field.component).data() + p * SD,1,&zero,scratch.right_contribution.data(),1);
                            for (ssize_t rc = 0; rc < R * C; ++rc) target[rc * P + p] += scratch.right_contribution[rc];
                        }
                    }
                } else for (ssize_t p = 0; p < P; ++p)
                    for (ssize_t c = 0; c < C; ++c) {
                        for (ssize_t s = 0; s < S; ++s)
                            scratch.intermediate_s[s] = prefix_input(field.component)[p * SD + s * D + c];
                        cblas_zgemv(CblasRowMajor, CblasNoTrans, R, S, &one,
                            field.values.data() + (p * C + c) * R * S, S,
                            scratch.intermediate_s.data(), 1, &zero, scratch.right_contribution.data(), 1);
                        for (ssize_t r = 0; r < R; ++r)
                            target[(r * C + c) * P + p] += scratch.right_contribution[r];
                    }
                if (profile_phases && profile_phases[0] == '1') {
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    scratch.right_seconds += elapsed;
                    scratch.right_layout_seconds[0] += elapsed;
                }
                continue;
            }
            const ssize_t batch_index = channel_batch[work];
            const PrefixBatch* batch = batch_index < 0 ? nullptr : &prefix_batches[batch_index];
            if (batch && batch->channels.front() != work) continue;
            const ssize_t packed_index = packed_left_batch_index[work];
            const PackedLeftBatch* left_batch = packed_index < 0 ? nullptr : &packed_left_batches[packed_index];
            if (left_batch && left_batch->channels.front() != work) continue;
            std::chrono::steady_clock::time_point batch_start;
            if (profile_phases && profile_phases[0] == '1') batch_start = std::chrono::steady_clock::now();
            if (left_batch) {
                const ssize_t rows = left_batch->channels.size() * AP;
                const ssize_t size = rows * Q * B;
                const double* parts = left_batch->parts.data();
                if (stream_left_banks) {
                    const ssize_t block = AP * Q * B;
                    for (std::size_t i = 0; i < left_batch->channels.size(); ++i) {
                        const auto& source = split_left[left_batch->channels[i]];
                        std::copy_n(source.data(), block, scratch.packed_left_tile.data() + i * block);
                        std::copy_n(source.data() + block, block, scratch.packed_left_tile.data() + size + i * block);
                    }
                    parts = scratch.packed_left_tile.data();
                }
                cdouble* product = scratch.packed_batch_products.data();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    fused_left_products ? 2 * rows : rows, 2 * SD, Q * B, 1., parts, Q * B,
                    reinterpret_cast<const double*>(x_by_q.data()), 2 * SD, 0.,
                    reinterpret_cast<double*>(product), 2 * SD);
                if (!fused_left_products) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, 2 * SD, Q * B, 1., parts + size, Q * B,
                    reinterpret_cast<const double*>(x_by_q.data()), 2 * SD, 0.,
                    reinterpret_cast<double*>(product + rows * SD), 2 * SD);
                for (ssize_t i = 0; i < rows * SD; ++i) {
                    const auto zr = product[i], zi = product[rows * SD + i];
                    product[i] = cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                }
            }
            if (batch) {
                const int batch_rows = static_cast<int>(batch->channels.size() * P);
                if (!batch->diagonal.empty()) {
                    for (std::size_t i = 0; i < batch->channels.size(); ++i)
                        for (ssize_t p = 0; p < P; ++p) {
                            const cdouble value = batch->diagonal[i * P + p];
                            const cdouble* source = prefix_input(batch->component).data() + p * SD;
                            cdouble* target = scratch.batch_out.data() + (i * P + p) * SD;
                            for (ssize_t sd = 0; sd < SD; ++sd) target[sd] = value * source[sd];
                        }
                } else if (!batch->real.empty()) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    batch_rows, static_cast<int>(2 * SD), static_cast<int>(Q),
                    1.0, batch->real.data(), static_cast<int>(Q),
                    reinterpret_cast<const double*>(prefix_input(batch->component).data()), static_cast<int>(2 * SD),
                    0.0, reinterpret_cast<double*>(scratch.batch_out.data()), static_cast<int>(2 * SD));
                else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    batch_rows, static_cast<int>(SD), static_cast<int>(Q), &one,
                    batch->complex.data(), static_cast<int>(Q), prefix_input(batch->component).data(), static_cast<int>(SD),
                    &zero, scratch.batch_out.data(), static_cast<int>(SD));
            }
            if (batch && (!batch->right_diagonal.empty() || !batch->right_parts.empty()) && output_group[batch->component] >= 0) {
                std::chrono::steady_clock::time_point right_start;
                if (profile_phases && profile_phases[0] == '1') {
                    right_start = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(right_start - batch_start).count();
                    scratch.left_seconds += elapsed;
                    scratch.left_layout_seconds[0] += elapsed;
                }
                cdouble* target = scratch.prefix_out.data() + output_group[batch->component] * PRC;
                const ssize_t count = batch->channels.size();
                for (ssize_t c = 0; c < C; ++c) {
                    for (ssize_t i = 0; i < count; ++i)
                        for (ssize_t s = 0; s < S; ++s)
                            for (ssize_t p = 0; p < P; ++p)
                                scratch.batch_right_input[(i * S + s) * P + p] =
                                    scratch.batch_out[(i * P + p) * SD + s * D + c];
                    if (!batch->right_parts.empty()) {
                        const ssize_t size = C * R * count * S;
                        const bool complex = batch->right_parts.size() > static_cast<std::size_t>(size);
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(R), static_cast<int>(2 * P), static_cast<int>(count * S),
                            1.0, batch->right_parts.data() + c * R * count * S, static_cast<int>(count * S),
                            reinterpret_cast<const double*>(scratch.batch_right_input.data()), static_cast<int>(2 * P),
                            0.0, reinterpret_cast<double*>(scratch.right_contribution.data()), static_cast<int>(2 * P));
                        if (complex) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(R), static_cast<int>(2 * P), static_cast<int>(count * S),
                            1.0, batch->right_parts.data() + size + c * R * count * S, static_cast<int>(count * S),
                            reinterpret_cast<const double*>(scratch.batch_right_input.data()), static_cast<int>(2 * P),
                            0.0, reinterpret_cast<double*>(scratch.imaginary_contribution.data()), static_cast<int>(2 * P));
                        for (ssize_t r = 0; r < R; ++r)
                            for (ssize_t p = 0; p < P; ++p) {
                                const cdouble zr = scratch.right_contribution[r * P + p];
                                const cdouble zi = complex ? scratch.imaginary_contribution[r * P + p] : zero;
                                target[(r * C + c) * P + p] += cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                            }
                    } else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(R), static_cast<int>(P), static_cast<int>(count * S),
                        &one, batch->right_diagonal.data() + c * R * count * S, static_cast<int>(count * S),
                        scratch.batch_right_input.data(), static_cast<int>(P),
                        &one, target + c * P, static_cast<int>(C * P));
                }
                if (profile_phases && profile_phases[0] == '1') {
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - right_start).count();
                    scratch.right_seconds += elapsed;
                    scratch.right_layout_seconds[0] += elapsed;
                }
                continue;
            }
            for (std::size_t member = 0; member < (batch ? batch->channels.size() : left_batch ? left_batch->channels.size() : 1); ++member) {
            const ssize_t n = batch ? batch->channels[member] : left_batch ? left_batch->channels[member] : work;
            std::chrono::steady_clock::time_point action_start;
            if (profile_phases && profile_phases[0] == '1') action_start = std::chrono::steady_clock::now();
            if ((batch || left_batch) && member == 0) action_start = batch_start;
            const ssize_t prefix = prefix_component[n];
            const auto* spectator = defer_spectator && !spectator_fields[n].empty() ? component_for_n[n] : nullptr;
            const ssize_t rows = spectator ? spectator->spectator_size * P : prefix < 0 ? AP : P;
            cdouble* spectator_sum = spectator ? scratch.spectator_sum.data()
                + spectator_offsets[spectator - sum_components->data()] : nullptr;
            const cdouble* bra = prefix < 0 ? nullptr : (*sum_components)[prefix].prefix_bra;
            const ssize_t group = prefix < 0 ? -1 : output_group[prefix];
            cdouble* prefix_out = group < 0 ? nullptr : scratch.prefix_out.data() + group * PRC;
            const cdouble* intermediate = scratch.intermediate.data();
            const auto& packed = packed_left[static_cast<std::size_t>(n)];
            if (!batch && !left_batch && !diagonal_spectator_fields[n]) std::fill_n(scratch.intermediate.data(),
                clear_workspace ? scratch.intermediate.size() : rows * SD, cdouble(0.0, 0.0));
            if (batch) intermediate = scratch.batch_out.data() + member * P * SD;
            else if (left_batch) intermediate = scratch.packed_batch_products.data() + member * AP * SD;
            else if (prefix >= 0) {
                if (!prefix_diagonal[n].empty()) {
                    for (ssize_t p = 0; p < P; ++p) {
                        const cdouble value = prefix_diagonal[n][p];
                        const cdouble* source = prefix_input(prefix).data() + p * SD;
                        cdouble* target = scratch.intermediate.data() + p * SD;
                        for (ssize_t sd = 0; sd < SD; ++sd) target[sd] = value * source[sd];
                    }
                } else if (!real_physical[n].empty())
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(P), static_cast<int>(2 * SD), static_cast<int>(Q),
                        1.0, real_physical[n].data(), static_cast<int>(Q),
                        reinterpret_cast<const double*>(prefix_input(prefix).data()), static_cast<int>(2 * SD),
                        0.0, reinterpret_cast<double*>(scratch.intermediate.data()), static_cast<int>(2 * SD));
                else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(P), static_cast<int>(SD), static_cast<int>(Q),
                    &one, prefix_physical[n].data(), static_cast<int>(Q),
                    prefix_input(prefix).data(), static_cast<int>(SD), &zero,
                    scratch.intermediate.data(), static_cast<int>(SD));
            } else if (!spectator_fields[n].empty()) {
                const auto* component = component_for_n[n];
                const ssize_t K = component->spectator_size;
                const auto component_index = ket_owner[component - sum_components->data()];
                for (ssize_t k = 0; k < K; ++k) {
                    if (diagonal_spectator_fields[n]) {
                        for (ssize_t p = 0; p < P; ++p) {
                            const cdouble value = spectator_fields[n][k * P + p];
                            const cdouble* input = spectator_inputs[component_index].data() + (k * Q + p) * SD;
                            cdouble* output = scratch.spectator_output.data() + (k * P + p) * SD;
                            for (ssize_t sd = 0; sd < SD; ++sd) output[sd] = value * input[sd];
                        }
                    } else if (!real_spectator_fields[n].empty())
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            P, 2 * SD, Q, 1.0, real_spectator_fields[n].data() + k * P * Q, Q,
                            reinterpret_cast<const double*>(spectator_inputs[component_index].data() + k * Q * SD), 2 * SD,
                            0.0, reinterpret_cast<double*>(scratch.spectator_output.data() + k * P * SD), 2 * SD);
                    else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        P, SD, Q, &one, spectator_fields[n].data() + k * P * Q, Q,
                        spectator_inputs[component_index].data() + k * Q * SD, SD,
                        &zero, scratch.spectator_output.data() + k * P * SD, SD);
                }
                if (spectator) intermediate = scratch.spectator_output.data();
                else {
                cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    A, P * SD, K, &one, component->spectator_bra, A,
                    scratch.spectator_output.data(), P * SD, &zero,
                    scratch.intermediate.data(), P * SD);
                // Right actions expect (p,a,sd), not the frame product's (a,p,sd).
                for (ssize_t a = 0; a < A; ++a)
                    for (ssize_t p = 0; p < P; ++p)
                        std::copy_n(scratch.intermediate.data() + (a * P + p) * SD, SD,
                                    scratch.imaginary_intermediate.data() + (p * A + a) * SD);
                intermediate = scratch.imaginary_intermediate.data();
                }
            } else if (!split_left[n].empty()) {
                const ssize_t count = AP * Q * B;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(AP), static_cast<int>(2 * SD), static_cast<int>(Q * B),
                    1.0, split_left[n].data(), static_cast<int>(Q * B),
                    reinterpret_cast<const double*>(x_by_q.data()), static_cast<int>(2 * SD),
                    0.0, reinterpret_cast<double*>(scratch.intermediate.data()), static_cast<int>(2 * SD));
                if (split_left[n].size() > static_cast<std::size_t>(count)) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(AP), static_cast<int>(2 * SD), static_cast<int>(Q * B),
                        1.0, split_left[n].data() + count, static_cast<int>(Q * B),
                        reinterpret_cast<const double*>(x_by_q.data()), static_cast<int>(2 * SD),
                        0.0, reinterpret_cast<double*>(scratch.imaginary_intermediate.data()), static_cast<int>(2 * SD));
                    for (ssize_t i = 0; i < AP * SD; ++i) {
                        const auto zr = scratch.intermediate[i], zi = scratch.imaginary_intermediate[i];
                        scratch.intermediate[i] = cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                    }
                }
            } else if (!packed.empty()) {
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(AP), static_cast<int>(SD), static_cast<int>(Q * B),
                    &one, packed.data(), static_cast<int>(Q * B),
                    x_by_q.data(), static_cast<int>(SD), &zero,
                    scratch.intermediate.data(), static_cast<int>(SD));
            } else for (ssize_t q = 0; q < Q; ++q) {
                const TwoSitePhysicalGroup& left_group =
                    left_groups[static_cast<std::size_t>(n) * Q + q];
                if (left_group.outputs.empty()) {
                    continue;
                }
                const ssize_t left_rows =
                    static_cast<ssize_t>(left_group.outputs.size()) * A;
                if (!left_group.split_blocks.empty()) {
                    const ssize_t count = left_rows * B;
                    const cdouble* input = x_by_q.data() + q * B * SD;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(left_rows), static_cast<int>(2 * SD), static_cast<int>(B),
                        1.0, left_group.split_blocks.data(), static_cast<int>(B),
                        reinterpret_cast<const double*>(input), static_cast<int>(2 * SD),
                        0.0, reinterpret_cast<double*>(scratch.left_contribution.data()), static_cast<int>(2 * SD));
                    if (left_group.split_blocks.size() > static_cast<std::size_t>(count)) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(left_rows), static_cast<int>(2 * SD), static_cast<int>(B),
                            1.0, left_group.split_blocks.data() + count, static_cast<int>(B),
                            reinterpret_cast<const double*>(input), static_cast<int>(2 * SD),
                            0.0, reinterpret_cast<double*>(scratch.imaginary_intermediate.data()), static_cast<int>(2 * SD));
                        for (ssize_t i = 0; i < left_rows * SD; ++i) {
                            const auto zr = scratch.left_contribution[i], zi = scratch.imaginary_intermediate[i];
                            scratch.left_contribution[i] = cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                        }
                    }
                } else cblas_zgemm(
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
            std::chrono::steady_clock::time_point right_start;
            if (profile_phases && profile_phases[0] == '1') {
                right_start = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(right_start - action_start).count();
                scratch.left_seconds += elapsed;
                const int layout = prefix >= 0 ? 0 : left_batch || !split_left[n].empty() ? 1 : !packed.empty() ? 2 : 3;
                scratch.left_layout_seconds[layout] += elapsed;
            }
            const auto& right_packed = packed_right[static_cast<std::size_t>(n)];
            const bool interleaved = static_cast<bool>(interleaved_right[n]);
            if (!right_packed.empty() || !real_right[n].empty() || !split_right[n].empty() || interleaved || !diagonal_right[n].values.empty()) {
                const bool real = !real_right[n].empty();
                const bool split = !split_right[n].empty();
                const bool transposed = (real || split) && !interleaved;
                if (!diagonal_right[n].values.empty()) {
                    std::fill_n(scratch.right_contribution.data(), rows * R * C, cdouble(0.0, 0.0));
                    const auto& factor = diagonal_right[n];
                    for (const auto& transition : factor.transitions) {
                        const ssize_t r = transition.first, s = transition.second;
                        const cdouble* values = factor.values.data() + (r * S + s) * C;
                        for (ssize_t row = 0; row < rows; ++row) {
                            const cdouble* input = intermediate + row * S * D + s * D;
                            cdouble* output = scratch.right_contribution.data() + (row * R + r) * C;
                            for (ssize_t c = 0; c < C; ++c) output[c] += values[c] * input[c];
                        }
                    }
                } else if (interleaved) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        rows, 2 * R * C, 2 * SD, 1.0,
                        reinterpret_cast<const double*>(intermediate), 2 * SD,
                        interleaved_right[n]->data(), 2 * R * C, 0.0,
                        reinterpret_cast<double*>(scratch.right_contribution.data()), 2 * R * C);
                } else if (real || split) {
                    for (ssize_t sd = 0; sd < SD; ++sd)
                        for (ssize_t row = 0; row < rows; ++row)
                            scratch.intermediate_s[sd * rows + row] = intermediate[row * SD + sd];
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(R * C), static_cast<int>(2 * rows), static_cast<int>(SD),
                        1.0, real ? real_right[n].data() : split_right[n].data(), static_cast<int>(SD),
                        reinterpret_cast<const double*>(scratch.intermediate_s.data()), static_cast<int>(2 * rows),
                        0.0, reinterpret_cast<double*>(scratch.right_contribution.data()), static_cast<int>(2 * rows));
                    if (split) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(R * C), static_cast<int>(2 * rows), static_cast<int>(SD),
                            1.0, split_right[n].data() + R * C * SD, static_cast<int>(SD),
                            reinterpret_cast<const double*>(scratch.intermediate_s.data()), static_cast<int>(2 * rows),
                            0.0, reinterpret_cast<double*>(scratch.imaginary_contribution.data()), static_cast<int>(2 * rows));
                        for (ssize_t i = 0; i < R * C * rows; ++i) {
                            const auto zr = scratch.right_contribution[i];
                            const auto zi = scratch.imaginary_contribution[i];
                            scratch.right_contribution[i] = cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                        }
                    }
                } else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(rows), static_cast<int>(R * C), static_cast<int>(SD),
                    &one, intermediate, static_cast<int>(SD),
                    right_packed.data(), static_cast<int>(SD), &zero,
                    scratch.right_contribution.data(), static_cast<int>(R * C));
                if (spectator_sum) {
                    for (ssize_t row = 0; row < rows; ++row)
                        for (ssize_t rc = 0; rc < R * C; ++rc)
                            spectator_sum[row * R * C + rc] += scratch.right_contribution[
                                transposed ? rc * rows + row : row * R * C + rc];
                } else if (prefix_out) {
                    if (transposed) {
                        for (ssize_t i = 0; i < PRC; ++i)
                            prefix_out[i] += scratch.right_contribution[i];
                    } else for (ssize_t rc = 0; rc < R * C; ++rc)
                        for (ssize_t p = 0; p < P; ++p)
                            prefix_out[rc * P + p] += scratch.right_contribution[p * R * C + rc];
                } else for (ssize_t p = 0; p < P; ++p)
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t rc = 0; rc < R * C; ++rc)
                            scratch.out[(a * P + p) * R * C + rc] +=
                                (bra ? bra[a] : one) * scratch.right_contribution[
                                    transposed ? rc * rows + (bra ? p : p * A + a)
                                         : (bra ? p : p * A + a) * R * C + rc];
            } else for (ssize_t s = 0; s < S; ++s) {
                const TwoSitePhysicalGroup& right_group =
                    right_groups[static_cast<std::size_t>(n) * S + s];
                if (right_group.outputs.empty()) {
                    continue;
                }
                const ssize_t right_rows =
                    static_cast<ssize_t>(right_group.outputs.size()) * C;
                const bool real = !right_group.real_blocks.empty();
                const bool split = !right_group.split_blocks.empty();
                const bool transposed = real || split;
                for (ssize_t ap = 0; ap < rows; ++ap) {
                    const cdouble* source =
                        intermediate + ap * SD + s * D;
                    cdouble* target = scratch.intermediate_s.data() + ap * D;
                    for (ssize_t d = 0; d < D; ++d) {
                        if (transposed) scratch.intermediate_s[d * rows + ap] = source[d];
                        else target[d] = source[d];
                    }
                }
                if (transposed) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(right_rows), static_cast<int>(2 * rows), static_cast<int>(D),
                    1.0, real ? right_group.real_blocks.data() : right_group.split_blocks.data(), static_cast<int>(D),
                    reinterpret_cast<const double*>(scratch.intermediate_s.data()), static_cast<int>(2 * rows),
                    0.0, reinterpret_cast<double*>(scratch.right_contribution.data()), static_cast<int>(2 * rows));
                    if (split) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(right_rows), static_cast<int>(2 * rows), static_cast<int>(D),
                            1.0, right_group.split_blocks.data() + right_rows * D, static_cast<int>(D),
                            reinterpret_cast<const double*>(scratch.intermediate_s.data()), static_cast<int>(2 * rows),
                            0.0, reinterpret_cast<double*>(scratch.imaginary_contribution.data()), static_cast<int>(2 * rows));
                        for (ssize_t i = 0; i < right_rows * rows; ++i) {
                            const auto zr = scratch.right_contribution[i], zi = scratch.imaginary_contribution[i];
                            scratch.right_contribution[i] = cdouble(zr.real() - zi.imag(), zr.imag() + zi.real());
                        }
                    }
                } else cblas_zgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    static_cast<int>(rows),
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
                if (spectator_sum) {
                    for (ssize_t row = 0; row < rows; ++row)
                        for (std::size_t rr = 0; rr < right_group.outputs.size(); ++rr)
                            for (ssize_t c = 0; c < C; ++c)
                                spectator_sum[(row * R + right_group.outputs[rr]) * C + c] +=
                                    scratch.right_contribution[transposed ? (rr * C + c) * rows + row
                                        : row * right_rows + rr * C + c];
                } else if (prefix_out) {
                    for (ssize_t p = 0; p < P; ++p)
                        for (std::size_t rr = 0; rr < right_group.outputs.size(); ++rr)
                            for (ssize_t c = 0; c < C; ++c)
                                prefix_out[(right_group.outputs[rr] * C + c) * P + p] +=
                                    scratch.right_contribution[transposed ? (rr * C + c) * rows + p
                                        : p * right_rows + rr * C + c];
                } else for (ssize_t p = 0; p < P; ++p) {
                    for (ssize_t a = 0; a < A; ++a) {
                        const ssize_t ap = bra ? p : p * A + a;
                        for (
                            std::size_t rr = 0;
                            rr < right_group.outputs.size();
                            ++rr
                        ) {
                            const ssize_t r = right_group.outputs[rr];
                            for (ssize_t c = 0; c < C; ++c) {
                                scratch.out[((a * P + p) * R + r) * C + c] +=
                                    (bra ? bra[a] : one) * scratch.right_contribution[
                                        transposed ? (static_cast<ssize_t>(rr) * C + c) * rows + ap
                                             : static_cast<std::size_t>(ap) * right_rows
                                               + static_cast<ssize_t>(rr) * C + c
                                    ];
                            }
                        }
                    }
                }
            }
            if (profile_phases && profile_phases[0] == '1') {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - right_start).count();
                scratch.right_seconds += elapsed;
                const int layout = !diagonal_right[n].values.empty() ? 0 : !real_right[n].empty() ? 1
                    : (!split_right[n].empty() || interleaved) ? 2 : !right_packed.empty() ? 3 : 4;
                scratch.right_layout_seconds[layout] += elapsed;
            }
            }
        }
        };

        auto apply_diagonal = [&](ssize_t p, Scratch& scratch) {
            const ssize_t count = diagonal_left_batch.size();
            // Keep channel intermediates cache-sized rather than streaming a
            // full channel-by-state output through memory twice per coordinate.
            for (ssize_t begin = 0; begin < count; begin += 32) {
            const ssize_t tile = std::min<ssize_t>(count - begin, 32);
            const auto start = std::chrono::steady_clock::now();
            if (split_diagonal_batch) {
                const ssize_t size = P * count * A * B;
                const cdouble* input = x_by_q.data() + p * B * SD;
                for (ssize_t i = 0; i < B * SD; ++i)
                    scratch.diagonal_rotated_input[i] = cdouble(-input[i].imag(), input[i].real());
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tile * A, 2 * SD, B, 1.0, diagonal_left_parts.data() + (p * count + begin) * A * B, B,
                    reinterpret_cast<const double*>(input), 2 * SD, 0.0,
                    reinterpret_cast<double*>(scratch.diagonal_left_output.data()), 2 * SD);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tile * A, 2 * SD, B, 1.0, diagonal_left_parts.data() + size + (p * count + begin) * A * B, B,
                    reinterpret_cast<const double*>(scratch.diagonal_rotated_input.data()), 2 * SD, 1.0,
                    reinterpret_cast<double*>(scratch.diagonal_left_output.data()), 2 * SD);
            } else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                tile * A, SD, B, &one, diagonal_left_bank.data() + (p * count + begin) * A * B, B,
                x_by_q.data() + p * B * SD, SD, &zero, scratch.diagonal_left_output.data(), SD);
            if (point_batch) {
                const auto right_start = std::chrono::steady_clock::now();
                for (ssize_t k = 0; k < tile; ++k)
                    for (ssize_t a = 0; a < A; ++a)
                        for (ssize_t s = 0; s < S; ++s)
                            scratch.out[(a * P + p) * S + s] +=
                                point_right_weights[(begin + k) * S + s]
                                * scratch.diagonal_left_output[(k * A + a) * S + s];
                if (profile_phases && profile_phases[0] == '1') {
                    const double left = std::chrono::duration<double>(right_start - start).count();
                    const double right = std::chrono::duration<double>(std::chrono::steady_clock::now() - right_start).count();
                    scratch.left_seconds += left; scratch.left_layout_seconds[3] += left;
                    scratch.right_seconds += right; scratch.right_layout_seconds[4] += right;
                }
                continue;
            }
            for (ssize_t k = 0; k < tile; ++k)
                for (ssize_t a = 0; a < A; ++a)
                    for (ssize_t sd = 0; sd < SD; ++sd)
                        scratch.diagonal_right_input[(k * SD + sd) * A + a] =
                            scratch.diagonal_left_output[(k * A + a) * SD + sd];
            const auto right_start = std::chrono::steady_clock::now();
            if (split_diagonal_batch) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    R * C, 2 * A, tile * SD, 1.0, diagonal_right_parts.data() + begin * SD, count * SD,
                    reinterpret_cast<const double*>(scratch.diagonal_right_input.data()), 2 * A, 0.0,
                    reinterpret_cast<double*>(scratch.diagonal_batch_output.data()), 2 * A);
                for (ssize_t i = 0; i < tile * SD * A; ++i) {
                    auto& value = scratch.diagonal_right_input[i];
                    value = cdouble(-value.imag(), value.real());
                }
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    R * C, 2 * A, tile * SD, 1.0, diagonal_right_parts.data() + R * C * count * SD + begin * SD, count * SD,
                    reinterpret_cast<const double*>(scratch.diagonal_right_input.data()), 2 * A, 1.0,
                    reinterpret_cast<double*>(scratch.diagonal_batch_output.data()), 2 * A);
            } else cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                R * C, A, tile * SD, &one, diagonal_right_bank.data() + begin * SD, count * SD,
                scratch.diagonal_right_input.data(), A, &zero, scratch.diagonal_batch_output.data(), A);
            for (ssize_t a = 0; a < A; ++a)
                for (ssize_t rc = 0; rc < R * C; ++rc)
                    scratch.out[(a * P + p) * R * C + rc] += scratch.diagonal_batch_output[rc * A + a];
            if (profile_phases && profile_phases[0] == '1') {
                const double left = std::chrono::duration<double>(right_start - start).count();
                const double right = std::chrono::duration<double>(std::chrono::steady_clock::now() - right_start).count();
                scratch.left_seconds += left; scratch.left_layout_seconds[3] += left;
                scratch.right_seconds += right; scratch.right_layout_seconds[4] += right;
            }
            }
        };
        if (thread_count == 1) {
            apply_range(0, N, scratches[0]);
            if (!diagonal_left_batch.empty()) for (ssize_t p = 0; p < P; ++p) apply_diagonal(p, scratches[0]);
        } else {
            std::atomic<ssize_t> next_n(0);
            std::atomic<ssize_t> next_p(0);
            auto apply_work = [&](int worker) {
                        Scratch& scratch = scratches[static_cast<std::size_t>(worker)];
                        while (true) {
                            const ssize_t n = next_n.fetch_add(
                                1, std::memory_order_relaxed
                            );
                            if (n >= N) {
                                break;
                            }
                            apply_range(n, n + 1, scratch);
                        }
                        if (!diagonal_left_batch.empty()) while (true) {
                            const ssize_t p = next_p.fetch_add(1, std::memory_order_relaxed);
                            if (p >= P) break;
                            apply_diagonal(p, scratch);
                        }
            };
            if (action_workers.active()) action_workers.run(apply_work);
            else {
                std::vector<std::thread> threads;
                threads.reserve(static_cast<std::size_t>(thread_count));
                for (int worker = 0; worker < thread_count; ++worker)
                    threads.emplace_back(apply_work, worker);
                for (auto& thread : threads) thread.join();
            }
        }
        out.assign(dim, cdouble(0.0, 0.0));
        for (const Scratch& scratch : scratches) {
            if (defer_spectator)
                for (std::size_t k = 0; k < sum_components->size(); ++k)
                    if (spectator_components[k] && bra_owner[k] == k)
                        cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            A, PRC, (*sum_components)[k].spectator_size, &one,
                            (*sum_components)[k].spectator_bra, A,
                            scratch.spectator_sum.data() + spectator_offsets[k], PRC,
                            &one, out.data(), PRC);
            for (std::size_t index = 0; index < dim; ++index) {
                out[index] += scratch.out[index];
            }
            for (std::size_t group = 0; group < output_bras.size(); ++group)
                for (ssize_t a = 0; a < A; ++a)
                    for (ssize_t p = 0; p < P; ++p)
                        for (ssize_t rc = 0; rc < R * C; ++rc)
                            out[(a * P + p) * R * C + rc] += output_bras[group][a]
                                * scratch.prefix_out[group * PRC + rc * P + p];
        }
    };

    auto factorized_matvec = [&](const std::vector<cdouble>& local, std::vector<cdouble>& out) {
        matvec(local, out);
        if (diagonal_factors.empty()) return;
        std::vector<cdouble> input(static_cast<std::size_t>(B) * D);
        std::vector<cdouble> output(static_cast<std::size_t>(A) * C);
        for (ssize_t p = 0; p < P; ++p)
            for (ssize_t r = 0; r < R; ++r) {
                for (ssize_t b = 0; b < B; ++b)
                    std::copy_n(local.data() + ((b * Q + p) * S + r) * D, D,
                        input.data() + b * D);
                cblas_zgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(A * C),
                    static_cast<int>(B * D), &one,
                    diagonal_factors.data() + (p * R + r) * A * C * B * D,
                    static_cast<int>(B * D), input.data(), 1, &zero, output.data(), 1);
                for (ssize_t a = 0; a < A; ++a)
                    for (ssize_t c = 0; c < C; ++c)
                        out[((a * P + p) * R + r) * C + c] += output[a * C + c];
            }
    };
    const auto solve_start = std::chrono::steady_clock::now();
    if (profile_phases && profile_phases[0] == '1')
        phases.emplace_back("workspace", std::chrono::duration<double>(solve_start - phase_start).count());
    auto result = lanczos_apply(vec, dt, krylov_dim, tol, factorized_matvec);
    if (midpoint) {
        auto projected = midpoint(result);
        if (projected.size() != vec.size())
            throw std::runtime_error("pair midpoint changed the local state dimension");
        result = lanczos_apply(projected, dt, krylov_dim, tol, factorized_matvec);
        reused_pair_actions.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<std::pair<std::string, double>> worker_times;
    if (profile_phases && profile_phases[0] == '1') {
        double left = 0.0, right = 0.0;
        for (const auto& scratch : scratches) { left += scratch.left_seconds; right += scratch.right_seconds; }
        worker_times = {{"left_action", left}, {"right_action", right}};
        for (std::size_t worker = 0; worker < scratches.size(); ++worker)
            worker_times.emplace_back("worker_" + std::to_string(worker),
                scratches[worker].left_seconds + scratches[worker].right_seconds);
        const char* left_names[] = {"left_prefix", "left_split", "left_packed", "left_sparse"};
        const char* right_names[] = {"right_diagonal", "right_real", "right_split", "right_packed", "right_sparse"};
        for (int i = 0; i < 4; ++i) {
            double seconds = 0;
            for (const auto& scratch : scratches) seconds += scratch.left_layout_seconds[i];
            worker_times.emplace_back(left_names[i], seconds);
        }
        for (int i = 0; i < 5; ++i) {
            double seconds = 0;
            for (const auto& scratch : scratches) seconds += scratch.right_layout_seconds[i];
            worker_times.emplace_back(right_names[i], seconds);
        }
    }
    record_contraction(true, {B, Q, S, D}, N, contraction_start, solve_start, std::move(phases), std::move(worker_times), {}, setup_threads);
    return result;
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
    int workers,
    const std::vector<CArray>& prefix_factors,
    py::object midpoint,
    const std::vector<std::vector<CArray>>& spectator_factors
) {
    const std::size_t count = lefts.size();
    if (!prefix_factors.empty() && prefix_factors.size() != count)
        throw std::runtime_error("prefix factor count must match component count");
    if (!spectator_factors.empty() && spectator_factors.size() != count)
        throw std::runtime_error("spectator factor count must match component count");
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
            left.ndim() != 3 || (first.ndim() != 3 && first.ndim() != 4)
            || (second.ndim() != 3 && second.ndim() != 4)
            || right.ndim() != 3
            || left.shape(0) != B || left.shape(2) != B
            || first.shape(0) != left.shape(1)
            || first.shape(2) != Q || (first.ndim() == 4 && first.shape(3) != Q)
            || second.shape(0) != first.shape(1)
            || second.shape(2) != S || (second.ndim() == 4 && second.shape(3) != S)
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
        components.back().first_diagonal = first_operators[k].ndim() == 3;
        components.back().second_diagonal = second_operators[k].ndim() == 3;
        if (!spectator_factors.empty() && !spectator_factors[k].empty()) {
            const auto& frame = spectator_factors[k];
            if (frame.size() != 3 || frame[0].ndim() != 2 || frame[1].ndim() != 2
                || frame[2].ndim() != 2 || frame[0].shape(0) > 2 * B
                || frame[0].shape(1) != B || frame[1].shape(1) != B
                || frame[1].shape(0) != frame[0].shape(0)
                || frame[2].shape(0) != lefts[k].shape(1)
                || frame[2].shape(1) != frame[0].shape(0))
                throw std::runtime_error("spectator factors require bra(K,B), ket(K,B), coefficients(M,K), K<=2B");
            components.back().spectator_size = frame[0].shape(0);
            components.back().spectator_bra = frame[0].data();
            components.back().spectator_ket = frame[1].data();
            components.back().spectator_coefficients = frame[2].data();
        }
        compact_pair_cores.fetch_add(components.back().first_diagonal + components.back().second_diagonal,
                                     std::memory_order_relaxed);
        if (!prefix_factors.empty() && prefix_factors[k].size()) {
            const auto& prefix = prefix_factors[k];
            if (prefix.ndim() != 2 || prefix.shape(0) != 2 || prefix.shape(1) != B
                || lefts[k].shape(1) != 1)
                throw std::runtime_error("prefix factors require shape (2,B) and MPO bond one");
            components.back().prefix_bra = prefix.data();
            components.back().prefix_ket = prefix.data() + B;
        }
    }
    std::vector<cdouble> vec(static_cast<std::size_t>(theta.size()));
    std::copy(theta.data(), theta.data() + theta.size(), vec.begin());
    std::vector<cdouble> result;
    std::function<std::vector<cdouble>(const std::vector<cdouble>&)> project;
    if (!midpoint.is_none()) {
        project = [&](const std::vector<cdouble>& local) {
            py::gil_scoped_acquire acquire;
            CArray state({B, Q, S, D});
            std::copy(local.begin(), local.end(), state.mutable_data());
            CArray projected = midpoint(state).cast<CArray>();
            if (projected.ndim() != 4 || projected.shape(0) != B || projected.shape(1) != Q
                || projected.shape(2) != S || projected.shape(3) != D)
                throw std::runtime_error("pair midpoint must preserve theta shape");
            return std::vector<cdouble>(projected.data(), projected.data() + projected.size());
        };
    }
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
            &components,
            project
        );
    }
    py::array_t<cdouble> compact_out({B, Q, S, D});
    std::copy(result.begin(), result.end(), compact_out.mutable_data());
    return compact_out;
#endif

    if (!midpoint.is_none())
        throw std::runtime_error("pair action reuse requires BLAS kernels");
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
                        ] = first.ndim() == 3 ? (p == q ? first.data()[(m * N + n) * Q + p] : cdouble(0.0, 0.0))
                                              : first.data()[((m * N + n) * Q + p) * Q + q];
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
                        ] = second.ndim() == 3 ? (r == s ? second.data()[(n * O + o) * S + r] : cdouble(0.0, 0.0))
                                               : second.data()[((n * O + o) * S + r) * S + s];
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
        reinterpret_cast<lapack_complex*>(a.data()),
        &lda_arg,
        reinterpret_cast<lapack_complex*>(tau.data()),
        reinterpret_cast<lapack_complex*>(&work_query),
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
        reinterpret_cast<lapack_complex*>(a.data()),
        &lda_arg,
        reinterpret_cast<lapack_complex*>(tau.data()),
        reinterpret_cast<lapack_complex*>(work.data()),
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
        reinterpret_cast<lapack_complex*>(a.data()),
        &lda_arg,
        reinterpret_cast<lapack_complex*>(tau.data()),
        reinterpret_cast<lapack_complex*>(&work_query),
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
        reinterpret_cast<lapack_complex*>(a.data()),
        &lda_arg,
        reinterpret_cast<lapack_complex*>(tau.data()),
        reinterpret_cast<lapack_complex*>(work.data()),
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
        py::arg("imaginary_time") = false,
        py::arg("spectator_factors") = std::vector<std::vector<CArray>>()
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
        "Lanczos evolution for a sum of two-site effective MPO operators",
        py::arg("theta"), py::arg("lefts"), py::arg("first_operators"),
        py::arg("second_operators"), py::arg("rights"), py::arg("dt"),
        py::arg("krylov_dim"), py::arg("tol"), py::arg("max_direct_sum_elements"),
        py::arg("workers"), py::arg("prefix_factors") = std::vector<CArray>(),
        py::arg("midpoint") = py::none(),
        py::arg("spectator_factors") = std::vector<std::vector<CArray>>()
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
