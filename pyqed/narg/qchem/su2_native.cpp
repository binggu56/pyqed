#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../../mps/nonabelian/su2_coupling_core.hpp"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using cdouble = std::complex<double>;

static int narg_openmp_threads = 1;
static long long narg_openmp_parallel_regions = 0;
static long long narg_openmp_tasks = 0;
static long long narg_growth_graph_calls = 0;
static long long narg_growth_graph_plans = 0;
static long long narg_growth_graph_outputs = 0;
static long long narg_growth_graph_candidates = 0;
static long long narg_growth_graph_output_bytes = 0;
static long long narg_growth_graph_mapped_bytes = 0;

static bool openmp_available_cpp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

static int set_narg_openmp_threads(int threads) {
    if (threads < 1) {
        throw std::invalid_argument("SU2-NARG thread count must be positive");
    }
#ifdef _OPENMP
    narg_openmp_threads = threads;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    narg_openmp_threads = 1;
#endif
    return narg_openmp_threads;
}

static py::dict narg_openmp_info() {
    py::dict out;
    out["available"] = py::bool_(openmp_available_cpp());
    out["threads"] = py::int_(narg_openmp_threads);
#ifdef _OPENMP
    out["max_threads"] = py::int_(omp_get_max_threads());
    out["version"] = py::int_(_OPENMP);
#else
    out["max_threads"] = py::int_(1);
    out["version"] = py::int_(0);
#endif
    out["parallel_regions"] = py::int_(narg_openmp_parallel_regions);
    out["tasks"] = py::int_(narg_openmp_tasks);
    out["growth_graph_calls"] = py::int_(narg_growth_graph_calls);
    out["growth_graph_plans"] = py::int_(narg_growth_graph_plans);
    out["growth_graph_outputs"] = py::int_(narg_growth_graph_outputs);
    out["growth_graph_candidates"] = py::int_(narg_growth_graph_candidates);
    out["growth_graph_output_bytes"] = py::int_(narg_growth_graph_output_bytes);
    out["growth_graph_mapped_bytes"] = py::int_(narg_growth_graph_mapped_bytes);
    return out;
}

static double cg(
    int left_j2,
    int left_m2,
    int right_j2,
    int right_m2,
    int coupled_j2,
    int coupled_m2
) {
    return pyqed::su2::clebsch_gordan_doubled(
        left_j2,
        right_j2,
        coupled_j2,
        left_m2,
        right_m2,
        coupled_m2
    );
}

static double product_tensor_pair_coeff(
    int bra_total_j2,
    int ket_total_j2,
    int total_rank2,
    int total_q2,
    int ket_total_m2,
    int bra_block_j2,
    int bra_local_j2,
    int ket_block_j2,
    int ket_local_j2,
    int block_rank2,
    int local_rank2,
    double atol
) {
    const int bra_total_m2 = ket_total_m2 + total_q2;
    if (bra_total_m2 < -bra_total_j2 || bra_total_m2 > bra_total_j2) {
        return 0.0;
    }

    double value = 0.0;
    for (int ket_block_m2 = -ket_block_j2; ket_block_m2 <= ket_block_j2; ket_block_m2 += 2) {
        const int ket_local_m2 = ket_total_m2 - ket_block_m2;
        if (ket_local_m2 < -ket_local_j2 || ket_local_m2 > ket_local_j2) {
            continue;
        }
        const double ket_cg = cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            ket_total_j2,
            ket_total_m2
        );
        if (std::abs(ket_cg) <= atol) {
            continue;
        }

        for (int q_block2 = -block_rank2; q_block2 <= block_rank2; q_block2 += 2) {
            const int q_local2 = total_q2 - q_block2;
            if (q_local2 < -local_rank2 || q_local2 > local_rank2) {
                continue;
            }
            const double tensor_cg = cg(
                block_rank2,
                q_block2,
                local_rank2,
                q_local2,
                total_rank2,
                total_q2
            );
            if (std::abs(tensor_cg) <= atol) {
                continue;
            }

            const int bra_block_m2 = ket_block_m2 + q_block2;
            const int bra_local_m2 = ket_local_m2 + q_local2;
            if (bra_block_m2 < -bra_block_j2 || bra_block_m2 > bra_block_j2) {
                continue;
            }
            if (bra_local_m2 < -bra_local_j2 || bra_local_m2 > bra_local_j2) {
                continue;
            }

            const double bra_cg = cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                bra_total_j2,
                bra_total_m2
            );
            const double block_cg = cg(
                ket_block_j2,
                ket_block_m2,
                block_rank2,
                q_block2,
                bra_block_j2,
                bra_block_m2
            );
            const double local_cg = cg(
                ket_local_j2,
                ket_local_m2,
                local_rank2,
                q_local2,
                bra_local_j2,
                bra_local_m2
            );
            if (
                std::abs(bra_cg) <= atol
                || std::abs(block_cg) <= atol
                || std::abs(local_cg) <= atol
            ) {
                continue;
            }

            value += (
                bra_cg
                * ket_cg
                * tensor_cg
                * block_cg
                * local_cg
                / std::sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            );
        }
    }
    return value;
}

#ifdef __APPLE__
extern "C" void cblas_zgemm(
    const int order,
    const int trans_a,
    const int trans_b,
    const int m,
    const int n,
    const int k,
    const void* alpha,
    const void* a,
    const int lda,
    const void* b,
    const int ldb,
    const void* beta,
    void* c,
    const int ldc
);
#endif

struct RotationJob {
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> u_bra;
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> block;
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> u_ket;
    py::array_t<cdouble> out;
    ssize_t source_bra;
    ssize_t source_ket;
    ssize_t kept_bra;
    ssize_t kept_ket;
};

static void rotate_operator_job(RotationJob& job) {
    const cdouble* u_bra = static_cast<const cdouble*>(job.u_bra.data());
    const cdouble* block = static_cast<const cdouble*>(job.block.data());
    const cdouble* u_ket = static_cast<const cdouble*>(job.u_ket.data());
    cdouble* out = static_cast<cdouble*>(job.out.mutable_data());
    std::vector<cdouble> temporary(
        static_cast<size_t>(job.kept_bra * job.source_ket),
        cdouble(0.0, 0.0)
    );

#ifdef __APPLE__
    constexpr int CblasRowMajor = 101;
    constexpr int CblasNoTrans = 111;
    constexpr int CblasConjTrans = 113;
    const cdouble alpha(1.0, 0.0);
    const cdouble beta_zero(0.0, 0.0);
    cblas_zgemm(
        CblasRowMajor,
        CblasConjTrans,
        CblasNoTrans,
        static_cast<int>(job.kept_bra),
        static_cast<int>(job.source_ket),
        static_cast<int>(job.source_bra),
        &alpha,
        u_bra,
        static_cast<int>(job.kept_bra),
        block,
        static_cast<int>(job.source_ket),
        &beta_zero,
        temporary.data(),
        static_cast<int>(job.source_ket)
    );
    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(job.kept_bra),
        static_cast<int>(job.kept_ket),
        static_cast<int>(job.source_ket),
        &alpha,
        temporary.data(),
        static_cast<int>(job.source_ket),
        u_ket,
        static_cast<int>(job.kept_ket),
        &beta_zero,
        out,
        static_cast<int>(job.kept_ket)
    );
#else
    for (ssize_t i = 0; i < job.kept_bra; ++i) {
        for (ssize_t k = 0; k < job.source_bra; ++k) {
            const cdouble left = std::conj(u_bra[k * job.kept_bra + i]);
            for (ssize_t j = 0; j < job.source_ket; ++j) {
                temporary[static_cast<size_t>(i * job.source_ket + j)] +=
                    left * block[k * job.source_ket + j];
            }
        }
    }
    for (ssize_t i = 0; i < job.kept_bra; ++i) {
        for (ssize_t k = 0; k < job.source_ket; ++k) {
            const cdouble left = temporary[static_cast<size_t>(i * job.source_ket + k)];
            for (ssize_t j = 0; j < job.kept_ket; ++j) {
                out[i * job.kept_ket + j] += left * u_ket[k * job.kept_ket + j];
            }
        }
    }
#endif
}

static py::list rotate_operator_blocks(py::sequence specs) {
    const ssize_t count = py::len(specs);
    std::vector<RotationJob> jobs;
    jobs.reserve(static_cast<size_t>(count));
    long long work = 0;

    for (ssize_t index = 0; index < count; ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(specs[index]);
        if (py::len(spec) != 3) {
            throw std::runtime_error(
                "operator rotation specs must be (U_bra, block, U_ket)"
            );
        }
        auto u_bra = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(spec[0]);
        auto block = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(spec[1]);
        auto u_ket = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(spec[2]);
        if (!u_bra || !block || !u_ket || u_bra.ndim() != 2 || block.ndim() != 2 || u_ket.ndim() != 2) {
            throw std::runtime_error("operator rotation inputs must be rank-2 complex arrays");
        }
        const ssize_t source_bra = block.shape(0);
        const ssize_t source_ket = block.shape(1);
        const ssize_t kept_bra = u_bra.shape(1);
        const ssize_t kept_ket = u_ket.shape(1);
        if (u_bra.shape(0) != source_bra || u_ket.shape(0) != source_ket) {
            throw std::runtime_error("operator rotation dimensions are incompatible");
        }
        if (
            source_bra > std::numeric_limits<int>::max()
            || source_ket > std::numeric_limits<int>::max()
            || kept_bra > std::numeric_limits<int>::max()
            || kept_ket > std::numeric_limits<int>::max()
        ) {
            throw std::runtime_error("operator rotation dimensions exceed native integer limits");
        }
        py::array_t<cdouble> out({kept_bra, kept_ket});
        work += static_cast<long long>(kept_bra) * source_ket
            * (source_bra + kept_ket);
        jobs.push_back(RotationJob{
            std::move(u_bra),
            std::move(block),
            std::move(u_ket),
            std::move(out),
            source_bra,
            source_ket,
            kept_bra,
            kept_ket,
        });
    }

    const bool parallel = openmp_available_cpp()
        && narg_openmp_threads > 1
        && count >= 4
        && work >= 32768;
    if (parallel) {
        ++narg_openmp_parallel_regions;
        narg_openmp_tasks += count;
    }
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(narg_openmp_threads) if(parallel)
#endif
        for (ssize_t index = 0; index < count; ++index) {
            rotate_operator_job(jobs[static_cast<size_t>(index)]);
        }
    }

    py::list out;
    for (auto& job : jobs) {
        out.append(job.out);
    }
    return out;
}

static py::array_t<cdouble> reduced_product_block_sum(
    const ssize_t rows,
    const ssize_t cols,
    py::sequence left_blocks,
    py::sequence right_blocks,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> weights
) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("rows and cols must be non-negative");
    }
    const ssize_t nterms = py::len(left_blocks);
    if (py::len(right_blocks) != nterms) {
        throw std::runtime_error("left_blocks and right_blocks must have the same length");
    }
    if (weights.ndim() != 1 || weights.shape(0) != nterms) {
        throw std::runtime_error("weights must be a one-dimensional array matching the term count");
    }

    py::array_t<cdouble> out({rows, cols});
    auto out_view = out.mutable_unchecked<2>();
    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            out_view(i, j) = cdouble(0.0, 0.0);
        }
    }
    if (rows == 0 || cols == 0 || nterms == 0) {
        return out;
    }

    std::vector<py::array_t<cdouble, py::array::c_style | py::array::forcecast>> left_arrays;
    std::vector<py::array_t<cdouble, py::array::c_style | py::array::forcecast>> right_arrays;
    std::vector<cdouble> packed_weights;
    left_arrays.reserve(static_cast<size_t>(nterms));
    right_arrays.reserve(static_cast<size_t>(nterms));
    packed_weights.reserve(static_cast<size_t>(nterms));

    auto w = weights.unchecked<1>();
    for (ssize_t term = 0; term < nterms; ++term) {
        auto left = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(
            left_blocks[term]
        );
        auto right = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(
            right_blocks[term]
        );
        if (!left || !right) {
            throw std::runtime_error("native reduced product expects dense complex arrays");
        }
        if (left.ndim() != 2 || right.ndim() != 2) {
            throw std::runtime_error("native reduced product expects rank-2 blocks");
        }
        if (left.shape(0) != rows || right.shape(1) != cols || left.shape(1) != right.shape(0)) {
            throw std::runtime_error("incompatible reduced product block dimensions");
        }
        if (left.shape(1) == 0 || w(term) == cdouble(0.0, 0.0)) {
            continue;
        }
        left_arrays.push_back(std::move(left));
        right_arrays.push_back(std::move(right));
        packed_weights.push_back(w(term));
    }

    {
        py::gil_scoped_release release;
#ifdef __APPLE__
        constexpr int CblasRowMajor = 101;
        constexpr int CblasNoTrans = 111;
        const cdouble beta_one(1.0, 0.0);
        cdouble* out_ptr = static_cast<cdouble*>(out.mutable_data());
        for (size_t term = 0; term < left_arrays.size(); ++term) {
            const auto& left = left_arrays[term];
            const auto& right = right_arrays[term];
            const int m = static_cast<int>(rows);
            const int n = static_cast<int>(cols);
            const int k = static_cast<int>(left.shape(1));
            const cdouble alpha = packed_weights[term];
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                &alpha,
                static_cast<const cdouble*>(left.data()),
                k,
                static_cast<const cdouble*>(right.data()),
                n,
                &beta_one,
                out_ptr,
                n
            );
        }
#else
        auto out_fallback = out.mutable_unchecked<2>();
        for (size_t term = 0; term < left_arrays.size(); ++term) {
            const cdouble weight = packed_weights[term];
            const auto& left = left_arrays[term];
            const auto& right = right_arrays[term];
            auto a = left.unchecked<2>();
            auto b = right.unchecked<2>();
            const ssize_t inner = left.shape(1);
            for (ssize_t i = 0; i < rows; ++i) {
                for (ssize_t kk = 0; kk < inner; ++kk) {
                    const cdouble scaled = weight * a(i, kk);
                    if (scaled == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    for (ssize_t j = 0; j < cols; ++j) {
                        out_fallback(i, j) += scaled * b(kk, j);
                    }
                }
            }
        }
#endif
    }

    return out;
}

static py::list reduced_product_block_sum_batch(py::sequence specs) {
    py::list out;
    const ssize_t nspecs = py::len(specs);
    for (ssize_t index = 0; index < nspecs; ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(specs[index]);
        if (py::len(spec) != 5) {
            std::ostringstream message;
            message << "batch spec " << index
                    << " must be (rows, cols, left_blocks, right_blocks, weights)";
            throw std::runtime_error(message.str());
        }
        auto weights = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(
            spec[4]
        );
        if (!weights) {
            throw std::runtime_error("batch weights must be convertible to complex128 arrays");
        }
        out.append(
            reduced_product_block_sum(
                spec[0].cast<ssize_t>(),
                spec[1].cast<ssize_t>(),
                py::reinterpret_borrow<py::sequence>(spec[2]),
                py::reinterpret_borrow<py::sequence>(spec[3]),
                weights
            )
        );
    }
    return out;
}

static void validate_state_table(const py::array& array, const char* name) {
    if (array.ndim() != 2 || array.shape(1) < 6) {
        std::ostringstream message;
        message << name << " must have shape (n, >=6)";
        throw std::runtime_error(message.str());
    }
}

static py::tuple product_tensor_pair_entries(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> bra_states,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ket_states,
    int bra_total_j2,
    int ket_total_j2,
    int total_rank2,
    int block_dnelec,
    int block_rank2,
    int local_dnelec,
    int local_rank2,
    double atol
) {
    validate_state_table(bra_states, "bra_states");
    validate_state_table(ket_states, "ket_states");

    const ssize_t bra_count = bra_states.shape(0);
    const ssize_t ket_count = ket_states.shape(0);
    ssize_t count = 0;
    ssize_t estimate_count = 0;

    auto bra = bra_states.unchecked<2>();
    auto ket = ket_states.unchecked<2>();

    {
        py::gil_scoped_release release;
        for (int total_q2 = -total_rank2; total_q2 <= total_rank2; total_q2 += 2) {
            for (int ket_total_m2 = -ket_total_j2; ket_total_m2 <= ket_total_j2; ket_total_m2 += 2) {
                const int bra_total_m2 = ket_total_m2 + total_q2;
                if (bra_total_m2 < -bra_total_j2 || bra_total_m2 > bra_total_j2) {
                    continue;
                }
                const double out_coeff = cg(
                    ket_total_j2,
                    ket_total_m2,
                    total_rank2,
                    total_q2,
                    bra_total_j2,
                    bra_total_m2
                );
                if (std::abs(out_coeff) <= atol) {
                    continue;
                }

                ssize_t local_count = 0;
                for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
                    const int bra_block_nelec = static_cast<int>(bra(bra_pos, 0));
                    const int bra_block_j2 = static_cast<int>(bra(bra_pos, 1));
                    const int bra_local_nelec = static_cast<int>(bra(bra_pos, 3));
                    const int bra_local_j2 = static_cast<int>(bra(bra_pos, 4));
                    for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
                        const int ket_block_nelec = static_cast<int>(ket(ket_pos, 0));
                        const int ket_block_j2 = static_cast<int>(ket(ket_pos, 1));
                        const int ket_local_nelec = static_cast<int>(ket(ket_pos, 3));
                        const int ket_local_j2 = static_cast<int>(ket(ket_pos, 4));
                        if (bra_block_nelec != ket_block_nelec + block_dnelec) {
                            continue;
                        }
                        if (bra_local_nelec != ket_local_nelec + local_dnelec) {
                            continue;
                        }
                        const double coeff = product_tensor_pair_coeff(
                            bra_total_j2,
                            ket_total_j2,
                            total_rank2,
                            total_q2,
                            ket_total_m2,
                            bra_block_j2,
                            bra_local_j2,
                            ket_block_j2,
                            ket_local_j2,
                            block_rank2,
                            local_rank2,
                            atol
                        );
                        if (std::abs(coeff) <= atol) {
                            continue;
                        }
                        ++local_count;
                    }
                }
                if (local_count != 0) {
                    ++estimate_count;
                    count += local_count;
                }
            }
        }
    }

    py::array_t<int64_t> rows({count});
    py::array_t<int64_t> cols({count});
    py::array_t<int64_t> block_rows({count});
    py::array_t<int64_t> block_cols({count});
    py::array_t<int64_t> local_rows({count});
    py::array_t<int64_t> local_cols({count});
    py::array_t<double> coeffs({count});

    if (count == 0 || estimate_count == 0) {
        return py::make_tuple(rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols);
    }

    auto rows_view = rows.mutable_unchecked<1>();
    auto cols_view = cols.mutable_unchecked<1>();
    auto block_rows_view = block_rows.mutable_unchecked<1>();
    auto block_cols_view = block_cols.mutable_unchecked<1>();
    auto local_rows_view = local_rows.mutable_unchecked<1>();
    auto local_cols_view = local_cols.mutable_unchecked<1>();
    auto coeffs_view = coeffs.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        ssize_t out_pos = 0;
        for (int total_q2 = -total_rank2; total_q2 <= total_rank2; total_q2 += 2) {
            for (int ket_total_m2 = -ket_total_j2; ket_total_m2 <= ket_total_j2; ket_total_m2 += 2) {
                const int bra_total_m2 = ket_total_m2 + total_q2;
                if (bra_total_m2 < -bra_total_j2 || bra_total_m2 > bra_total_j2) {
                    continue;
                }
                const double out_coeff = cg(
                    ket_total_j2,
                    ket_total_m2,
                    total_rank2,
                    total_q2,
                    bra_total_j2,
                    bra_total_m2
                );
                if (std::abs(out_coeff) <= atol) {
                    continue;
                }

                ssize_t local_count = 0;
                for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
                    const int bra_block_nelec = static_cast<int>(bra(bra_pos, 0));
                    const int bra_block_j2 = static_cast<int>(bra(bra_pos, 1));
                    const int bra_local_nelec = static_cast<int>(bra(bra_pos, 3));
                    const int bra_local_j2 = static_cast<int>(bra(bra_pos, 4));
                    for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
                        const int ket_block_nelec = static_cast<int>(ket(ket_pos, 0));
                        const int ket_block_j2 = static_cast<int>(ket(ket_pos, 1));
                        const int ket_local_nelec = static_cast<int>(ket(ket_pos, 3));
                        const int ket_local_j2 = static_cast<int>(ket(ket_pos, 4));
                        if (bra_block_nelec != ket_block_nelec + block_dnelec) {
                            continue;
                        }
                        if (bra_local_nelec != ket_local_nelec + local_dnelec) {
                            continue;
                        }
                        const double coeff = product_tensor_pair_coeff(
                            bra_total_j2,
                            ket_total_j2,
                            total_rank2,
                            total_q2,
                            ket_total_m2,
                            bra_block_j2,
                            bra_local_j2,
                            ket_block_j2,
                            ket_local_j2,
                            block_rank2,
                            local_rank2,
                            atol
                        );
                        if (std::abs(coeff) <= atol) {
                            continue;
                        }
                        ++local_count;
                    }
                }
                if (local_count == 0) {
                    continue;
                }

                const double scale = (
                    std::sqrt(bra_total_j2 + 1.0)
                    / (out_coeff * static_cast<double>(estimate_count))
                );
                for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
                    const int bra_block_nelec = static_cast<int>(bra(bra_pos, 0));
                    const int bra_block_j2 = static_cast<int>(bra(bra_pos, 1));
                    const int bra_local_nelec = static_cast<int>(bra(bra_pos, 3));
                    const int bra_local_j2 = static_cast<int>(bra(bra_pos, 4));
                    for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
                        const int ket_block_nelec = static_cast<int>(ket(ket_pos, 0));
                        const int ket_block_j2 = static_cast<int>(ket(ket_pos, 1));
                        const int ket_local_nelec = static_cast<int>(ket(ket_pos, 3));
                        const int ket_local_j2 = static_cast<int>(ket(ket_pos, 4));
                        if (bra_block_nelec != ket_block_nelec + block_dnelec) {
                            continue;
                        }
                        if (bra_local_nelec != ket_local_nelec + local_dnelec) {
                            continue;
                        }
                        const double coeff = product_tensor_pair_coeff(
                            bra_total_j2,
                            ket_total_j2,
                            total_rank2,
                            total_q2,
                            ket_total_m2,
                            bra_block_j2,
                            bra_local_j2,
                            ket_block_j2,
                            ket_local_j2,
                            block_rank2,
                            local_rank2,
                            atol
                        );
                        if (std::abs(coeff) <= atol) {
                            continue;
                        }
                        rows_view(out_pos) = bra_pos;
                        cols_view(out_pos) = ket_pos;
                        block_rows_view(out_pos) = bra(bra_pos, 2);
                        block_cols_view(out_pos) = ket(ket_pos, 2);
                        local_rows_view(out_pos) = bra(bra_pos, 5);
                        local_cols_view(out_pos) = ket(ket_pos, 5);
                        coeffs_view(out_pos) = coeff * scale;
                        ++out_pos;
                    }
                }
            }
        }
    }

    return py::make_tuple(rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols);
}

struct ProductPairJob {
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> bra_states;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ket_states;
    int bra_total_j2;
    int ket_total_j2;
    int total_rank2;
    int block_dnelec;
    int block_rank2;
    int local_dnelec;
    int local_rank2;
    double atol;
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<int64_t> block_rows;
    std::vector<int64_t> block_cols;
    std::vector<int64_t> local_rows;
    std::vector<int64_t> local_cols;
    std::vector<double> coeffs;
};

static void product_tensor_pair_job(ProductPairJob& job) {
    const ssize_t bra_count = job.bra_states.shape(0);
    const ssize_t ket_count = job.ket_states.shape(0);
    const ssize_t bra_stride = job.bra_states.shape(1);
    const ssize_t ket_stride = job.ket_states.shape(1);
    const int64_t* bra = static_cast<const int64_t*>(job.bra_states.data());
    const int64_t* ket = static_cast<const int64_t*>(job.ket_states.data());
    ssize_t estimate_count = 0;

    for (int total_q2 = -job.total_rank2; total_q2 <= job.total_rank2; total_q2 += 2) {
        for (
            int ket_total_m2 = -job.ket_total_j2;
            ket_total_m2 <= job.ket_total_j2;
            ket_total_m2 += 2
        ) {
            const int bra_total_m2 = ket_total_m2 + total_q2;
            if (bra_total_m2 < -job.bra_total_j2 || bra_total_m2 > job.bra_total_j2) {
                continue;
            }
            const double out_coeff = cg(
                job.ket_total_j2,
                ket_total_m2,
                job.total_rank2,
                total_q2,
                job.bra_total_j2,
                bra_total_m2
            );
            if (std::abs(out_coeff) <= job.atol) {
                continue;
            }

            const size_t estimate_start = job.coeffs.size();
            for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
                const int64_t* bra_row = bra + bra_stride * bra_pos;
                for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
                    const int64_t* ket_row = ket + ket_stride * ket_pos;
                    if (bra_row[0] != ket_row[0] + job.block_dnelec) {
                        continue;
                    }
                    if (bra_row[3] != ket_row[3] + job.local_dnelec) {
                        continue;
                    }
                    const double coeff = product_tensor_pair_coeff(
                        job.bra_total_j2,
                        job.ket_total_j2,
                        job.total_rank2,
                        total_q2,
                        ket_total_m2,
                        static_cast<int>(bra_row[1]),
                        static_cast<int>(bra_row[4]),
                        static_cast<int>(ket_row[1]),
                        static_cast<int>(ket_row[4]),
                        job.block_rank2,
                        job.local_rank2,
                        job.atol
                    );
                    if (std::abs(coeff) <= job.atol) {
                        continue;
                    }
                    job.rows.push_back(bra_pos);
                    job.cols.push_back(ket_pos);
                    job.block_rows.push_back(bra_row[2]);
                    job.block_cols.push_back(ket_row[2]);
                    job.local_rows.push_back(bra_row[5]);
                    job.local_cols.push_back(ket_row[5]);
                    job.coeffs.push_back(
                        coeff * std::sqrt(job.bra_total_j2 + 1.0) / out_coeff
                    );
                }
            }
            if (job.coeffs.size() != estimate_start) {
                ++estimate_count;
            }
        }
    }
    if (estimate_count != 0) {
        const double inverse = 1.0 / static_cast<double>(estimate_count);
        for (double& coeff : job.coeffs) {
            coeff *= inverse;
        }
    }
}

static py::list product_tensor_pair_entries_batch(py::sequence specs) {
    const ssize_t count = py::len(specs);
    std::vector<ProductPairJob> jobs;
    jobs.reserve(static_cast<size_t>(count));
    long long work = 0;
    for (ssize_t index = 0; index < count; ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(specs[index]);
        if (py::len(spec) != 10) {
            throw std::runtime_error("product pair batch specs must contain 10 fields");
        }
        auto bra = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(spec[0]);
        auto ket = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(spec[1]);
        if (!bra || !ket) {
            throw std::runtime_error("product pair batch states must be int64 arrays");
        }
        validate_state_table(bra, "bra_states");
        validate_state_table(ket, "ket_states");
        const int total_rank2 = spec[4].cast<int>();
        const int ket_total_j2 = spec[3].cast<int>();
        work += static_cast<long long>(bra.shape(0)) * ket.shape(0)
            * (total_rank2 + 1) * (ket_total_j2 + 1);
        jobs.push_back(ProductPairJob{
            std::move(bra),
            std::move(ket),
            spec[2].cast<int>(),
            ket_total_j2,
            total_rank2,
            spec[5].cast<int>(),
            spec[6].cast<int>(),
            spec[7].cast<int>(),
            spec[8].cast<int>(),
            spec[9].cast<double>(),
        });
    }

    const bool parallel = openmp_available_cpp()
        && narg_openmp_threads > 1
        && count >= 4
        && work >= 32768;
    if (parallel) {
        ++narg_openmp_parallel_regions;
        narg_openmp_tasks += count;
    }
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(narg_openmp_threads) if(parallel)
#endif
        for (ssize_t index = 0; index < count; ++index) {
            product_tensor_pair_job(jobs[static_cast<size_t>(index)]);
        }
    }

    py::list out;
    for (const auto& job : jobs) {
        const ssize_t size = static_cast<ssize_t>(job.coeffs.size());
        py::array_t<int64_t> rows({size});
        py::array_t<int64_t> cols({size});
        py::array_t<int64_t> block_rows({size});
        py::array_t<int64_t> block_cols({size});
        py::array_t<int64_t> local_rows({size});
        py::array_t<int64_t> local_cols({size});
        py::array_t<double> coeffs({size});
        if (size != 0) {
            std::memcpy(rows.mutable_data(), job.rows.data(), sizeof(int64_t) * size);
            std::memcpy(cols.mutable_data(), job.cols.data(), sizeof(int64_t) * size);
            std::memcpy(block_rows.mutable_data(), job.block_rows.data(), sizeof(int64_t) * size);
            std::memcpy(block_cols.mutable_data(), job.block_cols.data(), sizeof(int64_t) * size);
            std::memcpy(local_rows.mutable_data(), job.local_rows.data(), sizeof(int64_t) * size);
            std::memcpy(local_cols.mutable_data(), job.local_cols.data(), sizeof(int64_t) * size);
            std::memcpy(coeffs.mutable_data(), job.coeffs.data(), sizeof(double) * size);
        }
        out.append(py::make_tuple(
            rows,
            cols,
            coeffs,
            block_rows,
            block_cols,
            local_rows,
            local_cols
        ));
    }
    return out;
}

struct ChargePairKey {
    int values[4];

    bool operator==(const ChargePairKey& other) const {
        for (int index = 0; index < 4; ++index) {
            if (values[index] != other.values[index]) {
                return false;
            }
        }
        return true;
    }
};

struct ChargePairKeyHash {
    size_t operator()(const ChargePairKey& key) const {
        size_t seed = 1469598103934665603ull;
        for (int index = 0; index < 4; ++index) {
            seed ^= static_cast<size_t>(key.values[index])
                + 0x9e3779b97f4a7c15ull
                + (seed << 6)
                + (seed >> 2);
        }
        return seed;
    }
};

struct AngularJKey {
    int values[4];

    bool operator==(const AngularJKey& other) const {
        for (int index = 0; index < 4; ++index) {
            if (values[index] != other.values[index]) {
                return false;
            }
        }
        return true;
    }
};

struct AngularJKeyHash {
    size_t operator()(const AngularJKey& key) const {
        size_t seed = 1469598103934665603ull;
        for (int index = 0; index < 4; ++index) {
            seed ^= static_cast<size_t>(key.values[index])
                + 0x9e3779b97f4a7c15ull
                + (seed << 6)
                + (seed >> 2);
        }
        return seed;
    }
};

struct GrowthMatrix {
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> values;
};

using GrowthMatrixTable = std::unordered_map<
    ChargePairKey,
    GrowthMatrix,
    ChargePairKeyHash
>;

struct GrowthSector {
    int nelec;
    int j2;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> states;
};

struct GrowthRequest {
    int block_dnelec;
    int block_rank2;
    int local_dnelec;
    int local_rank2;
    int total_rank2;
    GrowthMatrixTable block_matrices;
    GrowthMatrixTable local_matrices;
};

struct GrowthEstimate {
    int total_q2;
    int ket_total_m2;
    double scale;
};

struct GrowthAngularPlanKey {
    int values[7];

    bool operator==(const GrowthAngularPlanKey& other) const {
        for (int index = 0; index < 7; ++index) {
            if (values[index] != other.values[index]) {
                return false;
            }
        }
        return true;
    }
};

struct GrowthAngularPlanKeyHash {
    size_t operator()(const GrowthAngularPlanKey& key) const {
        size_t seed = 1469598103934665603ull;
        for (int index = 0; index < 7; ++index) {
            seed ^= static_cast<size_t>(key.values[index])
                + 0x9e3779b97f4a7c15ull
                + (seed << 6)
                + (seed >> 2);
        }
        return seed;
    }
};

struct GrowthAngularPlan {
    std::unordered_map<AngularJKey, double, AngularJKeyHash> coefficients;
};

struct MappedGrowthBuffer {
    void* data;
    size_t bytes;
};

static py::array_t<cdouble> allocate_growth_output(ssize_t rows, ssize_t cols) {
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols)
        * sizeof(cdouble);
#if defined(__APPLE__) || defined(__linux__)
    constexpr size_t map_threshold = 32 * 1024;
    if (bytes >= map_threshold) {
        void* data = mmap(
            nullptr,
            bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (data != MAP_FAILED) {
            auto* allocation = new MappedGrowthBuffer{data, bytes};
            py::capsule owner(allocation, [](void* pointer) {
                auto* owned = static_cast<MappedGrowthBuffer*>(pointer);
                munmap(owned->data, owned->bytes);
                delete owned;
            });
            narg_growth_graph_mapped_bytes += static_cast<long long>(bytes);
            return py::array_t<cdouble>(
                {rows, cols},
                {
                    static_cast<ssize_t>(cols * sizeof(cdouble)),
                    static_cast<ssize_t>(sizeof(cdouble)),
                },
                static_cast<cdouble*>(data),
                owner
            );
        }
    }
#endif
    return py::array_t<cdouble>({rows, cols});
}

struct GrowthOutputJob {
    size_t request_index;
    size_t bra_sector_index;
    size_t ket_sector_index;
    size_t angular_plan_index;
    py::array_t<cdouble> out;
    long long work;
    double max_abs;
};

static GrowthMatrixTable parse_growth_matrix_table(
    py::sequence specs,
    const char* name
) {
    GrowthMatrixTable out;
    out.reserve(static_cast<size_t>(py::len(specs)));
    for (ssize_t index = 0; index < py::len(specs); ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(specs[index]);
        if (py::len(spec) != 5) {
            std::ostringstream message;
            message << name << " block specs must be (bra_nelec, bra_j2, "
                    << "ket_nelec, ket_j2, matrix)";
            throw std::runtime_error(message.str());
        }
        auto matrix = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(
            spec[4]
        );
        if (!matrix || matrix.ndim() != 2) {
            std::ostringstream message;
            message << name << " matrices must be rank-2 complex arrays";
            throw std::runtime_error(message.str());
        }
        ChargePairKey key{{
            spec[0].cast<int>(),
            spec[1].cast<int>(),
            spec[2].cast<int>(),
            spec[3].cast<int>(),
        }};
        out.insert_or_assign(key, GrowthMatrix{std::move(matrix)});
    }
    return out;
}

static bool growth_sector_pair_allowed(
    const GrowthSector& bra,
    const GrowthSector& ket,
    const GrowthRequest& request
) {
    const int dnelec = request.block_dnelec + request.local_dnelec;
    if (bra.nelec != ket.nelec + dnelec) {
        return false;
    }
    if (std::abs(ket.j2 - request.total_rank2) > bra.j2) {
        return false;
    }
    if (bra.j2 > ket.j2 + request.total_rank2) {
        return false;
    }
    return (ket.j2 + request.total_rank2 + bra.j2) % 2 == 0;
}

static GrowthAngularPlan build_growth_angular_plan(
    const GrowthSector& bra_sector,
    const GrowthSector& ket_sector,
    const GrowthRequest& request,
    double atol
) {
    const ssize_t bra_count = bra_sector.states.shape(0);
    const ssize_t ket_count = ket_sector.states.shape(0);
    const ssize_t bra_stride = bra_sector.states.shape(1);
    const ssize_t ket_stride = ket_sector.states.shape(1);
    const int64_t* bra_states = static_cast<const int64_t*>(bra_sector.states.data());
    const int64_t* ket_states = static_cast<const int64_t*>(ket_sector.states.data());
    std::vector<GrowthEstimate> estimates;

    for (
        int total_q2 = -request.total_rank2;
        total_q2 <= request.total_rank2;
        total_q2 += 2
    ) {
        for (
            int ket_total_m2 = -ket_sector.j2;
            ket_total_m2 <= ket_sector.j2;
            ket_total_m2 += 2
        ) {
            const int bra_total_m2 = ket_total_m2 + total_q2;
            if (bra_total_m2 < -bra_sector.j2 || bra_total_m2 > bra_sector.j2) {
                continue;
            }
            const double out_coeff = cg(
                ket_sector.j2,
                ket_total_m2,
                request.total_rank2,
                total_q2,
                bra_sector.j2,
                bra_total_m2
            );
            if (std::abs(out_coeff) <= atol) {
                continue;
            }
            bool has_entry = false;
            for (ssize_t bra_pos = 0; bra_pos < bra_count && !has_entry; ++bra_pos) {
                const int64_t* bra = bra_states + bra_stride * bra_pos;
                for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
                    const int64_t* ket = ket_states + ket_stride * ket_pos;
                    if (bra[0] != ket[0] + request.block_dnelec) {
                        continue;
                    }
                    if (bra[3] != ket[3] + request.local_dnelec) {
                        continue;
                    }
                    const double coeff = product_tensor_pair_coeff(
                        bra_sector.j2,
                        ket_sector.j2,
                        request.total_rank2,
                        total_q2,
                        ket_total_m2,
                        static_cast<int>(bra[1]),
                        static_cast<int>(bra[4]),
                        static_cast<int>(ket[1]),
                        static_cast<int>(ket[4]),
                        request.block_rank2,
                        request.local_rank2,
                        atol
                    );
                    if (std::abs(coeff) > atol) {
                        has_entry = true;
                        break;
                    }
                }
            }
            if (has_entry) {
                estimates.push_back(GrowthEstimate{
                    total_q2,
                    ket_total_m2,
                    std::sqrt(bra_sector.j2 + 1.0) / out_coeff,
                });
            }
        }
    }

    GrowthAngularPlan plan;
    if (estimates.empty()) {
        return plan;
    }
    const double inverse_estimates = 1.0 / static_cast<double>(estimates.size());
    for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
        const int64_t* bra = bra_states + bra_stride * bra_pos;
        for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
            const int64_t* ket = ket_states + ket_stride * ket_pos;
            if (bra[0] != ket[0] + request.block_dnelec) {
                continue;
            }
            if (bra[3] != ket[3] + request.local_dnelec) {
                continue;
            }
            const AngularJKey angular_key{{
                static_cast<int>(bra[1]),
                static_cast<int>(bra[4]),
                static_cast<int>(ket[1]),
                static_cast<int>(ket[4]),
            }};
            if (plan.coefficients.find(angular_key) != plan.coefficients.end()) {
                continue;
            }
            double angular = 0.0;
            for (const GrowthEstimate& estimate : estimates) {
                angular += estimate.scale * product_tensor_pair_coeff(
                    bra_sector.j2,
                    ket_sector.j2,
                    request.total_rank2,
                    estimate.total_q2,
                    estimate.ket_total_m2,
                    static_cast<int>(bra[1]),
                    static_cast<int>(bra[4]),
                    static_cast<int>(ket[1]),
                    static_cast<int>(ket[4]),
                    request.block_rank2,
                    request.local_rank2,
                    atol
                );
            }
            plan.coefficients.emplace(angular_key, angular * inverse_estimates);
        }
    }
    return plan;
}

static void evaluate_reduced_growth_output(
    GrowthOutputJob& output,
    const std::vector<GrowthSector>& sectors,
    const std::vector<GrowthRequest>& requests,
    const std::vector<GrowthAngularPlan>& angular_plans,
    double atol
) {
    const GrowthSector& bra_sector = sectors[output.bra_sector_index];
    const GrowthSector& ket_sector = sectors[output.ket_sector_index];
    const GrowthRequest& request = requests[output.request_index];
    const GrowthAngularPlan& angular_plan = angular_plans[output.angular_plan_index];
    const ssize_t bra_count = bra_sector.states.shape(0);
    const ssize_t ket_count = ket_sector.states.shape(0);
    const ssize_t bra_stride = bra_sector.states.shape(1);
    const ssize_t ket_stride = ket_sector.states.shape(1);
    const int64_t* bra_states = static_cast<const int64_t*>(bra_sector.states.data());
    const int64_t* ket_states = static_cast<const int64_t*>(ket_sector.states.data());
    cdouble* out = static_cast<cdouble*>(output.out.mutable_data());
    std::fill(
        out,
        out + static_cast<size_t>(bra_count * ket_count),
        cdouble(0.0, 0.0)
    );
    double max_abs = 0.0;

    for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
        const int64_t* bra = bra_states + bra_stride * bra_pos;
        for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
            const int64_t* ket = ket_states + ket_stride * ket_pos;
            if (bra[0] != ket[0] + request.block_dnelec) {
                continue;
            }
            if (bra[3] != ket[3] + request.local_dnelec) {
                continue;
            }

            const ChargePairKey block_key{{
                static_cast<int>(bra[0]),
                static_cast<int>(bra[1]),
                static_cast<int>(ket[0]),
                static_cast<int>(ket[1]),
            }};
            const ChargePairKey local_key{{
                static_cast<int>(bra[3]),
                static_cast<int>(bra[4]),
                static_cast<int>(ket[3]),
                static_cast<int>(ket[4]),
            }};
            const auto block_found = request.block_matrices.find(block_key);
            if (block_found == request.block_matrices.end()) {
                continue;
            }
            const auto local_found = request.local_matrices.find(local_key);
            if (local_found == request.local_matrices.end()) {
                continue;
            }

            const AngularJKey angular_key{{
                static_cast<int>(bra[1]),
                static_cast<int>(bra[4]),
                static_cast<int>(ket[1]),
                static_cast<int>(ket[4]),
            }};
            const auto angular_found = angular_plan.coefficients.find(angular_key);
            if (angular_found == angular_plan.coefficients.end()) {
                continue;
            }
            const double angular = angular_found->second;
            if (std::abs(angular) <= atol) {
                continue;
            }

            const auto& block = block_found->second.values;
            const auto& local = local_found->second.values;
            const ssize_t block_row = bra[2];
            const ssize_t block_col = ket[2];
            const ssize_t local_row = bra[5];
            const ssize_t local_col = ket[5];
            if (
                block_row < 0 || block_row >= block.shape(0)
                || block_col < 0 || block_col >= block.shape(1)
                || local_row < 0 || local_row >= local.shape(0)
                || local_col < 0 || local_col >= local.shape(1)
            ) {
                continue;
            }
            const cdouble* block_values = static_cast<const cdouble*>(block.data());
            const cdouble* local_values = static_cast<const cdouble*>(local.data());
            const cdouble value = (
                angular
                * block_values[block_row * block.shape(1) + block_col]
                * local_values[local_row * local.shape(1) + local_col]
            );
            out[bra_pos * ket_count + ket_pos] = value;
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    output.max_abs = max_abs;
}

static bool growth_output_has_support(
    const GrowthSector& bra_sector,
    const GrowthSector& ket_sector,
    const GrowthRequest& request,
    const GrowthAngularPlan& angular_plan,
    double atol
) {
    const ssize_t bra_count = bra_sector.states.shape(0);
    const ssize_t ket_count = ket_sector.states.shape(0);
    const ssize_t bra_stride = bra_sector.states.shape(1);
    const ssize_t ket_stride = ket_sector.states.shape(1);
    const int64_t* bra_states = static_cast<const int64_t*>(bra_sector.states.data());
    const int64_t* ket_states = static_cast<const int64_t*>(ket_sector.states.data());

    for (ssize_t bra_pos = 0; bra_pos < bra_count; ++bra_pos) {
        const int64_t* bra = bra_states + bra_stride * bra_pos;
        for (ssize_t ket_pos = 0; ket_pos < ket_count; ++ket_pos) {
            const int64_t* ket = ket_states + ket_stride * ket_pos;
            if (bra[0] != ket[0] + request.block_dnelec) {
                continue;
            }
            if (bra[3] != ket[3] + request.local_dnelec) {
                continue;
            }
            const ChargePairKey block_key{{
                static_cast<int>(bra[0]),
                static_cast<int>(bra[1]),
                static_cast<int>(ket[0]),
                static_cast<int>(ket[1]),
            }};
            const ChargePairKey local_key{{
                static_cast<int>(bra[3]),
                static_cast<int>(bra[4]),
                static_cast<int>(ket[3]),
                static_cast<int>(ket[4]),
            }};
            const auto block_found = request.block_matrices.find(block_key);
            const auto local_found = request.local_matrices.find(local_key);
            if (
                block_found == request.block_matrices.end()
                || local_found == request.local_matrices.end()
            ) {
                continue;
            }
            const AngularJKey angular_key{{
                static_cast<int>(bra[1]),
                static_cast<int>(bra[4]),
                static_cast<int>(ket[1]),
                static_cast<int>(ket[4]),
            }};
            const auto angular_found = angular_plan.coefficients.find(angular_key);
            if (
                angular_found == angular_plan.coefficients.end()
                || std::abs(angular_found->second) <= atol
            ) {
                continue;
            }
            const auto& block = block_found->second.values;
            const auto& local = local_found->second.values;
            const ssize_t block_row = bra[2];
            const ssize_t block_col = ket[2];
            const ssize_t local_row = bra[5];
            const ssize_t local_col = ket[5];
            if (
                block_row < 0 || block_row >= block.shape(0)
                || block_col < 0 || block_col >= block.shape(1)
                || local_row < 0 || local_row >= local.shape(0)
                || local_col < 0 || local_col >= local.shape(1)
            ) {
                continue;
            }
            const cdouble* block_values = static_cast<const cdouble*>(block.data());
            const cdouble* local_values = static_cast<const cdouble*>(local.data());
            if (
                block_values[block_row * block.shape(1) + block_col] != cdouble(0.0, 0.0)
                && local_values[local_row * local.shape(1) + local_col] != cdouble(0.0, 0.0)
            ) {
                return true;
            }
        }
    }
    return false;
}

static py::list reduced_growth_graph(
    py::sequence sector_specs,
    py::sequence request_specs,
    double atol
) {
    std::vector<GrowthSector> sectors;
    sectors.reserve(static_cast<size_t>(py::len(sector_specs)));
    for (ssize_t index = 0; index < py::len(sector_specs); ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(sector_specs[index]);
        if (py::len(spec) != 3) {
            throw std::runtime_error(
                "growth sectors must be (nelec, j2, state_table)"
            );
        }
        auto states = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(
            spec[2]
        );
        if (!states) {
            throw std::runtime_error("growth state tables must be int64 arrays");
        }
        validate_state_table(states, "growth states");
        sectors.push_back(GrowthSector{
            spec[0].cast<int>(),
            spec[1].cast<int>(),
            std::move(states),
        });
    }

    std::vector<GrowthRequest> requests;
    requests.reserve(static_cast<size_t>(py::len(request_specs)));
    for (ssize_t index = 0; index < py::len(request_specs); ++index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(request_specs[index]);
        if (py::len(spec) != 7) {
            throw std::runtime_error(
                "growth requests must contain five charges/ranks and two block tables"
            );
        }
        requests.push_back(GrowthRequest{
            spec[0].cast<int>(),
            spec[1].cast<int>(),
            spec[2].cast<int>(),
            spec[3].cast<int>(),
            spec[4].cast<int>(),
            parse_growth_matrix_table(
                py::reinterpret_borrow<py::sequence>(spec[5]),
                "retained"
            ),
            parse_growth_matrix_table(
                py::reinterpret_borrow<py::sequence>(spec[6]),
                "local"
            ),
        });
    }

    std::vector<GrowthOutputJob> outputs;
    std::vector<GrowthAngularPlan> angular_plans;
    std::unordered_map<
        GrowthAngularPlanKey,
        size_t,
        GrowthAngularPlanKeyHash
    > angular_plan_indices;
    long long total_work = 0;
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const GrowthRequest& request = requests[request_index];
        for (size_t bra_index = 0; bra_index < sectors.size(); ++bra_index) {
            const GrowthSector& bra = sectors[bra_index];
            for (size_t ket_index = 0; ket_index < sectors.size(); ++ket_index) {
                const GrowthSector& ket = sectors[ket_index];
                if (!growth_sector_pair_allowed(bra, ket, request)) {
                    continue;
                }
                const GrowthAngularPlanKey plan_key{{
                    request.block_dnelec,
                    request.block_rank2,
                    request.local_dnelec,
                    request.local_rank2,
                    request.total_rank2,
                    static_cast<int>(bra_index),
                    static_cast<int>(ket_index),
                }};
                auto plan_found = angular_plan_indices.find(plan_key);
                size_t plan_index = 0;
                if (plan_found == angular_plan_indices.end()) {
                    plan_index = angular_plans.size();
                    angular_plans.push_back(
                        build_growth_angular_plan(bra, ket, request, atol)
                    );
                    angular_plan_indices.emplace(plan_key, plan_index);
                } else {
                    plan_index = plan_found->second;
                }
                const ssize_t bra_count = bra.states.shape(0);
                const ssize_t ket_count = ket.states.shape(0);
                ++narg_growth_graph_candidates;
                if (!growth_output_has_support(
                    bra,
                    ket,
                    request,
                    angular_plans[plan_index],
                    atol
                )) {
                    continue;
                }
                const long long work = static_cast<long long>(bra_count) * ket_count
                    * (request.total_rank2 + 1) * (ket.j2 + 1);
                outputs.push_back(GrowthOutputJob{
                    request_index,
                    bra_index,
                    ket_index,
                    plan_index,
                    allocate_growth_output(bra_count, ket_count),
                    work,
                    0.0,
                });
                narg_growth_graph_output_bytes += static_cast<long long>(
                    sizeof(cdouble) * bra_count * ket_count
                );
                total_work += work;
            }
        }
    }
    ++narg_growth_graph_calls;
    narg_growth_graph_plans += static_cast<long long>(angular_plans.size());
    narg_growth_graph_outputs += static_cast<long long>(outputs.size());

    const bool parallel = openmp_available_cpp()
        && narg_openmp_threads > 1
        && outputs.size() >= 4
        && total_work >= 32768;
    if (parallel) {
        ++narg_openmp_parallel_regions;
        narg_openmp_tasks += static_cast<long long>(outputs.size());
    }
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(narg_openmp_threads) if(parallel)
#endif
        for (ssize_t index = 0; index < static_cast<ssize_t>(outputs.size()); ++index) {
            evaluate_reduced_growth_output(
                outputs[static_cast<size_t>(index)],
                sectors,
                requests,
                angular_plans,
                atol
            );
        }
    }

    std::vector<py::list> result_by_request;
    result_by_request.reserve(requests.size());
    for (size_t index = 0; index < requests.size(); ++index) {
        result_by_request.emplace_back();
    }
    for (auto& output : outputs) {
        result_by_request[output.request_index].append(py::make_tuple(
            output.bra_sector_index,
            output.ket_sector_index,
            output.out,
            output.max_abs
        ));
    }
    py::list result;
    for (auto& request_result : result_by_request) {
        result.append(request_result);
    }
    return result;
}

static py::tuple scalar_product_pair_entries(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> states,
    int total_j2,
    int block_dnelec,
    int block_rank2,
    int local_dnelec,
    int local_rank2,
    double atol
) {
    validate_state_table(states, "states");
    const ssize_t state_count = states.shape(0);
    const bool rank0_case = block_rank2 == 0 && local_rank2 == 0;
    auto state = states.unchecked<2>();

    auto coefficient = [&](ssize_t bra_pos, ssize_t ket_pos) {
        const int bra_block_j2 = static_cast<int>(state(bra_pos, 1));
        const int bra_local_j2 = static_cast<int>(state(bra_pos, 4));
        const int ket_block_j2 = static_cast<int>(state(ket_pos, 1));
        const int ket_local_j2 = static_cast<int>(state(ket_pos, 4));
        if (rank0_case) {
            if (bra_block_j2 != ket_block_j2 || bra_local_j2 != ket_local_j2) {
                return 0.0;
            }
            return 1.0 / std::sqrt(
                (bra_block_j2 + 1.0) * (bra_local_j2 + 1.0)
            );
        }
        return product_tensor_pair_coeff(
            total_j2,
            total_j2,
            0,
            0,
            total_j2,
            bra_block_j2,
            bra_local_j2,
            ket_block_j2,
            ket_local_j2,
            block_rank2,
            local_rank2,
            atol
        );
    };

    ssize_t count = 0;
    {
        py::gil_scoped_release release;
        for (ssize_t bra_pos = 0; bra_pos < state_count; ++bra_pos) {
            for (ssize_t ket_pos = 0; ket_pos < state_count; ++ket_pos) {
                if (state(bra_pos, 0) != state(ket_pos, 0) + block_dnelec) {
                    continue;
                }
                if (state(bra_pos, 3) != state(ket_pos, 3) + local_dnelec) {
                    continue;
                }
                if (std::abs(coefficient(bra_pos, ket_pos)) > atol) {
                    ++count;
                }
            }
        }
    }

    py::array_t<int64_t> rows({count});
    py::array_t<int64_t> cols({count});
    py::array_t<int64_t> block_rows({count});
    py::array_t<int64_t> block_cols({count});
    py::array_t<int64_t> local_rows({count});
    py::array_t<int64_t> local_cols({count});
    py::array_t<double> coeffs({count});
    auto rows_view = rows.mutable_unchecked<1>();
    auto cols_view = cols.mutable_unchecked<1>();
    auto block_rows_view = block_rows.mutable_unchecked<1>();
    auto block_cols_view = block_cols.mutable_unchecked<1>();
    auto local_rows_view = local_rows.mutable_unchecked<1>();
    auto local_cols_view = local_cols.mutable_unchecked<1>();
    auto coeffs_view = coeffs.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        ssize_t out_pos = 0;
        for (ssize_t bra_pos = 0; bra_pos < state_count; ++bra_pos) {
            for (ssize_t ket_pos = 0; ket_pos < state_count; ++ket_pos) {
                if (state(bra_pos, 0) != state(ket_pos, 0) + block_dnelec) {
                    continue;
                }
                if (state(bra_pos, 3) != state(ket_pos, 3) + local_dnelec) {
                    continue;
                }
                const double coeff = coefficient(bra_pos, ket_pos);
                if (std::abs(coeff) <= atol) {
                    continue;
                }
                rows_view(out_pos) = bra_pos;
                cols_view(out_pos) = ket_pos;
                block_rows_view(out_pos) = state(bra_pos, 2);
                block_cols_view(out_pos) = state(ket_pos, 2);
                local_rows_view(out_pos) = state(bra_pos, 5);
                local_cols_view(out_pos) = state(ket_pos, 5);
                coeffs_view(out_pos) = coeff;
                ++out_pos;
            }
        }
    }
    return py::make_tuple(rows, cols, coeffs, block_rows, block_cols, local_rows, local_cols);
}

struct GroupKey {
    int64_t values[8];

    bool operator==(const GroupKey& other) const {
        for (int i = 0; i < 8; ++i) {
            if (values[i] != other.values[i]) {
                return false;
            }
        }
        return true;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& key) const {
        size_t seed = 1469598103934665603ull;
        for (int i = 0; i < 8; ++i) {
            size_t value = static_cast<size_t>(key.values[i]);
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

static py::tuple product_tensor_group_indices(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> bra_states,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ket_states,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols
) {
    validate_state_table(bra_states, "bra_states");
    validate_state_table(ket_states, "ket_states");
    if (rows.ndim() != 1 || cols.ndim() != 1 || rows.shape(0) != cols.shape(0)) {
        throw std::runtime_error("rows and cols must be matching one-dimensional arrays");
    }

    const ssize_t size = rows.shape(0);
    auto bra = bra_states.unchecked<2>();
    auto ket = ket_states.unchecked<2>();
    auto rows_view = rows.unchecked<1>();
    auto cols_view = cols.unchecked<1>();

    std::vector<GroupKey> keys;
    std::vector<int64_t> group_counts;
    std::vector<int64_t> group_ids(static_cast<size_t>(size), 0);
    std::unordered_map<GroupKey, int64_t, GroupKeyHash> index_by_key;
    keys.reserve(static_cast<size_t>(size));
    group_counts.reserve(static_cast<size_t>(size));

    {
        py::gil_scoped_release release;
        for (ssize_t entry = 0; entry < size; ++entry) {
            const ssize_t row = rows_view(entry);
            const ssize_t col = cols_view(entry);
            GroupKey key{{
                bra(row, 0),
                bra(row, 1),
                ket(col, 0),
                ket(col, 1),
                bra(row, 3),
                bra(row, 4),
                ket(col, 3),
                ket(col, 4),
            }};

            auto found = index_by_key.find(key);
            int64_t group = 0;
            if (found == index_by_key.end()) {
                group = static_cast<int64_t>(keys.size());
                keys.push_back(key);
                group_counts.push_back(0);
                index_by_key.emplace(key, group);
            } else {
                group = found->second;
            }
            group_ids[static_cast<size_t>(entry)] = group;
            group_counts[static_cast<size_t>(group)] += 1;
        }
    }

    const ssize_t ngroups = static_cast<ssize_t>(keys.size());
    py::array_t<int64_t> group_keys({ngroups, static_cast<ssize_t>(8)});
    py::array_t<int64_t> group_starts({ngroups + 1});
    py::array_t<int64_t> order({size});
    auto group_keys_view = group_keys.mutable_unchecked<2>();
    auto group_starts_view = group_starts.mutable_unchecked<1>();
    auto order_view = order.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        group_starts_view(0) = 0;
        for (ssize_t group = 0; group < ngroups; ++group) {
            for (int field = 0; field < 8; ++field) {
                group_keys_view(group, field) = keys[static_cast<size_t>(group)].values[field];
            }
            group_starts_view(group + 1) = (
                group_starts_view(group) + group_counts[static_cast<size_t>(group)]
            );
            group_counts[static_cast<size_t>(group)] = group_starts_view(group);
        }

        for (ssize_t entry = 0; entry < size; ++entry) {
            const int64_t group = group_ids[static_cast<size_t>(entry)];
            const int64_t pos = group_counts[static_cast<size_t>(group)];
            order_view(pos) = entry;
            group_counts[static_cast<size_t>(group)] = pos + 1;
        }
    }

    return py::make_tuple(group_keys, group_starts, order);
}

static py::array_t<cdouble> accumulate_bilinear(
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> out,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> block_rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> block_cols,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> local_rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> local_cols,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> coeffs,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> block,
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> local,
    cdouble prefactor
) {
    if (out.ndim() != 2 || block.ndim() != 2 || local.ndim() != 2) {
        throw std::runtime_error("out, block, and local must be rank-2 arrays");
    }
    const ssize_t size = rows.shape(0);
    if (
        rows.ndim() != 1
        || cols.ndim() != 1
        || block_rows.ndim() != 1
        || block_cols.ndim() != 1
        || local_rows.ndim() != 1
        || local_cols.ndim() != 1
        || coeffs.ndim() != 1
        || cols.shape(0) != size
        || block_rows.shape(0) != size
        || block_cols.shape(0) != size
        || local_rows.shape(0) != size
        || local_cols.shape(0) != size
        || coeffs.shape(0) != size
    ) {
        throw std::runtime_error("bilinear index arrays must be matching one-dimensional arrays");
    }

    auto out_view = out.mutable_unchecked<2>();
    auto rows_view = rows.unchecked<1>();
    auto cols_view = cols.unchecked<1>();
    auto block_rows_view = block_rows.unchecked<1>();
    auto block_cols_view = block_cols.unchecked<1>();
    auto local_rows_view = local_rows.unchecked<1>();
    auto local_cols_view = local_cols.unchecked<1>();
    auto coeffs_view = coeffs.unchecked<1>();
    auto block_view = block.unchecked<2>();
    auto local_view = local.unchecked<2>();

    {
        py::gil_scoped_release release;
        for (ssize_t n = 0; n < size; ++n) {
            out_view(rows_view(n), cols_view(n)) += (
                prefactor
                * coeffs_view(n)
                * block_view(block_rows_view(n), block_cols_view(n))
                * local_view(local_rows_view(n), local_cols_view(n))
            );
        }
    }
    return out;
}

struct BilinearGroupJob {
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> block_rows;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> block_cols;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> local_rows;
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> local_cols;
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> coeffs;
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> block;
    py::array_t<cdouble, py::array::c_style | py::array::forcecast> local;
    cdouble prefactor;
    ssize_t size;
};

struct BilinearOutputJob {
    py::array_t<cdouble> out;
    std::vector<BilinearGroupJob> groups;
    ssize_t rows;
    ssize_t cols;
    long long entries;
};

static void accumulate_bilinear_output_job(BilinearOutputJob& job) {
    cdouble* out = static_cast<cdouble*>(job.out.mutable_data());
    std::fill(
        out,
        out + static_cast<size_t>(job.rows * job.cols),
        cdouble(0.0, 0.0)
    );
    for (const auto& group : job.groups) {
        const int64_t* rows = static_cast<const int64_t*>(group.rows.data());
        const int64_t* cols = static_cast<const int64_t*>(group.cols.data());
        const int64_t* block_rows = static_cast<const int64_t*>(group.block_rows.data());
        const int64_t* block_cols = static_cast<const int64_t*>(group.block_cols.data());
        const int64_t* local_rows = static_cast<const int64_t*>(group.local_rows.data());
        const int64_t* local_cols = static_cast<const int64_t*>(group.local_cols.data());
        const cdouble* coeffs = static_cast<const cdouble*>(group.coeffs.data());
        const cdouble* block = static_cast<const cdouble*>(group.block.data());
        const cdouble* local = static_cast<const cdouble*>(group.local.data());
        const ssize_t block_cols_dim = group.block.shape(1);
        const ssize_t local_cols_dim = group.local.shape(1);
        for (ssize_t n = 0; n < group.size; ++n) {
            out[rows[n] * job.cols + cols[n]] += (
                group.prefactor
                * coeffs[n]
                * block[block_rows[n] * block_cols_dim + block_cols[n]]
                * local[local_rows[n] * local_cols_dim + local_cols[n]]
            );
        }
    }
}

static py::list accumulate_bilinear_wave(py::sequence specs) {
    const ssize_t count = py::len(specs);
    std::vector<BilinearOutputJob> jobs;
    jobs.reserve(static_cast<size_t>(count));
    long long total_entries = 0;

    for (ssize_t job_index = 0; job_index < count; ++job_index) {
        py::sequence spec = py::reinterpret_borrow<py::sequence>(specs[job_index]);
        if (py::len(spec) != 3) {
            throw std::runtime_error(
                "bilinear wave specs must be (rows, cols, groups)"
            );
        }
        const ssize_t rows = spec[0].cast<ssize_t>();
        const ssize_t cols = spec[1].cast<ssize_t>();
        if (rows < 0 || cols < 0) {
            throw std::runtime_error("bilinear output dimensions must be non-negative");
        }
        py::sequence group_specs = py::reinterpret_borrow<py::sequence>(spec[2]);
        std::vector<BilinearGroupJob> groups;
        groups.reserve(static_cast<size_t>(py::len(group_specs)));
        long long job_entries = 0;

        for (ssize_t group_index = 0; group_index < py::len(group_specs); ++group_index) {
            py::sequence group = py::reinterpret_borrow<py::sequence>(group_specs[group_index]);
            if (py::len(group) != 10) {
                throw std::runtime_error(
                    "bilinear groups must contain seven index/value arrays, "
                    "block, local, and prefactor"
                );
            }
            auto row_index = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[0]);
            auto col_index = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[1]);
            auto block_rows = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[2]);
            auto block_cols = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[3]);
            auto local_rows = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[4]);
            auto local_cols = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(group[5]);
            auto coeffs = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(group[6]);
            auto block = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(group[7]);
            auto local = py::array_t<cdouble, py::array::c_style | py::array::forcecast>::ensure(group[8]);
            if (
                !row_index || !col_index || !block_rows || !block_cols
                || !local_rows || !local_cols || !coeffs || !block || !local
            ) {
                throw std::runtime_error("bilinear wave inputs have invalid array dtypes");
            }
            const ssize_t size = row_index.shape(0);
            if (
                row_index.ndim() != 1 || col_index.ndim() != 1
                || block_rows.ndim() != 1 || block_cols.ndim() != 1
                || local_rows.ndim() != 1 || local_cols.ndim() != 1
                || coeffs.ndim() != 1 || block.ndim() != 2 || local.ndim() != 2
                || col_index.shape(0) != size || block_rows.shape(0) != size
                || block_cols.shape(0) != size || local_rows.shape(0) != size
                || local_cols.shape(0) != size || coeffs.shape(0) != size
            ) {
                throw std::runtime_error("bilinear wave arrays have incompatible shapes");
            }
            groups.push_back(BilinearGroupJob{
                std::move(row_index),
                std::move(col_index),
                std::move(block_rows),
                std::move(block_cols),
                std::move(local_rows),
                std::move(local_cols),
                std::move(coeffs),
                std::move(block),
                std::move(local),
                group[9].cast<cdouble>(),
                size,
            });
            job_entries += size;
        }
        py::array_t<cdouble> out({rows, cols});
        jobs.push_back(BilinearOutputJob{
            std::move(out),
            std::move(groups),
            rows,
            cols,
            job_entries,
        });
        total_entries += job_entries;
    }

    const bool parallel = openmp_available_cpp()
        && narg_openmp_threads > 1
        && count >= 4
        && total_entries >= 2048;
    if (parallel) {
        ++narg_openmp_parallel_regions;
        narg_openmp_tasks += count;
    }
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(narg_openmp_threads) if(parallel)
#endif
        for (ssize_t index = 0; index < count; ++index) {
            accumulate_bilinear_output_job(jobs[static_cast<size_t>(index)]);
        }
    }

    py::list out;
    for (auto& job : jobs) {
        out.append(job.out);
    }
    return out;
}

PYBIND11_MODULE(_su2_native, m) {
    m.doc() = "Native kernels for SU(2)-NARG reduced tensor products";
    m.def("openmp_available", &openmp_available_cpp);
    m.def("set_num_threads", &set_narg_openmp_threads, py::arg("threads"));
    m.def("get_num_threads", []() { return narg_openmp_threads; });
    m.def("openmp_info", &narg_openmp_info);
    m.def("rotate_operator_blocks", &rotate_operator_blocks, py::arg("specs"));
    m.def(
        "clebsch_gordan_doubled",
        [](
            int left_j2,
            int right_j2,
            int fused_j2,
            int left_m2,
            int right_m2,
            int fused_m2
        ) {
            return cg(
                left_j2,
                left_m2,
                right_j2,
                right_m2,
                fused_j2,
                fused_m2
            );
        },
        py::arg("left_j2"),
        py::arg("right_j2"),
        py::arg("fused_j2"),
        py::arg("left_m2"),
        py::arg("right_m2"),
        py::arg("fused_m2")
    );
    m.def(
        "reduced_product_block_sum",
        &reduced_product_block_sum,
        py::arg("rows"),
        py::arg("cols"),
        py::arg("left_blocks"),
        py::arg("right_blocks"),
        py::arg("weights")
    );
    m.def(
        "reduced_product_block_sum_batch",
        &reduced_product_block_sum_batch,
        py::arg("specs")
    );
    m.def(
        "product_tensor_pair_entries",
        &product_tensor_pair_entries,
        py::arg("bra_states"),
        py::arg("ket_states"),
        py::arg("bra_total_j2"),
        py::arg("ket_total_j2"),
        py::arg("total_rank2"),
        py::arg("block_dnelec"),
        py::arg("block_rank2"),
        py::arg("local_dnelec"),
        py::arg("local_rank2"),
        py::arg("atol")
    );
    m.def(
        "scalar_product_pair_entries",
        &scalar_product_pair_entries,
        py::arg("states"),
        py::arg("total_j2"),
        py::arg("block_dnelec"),
        py::arg("block_rank2"),
        py::arg("local_dnelec"),
        py::arg("local_rank2"),
        py::arg("atol")
    );
    m.def(
        "product_tensor_pair_entries_batch",
        &product_tensor_pair_entries_batch,
        py::arg("specs")
    );
    m.def(
        "reduced_growth_graph",
        &reduced_growth_graph,
        py::arg("sectors"),
        py::arg("requests"),
        py::arg("atol") = 1.0e-12
    );
    m.def(
        "product_tensor_group_indices",
        &product_tensor_group_indices,
        py::arg("bra_states"),
        py::arg("ket_states"),
        py::arg("rows"),
        py::arg("cols")
    );
    m.def(
        "accumulate_bilinear",
        &accumulate_bilinear,
        py::arg("out"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("block_rows"),
        py::arg("block_cols"),
        py::arg("local_rows"),
        py::arg("local_cols"),
        py::arg("coeffs"),
        py::arg("block"),
        py::arg("local"),
        py::arg("prefactor")
    );
    m.def(
        "accumulate_bilinear_wave",
        &accumulate_bilinear_wave,
        py::arg("specs")
    );
}
