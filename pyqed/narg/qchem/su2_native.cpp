#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../../mps/nonabelian/su2_coupling_core.hpp"

#include <complex>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

using cdouble = std::complex<double>;

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

PYBIND11_MODULE(_su2_native, m) {
    m.doc() = "Native kernels for SU(2)-NARG reduced tensor products";
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
}
