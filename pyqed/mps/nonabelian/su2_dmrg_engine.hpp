#pragma once

#include "../dmrg_linalg_core.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyqed::su2 {

using Complex = std::complex<double>;

double reduced_environment_recoupling_coefficient(
    bool left,
    std::int64_t physical_output_charge,
    std::int64_t physical_input_charge,
    std::int32_t boundary_bra_two_j,
    std::int32_t boundary_ket_two_j,
    std::int32_t physical_bra_two_j,
    std::int32_t physical_ket_two_j,
    std::int32_t next_bra_two_j,
    std::int32_t next_ket_two_j,
    std::int32_t left_channel_two_j,
    std::int32_t right_channel_two_j,
    std::int32_t operator_two_j,
    std::int32_t left_channel_two_m,
    std::int32_t right_channel_two_m,
    std::int32_t operator_two_m
) noexcept;

enum class Family : std::int32_t {
    S = 0,
    R = 1,
    A = 2,
    P = 3,
    B = 4,
    Q = 5,
};

// Reduced local operators used by the normal/complementary QC sweep.
// Their numerical site tensors are supplied by the SU(2) tensor layer; the
// route topology and all integral coefficients live permanently in System.
enum class NCOperator : std::int32_t {
    Identity = 0,
    Hamiltonian = 1,
    Create = 2,
    Destroy = 3,
    CreateNegative = 4,
    DestroyNegative = 5,
    ReducedCreate = 6,
    ReducedDestroy = 7,
    PairCreate0 = 8,
    PairCreate1 = 9,
    PairDestroy0 = 10,
    PairDestroy1 = 11,
    Hole0 = 12,
    Hole1 = 13,
    ComplementPairCreate0 = 14,
    ComplementPairCreate1 = 15,
    ComplementPairDestroy0 = 16,
    ComplementPairDestroy1 = 17,
    ComplementHole0 = 18,
    ComplementHole1 = 19,
};

struct NormalComplementaryTransition {
    std::int32_t source = 0;
    std::int32_t target = 0;
    NCOperator local_operator = NCOperator::Identity;
    std::int64_t first_index = -1;
    std::int64_t second_index = -1;
    double coefficient = 0.0;
    std::uint64_t family_mask = 0;
};

struct NormalComplementaryChannel {
    std::int64_t charge = 0;
    std::int32_t two_j = 0;
    std::int64_t point_group = 0;
};

struct NormalComplementaryComponentAction {
    std::int32_t transition = 0;
    std::int32_t source_component = 0;
    std::int32_t target_component = 0;
    std::int32_t local_two_m = 0;
    double coefficient = 0.0;
};

struct NormalComplementaryPlan {
    std::int64_t site = 0;
    std::int64_t left_channels = 0;
    std::int64_t right_channels = 0;
    std::vector<NormalComplementaryChannel> left_channel_quantum_numbers;
    std::vector<NormalComplementaryChannel> right_channel_quantum_numbers;
    std::vector<std::int64_t> left_component_offsets;
    std::vector<std::int64_t> right_component_offsets;
    std::vector<NormalComplementaryTransition> transitions;
    std::vector<NormalComplementaryComponentAction> component_actions;
    std::vector<std::int64_t> component_action_offsets;
    std::array<std::vector<std::int64_t>, 9>
        active_transitions_by_physical_pair;
    std::array<std::vector<std::int64_t>, 9>
        active_source_transition_offsets;
    std::array<std::vector<std::int64_t>, 9>
        active_source_transitions;
    std::array<std::vector<std::int64_t>, 9>
        active_target_transition_offsets;
    std::array<std::vector<std::int64_t>, 9>
        active_target_transitions;
    std::array<std::vector<std::int32_t>, 9>
        active_transition_block_ids;
    std::array<std::vector<std::array<std::int32_t, 2>>, 9>
        active_block_channels;
    std::array<std::uint64_t, 6> family_transition_counts{};
};

struct FamilyData {
    std::int32_t rank = 0;
    std::vector<std::int64_t> indices;
    std::vector<double> values;
};

class System {
public:
    System(
        const double* h1,
        const double* eri,
        std::size_t n_sites,
        double ecore,
        std::int64_t n_elec,
        std::int64_t two_s,
        const std::int64_t* orb_sym,
        double cutoff,
        bool include_half
    );

    std::size_t n_sites() const noexcept;
    std::int64_t n_elec() const noexcept;
    std::int64_t two_s() const noexcept;
    double ecore() const noexcept;
    double cutoff() const noexcept;
    bool include_half() const noexcept;
    std::uint64_t revision() const noexcept;
    std::size_t stored_integral_elements() const noexcept;
    std::size_t stored_family_terms() const noexcept;
    std::size_t memory_bytes() const noexcept;
    void update_h1(const double* h1, std::size_t n_values);
    void update_integrals(
        const double* h1,
        std::size_t n_h1_values,
        const double* eri,
        std::size_t n_eri_values,
        double ecore
    );
    void family_partition_counts(
        std::int32_t family_id,
        bool left,
        std::int64_t bond,
        std::size_t* internal,
        std::size_t* cross,
        std::size_t* external
    ) const;
    const NormalComplementaryPlan& normal_complementary_plan(
        std::int64_t site
    ) const;
    std::size_t normal_complementary_transition_count() const noexcept;
    std::size_t normal_complementary_component_action_count() const noexcept;
    std::size_t normal_complementary_memory_bytes() const noexcept;
    std::size_t normal_complementary_left_component_count(
        std::int64_t site
    ) const;
    std::size_t normal_complementary_right_component_count(
        std::int64_t site
    ) const;
    const double* normal_complementary_primitives(
        std::int64_t site
    ) const;
    std::size_t normal_complementary_primitive_count() const noexcept;
    std::size_t normal_complementary_primitive_bytes() const noexcept;
    void apply_normal_complementary_core(
        std::int64_t site,
        const double* primitive_components,
        std::size_t n_primitive_components,
        const double* input,
        std::size_t n_input,
        std::size_t batch_size,
        double* output,
        std::size_t n_output
    ) const;

    const std::vector<double>& h1() const noexcept;
    const std::vector<double>& eri() const noexcept;
    const std::vector<std::int64_t>& orb_sym() const noexcept;
    const FamilyData& family(std::int32_t family_id) const;

private:
    void build_families();
    void build_normal_complementary_plans();
    void build_normal_complementary_primitives();

    std::size_t n_sites_ = 0;
    std::int64_t n_elec_ = 0;
    std::int64_t two_s_ = 0;
    double ecore_ = 0.0;
    double cutoff_ = 0.0;
    bool include_half_ = true;
    std::uint64_t revision_ = 1;
    std::vector<double> h1_;
    std::vector<double> eri_;
    std::vector<std::int64_t> orb_sym_;
    std::map<std::int32_t, FamilyData> families_;
    std::vector<NormalComplementaryPlan> normal_complementary_plans_;
    std::vector<double> normal_complementary_primitives_;
};

struct PackedArena {
    const double* values = nullptr;
    std::size_t n_values = 0;
    std::shared_ptr<std::vector<double>> owned_values;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> shape_offsets;
    std::vector<std::int64_t> shapes;
    std::vector<std::int64_t> leg_sector_offsets;
    std::vector<std::int64_t> leg_sector_labels;
    std::vector<std::int64_t> leg_sector_dims;
    std::uint64_t topology_revision = 0;
    std::uint64_t numeric_revision = 0;
    bool owns_values = false;
    bool symbolic_identity = false;
};

struct PackedSiteTensor {
    std::vector<double> values;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> shape_offsets;
    std::vector<std::int64_t> shapes;

    std::size_t memory_bytes() const noexcept {
        return values.capacity() * sizeof(double)
            + (
                offsets.capacity()
                + labels.capacity()
                + shape_offsets.capacity()
                + shapes.capacity()
            ) * sizeof(std::int64_t);
    }
};

struct BoundaryBufferHandle {
    std::shared_ptr<std::vector<double>> values;
};

struct ComplexPackedArena {
    std::shared_ptr<std::vector<Complex>> owned_values;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> shape_offsets;
    std::vector<std::int64_t> shapes;
    std::uint64_t topology_revision = 0;
    std::uint64_t numeric_revision = 0;
};

struct ComplexBoundaryBufferHandle {
    std::shared_ptr<std::vector<Complex>> values;
};

struct NormalComplementaryBoundaryRoute {
    std::uint16_t parent = 0;
    std::uint16_t bra = 0;
    std::uint16_t ket = 0;
    std::uint16_t output = 0;
    std::uint32_t core = 0;
};
static_assert(sizeof(NormalComplementaryBoundaryRoute) == 12);

struct NormalComplementaryBoundaryAction {
    std::string side;
    std::int64_t parent_bond = -1;
    std::int64_t child_bond = -1;
    std::int64_t site = -1;
    bool left = false;
    bool reduced_physical = false;
    bool dual_right_boundary = false;
    std::uint64_t route_topology_revision = 0;
    std::uint64_t output_topology_revision = 0;
    std::uint64_t parent_topology_revision = 0;
    std::uint64_t site_topology_revision = 0;
    std::uint64_t parent_structure_revision = 0;
    std::uint64_t site_structure_revision = 0;
    std::vector<NormalComplementaryBoundaryRoute> routes;
    std::vector<std::int64_t> output_offsets;
    std::vector<std::int64_t> output_shape_offsets;
    std::vector<std::int64_t> output_shapes;
    std::vector<std::int64_t> output_labels;
    std::vector<double> mpo_values;
    std::vector<std::int64_t> mpo_offsets;
    std::vector<std::int64_t> mpo_shape_offsets;
    std::vector<std::int64_t> mpo_shapes;
    std::vector<std::array<std::uint32_t, 3>> parent_block_dims;
    std::vector<std::array<std::uint32_t, 3>> bra_block_dims;
    std::vector<std::array<std::uint32_t, 3>> ket_block_dims;
    std::vector<std::array<std::uint32_t, 4>> mpo_block_dims;
    std::vector<std::array<std::uint32_t, 3>> output_block_dims;
    bool compiled = false;

    std::size_t memory_bytes() const noexcept {
        std::size_t total =
            routes.capacity() * sizeof(NormalComplementaryBoundaryRoute)
            + mpo_values.capacity() * sizeof(double)
            + (
                output_offsets.capacity()
                + output_shape_offsets.capacity()
                + output_shapes.capacity()
                + output_labels.capacity()
                + mpo_offsets.capacity()
                + mpo_shape_offsets.capacity()
                + mpo_shapes.capacity()
            ) * sizeof(std::int64_t);
        total += (
            parent_block_dims.capacity()
            + bra_block_dims.capacity()
            + ket_block_dims.capacity()
            + output_block_dims.capacity()
        ) * sizeof(std::array<std::uint32_t, 3>);
        total += mpo_block_dims.capacity()
            * sizeof(std::array<std::uint32_t, 4>);
        return total;
    }
};

struct MetricBoundaryAction {
    std::string side;
    std::int64_t parent_bond = -1;
    std::int64_t child_bond = -1;
    std::int64_t site = -1;
    bool left = false;
    std::uint64_t route_topology_revision = 0;
    std::uint64_t output_topology_revision = 0;
    std::uint64_t parent_topology_revision = 0;
    std::uint64_t site_topology_revision = 0;
    std::vector<std::int64_t> routes;
    std::vector<double> route_coefficients;
    std::vector<double> mpo_values;
    std::vector<std::int64_t> mpo_offsets;
    std::vector<std::int64_t> mpo_shape_offsets;
    std::vector<std::int64_t> mpo_shapes;
    std::vector<std::int64_t> output_offsets;
    std::vector<std::int64_t> output_shape_offsets;
    std::vector<std::int64_t> output_shapes;
    std::vector<std::int64_t> output_labels;

    std::size_t memory_bytes() const noexcept {
        return (
                mpo_values.capacity()
                + route_coefficients.capacity()
            ) * sizeof(double)
            + (
                routes.capacity()
                + mpo_offsets.capacity()
                + mpo_shape_offsets.capacity()
                + mpo_shapes.capacity()
                + output_offsets.capacity()
                + output_shape_offsets.capacity()
                + output_shapes.capacity()
                + output_labels.capacity()
            ) * sizeof(std::int64_t);
    }
};

const double* boundary_buffer_data(
    const BoundaryBufferHandle* handle
) noexcept;
std::size_t boundary_buffer_size(
    const BoundaryBufferHandle* handle
) noexcept;
void release_boundary_buffer(BoundaryBufferHandle* handle) noexcept;
const Complex* complex_boundary_buffer_data(
    const ComplexBoundaryBufferHandle* handle
) noexcept;
std::size_t complex_boundary_buffer_size(
    const ComplexBoundaryBufferHandle* handle
) noexcept;
void release_complex_boundary_buffer(
    ComplexBoundaryBufferHandle* handle
) noexcept;

struct LocalBlock {
    const Complex* values = nullptr;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t input_start = 0;
    std::int64_t output_start = 0;
};

struct ContextualCoreBlock {
    std::int64_t left_channel = 0;
    std::int64_t right_channel = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<double> values;
};

struct FactorRoute {
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::int64_t left_offset = 0;
    std::int64_t right_offset = 0;
    std::int64_t l = 0;
    std::int64_t k = 0;
    std::int64_t w = 0;
    std::int64_t a = 0;
    std::int64_t b = 0;
    std::int64_t q = 0;
    std::int64_t r = 0;
    std::int64_t d = 0;
    std::int64_t c = 0;
    std::int64_t input_entry = 0;
    std::int64_t output_entry = 0;
    std::int64_t left_factor = -1;
    std::int64_t right_factor = -1;
    std::uint64_t family_mask = 0;
};

struct RawFactorSource {
    std::vector<std::int64_t> owned_boundary_ids;
    std::vector<std::int64_t> owned_w_ids;
    std::vector<std::int64_t> owned_boundary_offsets;
    std::vector<std::int64_t> owned_boundary_shape_offsets;
    std::vector<std::int64_t> owned_boundary_shapes;
    std::vector<std::int64_t> owned_w_offsets;
    std::vector<std::int64_t> owned_w_shape_offsets;
    std::vector<std::int64_t> owned_w_shapes;
    const std::uint32_t* compact_boundary_ids = nullptr;
    const std::uint32_t* compact_w_ids = nullptr;
    const std::uint32_t* compact_w_offsets = nullptr;
    const std::uint16_t* compact_w_rows = nullptr;
    const std::uint16_t* compact_w_cols = nullptr;
    bool compact_contextual = false;
    const std::int64_t* boundary_ids = nullptr;
    const std::int64_t* w_ids = nullptr;
    std::size_t n_factors = 0;
    const std::int64_t* boundary_offsets = nullptr;
    const std::int64_t* boundary_shape_offsets = nullptr;
    const std::int64_t* boundary_shapes = nullptr;
    std::size_t n_boundary_arrays = 0;
    std::size_t n_boundary_shape_values = 0;
    const double* boundary_data = nullptr;
    std::size_t n_boundary_data = 0;
    const std::int64_t* w_offsets = nullptr;
    const std::int64_t* w_shape_offsets = nullptr;
    const std::int64_t* w_shapes = nullptr;
    std::size_t n_w_arrays = 0;
    std::size_t n_w_shape_values = 0;
    const double* w_data = nullptr;
    std::size_t n_w_data = 0;
    std::vector<std::uint8_t> cacheable;
    std::vector<std::uint8_t> cache_valid;
    std::vector<std::int64_t> cache_starts;
    std::vector<std::int64_t> cache_sizes;
    double* cache_data = nullptr;
    std::size_t cache_data_size = 0;
    std::vector<double> transient;
    std::vector<double> build_scratch;
    std::int64_t transient_factor = -1;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    double build_seconds = 0.0;
};

struct ContextualFactorStorage {
    std::vector<std::uint32_t> boundary_ids;
    std::vector<std::uint32_t> w_ids;
    std::vector<std::uint32_t> w_offsets;
    std::vector<std::uint16_t> w_rows;
    std::vector<std::uint16_t> w_cols;
    std::vector<const double*> w_views;
    std::vector<double> w_data;

    bool core(
        std::size_t id,
        std::int64_t& rows,
        std::int64_t& cols,
        const double*& values
    ) const noexcept {
        if (
            id >= w_rows.size() ||
            id >= w_cols.size() ||
            id >= w_views.size() ||
            id + 1 >= w_offsets.size()
        ) {
            return false;
        }
        rows = w_rows[id];
        cols = w_cols[id];
        const std::size_t elements = static_cast<std::size_t>(rows * cols);
        if (w_views[id] != nullptr) {
            values = w_views[id];
            return true;
        }
        const std::size_t offset = w_offsets[id];
        const std::size_t stop = w_offsets[id + 1];
        if (stop - offset != elements || stop > w_data.size()) {
            return false;
        }
        values = w_data.data() + offset;
        return true;
    }

    void materialize_views() {
        if (std::none_of(
            w_views.begin(),
            w_views.end(),
            [](const double* values) { return values != nullptr; }
        )) {
            return;
        }
        std::size_t total = 0;
        for (std::size_t id = 0; id < w_rows.size(); ++id) {
            total += static_cast<std::size_t>(w_rows[id])
                * static_cast<std::size_t>(w_cols[id]);
        }
        std::vector<double> packed;
        packed.reserve(total);
        std::vector<std::uint32_t> offsets(1, 0);
        offsets.reserve(w_rows.size() + 1);
        for (std::size_t id = 0; id < w_rows.size(); ++id) {
            std::int64_t rows = 0;
            std::int64_t cols = 0;
            const double* values = nullptr;
            if (!core(id, rows, cols, values)) {
                throw std::logic_error(
                    "Contextual core view cannot be materialized."
                );
            }
            const std::size_t elements =
                static_cast<std::size_t>(rows * cols);
            packed.insert(packed.end(), values, values + elements);
            if (packed.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Contextual core arena exceeds uint32 storage."
                );
            }
            offsets.push_back(static_cast<std::uint32_t>(packed.size()));
        }
        w_data.swap(packed);
        w_offsets.swap(offsets);
        std::fill(w_views.begin(), w_views.end(), nullptr);
    }

    std::size_t metadata_memory_bytes() const noexcept {
        return
            boundary_ids.capacity() * sizeof(std::uint32_t) +
            w_ids.capacity() * sizeof(std::uint32_t) +
            w_offsets.capacity() * sizeof(std::uint32_t) +
            w_rows.capacity() * sizeof(std::uint16_t) +
            w_cols.capacity() * sizeof(std::uint16_t) +
            w_views.capacity() * sizeof(const double*);
    }

    std::size_t value_memory_bytes() const noexcept {
        return w_data.capacity() * sizeof(double);
    }

    std::size_t memory_bytes() const noexcept {
        return metadata_memory_bytes() + value_memory_bytes();
    }
};

using ContextualCoreKey = std::array<std::int64_t, 8>;
using ContextualRecouplingKey = std::array<std::int32_t, 16>;

struct ContextualRecouplingKeyHash {
    std::size_t operator()(
        const ContextualRecouplingKey& key
    ) const noexcept {
        std::size_t result = 1469598103934665603ULL;
        for (const std::int32_t value : key) {
            result ^= std::hash<std::int32_t>{}(value);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct ContextualCoreReference {
    std::int64_t left_channel = 0;
    std::int64_t right_channel = 0;
    std::int64_t w_id = -1;
    const double* values = nullptr;
    std::uint16_t rows = 0;
    std::uint16_t cols = 0;
    const ContextualCoreBlock* block = nullptr;
};

struct ContextualCompiledRouteGroup {
    std::uint16_t input_entry = 0;
    std::uint16_t output_entry = 0;
    std::uint32_t factor_start = 0;
    std::uint32_t factor_stop = 0;
    std::uint32_t fused_left_start = 0;
    std::uint32_t fused_left_stop = 0;
    std::uint32_t fused_right_start = 0;
    std::uint32_t fused_right_stop = 0;
    std::uint8_t flags = 0;
};

struct ContextualCompiledRouteSchedule {
    std::vector<ContextualCompiledRouteGroup> groups;
    std::vector<std::uint32_t> left_factors;
    std::vector<std::uint32_t> right_factors;
    std::vector<std::uint32_t> fused_left_factors;
    std::vector<std::uint32_t> fused_right_factors;
    std::array<std::uint64_t, 6> family_route_counts{};
    std::uint64_t unlabeled_family_route_count = 0;
    std::uint64_t fused_group_count = 0;
    std::uint64_t fused_route_count = 0;
    bool ready = false;

    std::size_t memory_bytes() const noexcept {
        return
            groups.capacity() * sizeof(ContextualCompiledRouteGroup) +
            left_factors.capacity() * sizeof(std::uint32_t) +
            right_factors.capacity() * sizeof(std::uint32_t) +
            fused_left_factors.capacity() * sizeof(std::uint32_t) +
            fused_right_factors.capacity() * sizeof(std::uint32_t);
    }
};

struct ReducedContextualMatrix {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<double> values;
    const double* borrowed_values = nullptr;
    std::size_t borrowed_elements = 0;
    bool transposed_storage = false;

    const double* data() const noexcept {
        return borrowed_values == nullptr
            ? values.data()
            : borrowed_values;
    }

    std::size_t size() const noexcept {
        return borrowed_values == nullptr
            ? values.size()
            : borrowed_elements;
    }

    bool borrowed() const noexcept {
        return borrowed_values != nullptr;
    }

    bool transposed() const noexcept {
        return transposed_storage;
    }

    std::int64_t storage_stride() const noexcept {
        return transposed_storage ? rows : cols;
    }

    double value(std::int64_t row, std::int64_t col) const noexcept {
        return transposed_storage
            ? data()[static_cast<std::size_t>(col * rows + row)]
            : data()[static_cast<std::size_t>(row * cols + col)];
    }

    double flat_value(std::size_t index) const noexcept {
        if (!transposed_storage) {
            return data()[index];
        }
        const std::size_t logical_cols = static_cast<std::size_t>(cols);
        const std::size_t row = index / logical_cols;
        const std::size_t col = index - row * logical_cols;
        return data()[col * static_cast<std::size_t>(rows) + row];
    }

    void copy_logical_to(double* target) const noexcept {
        if (!transposed_storage) {
            const double* source = data();
            for (std::size_t index = 0; index < size(); ++index) {
                target[index] = source[index];
            }
            return;
        }
        for (std::int64_t row = 0; row < rows; ++row) {
            for (std::int64_t col = 0; col < cols; ++col) {
                target[static_cast<std::size_t>(row * cols + col)] =
                    data()[static_cast<std::size_t>(col * rows + row)];
            }
        }
    }

    void own() noexcept {
        borrowed_values = nullptr;
        borrowed_elements = 0;
        transposed_storage = false;
    }

    void borrow(
        const double* source,
        std::size_t elements,
        bool transposed = false
    ) noexcept {
        values.clear();
        borrowed_values = source;
        borrowed_elements = elements;
        transposed_storage = transposed;
    }
};

struct ComplementaryLocalTerm {
    std::int32_t left_matrix = -1;
    std::int32_t right_matrix = -1;
    double scale = 0.0;
    std::uint8_t family_mask = 0;
};

struct ComplementaryLocalAction {
    std::int64_t input_entry = 0;
    std::int64_t output_entry = 0;
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::array<std::int64_t, 9> dims{};
    std::uint32_t term_start = 0;
    std::uint32_t term_stop = 0;
    std::uint8_t family_mask = 0;
    bool diagonal = false;
};

struct ContextualBoundaryDecomposition {
    std::int64_t components = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t rank = 0;
    bool zero = false;
    bool pivot_basis = false;
    std::size_t component_offset = 0;
    std::size_t matrix_offset = 0;
    std::size_t pivot_offset = 0;
};

struct ContextualDecompositionWorkspace {
    std::vector<ContextualBoundaryDecomposition> values;
    std::vector<std::uint8_t> ready;
    std::vector<double> component_values;
    std::vector<double> matrix_values;
    std::vector<std::uint16_t> pivot_components;
    std::vector<double> square_scratch;
    std::vector<double> residual_scratch;

    void reset(std::size_t size) {
        values.resize(size);
        ready.resize(size);
        std::fill(ready.begin(), ready.end(), std::uint8_t{0});
        component_values.clear();
        matrix_values.clear();
        pivot_components.clear();
    }

    ContextualBoundaryDecomposition* cached(
        std::int64_t block
    ) noexcept {
        if (
            block < 0
            || static_cast<std::size_t>(block) >= values.size()
            || ready[static_cast<std::size_t>(block)] == 0
        ) {
            return nullptr;
        }
        return &values[static_cast<std::size_t>(block)];
    }

    const double* components(
        const ContextualBoundaryDecomposition& value
    ) const noexcept {
        return component_values.data() + value.component_offset;
    }

    const double* matrices(
        const ContextualBoundaryDecomposition& value
    ) const noexcept {
        return matrix_values.data() + value.matrix_offset;
    }

    void clear() noexcept {
        std::fill(ready.begin(), ready.end(), std::uint8_t{0});
        component_values.clear();
        matrix_values.clear();
        pivot_components.clear();
    }

    void release() noexcept {
        std::vector<ContextualBoundaryDecomposition>().swap(values);
        std::vector<std::uint8_t>().swap(ready);
        std::vector<double>().swap(component_values);
        std::vector<double>().swap(matrix_values);
        std::vector<std::uint16_t>().swap(pivot_components);
        std::vector<double>().swap(square_scratch);
        std::vector<double>().swap(residual_scratch);
    }
};

struct ContextualBoundaryMatrixBinding {
    std::uint32_t boundary_block = 0;
    std::uint16_t component = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    bool transpose = false;
};

struct ContextualReducedActionPlan {
    std::vector<ContextualBoundaryMatrixBinding> left_matrices;
    std::vector<ContextualBoundaryMatrixBinding> right_matrices;
    std::vector<ComplementaryLocalAction> actions;
    std::vector<ComplementaryLocalTerm> terms;
    std::array<std::uint64_t, 6> family_counts{};
    std::uint64_t unlabeled_count = 0;
    bool ready = false;

    std::size_t memory_bytes() const noexcept {
        return
            left_matrices.capacity()
                * sizeof(ContextualBoundaryMatrixBinding) +
            right_matrices.capacity()
                * sizeof(ContextualBoundaryMatrixBinding) +
            actions.capacity() * sizeof(ComplementaryLocalAction) +
            terms.capacity() * sizeof(ComplementaryLocalTerm);
    }
};

struct ContextualDecomposedMatrixBinding {
    std::uint32_t boundary_block = 0;
    std::uint16_t rank_index = 0;
    std::uint16_t expected_rank = 0;
    std::uint16_t expected_components = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t reduction_offset = 0;
    std::array<std::int64_t, 5> boundary_identity{};
    bool transpose = false;
};

struct ContextualDecomposedScaleRecipe {
    std::uint32_t term = 0;
    std::uint32_t core_offset = 0;
};
static_assert(sizeof(ContextualDecomposedScaleRecipe) == 8);

using ContextualActionFragmentKey = std::array<std::int64_t, 22>;
using ContextualRouteSkeletonKey = std::array<std::int64_t, 16>;
using ContextualCompactBoundaryIdentity = std::array<std::int32_t, 5>;

struct ContextualActionFragmentKeyHash {
    std::size_t operator()(
        const ContextualActionFragmentKey& key
    ) const noexcept {
        std::size_t result = 1469598103934665603ULL;
        for (const std::int64_t value : key) {
            result ^= std::hash<std::int64_t>{}(value);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct ContextualRouteSkeletonKeyHash {
    std::size_t operator()(
        const ContextualRouteSkeletonKey& key
    ) const noexcept {
        std::size_t result = 1469598103934665603ULL;
        for (const std::int64_t value : key) {
            result ^= std::hash<std::int64_t>{}(value);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct ContextualRouteSkeletonTerm {
    std::uint16_t left_core = 0;
    std::uint16_t right_core = 0;
};

struct ContextualRouteSkeleton {
    std::vector<ContextualRouteSkeletonTerm> terms;

    std::size_t memory_bytes() const noexcept {
        return sizeof(ContextualRouteSkeleton)
            + terms.capacity() * sizeof(ContextualRouteSkeletonTerm);
    }
};

struct ContextualActionFragmentTerm {
    ContextualCompactBoundaryIdentity left_identity{};
    ContextualCompactBoundaryIdentity right_identity{};
    std::uint32_t core_offset = 0;
    std::uint16_t left_rank = 0;
    std::uint16_t right_rank = 0;
    std::uint16_t left_expected_rank = 0;
    std::uint16_t right_expected_rank = 0;
    std::uint16_t left_components = 0;
    std::uint16_t right_components = 0;
    std::uint8_t family_mask = 0;
};

struct ContextualActionFragment {
    std::vector<ContextualActionFragmentTerm> terms;
    std::vector<double> scale_cores;
    std::array<std::uint64_t, 6> family_counts{};
    std::uint64_t unlabeled_count = 0;
    std::uint8_t family_mask = 0;
    bool diagonal = false;

    std::size_t memory_bytes() const noexcept {
        return sizeof(ContextualActionFragment)
            + terms.capacity() * sizeof(ContextualActionFragmentTerm)
            + scale_cores.capacity() * sizeof(double);
    }
};

struct ContextualActionFragmentUse {
    std::uint16_t input_entry = 0;
    std::uint16_t output_entry = 0;
    std::shared_ptr<const ContextualActionFragment> fragment;
};

struct ContextualDecomposedActionPlan {
    std::vector<ContextualDecomposedMatrixBinding> left_matrices;
    std::vector<ContextualDecomposedMatrixBinding> right_matrices;
    std::vector<std::uint16_t> left_pivot_components;
    std::vector<std::uint16_t> right_pivot_components;
    std::vector<ContextualDecomposedScaleRecipe> scale_recipes;
    std::vector<double> scale_cores;
    std::vector<double> invariant_scales;
    std::array<std::uint64_t, 6> family_counts{};
    std::uint64_t unlabeled_count = 0;
    bool ready = false;

    std::size_t memory_bytes() const noexcept {
        return
            left_matrices.capacity()
                * sizeof(ContextualDecomposedMatrixBinding) +
            right_matrices.capacity()
                * sizeof(ContextualDecomposedMatrixBinding) +
            (
                left_pivot_components.capacity() +
                right_pivot_components.capacity()
            ) * sizeof(std::uint16_t) +
            scale_recipes.capacity()
                * sizeof(ContextualDecomposedScaleRecipe) +
            (
                scale_cores.capacity() + invariant_scales.capacity()
            ) * sizeof(double);
    }
};

struct ReducedContextualExecutionSchedule;

struct CompactRouteIndices {
    std::uint8_t width = 0;
    std::vector<std::uint8_t> values;

    std::size_t size() const noexcept {
        return width == 0 ? 0 : values.size() / width;
    }

    const void* data() const noexcept {
        return values.empty() ? nullptr : values.data();
    }

    std::uint32_t operator[](std::size_t index) const noexcept {
        const std::size_t offset = index * width;
        std::uint32_t value = 0;
        for (std::size_t byte = 0; byte < width; ++byte) {
            value |= static_cast<std::uint32_t>(
                values[offset + byte]
            ) << (8 * byte);
        }
        return value;
    }

    void assign(
        const std::vector<std::uint32_t>& source,
        std::uint8_t index_width
    ) {
        width = index_width;
        values.resize(source.size() * width);
        for (std::size_t index = 0; index < source.size(); ++index) {
            const std::uint32_t value = source[index];
            for (std::size_t byte = 0; byte < width; ++byte) {
                values[index * width + byte] = static_cast<std::uint8_t>(
                    value >> (8 * byte)
                );
            }
        }
    }

    std::size_t memory_bytes() const noexcept {
        return values.capacity() * sizeof(std::uint8_t);
    }
};

struct ContextualFactorRoutePlan {
    std::int64_t bond = -1;
    std::uint64_t topology_revision = 0;
    std::uint64_t structural_revision = 0;
    std::uint64_t left_boundary_topology_revision = 0;
    std::uint64_t right_boundary_topology_revision = 0;
    std::uint64_t last_use = 0;
    std::size_t n_basis = 0;
    std::size_t n_routes = 0;
    std::size_t total_dimension = 0;
    std::vector<std::int64_t> basis_offsets;
    std::vector<std::int64_t> basis_shapes;
    std::vector<std::int64_t> basis_quantum_numbers;
    std::vector<ContextualActionFragmentUse> cached_action_fragments;
    bool dual_right_basis = false;
    std::vector<std::uint16_t> in_indices;
    std::vector<std::uint16_t> out_indices;
    CompactRouteIndices left_indices;
    CompactRouteIndices right_indices;
    ContextualFactorStorage left;
    ContextualFactorStorage right;
    std::vector<std::uint8_t> left_family_masks;
    std::vector<std::uint8_t> right_family_masks;
    ContextualCompiledRouteSchedule compiled;
    ContextualReducedActionPlan reduced;
    ContextualDecomposedActionPlan decomposed;
    std::shared_ptr<ReducedContextualExecutionSchedule> execution;

    bool route_source_available() const noexcept {
        return n_routes != 0
            && in_indices.size() == n_routes
            && out_indices.size() == n_routes
            && left_indices.size() == n_routes
            && right_indices.size() == n_routes;
    }

    void retire_route_source() noexcept {
        std::vector<std::uint16_t>().swap(in_indices);
        std::vector<std::uint16_t>().swap(out_indices);
        left_indices = CompactRouteIndices{};
        right_indices = CompactRouteIndices{};
        left = ContextualFactorStorage{};
        right = ContextualFactorStorage{};
        std::vector<std::uint8_t>().swap(left_family_masks);
        std::vector<std::uint8_t>().swap(right_family_masks);
        compiled = ContextualCompiledRouteSchedule{};
        reduced = ContextualReducedActionPlan{};
    }

    std::size_t logical_route_count() const noexcept {
        return n_routes + cached_action_fragments.size();
    }

    std::size_t route_source_memory_bytes() const noexcept {
        return in_indices.capacity() * sizeof(std::uint16_t)
            + out_indices.capacity() * sizeof(std::uint16_t)
            + left_indices.memory_bytes()
            + right_indices.memory_bytes()
            + left.memory_bytes()
            + right.memory_bytes()
            + left_family_masks.capacity() * sizeof(std::uint8_t)
            + right_family_masks.capacity() * sizeof(std::uint8_t);
    }

    std::size_t route_memory_bytes() const noexcept {
        return
            basis_offsets.capacity() * sizeof(std::int64_t) +
            basis_shapes.capacity() * sizeof(std::int64_t) +
            basis_quantum_numbers.capacity() * sizeof(std::int64_t) +
            cached_action_fragments.capacity()
                * sizeof(ContextualActionFragmentUse) +
            route_source_memory_bytes() -
                left.value_memory_bytes() -
                right.value_memory_bytes() +
            compiled.memory_bytes() +
            reduced.memory_bytes() +
            decomposed.memory_bytes();
    }

    std::size_t core_value_memory_bytes() const noexcept {
        return left.value_memory_bytes() + right.value_memory_bytes();
    }

    std::size_t memory_bytes() const noexcept {
        return route_memory_bytes() + core_value_memory_bytes();
    }
};

struct RawFactorView {
    const double* values = nullptr;
    std::size_t count = 0;

    const double* data() const noexcept { return values; }
    const double* begin() const noexcept { return values; }
    const double* end() const noexcept { return values + count; }
    std::size_t size() const noexcept { return count; }
    const double& operator[](std::size_t index) const noexcept {
        return values[index];
    }
};

struct RawFactorPair {
    std::int32_t left = -1;
    std::int32_t right = -1;
};

struct RawRouteGroup {
    std::int64_t input_entry = 0;
    std::int64_t output_entry = 0;
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::array<std::int64_t, 9> dims{};
    std::vector<RawFactorPair> factors;
    std::vector<double> factor_scales;
    std::vector<std::int64_t> fused_left_factors;
    std::vector<std::int64_t> fused_right_factors;
    std::vector<double> fused_left_values;
    std::vector<double> fused_right_values;
    bool fused = false;
    bool reduced_contextual = false;
    bool diagonal = false;
    std::int64_t dense_pair = -1;
};

struct RawExecutionAction {
    std::uint32_t group = 0;
    std::int32_t factor = -1;
    std::uint32_t factor_stop = 0;
    std::uint32_t combined_left_start = 0;
    std::uint32_t combined_left_stop = 0;
    bool right_grouped = false;
    bool direct_preferred = false;
};

struct RawExecutionGroup {
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::array<std::int64_t, 8> dims{};
    std::uint32_t action_start = 0;
    std::uint32_t action_stop = 0;
};

struct RawExecutionTile {
    std::uint32_t execution = 0;
    std::uint32_t action_start = 0;
    std::uint32_t action_stop = 0;
    std::int64_t total_w = 0;
    std::size_t workspace_elements = 0;
    bool direct_preferred = false;
    bool requires_right = true;
    bool planned_requires_right = true;
};

struct RawCompactOutputProduct {
    std::uint32_t tile = 0;
    std::uint32_t panel = 0;
    std::int64_t temporary_row_offset = 0;
};

struct RawCompactRightPanel {
    std::uint32_t action_start = 0;
    std::uint32_t action_stop = 0;
    std::int64_t total_w = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::size_t value_offset = 0;
    std::uint32_t use_count = 0;
};

struct RawExecutionBatch {
    std::uint32_t tile_start = 0;
    std::uint32_t tile_stop = 0;
    std::int64_t total_left_rows = 0;
    std::int64_t packed_left_rows = 0;
    std::int64_t total_w = 0;
    std::int64_t dq = 0;
    std::size_t total_right_elements = 0;
    std::size_t workspace_elements = 0;
    bool right_first = false;
    bool combined_left = false;
    bool requires_right = true;
    std::size_t direct_action_count = 0;
    std::size_t direct_term_count = 0;
    std::vector<std::uint32_t> unique_left_actions;
    std::vector<std::int64_t> unique_left_row_offsets;
    std::vector<std::int64_t> action_left_row_offsets;
    std::vector<std::uint16_t> channel_left_indices;
    std::vector<RawCompactOutputProduct> compact_output_products;
    std::vector<std::uint32_t> compact_output_wave_offsets;
    std::size_t cached_left_offset =
        std::numeric_limits<std::size_t>::max();
    std::size_t cached_right_offset =
        std::numeric_limits<std::size_t>::max();
    std::vector<double> left_values;
    std::vector<double> right_values;
    std::size_t planned_total_right_elements = 0;
    bool planned_requires_right = true;
};

struct RawInputSuperchannel {
    std::int64_t input_offset = 0;
    std::int64_t kb = 0;
    std::int64_t cr = 0;
    std::vector<RawExecutionTile> tiles;
    std::vector<RawExecutionBatch> batches;
    std::vector<std::uint32_t> unique_left_actions;
    std::vector<std::uint32_t> unique_left_row_offsets;
    std::vector<std::int32_t> persistent_product_slots;
    std::int64_t unique_left_rows = 0;
    std::size_t cached_unique_left_offset =
        std::numeric_limits<std::size_t>::max();
};

struct RawOutputFusionBinding {
    std::uint32_t tile = 0;
    std::uint32_t group = 0;
    std::int64_t temporary_row_offset = 0;
    std::size_t right_offset = 0;
    std::int64_t group_k_offset = 0;
    std::uint32_t shared_right_start = 0;
    std::uint32_t shared_right_stop = 0;
};

struct RawGroupedOutputProductGroup {
    std::uint32_t binding_start = 0;
    std::uint32_t binding_stop = 0;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int32_t inner = 0;
    std::size_t persistent_right_offset =
        std::numeric_limits<std::size_t>::max();
};

struct RawSharedLeftOutputBinding {
    std::uint32_t binding = 0;
    std::uint32_t action = 0;
    std::int64_t source_k_offset = 0;
    std::int64_t column_offset = 0;
};

struct RawSharedLeftOutputGroup {
    std::uint32_t binding_start = 0;
    std::uint32_t binding_stop = 0;
    std::uint32_t reference_binding = 0;
    std::uint32_t channel = 0;
    std::int64_t reference_source_k_offset = 0;
    std::int64_t reference_stride = 0;
    std::int64_t rows = 0;
    std::int64_t inner = 0;
    std::int64_t total_columns = 0;
    std::size_t right_offset = 0;
    std::size_t output_offset = 0;
    bool channel_level = false;
};

struct RawOutputFusionBatch {
    std::uint32_t channel = 0;
    std::uint32_t batch = 0;
    std::uint32_t binding_start = 0;
    std::uint32_t binding_stop = 0;
    std::uint32_t grouped_product_start = 0;
    std::uint32_t grouped_product_stop = 0;
    std::uint32_t shared_left_group_start = 0;
    std::uint32_t shared_left_group_stop = 0;
    bool singleton_outputs = false;
    bool shared_right_panels = false;
    bool deferred_outputs = false;
    bool direct_channel_fusion = false;
    bool persistent_output_only = false;
};

struct RawChannelFusionTask {
    std::uint32_t batch = 0;
    std::uint32_t binding = 0;
    std::uint32_t channel_action_offset = 0;
};

struct RawOutputFusionGroup {
    std::int64_t output_offset = 0;
    std::int64_t la = 0;
    std::int64_t dq = 0;
    std::int64_t total_k = 0;
    std::size_t temporary_offset =
        std::numeric_limits<std::size_t>::max();
    std::size_t right_offset =
        std::numeric_limits<std::size_t>::max();
    std::uint32_t tile_count = 0;
    bool persistent_output = false;
};

struct RawPersistentOutputBinding {
    std::uint32_t wave = 0;
    std::uint32_t batch = 0;
    std::uint32_t binding = 0;
    std::uint32_t channel_action_offset = 0;
};

struct RawPersistentOutputReference {
    std::uint32_t wave = 0;
    std::uint32_t group = 0;
    std::uint32_t binding_start = 0;
    std::uint32_t binding_stop = 0;
    std::int64_t group_k_offset = 0;
};

struct RawPersistentOutputGroup {
    std::uint32_t reference_start = 0;
    std::uint32_t reference_stop = 0;
    std::uint64_t work = 0;
    std::int64_t output_offset = 0;
    std::int64_t la = 0;
    std::int64_t dq = 0;
    std::int64_t total_k = 0;
    std::size_t right_offset =
        std::numeric_limits<std::size_t>::max();
    bool combined = false;
};

struct RawPersistentProductCacheEntry {
    std::uint32_t channel = 0;
    std::uint32_t action = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::size_t value_offset = 0;
};

struct RawPersistentRightCacheEntry {
    std::uint32_t action = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::size_t value_offset = 0;
};

struct RawPersistentOutputBundle {
    std::uint32_t group_start = 0;
    std::uint32_t group_stop = 0;
    std::uint64_t work = 0;
};

struct RawPersistentOutputTask {
    std::uint32_t group = 0;
    std::uint32_t reference_start = 0;
    std::uint32_t reference_stop = 0;
    std::uint64_t work = 0;
    bool combined = false;
};

struct RawSharedRightBinding {
    std::uint32_t panel = 0;
    std::int64_t output_offset = 0;
    std::int64_t source_k_offset = 0;
    std::int64_t panel_row_offset = 0;
    std::int64_t rows = 0;
    double scale = 1.0;
};

struct RawSharedRightPanel {
    std::int32_t right_matrix = -1;
    std::int64_t inner = 0;
    std::int64_t columns = 0;
    std::int64_t total_rows = 0;
    std::uint32_t binding_start = 0;
    std::uint32_t binding_stop = 0;
    std::size_t input_offset = 0;
    std::size_t output_offset = 0;
};

struct RawOutputFusionWave {
    std::vector<RawOutputFusionBatch> batches;
    std::vector<RawOutputFusionBinding> bindings;
    std::vector<RawOutputFusionGroup> groups;
    std::vector<std::uint32_t> grouped_product_bindings;
    std::vector<RawGroupedOutputProductGroup> grouped_product_groups;
    std::vector<RawSharedLeftOutputBinding> shared_left_bindings;
    std::vector<RawSharedLeftOutputGroup> shared_left_groups;
    std::vector<RawSharedRightBinding> shared_right_bindings;
    std::vector<std::uint32_t> shared_right_tile_bindings;
    std::vector<RawSharedRightPanel> shared_right_panels;
    std::vector<std::size_t> shared_right_deferred_output_offsets;
    std::vector<RawChannelFusionTask> channel_fusion_tasks;
    std::size_t temporary_elements = 0;
    std::size_t right_elements = 0;
    std::size_t shared_right_input_elements = 0;
    std::size_t shared_right_output_elements = 0;
    std::size_t shared_right_deferred_output_elements = 0;
    std::size_t shared_right_tile_output_elements = 0;
    std::size_t shared_left_right_elements = 0;
    std::size_t shared_left_output_elements = 0;
    std::size_t grouped_product_right_elements = 0;
    std::size_t persistent_right_offset = 0;
    long double channel_fusion_work = 0.0L;
    bool channel_left_ready = false;

    std::size_t memory_bytes() const noexcept {
        return batches.capacity() * sizeof(RawOutputFusionBatch)
            + bindings.capacity() * sizeof(RawOutputFusionBinding)
            + groups.capacity() * sizeof(RawOutputFusionGroup)
            + grouped_product_bindings.capacity()
                * sizeof(std::uint32_t)
            + grouped_product_groups.capacity()
                * sizeof(RawGroupedOutputProductGroup)
            + shared_left_bindings.capacity()
                * sizeof(RawSharedLeftOutputBinding)
            + shared_left_groups.capacity()
                * sizeof(RawSharedLeftOutputGroup)
            + shared_right_bindings.capacity()
                * sizeof(RawSharedRightBinding)
            + shared_right_tile_bindings.capacity()
                * sizeof(std::uint32_t)
            + shared_right_panels.capacity()
                * sizeof(RawSharedRightPanel)
            + shared_right_deferred_output_offsets.capacity()
                * sizeof(std::size_t)
            + channel_fusion_tasks.capacity()
                * sizeof(RawChannelFusionTask);
    }
};

struct ReducedContextualExecutionSchedule {
    std::vector<RawExecutionGroup> groups;
    std::vector<RawExecutionAction> actions;
    std::vector<std::uint32_t> combined_left_terms;
    std::vector<RawInputSuperchannel> superchannels;
    std::vector<RawOutputFusionWave> output_waves;
    std::vector<ComplementaryLocalAction> local_actions;
    std::vector<ComplementaryLocalTerm> local_terms;
    std::vector<RawCompactRightPanel> compact_right_panels;
    std::size_t action_count = 0;
    std::size_t right_grouped_action_count = 0;
    std::size_t tile_count = 0;
    std::size_t batch_count = 0;
    std::uint64_t topology_revision = 0;

    std::size_t memory_bytes() const noexcept {
        std::size_t total =
            groups.capacity() * sizeof(RawExecutionGroup)
            + actions.capacity() * sizeof(RawExecutionAction)
            + combined_left_terms.capacity() * sizeof(std::uint32_t)
            + superchannels.capacity() * sizeof(RawInputSuperchannel)
            + output_waves.capacity() * sizeof(RawOutputFusionWave)
            + local_actions.capacity() * sizeof(ComplementaryLocalAction)
            + local_terms.capacity() * sizeof(ComplementaryLocalTerm)
            + compact_right_panels.capacity()
                * sizeof(RawCompactRightPanel);
        for (const RawInputSuperchannel& channel : superchannels) {
            total +=
                channel.tiles.capacity() * sizeof(RawExecutionTile)
                + channel.batches.capacity() * sizeof(RawExecutionBatch);
            total += channel.unique_left_actions.capacity()
                * sizeof(std::uint32_t);
            total += channel.unique_left_row_offsets.capacity()
                * sizeof(std::uint32_t);
            for (const RawExecutionBatch& batch : channel.batches) {
                total += (
                    batch.left_values.capacity()
                    + batch.right_values.capacity()
                ) * sizeof(double);
                total += batch.unique_left_actions.capacity()
                    * sizeof(std::uint32_t);
                total += (
                    batch.unique_left_row_offsets.capacity()
                    + batch.action_left_row_offsets.capacity()
                ) * sizeof(std::int64_t);
                total += batch.channel_left_indices.capacity()
                    * sizeof(std::uint16_t);
                total += batch.compact_output_products.capacity()
                    * sizeof(RawCompactOutputProduct);
                total += batch.compact_output_wave_offsets.capacity()
                    * sizeof(std::uint32_t);
            }
        }
        for (const RawOutputFusionWave& wave : output_waves) {
            total += wave.memory_bytes();
        }
        return total;
    }
};

struct DensePairKernel {
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::int64_t input_size = 0;
    std::int64_t output_size = 0;
    std::vector<double> values;
};

struct DensePairExecution {
    bool shared_input = true;
    std::int64_t common_offset = 0;
    std::int64_t common_size = 0;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> sizes;
    std::vector<double> values;
};

struct ProjectionComponent {
    std::vector<std::int64_t> parent_indices;
    std::vector<std::int64_t> orthonormal_indices;
    std::vector<double> owned_real_transform;
    std::vector<Complex> owned_complex_transform;
    const double* real_transform = nullptr;
    const Complex* complex_transform = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t orth_offset = 0;
    bool diagonal = false;
    bool kronecker = false;
    std::size_t parent_offset = 0;
    std::size_t left_dim = 0;
    std::size_t selected_dim = 0;
    std::size_t local_dim = 0;
    std::size_t right_dim = 0;
};

struct FactorizedMetricRoute {
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    std::array<std::int64_t, 4> input_shape{};
    std::array<std::int64_t, 4> output_shape{};
    const double* left_real = nullptr;
    const Complex* left_complex = nullptr;
    const double* right_real = nullptr;
    const Complex* right_complex = nullptr;
    double scale = 1.0;
    bool left_identity = false;
    bool right_identity = false;
};

struct CanonicalProjectionInfo {
    bool compatible = false;
    bool reused = false;
    std::string projection_key;
    std::size_t parent_dimension = 0;
    std::size_t orthonormal_dimension = 0;
    std::size_t components = 0;
    std::size_t max_component_dimension = 0;
    std::size_t transform_elements = 0;
    double whitening_residual = 0.0;
    double build_seconds = 0.0;
};

struct ActiveBondCanonicalSolveResult {
    CanonicalProjectionInfo projection;
    pyqed::dmrg::DavidsonResult davidson;
    int complementary_action_status = -1;
    std::size_t metric_routes = 0;
    std::size_t requested_restart_dimension = 0;
    std::size_t workspace_restart_dimension = 0;
    std::size_t estimated_workspace_bytes = 0;
    double solve_seconds = 0.0;
};

struct ActiveBondStateAverageSolveResult {
    CanonicalProjectionInfo projection;
    pyqed::dmrg::BlockDavidsonResult davidson;
    int complementary_action_status = -1;
    std::size_t metric_routes = 0;
    std::size_t requested_restart_dimension = 0;
    std::size_t workspace_restart_dimension = 0;
    std::size_t estimated_workspace_bytes = 0;
    double solve_seconds = 0.0;
};

struct CanonicalProjectionCacheComponent {
    std::vector<std::int64_t> parent_indices;
    std::vector<double> transform;
    std::size_t columns = 0;
    std::size_t orthonormal_offset = 0;
    bool diagonal = false;
};

struct CanonicalProjectionCacheEntry {
    std::vector<CanonicalProjectionCacheComponent> components;
    std::vector<double> metric_diagonal;
    std::vector<double> metric_probe_output;
    std::size_t parent_dimension = 0;
    std::size_t orthonormal_dimension = 0;
    std::size_t transform_elements = 0;
    std::size_t max_component_dimension = 0;
    double whitening_residual = 0.0;
    std::uint64_t last_use = 0;
};

struct BlockSVDResult {
    std::vector<Complex> left_values;
    std::vector<double> singular_values;
    std::vector<Complex> right_values;
    std::vector<std::int64_t> left_offsets;
    std::vector<std::int64_t> singular_offsets;
    std::vector<std::int64_t> right_offsets;
    std::vector<std::int64_t> kept_offsets;
    std::vector<std::int64_t> kept_indices;
    double truncation_error = 0.0;
    double full_squared_norm = 0.0;
    double kept_squared_norm = 0.0;
};

struct ActiveBondSplitResult {
    PackedSiteTensor left;
    PackedSiteTensor right;
    std::vector<std::int64_t> bond_labels;
    std::vector<std::int64_t> bond_dims;
    std::vector<double> singular_values;
    std::vector<std::int64_t> singular_offsets;
    double truncation_error = 0.0;
    double full_squared_norm = 0.0;
    double kept_squared_norm = 0.0;
    std::uint64_t kept_states = 0;
    std::uint64_t left_topology_revision = 0;
    std::uint64_t left_numeric_revision = 0;
    std::uint64_t right_topology_revision = 0;
    std::uint64_t right_numeric_revision = 0;
};

struct OwnedHalfSweepSplitSummary {
    double truncation_error = 0.0;
    double full_squared_norm = 0.0;
    double kept_squared_norm = 0.0;
    std::uint64_t kept_states = 0;
    std::uint64_t left_topology_revision = 0;
    std::uint64_t left_numeric_revision = 0;
    std::uint64_t right_topology_revision = 0;
    std::uint64_t right_numeric_revision = 0;
};

struct OwnedHalfSweepBondResult {
    std::int64_t bond = -1;
    ActiveBondCanonicalSolveResult solve;
    OwnedHalfSweepSplitSummary split;
    std::vector<double> state_energies;
    std::vector<double> state_residual_norms;
    bool state_average = false;
};

struct OwnedSplitSiteExport {
    std::int64_t site = -1;
    PackedSiteTensor tensor;
    std::vector<std::int64_t> leg_sector_offsets;
    std::vector<std::int64_t> leg_sector_labels;
    std::vector<std::int64_t> leg_sector_dims;
    std::uint64_t topology_revision = 0;
    std::uint64_t numeric_revision = 0;
};

struct SpatialNPDMResult {
    std::vector<double> rdm1;
    std::vector<double> rdm2;
    double norm = 0.0;
    std::size_t max_reduced_bond_dimension = 0;
    std::size_t max_component_bond_dimension = 0;
    std::size_t max_operator_channels = 0;
    std::int32_t max_operator_two_j = 0;
    std::uint64_t string_contractions = 0;
    bool spin_rotation_reduction = false;
    bool magnetic_component_expansion = false;
    double setup_seconds = 0.0;
    double environment_seconds = 0.0;
    double rdm1_seconds = 0.0;
    double rdm2_seconds = 0.0;
};

using HalfSweepBondExecutor = bool (*)(
    void* context,
    std::int64_t bond
);

class MovingEnvironment {
public:
    explicit MovingEnvironment(const System* system);

    void set_num_threads(int n_threads);
    int num_threads() const noexcept;
    bool openmp_available() const noexcept;
    int openmp_version() const noexcept;
    std::uint64_t openmp_parallel_regions() const noexcept;
    std::uint64_t openmp_tasks() const noexcept;

    bool install_boundary(
        const std::string& side,
        std::int64_t bond,
        const double* values,
        std::size_t n_values,
        const std::int64_t* offsets,
        std::size_t n_offsets,
        const std::int64_t* labels,
        std::size_t n_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    bool install_metric_boundary(
        const std::string& side,
        std::int64_t bond,
        const double* values,
        std::size_t n_values,
        const std::int64_t* offsets,
        std::size_t n_offsets,
        const std::int64_t* labels,
        std::size_t n_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    bool release_boundary(const std::string& side, std::int64_t bond);
    void clear_boundaries();
    bool boundary_installed(
        const std::string& side,
        std::int64_t bond,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    ) const noexcept;
    bool metric_boundary_installed(
        const std::string& side,
        std::int64_t bond,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    ) const noexcept;
    bool advance_boundary(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        bool left,
        const std::int64_t* routes,
        std::size_t n_routes,
        const double* bra_values,
        std::size_t n_bra_values,
        const std::int64_t* bra_offsets,
        std::size_t n_bra_offsets,
        const std::int64_t* bra_shape_offsets,
        std::size_t n_bra_shape_offsets,
        const std::int64_t* bra_shapes,
        std::size_t n_bra_shapes,
        const double* ket_values,
        std::size_t n_ket_values,
        const std::int64_t* ket_offsets,
        std::size_t n_ket_offsets,
        const std::int64_t* ket_shape_offsets,
        std::size_t n_ket_shape_offsets,
        const std::int64_t* ket_shapes,
        std::size_t n_ket_shapes,
        const double* mpo_values,
        std::size_t n_mpo_values,
        const std::int64_t* mpo_offsets,
        std::size_t n_mpo_offsets,
        const std::int64_t* mpo_shape_offsets,
        std::size_t n_mpo_shape_offsets,
        const std::int64_t* mpo_shapes,
        std::size_t n_mpo_shapes,
        const std::int64_t* output_offsets,
        std::size_t n_output_offsets,
        const std::int64_t* output_shape_offsets,
        std::size_t n_output_shape_offsets,
        const std::int64_t* output_shapes,
        std::size_t n_output_shapes,
        const std::int64_t* output_labels,
        std::size_t n_output_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        const double* route_coefficients,
        bool metric_boundary,
        bool accumulate_output = false,
        bool finalize_update = true,
        const NormalComplementaryBoundaryAction* compiled_action = nullptr
    );
    bool advance_boundary_complex(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        bool left,
        const std::int64_t* routes,
        std::size_t n_routes,
        const Complex* bra_values,
        std::size_t n_bra_values,
        const std::int64_t* bra_offsets,
        std::size_t n_bra_offsets,
        const std::int64_t* bra_shape_offsets,
        std::size_t n_bra_shape_offsets,
        const std::int64_t* bra_shapes,
        std::size_t n_bra_shapes,
        const Complex* ket_values,
        std::size_t n_ket_values,
        const std::int64_t* ket_offsets,
        std::size_t n_ket_offsets,
        const std::int64_t* ket_shape_offsets,
        std::size_t n_ket_shape_offsets,
        const std::int64_t* ket_shapes,
        std::size_t n_ket_shapes,
        const Complex* mpo_values,
        std::size_t n_mpo_values,
        const std::int64_t* mpo_offsets,
        std::size_t n_mpo_offsets,
        const std::int64_t* mpo_shape_offsets,
        std::size_t n_mpo_shape_offsets,
        const std::int64_t* mpo_shapes,
        std::size_t n_mpo_shapes,
        const std::int64_t* output_offsets,
        std::size_t n_output_offsets,
        const std::int64_t* output_shape_offsets,
        std::size_t n_output_shape_offsets,
        const std::int64_t* output_shapes,
        std::size_t n_output_shapes,
        const std::int64_t* output_labels,
        std::size_t n_output_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        const double* route_coefficients,
        bool metric_boundary
    );
    bool advance_normal_complementary_boundary(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        bool left,
        std::int64_t site,
        bool reduced_physical,
        bool dual_right_boundary,
        const std::int32_t* routes,
        std::size_t n_routes,
        const double* bra_values,
        std::size_t n_bra_values,
        const std::int64_t* bra_offsets,
        std::size_t n_bra_offsets,
        const std::int64_t* bra_shape_offsets,
        std::size_t n_bra_shape_offsets,
        const std::int64_t* bra_shapes,
        std::size_t n_bra_shapes,
        const double* ket_values,
        std::size_t n_ket_values,
        const std::int64_t* ket_offsets,
        std::size_t n_ket_offsets,
        const std::int64_t* ket_shape_offsets,
        std::size_t n_ket_shape_offsets,
        const std::int64_t* ket_shapes,
        std::size_t n_ket_shapes,
        const std::int64_t* output_offsets,
        std::size_t n_output_offsets,
        const std::int64_t* output_shape_offsets,
        std::size_t n_output_shape_offsets,
        const std::int64_t* output_shapes,
        std::size_t n_output_shapes,
        const std::int64_t* output_labels,
        std::size_t n_output_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        std::uint64_t route_topology_revision
    );
    bool install_split_site(
        std::int64_t site,
        std::vector<double>& values,
        std::vector<std::int64_t>& offsets,
        std::vector<std::int64_t>& labels,
        std::vector<std::int64_t>& shape_offsets,
        std::vector<std::int64_t>& shapes,
        std::vector<std::int64_t>& leg_sector_offsets,
        std::vector<std::int64_t>& leg_sector_labels,
        std::vector<std::int64_t>& leg_sector_dims,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    void configure_state_average(
        const double* weights,
        std::size_t nroots,
        std::int64_t center_site
    );
    void install_state_average_center(
        std::size_t root,
        const double* values,
        std::size_t n_values
    );
    bool state_average_installed() const noexcept;
    std::size_t state_average_roots() const noexcept;
    std::int64_t state_average_center_site() const noexcept;
    const std::vector<std::vector<double>>&
    state_average_center_values() const noexcept;
    bool split_site_installed(
        std::int64_t site,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    ) const noexcept;
    void merge_active_bond();
    const PackedSiteTensor& merged_site() const noexcept;
    const PackedSiteTensor& merged_channel_site() const noexcept;
    bool advance_normal_complementary_boundary_from_split_site(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        bool left,
        std::int64_t site,
        bool reduced_physical,
        bool dual_right_boundary,
        const std::int32_t* routes,
        std::size_t n_routes,
        const std::int64_t* output_offsets,
        std::size_t n_output_offsets,
        const std::int64_t* output_shape_offsets,
        std::size_t n_output_shape_offsets,
        const std::int64_t* output_shapes,
        std::size_t n_output_shapes,
        const std::int64_t* output_labels,
        std::size_t n_output_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        std::uint64_t route_topology_revision
    );
    bool cached_normal_complementary_boundary_ready(
        const std::string& side,
        std::int64_t child_bond,
        std::int64_t site,
        std::uint64_t output_topology_revision,
        std::uint64_t route_topology_revision
    ) const noexcept;
    bool replay_normal_complementary_boundary_from_split_site(
        const std::string& side,
        std::int64_t child_bond,
        std::uint64_t numeric_revision
    );
    bool advance_metric_boundary_from_split_site(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        bool left,
        std::int64_t site,
        const std::int64_t* routes,
        std::size_t n_routes,
        const double* route_coefficients,
        std::size_t n_route_coefficients,
        const double* mpo_values,
        std::size_t n_mpo_values,
        const std::int64_t* mpo_offsets,
        std::size_t n_mpo_offsets,
        const std::int64_t* mpo_shape_offsets,
        std::size_t n_mpo_shape_offsets,
        const std::int64_t* mpo_shapes,
        std::size_t n_mpo_shapes,
        const std::int64_t* output_offsets,
        std::size_t n_output_offsets,
        const std::int64_t* output_shape_offsets,
        std::size_t n_output_shape_offsets,
        const std::int64_t* output_shapes,
        std::size_t n_output_shapes,
        const std::int64_t* output_labels,
        std::size_t n_output_labels,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        std::uint64_t route_topology_revision
    );
    bool replay_metric_boundary_from_split_site(
        const std::string& side,
        std::int64_t child_bond,
        std::uint64_t numeric_revision
    );
    std::vector<ContextualCoreBlock> contextual_core(
        std::int64_t site,
        std::int64_t physical_output_charge,
        std::int64_t physical_input_charge,
        std::int32_t boundary_bra_two_j,
        std::int32_t boundary_ket_two_j,
        std::int32_t physical_bra_two_j,
        std::int32_t physical_ket_two_j,
        std::int32_t next_bra_two_j,
        std::int32_t next_ket_two_j,
        bool left,
        bool dual_right_basis
    ) const;
    void refresh_normal_complementary_numerics();
    std::size_t boundary_value_count(
        const std::string& side,
        std::int64_t bond
    ) const;
    void copy_boundary_values(
        const std::string& side,
        std::int64_t bond,
        double* output,
        std::size_t n_values
    ) const;
    std::size_t metric_boundary_value_count(
        const std::string& side,
        std::int64_t bond
    ) const;
    void copy_metric_boundary_values(
        const std::string& side,
        std::int64_t bond,
        double* output,
        std::size_t n_values
    ) const;
    BoundaryBufferHandle* retain_boundary_buffer(
        const std::string& side,
        std::int64_t bond
    ) const;
    BoundaryBufferHandle* retain_metric_boundary_buffer(
        const std::string& side,
        std::int64_t bond
    ) const;
    ComplexBoundaryBufferHandle* retain_complex_boundary_buffer(
        const std::string& side,
        std::int64_t bond,
        bool metric_boundary
    ) const;

    bool install_local_operator(
        const std::string& key,
        const Complex* values,
        std::size_t n_values,
        const std::int64_t* value_offsets,
        const std::int64_t* rows,
        const std::int64_t* cols,
        const std::int64_t* input_starts,
        const std::int64_t* output_starts,
        std::size_t n_blocks,
        std::size_t dimension,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    void clear_local_operator();
    void local_matvec(
        const std::string& key,
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void local_diagonal(
        const std::string& key,
        Complex* output,
        std::size_t dimension
    ) const;
    pyqed::dmrg::DavidsonResult local_davidson(
        const std::string& key,
        const Complex* diagonal,
        const Complex* guess,
        std::size_t dimension,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );

    bool install_factor_routes(
        const std::string& key,
        const std::int64_t* in_indices,
        const std::int64_t* out_indices,
        const std::int64_t* left_indices,
        const std::int64_t* right_indices,
        std::size_t n_routes,
        const std::int64_t* basis_offsets,
        const std::int64_t* basis_shapes,
        std::size_t n_basis,
        const std::int64_t* left_factor_indices,
        std::size_t n_left_factors,
        const std::int64_t* left_offsets,
        const std::int64_t* left_shape_offsets,
        std::size_t n_left_pool,
        const std::int64_t* left_shapes,
        std::size_t n_left_shapes,
        const double* left_data,
        std::size_t n_left_data,
        const std::int64_t* right_factor_indices,
        std::size_t n_right_factors,
        const std::int64_t* right_offsets,
        const std::int64_t* right_shape_offsets,
        std::size_t n_right_pool,
        const std::int64_t* right_shapes,
        std::size_t n_right_shapes,
        const double* right_data,
        std::size_t n_right_data,
        std::size_t total_dimension,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    bool install_raw_factor_routes(
        const std::string& key,
        const void* in_indices,
        const void* out_indices,
        const void* left_indices,
        const void* right_indices,
        std::size_t n_routes,
        std::size_t route_basis_index_bytes,
        std::size_t route_factor_index_bytes,
        const std::int64_t* basis_offsets,
        const std::int64_t* basis_shapes,
        std::size_t n_basis,
        const std::int64_t* left_factor_indices,
        std::size_t n_left_factors,
        const std::int64_t* left_boundary_ids,
        const std::int64_t* left_w_ids,
        std::size_t n_left_raw_factors,
        const std::int64_t* left_boundary_offsets,
        const std::int64_t* left_boundary_shape_offsets,
        std::size_t n_left_boundary_arrays,
        const std::int64_t* left_boundary_shapes,
        std::size_t n_left_boundary_shape_values,
        const double* left_boundary_data,
        std::size_t n_left_boundary_data,
        const std::int64_t* left_w_offsets,
        const std::int64_t* left_w_shape_offsets,
        std::size_t n_left_w_arrays,
        const std::int64_t* left_w_shapes,
        std::size_t n_left_w_shape_values,
        const double* left_w_data,
        std::size_t n_left_w_data,
        const std::int64_t* right_factor_indices,
        std::size_t n_right_factors,
        const std::int64_t* right_boundary_ids,
        const std::int64_t* right_w_ids,
        std::size_t n_right_raw_factors,
        const std::int64_t* right_boundary_offsets,
        const std::int64_t* right_boundary_shape_offsets,
        std::size_t n_right_boundary_arrays,
        const std::int64_t* right_boundary_shapes,
        std::size_t n_right_boundary_shape_values,
        const double* right_boundary_data,
        std::size_t n_right_boundary_data,
        const std::int64_t* right_w_offsets,
        const std::int64_t* right_w_shape_offsets,
        std::size_t n_right_w_arrays,
        const std::int64_t* right_w_shapes,
        std::size_t n_right_w_shape_values,
        const double* right_w_data,
        std::size_t n_right_w_data,
        const void* left_family_masks,
        const void* right_family_masks,
        std::size_t family_mask_bytes,
        std::size_t total_dimension,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision,
        bool direct_actions,
        bool sources_preconfigured
    );
    bool install_contextual_factor_routes(
        const std::string& key,
        std::int64_t bond,
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        const std::int64_t* basis_offsets,
        const std::int64_t* basis_shapes,
        const std::int64_t* basis_quantum_numbers,
        const std::int64_t* left_sector_ids,
        const std::int64_t* right_sector_ids,
        std::size_t n_basis,
        std::size_t total_dimension,
        std::uint64_t topology_revision,
        bool dual_right_basis
    );
    int prepare_active_bond_complementary_actions(
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        std::size_t expected_basis,
        std::size_t expected_dimension,
        bool dual_right_basis
    );
    const std::string& factor_route_key() const noexcept;
    std::size_t prepare_active_bond_metric_routes(
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond
    );
    const std::string& metric_key() const noexcept;
    void clear_factor_routes();
    void set_factor_routes_hermitianized(bool enabled) noexcept;
    void factor_route_matvec(
        const std::string& key,
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void factor_route_real_matvec(
        const std::string& key,
        const double* input,
        double* output,
        std::size_t dimension
    );
    void factor_route_diagonal(
        const std::string& key,
        double* output,
        std::size_t dimension
    );
    bool factor_route_installed(
        const std::string& key,
        std::size_t dimension
    ) const noexcept;
    pyqed::dmrg::DavidsonResult factor_route_davidson(
        const std::string& key,
        const Complex* diagonal,
        const Complex* guess,
        std::size_t dimension,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    pyqed::dmrg::DavidsonResult active_bond_complementary_davidson(
        const std::string& key,
        const Complex* guess,
        std::size_t dimension,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    ActiveBondCanonicalSolveResult solve_active_bond_canonical(
        const std::string& metric_key,
        double projection_tolerance,
        std::size_t max_component_elements,
        std::size_t max_transform_elements,
        double davidson_tolerance,
        int max_iterations,
        int requested_restart_dimension,
        std::size_t workspace_budget_bytes,
        std::size_t workspace_basis_arrays,
        bool accept_unconverged
    );
    ActiveBondCanonicalSolveResult
    prepare_and_solve_active_bond_canonical(
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        bool dual_right_basis,
        double projection_tolerance,
        std::size_t max_component_elements,
        std::size_t max_transform_elements,
        double davidson_tolerance,
        int max_iterations,
        int requested_restart_dimension,
        std::size_t workspace_budget_bytes,
        std::size_t workspace_basis_arrays,
        bool accept_unconverged
    );
    ActiveBondStateAverageSolveResult
    prepare_and_solve_active_bond_state_average(
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        bool dual_right_basis,
        double projection_tolerance,
        std::size_t max_component_elements,
        std::size_t max_transform_elements,
        double davidson_tolerance,
        int max_iterations,
        int requested_restart_dimension,
        std::size_t workspace_budget_bytes,
        std::size_t workspace_basis_arrays,
        bool accept_unconverged
    );
    std::vector<double>
    evaluate_truncated_state_average_active_bond();
    ActiveBondCanonicalSolveResult
    prepare_and_solve_active_bond_orthonormal(
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        bool dual_right_basis,
        double davidson_tolerance,
        int max_iterations,
        int requested_restart_dimension,
        std::size_t workspace_budget_bytes,
        std::size_t workspace_basis_arrays,
        bool accept_unconverged
    );
    bool active_bond_complementary_action_ready(
        const std::string& key,
        std::size_t dimension
    ) const noexcept;

    bool begin_factor_route_projection(
        const std::string& key,
        const std::string& factor_route_key,
        std::size_t parent_dimension,
        std::size_t orthonormal_dimension,
        std::size_t n_components,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    void install_factor_route_projection_component(
        std::size_t component,
        const std::int64_t* parent_indices,
        std::size_t n_indices,
        const double* real_transform,
        const Complex* complex_transform,
        std::size_t transform_columns,
        std::size_t orthonormal_offset
    );
    void install_factor_route_projection_indexed_component(
        std::size_t component,
        const std::int64_t* parent_indices,
        std::size_t n_parent_indices,
        const std::int64_t* orthonormal_indices,
        std::size_t n_orthonormal_indices,
        const double* real_transform,
        const Complex* complex_transform
    );
    void install_factor_route_projection_kronecker_component(
        std::size_t component,
        std::size_t parent_offset,
        const std::int64_t* orthonormal_indices,
        std::size_t n_orthonormal_indices,
        std::size_t left_dim,
        std::size_t selected_dim,
        std::size_t local_dim,
        std::size_t right_dim,
        const double* real_transform,
        const Complex* complex_transform
    );
    void finish_factor_route_projection();
    void clear_factor_route_projection();
    void factor_route_projected_matvec(
        const std::string& key,
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void factor_route_projected_real_matvec(
        const std::string& key,
        const double* input,
        double* output,
        std::size_t dimension
    );
    pyqed::dmrg::DavidsonResult factor_route_projected_davidson(
        const std::string& key,
        const Complex* diagonal,
        const Complex* guess,
        std::size_t dimension,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    bool begin_factorized_metric(
        const std::string& key,
        std::size_t dimension,
        std::size_t n_routes,
        std::uint64_t topology_revision,
        std::uint64_t numeric_revision
    );
    std::size_t install_contextual_metric_routes(
        const std::string& key,
        std::int64_t left_boundary_bond,
        std::int64_t right_boundary_bond,
        const std::int64_t* basis_offsets,
        const std::int64_t* basis_shapes,
        const std::int64_t* basis_quantum_numbers,
        const std::int64_t* left_sector_ids,
        const std::int64_t* right_sector_ids,
        std::size_t n_basis,
        std::size_t total_dimension,
        std::uint64_t topology_revision
    );
    void install_factorized_metric_route(
        std::size_t route,
        std::int64_t input_offset,
        std::int64_t output_offset,
        const std::int64_t* input_shape,
        const std::int64_t* output_shape,
        const double* left_real,
        const Complex* left_complex,
        const double* right_real,
        const Complex* right_complex
    );
    void finish_factorized_metric();
    void clear_factorized_metric();
    void factorized_metric_matvec(
        const std::string& key,
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void factorized_metric_real_matvec(
        const std::string& key,
        const double* input,
        double* output,
        std::size_t dimension
    );
    void factorized_metric_real_diagonal(
        const std::string& key,
        double* output,
        std::size_t dimension
    ) const;
    CanonicalProjectionInfo prepare_canonical_reduced_projection(
        const std::string& metric_key,
        double tolerance,
        std::size_t max_component_elements,
        std::size_t max_transform_elements
    );
    void canonical_reduced_projection_guess(
        const std::string& projection_key,
        const std::string& metric_key,
        const Complex* parent_guess,
        Complex* orthonormal_guess,
        std::size_t parent_dimension,
        std::size_t orthonormal_dimension
    );
    void lift_factor_route_projection_vector(
        const std::string& projection_key,
        const Complex* orthonormal_vector,
        Complex* parent_vector,
        std::size_t orthonormal_dimension,
        std::size_t parent_dimension
    );
    pyqed::dmrg::DavidsonResult factor_route_generalized_davidson(
        const std::string& factor_route_key,
        const std::string& metric_key,
        const Complex* h_diagonal,
        const Complex* n_diagonal,
        const Complex* guess,
        std::size_t dimension,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    pyqed::dmrg::DavidsonResult
    active_bond_complementary_generalized_davidson(
        const std::string& factor_route_key,
        const std::string& metric_key,
        const Complex* guess,
        std::size_t dimension,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    pyqed::dmrg::DavidsonResult
    factor_route_projected_generalized_davidson(
        const std::string& projection_key,
        const std::string& metric_key,
        const Complex* h_diagonal,
        const Complex* n_diagonal,
        const Complex* guess,
        std::size_t dimension,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bool accept_unconverged
    );
    BlockSVDResult blockwise_svd(
        const Complex* values,
        std::size_t n_values,
        const std::int64_t* value_offsets,
        const std::int64_t* rows,
        const std::int64_t* cols,
        const std::int64_t* state_weights,
        std::size_t n_blocks,
        double cutoff,
        std::int64_t max_bond,
        const std::string& max_bond_mode,
        bool retain_sector_topology
    );
    ActiveBondSplitResult split_active_bond_solution(
        double cutoff,
        std::int64_t max_bond,
        const std::string& max_bond_mode,
        bool retain_sector_topology,
        const std::string& absorb,
        bool install_and_stage,
        bool retain_result_tensors
    );
    bool active_bond_solution_ready() const noexcept;

    std::vector<std::int64_t> sweep_bonds(
        const std::string& direction,
        std::int64_t n_sites
    ) const;
    void begin_half_sweep(const std::string& direction, std::int64_t n_sites);
    void prepare_owned_half_sweep();
    int owned_half_sweep_readiness_code() const noexcept;
    bool owned_half_sweep_ready() const noexcept;
    std::vector<OwnedHalfSweepBondResult> execute_owned_half_sweep(
        double cutoff,
        std::int64_t max_bond,
        const std::string& max_bond_mode,
        bool retain_sector_topology,
        double projection_tolerance,
        std::size_t max_component_elements,
        std::size_t max_transform_elements,
        double davidson_tolerance,
        int max_iterations,
        int requested_restart_dimension,
        std::size_t workspace_budget_bytes,
        std::size_t workspace_basis_arrays,
        bool accept_unconverged
    );
    std::vector<OwnedSplitSiteExport> export_owned_split_sites() const;
    SpatialNPDMResult spatial_npdm(bool spin_rotation_reduction);
    SpatialNPDMResult spatial_npdm_component_reference(
        bool spin_rotation_reduction
    ) const;
    void release_workspaces();
    std::size_t execute_half_sweep(
        HalfSweepBondExecutor executor,
        void* context
    );
    std::int64_t claim_next_bond();
    void begin_bond(std::int64_t bond);
    void mark_bond_solved();
    void mark_bond_split(std::uint64_t kept_states, double truncation_seconds);
    void stage_bond_update(
        std::uint64_t kept_states,
        double truncation_seconds
    );
    void mark_bond_advanced();
    void commit_bond(
        std::uint64_t matvec_calls,
        std::uint64_t davidson_iterations,
        double matvec_seconds,
        double davidson_seconds,
        double energy
    );
    void commit_bond_update(
        std::uint64_t matvec_calls,
        std::uint64_t davidson_iterations,
        double matvec_seconds,
        double davidson_seconds,
        double energy
    );
    void record_bond(
        std::int64_t bond,
        std::uint64_t matvec_calls,
        std::uint64_t davidson_iterations,
        std::uint64_t kept_states,
        double matvec_seconds,
        double davidson_seconds,
        double truncation_seconds
    );
    void finish_half_sweep();
    void abort_half_sweep() noexcept;

    const System* system() const noexcept;
    std::uint64_t system_revision() const noexcept;
    std::uint64_t boundary_topology_builds() const noexcept;
    std::uint64_t boundary_numeric_refreshes() const noexcept;
    std::uint64_t boundary_reallocations() const noexcept;
    std::uint64_t boundary_update_topology_builds() const noexcept;
    std::uint64_t boundary_update_calls() const noexcept;
    std::uint64_t boundary_update_routes() const noexcept;
    double boundary_update_seconds() const noexcept;
    std::size_t normal_complementary_boundary_action_count() const noexcept;
    std::size_t normal_complementary_boundary_action_bytes() const noexcept;
    std::size_t metric_boundary_count() const noexcept;
    std::size_t metric_boundary_action_count() const noexcept;
    std::size_t metric_boundary_action_bytes() const noexcept;
    std::uint64_t local_topology_builds() const noexcept;
    std::uint64_t local_numeric_refreshes() const noexcept;
    std::uint64_t local_matvec_calls() const noexcept;
    std::uint64_t local_davidson_calls() const noexcept;
    std::uint64_t local_davidson_workspace_reuses() const noexcept;
    std::uint64_t factor_route_topology_builds() const noexcept;
    std::uint64_t factor_route_numeric_refreshes() const noexcept;
    std::uint64_t factor_route_matvec_calls() const noexcept;
    std::uint64_t real_factor_route_matvec_calls() const noexcept;
    std::uint64_t factor_route_diagonal_calls() const noexcept;
    std::uint64_t factor_route_davidson_calls() const noexcept;
    std::uint64_t factor_route_scratch_growths() const noexcept;
    std::uint64_t contextual_route_plan_builds() const noexcept;
    std::uint64_t contextual_route_plan_hits() const noexcept;
    std::uint64_t contextual_route_plan_shape_refreshes() const noexcept;
    std::uint64_t decomposed_action_plan_builds() const noexcept;
    std::uint64_t decomposed_action_plan_hits() const noexcept;
    std::uint64_t decomposed_action_plan_rebuilds() const noexcept;
    std::size_t complementary_execution_graph_bytes() const noexcept;
    std::uint64_t complementary_execution_graph_builds() const noexcept;
    std::uint64_t complementary_execution_graph_hits() const noexcept;
    std::size_t contextual_route_plan_count() const noexcept;
    std::size_t contextual_route_plan_bytes() const noexcept;
    std::size_t contextual_route_index_bytes() const noexcept;
    std::size_t contextual_route_core_value_bytes() const noexcept;
    std::size_t contextual_compiled_schedule_bytes() const noexcept;
    std::uint64_t contextual_compiled_schedule_builds() const noexcept;
    std::uint64_t contextual_compiled_schedule_hits() const noexcept;
    double contextual_compiled_schedule_restore_seconds() const noexcept;
    std::size_t contextual_route_core_elements() const noexcept;
    std::size_t contextual_route_core_nonzero_elements() const noexcept;
    std::size_t contextual_core_cache_count() const noexcept;
    std::size_t contextual_core_cache_bytes() const noexcept;
    std::uint64_t contextual_core_cache_hits() const noexcept;
    std::uint64_t contextual_core_reuse_hits() const noexcept;
    std::size_t contextual_route_skeleton_count() const noexcept;
    std::size_t contextual_route_skeleton_bytes() const noexcept;
    std::uint64_t contextual_route_skeleton_hits() const noexcept;
    double contextual_route_match_seconds() const noexcept;
    double contextual_route_activation_seconds() const noexcept;
    double contextual_core_build_seconds() const noexcept;
    double contextual_core_reuse_seconds() const noexcept;
    double raw_route_setup_seconds() const noexcept;
    double raw_route_group_seconds() const noexcept;
    double dense_pair_build_seconds() const noexcept;
    double fused_factor_build_seconds() const noexcept;
    double raw_execution_build_seconds() const noexcept;
    double raw_factor_matvec_seconds() const noexcept;
    double raw_input_pack_seconds() const noexcept;
    double dense_pair_matvec_seconds() const noexcept;
    double raw_execution_matvec_seconds() const noexcept;
    double raw_execution_pack_seconds() const noexcept;
    double raw_batch_expand_seconds() const noexcept;
    double raw_batch_right_prepare_seconds() const noexcept;
    double raw_batch_fallback_prepare_seconds() const noexcept;
    double raw_channel_first_stage_seconds() const noexcept;
    double raw_wave_batch_seconds() const noexcept;
    double raw_shared_left_output_seconds() const noexcept;
    double raw_grouped_output_seconds() const noexcept;
    double raw_binding_output_seconds() const noexcept;
    double raw_fusion_finalize_seconds() const noexcept;
    double raw_pointer_execution_matvec_seconds() const noexcept;
    std::uint64_t raw_pointer_execution_matvec_calls() const noexcept;
    double direct_complementary_action_seconds() const noexcept;
    std::uint64_t direct_complementary_action_calls() const noexcept;
    std::uint64_t direct_complementary_actions() const noexcept;
    double factorized_metric_matvec_seconds() const noexcept;
    std::uint64_t contextual_zero_core_cache_hits() const noexcept;
    std::size_t contextual_zero_core_cache_count() const noexcept;
    std::size_t contextual_zero_core_cache_bytes() const noexcept;
    bool raw_factor_routes() const noexcept;
    bool factor_routes_hermitianized() const noexcept;
    std::uint64_t raw_factor_cache_hits() const noexcept;
    std::uint64_t raw_factor_cache_misses() const noexcept;
    std::uint64_t raw_factor_gemm_calls() const noexcept;
    std::uint64_t raw_output_product_calls() const noexcept;
    std::uint64_t direct_source_factor_loads() const noexcept;
    std::size_t compact_right_panel_count() const noexcept;
    std::size_t compact_right_panel_value_bytes() const noexcept;
    std::size_t compact_right_panel_product_count() const noexcept;
    std::size_t compact_right_panel_batch_count() const noexcept;
    std::size_t compact_right_panel_budget_bytes() const noexcept;
    std::uint64_t compact_right_panel_registry_builds() const noexcept;
    std::uint64_t compact_right_panel_numeric_refreshes() const noexcept;
    std::uint64_t compact_right_panel_matvec_batches() const noexcept;
    std::uint64_t compact_right_panel_matvec_products() const noexcept;
    std::uint64_t complementary_family_route_count(
        std::int32_t family_id
    ) const;
    std::uint64_t unlabeled_family_route_count() const noexcept;
    double raw_factor_build_seconds() const noexcept;
    std::size_t raw_route_group_count() const noexcept;
    bool complementary_local_actions() const noexcept;
    std::size_t complementary_local_action_count() const noexcept;
    std::size_t complementary_local_term_count() const noexcept;
    std::size_t complementary_local_action_bytes() const noexcept;
    std::size_t fused_raw_route_group_count() const noexcept;
    std::size_t fused_raw_route_count() const noexcept;
    std::size_t dense_pair_kernel_count() const noexcept;
    std::size_t dense_pair_execution_count() const noexcept;
    std::size_t dense_pair_wave_count() const noexcept;
    std::size_t dense_pair_max_wave_width() const noexcept;
    std::size_t dense_pair_thread_workspace_bytes() const noexcept;
    std::size_t dense_pair_kernel_elements() const noexcept;
    std::size_t dense_pair_route_count() const noexcept;
    std::size_t dense_factor_pack_bytes() const noexcept;
    std::uint64_t dense_factor_pack_builds() const noexcept;
    std::uint64_t dense_factor_pack_reuses() const noexcept;
    std::size_t raw_execution_group_count() const noexcept;
    std::size_t raw_execution_action_count() const noexcept;
    std::size_t right_grouped_execution_action_count() const noexcept;
    std::size_t peak_right_grouped_execution_action_count() const noexcept;
    std::size_t peak_raw_execution_group_count() const noexcept;
    std::size_t peak_raw_execution_action_count() const noexcept;
    std::size_t raw_input_superchannel_count() const noexcept;
    std::size_t raw_input_superchannel_tile_count() const noexcept;
    std::size_t raw_input_superchannel_batch_count() const noexcept;
    std::size_t peak_raw_input_superchannel_batch_count() const noexcept;
    std::size_t peak_raw_channel_unique_left_count() const noexcept;
    std::size_t peak_raw_channel_left_occurrence_count() const noexcept;
    std::size_t peak_raw_shared_left_panel_count() const noexcept;
    std::size_t peak_raw_shared_left_occurrence_count() const noexcept;
    std::size_t peak_raw_output_fusion_wave_count() const noexcept;
    std::size_t peak_raw_output_fusion_group_count() const noexcept;
    std::size_t peak_raw_output_fusion_tile_count() const noexcept;
    std::size_t peak_raw_output_fusion_workspace_bytes() const noexcept;
    std::uint64_t raw_output_fusion_gemm_calls() const noexcept;
    std::uint64_t raw_output_fusion_copied_elements() const noexcept;
    std::size_t peak_persistent_output_batch_count() const noexcept;
    std::size_t peak_persistent_output_task_count() const noexcept;
    std::size_t peak_persistent_output_group_count() const noexcept;
    std::uint64_t private_output_executor_calls() const noexcept;
    std::uint64_t private_output_executor_fallbacks() const noexcept;
    std::size_t peak_private_output_task_count() const noexcept;
    std::size_t peak_private_output_workspace_bytes() const noexcept;
    std::uint64_t private_output_reduced_elements() const noexcept;
    bool grouped_output_product_backend() const noexcept;
    std::size_t grouped_output_product_group_count() const noexcept;
    std::size_t grouped_output_product_binding_count() const noexcept;
    std::size_t peak_grouped_output_candidate_binding_count()
        const noexcept;
    std::uint64_t peak_grouped_output_candidate_work_count(
        std::int32_t bin
    ) const;
    std::uint64_t grouped_output_product_batch_calls() const noexcept;
    std::uint64_t grouped_output_products() const noexcept;
    std::size_t peak_raw_shared_right_panel_count() const noexcept;
    std::size_t peak_raw_shared_right_binding_count() const noexcept;
    std::size_t peak_raw_shared_right_workspace_bytes() const noexcept;
    std::uint64_t raw_shared_right_gemm_calls() const noexcept;
    std::uint64_t raw_shared_right_copied_elements() const noexcept;
    bool reduced_contextual_routes() const noexcept;
    std::size_t reduced_contextual_execution_count() const noexcept;
    std::size_t reduced_contextual_matrix_elements() const noexcept;
    std::size_t peak_reduced_contextual_execution_count() const noexcept;
    std::size_t peak_reduced_contextual_matrix_elements() const noexcept;
    std::size_t peak_borrowed_reduced_contextual_right_elements()
        const noexcept;
    std::size_t complementary_execution_slab_bytes() const noexcept;
    std::size_t complementary_execution_slab_capacity_bytes() const noexcept;
    std::size_t complementary_execution_slab_budget_bytes() const noexcept;
    std::size_t complementary_execution_slab_required_bytes() const noexcept;
    std::size_t peak_complementary_execution_slab_required_bytes()
        const noexcept;
    std::size_t peak_complementary_left_required_bytes() const noexcept;
    std::size_t peak_complementary_right_required_bytes() const noexcept;
    std::size_t peak_complementary_left_cached_bytes() const noexcept;
    std::size_t peak_complementary_right_cached_bytes() const noexcept;
    std::uint64_t complementary_execution_slab_full_prepares() const noexcept;
    std::uint64_t complementary_execution_slab_partial_prepares()
        const noexcept;
    std::uint64_t complementary_execution_slab_matvec_repacks()
        const noexcept;
    std::int64_t peak_reduced_contextual_boundary_rank() const noexcept;
    std::uint64_t reduced_contextual_fallbacks() const noexcept;
    std::int32_t reduced_contextual_fallback_reason() const noexcept;
    double reduced_contextual_fallback_residual_norm() const noexcept;
    double reduced_contextual_fallback_boundary_norm() const noexcept;
    double reduced_contextual_build_seconds() const noexcept;
    double reduced_contextual_numeric_refresh_seconds() const noexcept;
    double reduced_contextual_boundary_refresh_seconds() const noexcept;
    double reduced_contextual_scale_refresh_seconds() const noexcept;
    double reduced_contextual_execution_refresh_seconds() const noexcept;
    double reduced_contextual_diagonal_seconds() const noexcept;
    double reduced_contextual_matvec_seconds() const noexcept;
    std::uint64_t factor_route_projection_topology_builds() const noexcept;
    std::uint64_t factor_route_projection_numeric_refreshes() const noexcept;
    std::uint64_t factor_route_projected_matvec_calls() const noexcept;
    std::uint64_t factor_route_projected_davidson_calls() const noexcept;
    std::uint64_t factor_route_generalized_davidson_calls() const noexcept;
    std::uint64_t real_generalized_davidson_calls() const noexcept;
    std::uint64_t factorized_metric_matvec_calls() const noexcept;
    std::uint64_t real_factorized_metric_matvec_calls() const noexcept;
    std::uint64_t canonical_projection_builds() const noexcept;
    std::uint64_t canonical_projection_reuses() const noexcept;
    std::uint64_t canonical_projection_davidson_calls() const noexcept;
    std::size_t canonical_projection_transform_elements() const noexcept;
    std::size_t canonical_projection_max_component_dimension() const noexcept;
    std::size_t canonical_projection_cache_entries() const noexcept;
    std::size_t canonical_projection_cache_transform_elements() const noexcept;
    std::uint64_t canonical_projection_cache_evictions() const noexcept;
    double canonical_projection_whitening_residual() const noexcept;
    double canonical_projection_build_seconds() const noexcept;
    std::uint64_t block_svd_calls() const noexcept;
    std::uint64_t block_svd_blocks() const noexcept;
    std::uint64_t block_svd_workspace_growths() const noexcept;
    double block_svd_seconds() const noexcept;
    std::size_t block_svd_workspace_bytes() const noexcept;
    std::uint64_t split_site_owner_revision() const noexcept;
    std::uint64_t split_site_installs() const noexcept;
    std::uint64_t split_site_topology_builds() const noexcept;
    std::uint64_t split_site_boundary_uses() const noexcept;
    std::uint64_t cached_boundary_replays() const noexcept;
    std::size_t split_site_count() const noexcept;
    std::size_t split_site_bytes() const noexcept;
    std::uint64_t site_merge_calls() const noexcept;
    std::uint64_t site_merge_blocks() const noexcept;
    double site_merge_seconds() const noexcept;
    std::size_t site_merge_bytes() const noexcept;
    std::uint64_t active_bond_complementary_prepares() const noexcept;
    std::uint64_t active_bond_complementary_fallbacks() const noexcept;
    int active_bond_complementary_fallback_reason() const noexcept;
    std::int64_t active_bond_complementary_fallback_bond() const noexcept;
    std::size_t active_bond_complementary_basis() const noexcept;
    std::size_t active_bond_complementary_dimension() const noexcept;
    std::size_t active_bond_complementary_expected_basis() const noexcept;
    std::size_t active_bond_complementary_expected_dimension() const noexcept;
    std::uint64_t active_bond_complementary_davidson_calls() const noexcept;
    std::uint64_t
    active_bond_complementary_generalized_davidson_calls() const noexcept;
    std::uint64_t active_bond_metric_prepares() const noexcept;
    std::uint64_t active_bond_cpp_splits() const noexcept;
    std::uint64_t half_sweeps() const noexcept;
    std::uint64_t half_sweep_executor_calls() const noexcept;
    std::uint64_t half_sweep_executor_bonds() const noexcept;
    std::uint64_t half_sweep_python_bond_callbacks() const noexcept;
    double half_sweep_executor_seconds() const noexcept;
    std::uint64_t owned_half_sweep_calls() const noexcept;
    std::uint64_t owned_half_sweep_bonds() const noexcept;
    double owned_half_sweep_seconds() const noexcept;
    std::uint64_t aborted_half_sweeps() const noexcept;
    std::uint64_t bond_steps() const noexcept;
    std::uint64_t bond_prepares() const noexcept;
    std::uint64_t bond_solves() const noexcept;
    std::uint64_t bond_splits() const noexcept;
    std::uint64_t bond_advances() const noexcept;
    std::uint64_t staged_bond_updates() const noexcept;
    std::uint64_t committed_bond_updates() const noexcept;
    std::uint64_t matvec_calls() const noexcept;
    std::uint64_t davidson_iterations() const noexcept;
    std::uint64_t kept_states() const noexcept;
    double matvec_seconds() const noexcept;
    double davidson_seconds() const noexcept;
    double truncation_seconds() const noexcept;
    double last_half_sweep_energy() const noexcept;
    std::size_t boundary_count() const noexcept;
    std::size_t borrowed_boundary_bytes() const noexcept;
    std::size_t owned_boundary_bytes() const noexcept;
    std::size_t local_operator_blocks() const noexcept;
    std::size_t borrowed_local_operator_bytes() const noexcept;
    std::size_t factor_route_count() const noexcept;
    std::size_t factor_route_table_bytes() const noexcept;
    std::size_t borrowed_factor_pool_bytes() const noexcept;
    std::size_t factor_route_scratch_bytes() const noexcept;
    std::size_t borrowed_raw_factor_source_bytes() const noexcept;
    std::size_t raw_factor_cache_bytes() const noexcept;
    std::size_t peak_raw_factor_cache_bytes() const noexcept;
    std::size_t factor_route_projection_components() const noexcept;
    std::size_t factor_route_projection_index_bytes() const noexcept;
    std::size_t borrowed_factor_route_transform_bytes() const noexcept;
    std::size_t factor_route_projection_scratch_bytes() const noexcept;
    std::size_t factorized_metric_route_bytes() const noexcept;
    std::size_t factorized_metric_scratch_bytes() const noexcept;
    std::size_t davidson_workspace_bytes() const noexcept;
    std::size_t peak_borrowed_local_operator_bytes() const noexcept;
    std::size_t peak_factor_route_table_bytes() const noexcept;
    std::size_t peak_borrowed_factor_pool_bytes() const noexcept;
    std::size_t peak_factor_route_scratch_bytes() const noexcept;
    std::size_t peak_factor_route_projection_index_bytes() const noexcept;
    std::size_t peak_borrowed_factor_route_transform_bytes() const noexcept;
    std::size_t peak_factor_route_projection_scratch_bytes() const noexcept;
    std::size_t memory_bytes() const noexcept;
    const std::string& direction() const noexcept;
    std::int32_t lifecycle_phase_code() const noexcept;
    std::int64_t active_bond() const noexcept;

private:
    static std::string boundary_key(const std::string& side, std::int64_t bond);
    static void decode_boundary_shapes(PackedArena& arena);
    void rebuild_normal_complementary_boundary_from_split_site(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        std::int64_t site,
        std::uint64_t numeric_revision
    );
    void rebuild_metric_boundary_from_split_site(
        const std::string& side,
        std::int64_t parent_bond,
        std::int64_t child_bond,
        std::int64_t site,
        std::uint64_t numeric_revision
    );
    void install_canonical_metric_boundary_from_split_site(
        const std::string& side,
        std::int64_t child_bond,
        std::int64_t site,
        std::uint64_t numeric_revision
    );
    void require_local_key(const std::string& key, std::size_t dimension) const;
    void require_factor_route_key(
        const std::string& key,
        std::size_t dimension
    ) const;
    bool activate_contextual_factor_route_plan(
        const std::string& key,
        ContextualFactorRoutePlan& plan,
        const PackedArena& left_boundary,
        const PackedArena& right_boundary
    );
    bool try_activate_reduced_contextual_factor_routes(
        const std::string& key,
        ContextualFactorRoutePlan& plan,
        const PackedArena& left_boundary,
        const PackedArena& right_boundary,
        std::uint64_t numeric_revision
    );
    bool try_activate_component_contextual_factor_routes(
        const std::string& key,
        ContextualFactorRoutePlan& plan,
        const PackedArena& left_boundary,
        const PackedArena& right_boundary,
        std::uint64_t numeric_revision
    );
    void capture_contextual_compiled_schedule(
        ContextualFactorRoutePlan& plan
    );
    void restore_contextual_compiled_schedule(
        const ContextualFactorRoutePlan& plan
    );
    const std::vector<ContextualCoreBlock>*
    cached_contextual_core(
        const std::array<std::int64_t, 11>& key
    ) const noexcept;
    const std::vector<ContextualCoreBlock>* contextual_core_view(
        std::int64_t site,
        std::int64_t physical_output_charge,
        std::int64_t physical_input_charge,
        std::int32_t boundary_bra_two_j,
        std::int32_t boundary_ket_two_j,
        std::int32_t physical_bra_two_j,
        std::int32_t physical_ket_two_j,
        std::int32_t next_bra_two_j,
        std::int32_t next_ket_two_j,
        bool left,
        bool dual_right_basis,
        std::vector<ContextualCoreBlock>& transient
    ) const;
    double cached_contextual_recoupling(
        const ContextualRecouplingKey& key
    ) const;
    void require_projection_key(
        const std::string& key,
        std::size_t dimension
    ) const;
    void require_metric_key(
        const std::string& key,
        std::size_t dimension
    ) const;
    void apply_factor_routes(
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void apply_factor_routes_real(
        const double* input,
        double* output,
        std::size_t dimension
    );
    static void clear_raw_factor_source(RawFactorSource& source);
    static std::size_t raw_factor_source_bytes(
        const RawFactorSource& source
    ) noexcept;
    static std::int64_t raw_boundary_id(
        const RawFactorSource& source,
        std::size_t factor
    );
    static std::int64_t raw_w_id(
        const RawFactorSource& source,
        std::size_t factor
    );
    static std::int64_t raw_w_offset(
        const RawFactorSource& source,
        std::size_t w_id
    );
    static std::array<std::int64_t, 4> raw_w_shape(
        const RawFactorSource& source,
        std::size_t w_id
    );
    std::array<std::int64_t, 5> raw_factor_shape(
        const RawFactorSource& source,
        std::int64_t factor,
        bool left
    ) const;
    RawFactorView raw_factor(
        RawFactorSource& source,
        std::int64_t factor,
        bool left
    );
    bool load_direct_source_factor(
        const RawFactorSource& source,
        std::int64_t factor,
        bool left,
        std::size_t count,
        std::size_t local,
        double* target,
        bool add
    );
    bool load_direct_source_factor_slice(
        const RawFactorSource& source,
        std::int64_t factor,
        bool left,
        std::int64_t total_w,
        std::int64_t w_offset,
        double* target,
        bool add
    );
    void apply_raw_factor_route(
        const FactorRoute& route,
        const Complex* input,
        Complex* output
    );
    void build_raw_route_groups();
    void build_dense_pair_kernels();
    void build_dense_pair_executions();
    void build_fused_factor_aggregates();
    void build_raw_execution_groups();
    void build_direct_execution_waves();
    void build_raw_input_superchannels();
    void build_compact_right_panel_registry();
    void refresh_compact_right_panels();
    void build_raw_output_fusion_waves();
    std::uint64_t reduced_contextual_action_topology_revision()
        const noexcept;
    bool raw_combined_left_schedule_valid() const;
    void stash_reduced_contextual_execution_schedule();
    bool restore_reduced_contextual_execution_schedule(
        ContextualFactorRoutePlan& plan,
        bool restore_local_actions = false
    );
    void refresh_complementary_execution_slab();
    void pack_cached_complementary_execution_slab();
    void prepare_output_fusion_right_slab();
    void select_direct_complementary_tiles();
    void build_persistent_output_group_schedule();
    void build_persistent_right_action_cache();
    void build_persistent_output_bundles(int thread_count);
    void build_persistent_output_tasks();
    std::uint64_t apply_persistent_channel_fusion_task(
        const RawOutputFusionWave& wave,
        const RawChannelFusionTask& task
    );
    std::uint64_t apply_persistent_output_group(
        const RawPersistentOutputGroup& scheduled,
        double* output,
        std::vector<double>& left_workspace,
        std::vector<double>& operator_workspace,
        std::vector<double>& right_workspace,
        std::vector<double>& product_workspace,
        std::vector<std::uint8_t>& product_valid,
        std::size_t reference_start,
        std::size_t reference_stop,
        bool combine_references
    );
    void pack_raw_execution_batch(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        double* left_values,
        double* right_values,
        std::vector<double>* panel_scratch = nullptr
    );
    bool prepare_raw_execution_batch_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* packed_input,
        double* output,
        bool allow_direct,
        bool require_right,
        const double*& temporary_values,
        const double*& right_values
    );
    bool apply_direct_complementary_batch_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* packed_input,
        double* output
    );
    bool apply_direct_complementary_temporary_batch_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* temporary_values,
        double* output
    );
    bool apply_compact_complementary_products_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* temporary_values,
        double* output
    );
    void apply_direct_complementary_temporary_tile_real(
        const RawInputSuperchannel& channel,
        const RawExecutionTile& tile,
        const double* temporary_values,
        double* output
    );
    void apply_selected_direct_complementary_temporary_tile_real(
        const RawInputSuperchannel& channel,
        const RawExecutionTile& tile,
        const double* temporary_values,
        double* output
    );
    void clear_raw_output_fusion_batch(
        const RawOutputFusionWave& wave,
        const RawOutputFusionBatch& scheduled,
        const RawInputSuperchannel& channel
    );
    bool apply_grouped_output_products(
        const RawOutputFusionWave& wave,
        const RawOutputFusionBatch& scheduled,
        const RawInputSuperchannel& channel,
        const double* temporary_values,
        const double* right_values,
        double* output
    );
    bool apply_shared_left_output_products(
        const RawOutputFusionWave& wave,
        const RawOutputFusionBatch& scheduled,
        const RawInputSuperchannel& channel,
        const double* temporary_values,
        const double* channel_temporary_values,
        double* output
    );
    const double* expand_raw_execution_batch_left_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* packed_temporary
    );
    const double* expand_raw_execution_batch_channel_left_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* channel_temporary,
        const RawOutputFusionWave& wave,
        const RawOutputFusionBatch& scheduled,
        bool skip_persistent_output
    );
    const double* prepare_raw_execution_batch_right_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch
    );
    void apply_raw_execution_batch_right_first_real(
        const RawInputSuperchannel& channel,
        const RawExecutionBatch& batch,
        const double* packed_input,
        double* output
    );
    void refresh_output_binding_right_offsets();
    void restore_planned_complementary_right_layout();
    void compact_selected_direct_right_layout();
    void prepare_complementary_execution_slab();
    bool build_complementary_execution_diagonal(
        std::int64_t total_dimension
    );
    void configure_raw_factor_caches();
    void pack_raw_execution_factor(
        RawFactorSource& source,
        std::int64_t factor,
        bool left,
        std::int64_t total_w,
        std::int64_t w_offset,
        double* target,
        bool add
    );
    void pack_raw_execution_action(
        const RawExecutionAction& action,
        std::int64_t total_w,
        std::int64_t w_offset
    );
    void pack_raw_execution_action_into(
        const RawExecutionAction& action,
        std::int64_t total_w,
        std::int64_t w_offset,
        double* left_target,
        double* right_target,
        bool horizontal_right = false,
        std::int64_t right_total_w = 0,
        std::int64_t right_w_offset = 0,
        std::vector<double>* panel_scratch = nullptr
    );
    std::int64_t raw_execution_action_left_key(
        const RawExecutionAction& action
    ) const;
    void raw_dgemm(
        bool transpose_left,
        bool transpose_right,
        std::int64_t rows,
        std::int64_t cols,
        std::int64_t inner,
        double alpha,
        const double* left,
        std::int64_t left_stride,
        const double* right,
        std::int64_t right_stride,
        double beta,
        double* output,
        std::int64_t output_stride
    );
    void accumulate_raw_output_product(
        std::int64_t rows,
        std::int64_t cols,
        std::int64_t inner,
        const double* left,
        const double* right,
        double* output
    );
    void accumulate_reduced_contextual_output_product(
        std::int64_t rows,
        std::int64_t cols,
        std::int64_t inner,
        double alpha,
        const double* left,
        std::int64_t left_stride,
        const ReducedContextualMatrix& right,
        double* output,
        std::int64_t output_stride
    );
    bool can_accumulate_factorized_output(
        const RawInputSuperchannel& channel,
        const RawExecutionTile& tile
    ) const;
    void accumulate_factorized_output(
        const RawInputSuperchannel& channel,
        const RawExecutionTile& tile,
        const double* temporary,
        double* output
    );
    void apply_raw_factor_groups(
        const Complex* input,
        Complex* output
    );
    void apply_raw_factor_groups_real(
        const double* input,
        double* output
    );
    void apply_direct_complementary_actions_real(
        const RawExecutionGroup& execution,
        std::size_t action_start,
        std::size_t action_stop,
        const double* packed_input,
        double* output
    );
    std::uint64_t apply_direct_complementary_actions_real_workspace(
        const RawExecutionGroup& execution,
        std::size_t action_start,
        std::size_t action_stop,
        const double* packed_input,
        double* output,
        std::vector<double>& left_workspace,
        std::vector<double>& temporary_workspace
    );
    void apply_raw_pointer_actions_real(
        const double* input,
        double* output
    );
    void add_raw_factor_adjoint(
        const Complex* input,
        Complex* output
    );
    void add_raw_factor_adjoint_real(
        const double* input,
        double* output
    );
    void apply_projection(
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void lift_projection(const Complex* input);
    void project_projection(Complex* output, std::size_t dimension);
    void apply_projected_metric(
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void apply_factorized_metric(
        const Complex* input,
        Complex* output,
        std::size_t dimension
    );
    void apply_factorized_metric_real(
        const double* input,
        double* output,
        std::size_t dimension
    );
    void apply_projection_real(
        const double* input,
        double* output,
        std::size_t dimension
    );
    void apply_projected_metric_real(
        const double* input,
        double* output,
        std::size_t dimension
    );
    void lift_projection_real(const double* input);
    void project_projection_real(double* output, std::size_t dimension);
    void project_parent_diagonal(
        const double* parent,
        double* output,
        std::size_t dimension
    ) const;
    bool projection_is_real() const noexcept;
    bool metric_is_real() const noexcept;

    const System* system_ = nullptr;
    int n_threads_ = 1;
    std::uint64_t openmp_parallel_regions_ = 0;
    std::uint64_t openmp_tasks_ = 0;
    std::unordered_map<std::string, PackedArena> boundaries_;
    std::unordered_map<std::string, PackedArena> metric_boundaries_;
    std::unordered_map<std::string, ComplexPackedArena> complex_boundaries_;
    std::unordered_map<std::string, ComplexPackedArena>
        complex_metric_boundaries_;
    std::unordered_map<std::int64_t, PackedArena> split_sites_;
    std::vector<std::vector<double>> state_average_center_values_;
    std::vector<double> state_average_weights_;
    std::int64_t state_average_center_site_ = -1;
    PackedSiteTensor merged_site_;
    PackedSiteTensor merged_channel_site_;
    std::unordered_map<std::string, NormalComplementaryBoundaryAction>
        normal_complementary_boundary_actions_;
    std::unordered_map<std::string, MetricBoundaryAction>
        metric_boundary_actions_;
    std::uint64_t split_site_owner_revision_ = 0;
    std::uint64_t split_site_installs_ = 0;
    std::uint64_t split_site_topology_builds_ = 0;
    std::uint64_t split_site_boundary_uses_ = 0;
    std::uint64_t cached_boundary_replays_ = 0;
    std::uint64_t site_merge_calls_ = 0;
    std::uint64_t site_merge_blocks_ = 0;
    double site_merge_seconds_ = 0.0;
    std::uint64_t active_bond_complementary_prepares_ = 0;
    std::uint64_t active_bond_complementary_fallbacks_ = 0;
    int active_bond_complementary_fallback_reason_ = 0;
    std::int64_t active_bond_complementary_fallback_bond_ = -1;
    std::size_t active_bond_complementary_basis_ = 0;
    std::size_t active_bond_complementary_dimension_ = 0;
    std::size_t active_bond_complementary_expected_basis_ = 0;
    std::size_t active_bond_complementary_expected_dimension_ = 0;
    std::uint64_t active_bond_complementary_davidson_calls_ = 0;
    std::uint64_t
        active_bond_complementary_generalized_davidson_calls_ = 0;
    std::uint64_t active_bond_metric_prepares_ = 0;
    std::vector<std::size_t> active_bond_basis_order_;
    std::vector<std::int64_t> active_bond_basis_offsets_;
    std::vector<std::int64_t> active_bond_basis_shapes_;
    std::vector<std::int64_t> active_bond_basis_quantum_numbers_;
    std::vector<Complex> active_bond_solution_;
    std::vector<std::vector<Complex>> active_bond_root_solutions_;
    std::int64_t active_bond_solution_bond_ = -1;
    std::uint64_t active_bond_cpp_splits_ = 0;
    std::uint64_t active_bond_split_numeric_revision_ = 0;
    std::vector<double> active_bond_h_diagonal_real_;
    std::vector<double> active_bond_n_diagonal_real_;
    std::vector<Complex> active_bond_h_diagonal_;
    std::vector<Complex> active_bond_n_diagonal_;

    std::string local_key_;
    std::vector<LocalBlock> local_blocks_;
    std::size_t local_dimension_ = 0;
    std::size_t local_borrowed_elements_ = 0;
    std::uint64_t local_topology_revision_ = 0;
    std::uint64_t local_numeric_revision_ = 0;

    std::string factor_route_key_;
    std::vector<FactorRoute> factor_routes_;
    std::size_t factor_route_count_ = 0;
    const double* left_factor_data_ = nullptr;
    const double* right_factor_data_ = nullptr;
    std::size_t left_factor_elements_ = 0;
    std::size_t right_factor_elements_ = 0;
    std::size_t factor_route_dimension_ = 0;
    std::uint64_t factor_route_topology_revision_ = 0;
    std::uint64_t factor_route_numeric_revision_ = 0;
    const std::uint64_t* reduced_contextual_left_revision_source_ = nullptr;
    const std::uint64_t* reduced_contextual_right_revision_source_ = nullptr;
    std::uint64_t reduced_contextual_left_revision_ = 0;
    std::uint64_t reduced_contextual_right_revision_ = 0;
    std::vector<Complex> factor_route_scratch_;
    std::vector<double> factor_route_real_scratch_;
    std::vector<double> real_matvec_input_;
    std::vector<double> real_matvec_output_;
    bool raw_factor_routes_ = false;
    bool reduced_contextual_routes_ = false;
    bool factor_routes_direct_actions_ = false;
    bool factor_routes_hermitianized_ = false;
    RawFactorSource left_raw_factor_source_;
    RawFactorSource right_raw_factor_source_;
    ContextualFactorStorage left_contextual_factors_;
    ContextualFactorStorage right_contextual_factors_;
    std::map<std::string, ContextualFactorRoutePlan> contextual_route_plans_;
    std::shared_ptr<ReducedContextualExecutionSchedule>
        active_reduced_contextual_execution_schedule_;
    std::uint64_t contextual_route_plan_clock_ = 0;
    std::uint64_t contextual_route_plan_builds_ = 0;
    std::uint64_t contextual_route_plan_hits_ = 0;
    std::uint64_t contextual_route_plan_shape_refreshes_ = 0;
    std::uint64_t decomposed_action_plan_builds_ = 0;
    std::uint64_t decomposed_action_plan_hits_ = 0;
    std::uint64_t decomposed_action_plan_rebuilds_ = 0;
    std::uint64_t complementary_execution_graph_builds_ = 0;
    std::uint64_t complementary_execution_graph_hits_ = 0;
    ContextualFactorRoutePlan* activating_contextual_plan_ = nullptr;
    std::uint64_t contextual_compiled_schedule_builds_ = 0;
    std::uint64_t contextual_compiled_schedule_hits_ = 0;
    double contextual_compiled_schedule_restore_seconds_ = 0.0;
    mutable std::set<std::array<std::int64_t, 11>>
        contextual_zero_core_cache_;
    mutable std::uint64_t contextual_zero_core_cache_hits_ = 0;
    mutable std::map<
        std::array<std::int64_t, 11>,
        std::vector<ContextualCoreBlock>
    > contextual_core_cache_;
    mutable std::size_t contextual_core_cache_elements_ = 0;
    mutable std::size_t contextual_core_cache_blocks_ = 0;
    mutable std::uint64_t contextual_core_cache_hits_ = 0;
    mutable std::unordered_map<
        ContextualRecouplingKey,
        double,
        ContextualRecouplingKeyHash
    > contextual_recoupling_cache_;
    mutable std::uint64_t contextual_recoupling_cache_hits_ = 0;
    std::unordered_map<
        ContextualActionFragmentKey,
        std::shared_ptr<const ContextualActionFragment>,
        ContextualActionFragmentKeyHash
    > contextual_action_fragments_;
    bool contextual_action_fragment_cache_enabled_ = false;
    std::size_t contextual_action_fragment_bytes_ = 0;
    std::uint64_t contextual_action_fragment_hits_ = 0;
    std::unordered_map<
        ContextualRouteSkeletonKey,
        ContextualRouteSkeleton,
        ContextualRouteSkeletonKeyHash
    > contextual_route_skeletons_;
    std::size_t contextual_route_skeleton_bytes_ = 0;
    std::uint64_t contextual_route_skeleton_hits_ = 0;
    std::uint64_t contextual_core_reuse_hits_ = 0;
    double contextual_route_match_seconds_ = 0.0;
    double contextual_route_activation_seconds_ = 0.0;
    double contextual_core_build_seconds_ = 0.0;
    double contextual_core_reuse_seconds_ = 0.0;
    double raw_route_setup_seconds_ = 0.0;
    double raw_route_group_seconds_ = 0.0;
    double dense_pair_build_seconds_ = 0.0;
    double fused_factor_build_seconds_ = 0.0;
    double raw_execution_build_seconds_ = 0.0;
    std::vector<double> raw_factor_cache_arena_;
    std::vector<std::int64_t> raw_basis_offsets_;
    std::vector<std::int64_t> raw_basis_shapes_;
    std::vector<double> raw_input_real_;
    std::vector<double> raw_input_imag_;
    std::vector<double> raw_packed_input_real_;
    std::vector<double> raw_packed_input_imag_;
    std::vector<double> raw_temporary_real_;
    std::vector<double> raw_temporary_imag_;
    std::vector<double> raw_output_real_;
    std::vector<double> raw_output_imag_;
    std::vector<RawRouteGroup> raw_route_groups_;
    std::vector<ReducedContextualMatrix> reduced_contextual_left_matrices_;
    std::vector<ReducedContextualMatrix> reduced_contextual_right_matrices_;
    ContextualDecompositionWorkspace left_decomposition_workspace_;
    ContextualDecompositionWorkspace right_decomposition_workspace_;
    std::vector<ComplementaryLocalAction> complementary_local_actions_;
    std::vector<ComplementaryLocalTerm> complementary_local_terms_;
    std::vector<double> complementary_panel_scratch_;
    std::vector<std::vector<double>> complementary_panel_thread_scratch_;
    std::vector<std::array<std::size_t, 3>> complementary_pack_tasks_;
    std::vector<double> reduced_contextual_diagonal_;
    std::uint64_t reduced_contextual_fallbacks_ = 0;
    std::int32_t reduced_contextual_fallback_reason_ = 0;
    double reduced_contextual_fallback_residual_norm_ = 0.0;
    double reduced_contextual_fallback_boundary_norm_ = 0.0;
    std::size_t peak_reduced_contextual_executions_ = 0;
    std::size_t peak_reduced_contextual_matrix_elements_ = 0;
    std::size_t peak_borrowed_reduced_contextual_right_elements_ = 0;
    std::int64_t peak_reduced_contextual_boundary_rank_ = 0;
    double reduced_contextual_build_seconds_ = 0.0;
    double reduced_contextual_numeric_refresh_seconds_ = 0.0;
    double reduced_contextual_boundary_refresh_seconds_ = 0.0;
    double reduced_contextual_scale_refresh_seconds_ = 0.0;
    double reduced_contextual_execution_refresh_seconds_ = 0.0;
    double reduced_contextual_diagonal_seconds_ = 0.0;
    double reduced_contextual_matvec_seconds_ = 0.0;
    std::vector<RawExecutionGroup> raw_execution_groups_;
    std::vector<RawExecutionAction> raw_execution_action_arena_;
    std::vector<RawExecutionAction> raw_pointer_action_arena_;
    std::vector<std::uint32_t> raw_combined_left_terms_;
    std::vector<std::size_t> direct_execution_wave_offsets_;
    std::vector<std::size_t> direct_execution_wave_indices_;
    std::size_t direct_execution_max_wave_width_ = 0;
    std::vector<std::vector<double>> direct_action_thread_left_;
    std::vector<std::vector<double>> direct_action_thread_temporary_;
    std::size_t raw_execution_actions_ = 0;
    std::size_t right_grouped_execution_actions_ = 0;
    std::size_t peak_right_grouped_execution_actions_ = 0;
    std::size_t peak_raw_execution_groups_ = 0;
    std::size_t peak_raw_execution_actions_ = 0;
    std::vector<RawInputSuperchannel> raw_input_superchannels_;
    std::vector<RawOutputFusionWave> raw_output_fusion_waves_;
    std::vector<RawPersistentOutputGroup> persistent_output_groups_;
    std::vector<RawPersistentOutputReference>
        persistent_output_references_;
    std::vector<RawPersistentOutputBinding>
        persistent_output_bindings_;
    std::vector<std::vector<double>> persistent_output_thread_left_;
    std::vector<std::vector<double>> persistent_output_thread_operator_;
    std::vector<std::vector<double>> persistent_output_thread_right_;
    std::vector<RawPersistentProductCacheEntry>
        persistent_product_cache_;
    std::size_t persistent_product_cache_elements_ = 0;
    std::vector<std::vector<double>> persistent_output_thread_products_;
    std::vector<std::vector<std::uint8_t>>
        persistent_output_thread_product_valid_;
    std::vector<std::int32_t> persistent_right_action_slots_;
    std::vector<RawPersistentRightCacheEntry> persistent_right_cache_;
    std::vector<double> persistent_right_cache_values_;
    std::vector<RawPersistentOutputBundle> persistent_output_bundles_;
    std::vector<std::uint32_t> persistent_output_bundle_groups_;
    std::vector<RawPersistentOutputTask> persistent_output_tasks_;
    std::vector<std::vector<double>> persistent_output_thread_outputs_;
    bool dynamic_persistent_output_right_ = false;
    std::vector<RawCompactRightPanel> compact_right_panels_;
    std::vector<double> compact_right_panel_values_;
    std::uint64_t compact_right_panel_registry_builds_ = 0;
    std::uint64_t compact_right_panel_numeric_refreshes_ = 0;
    std::uint64_t compact_right_panel_matvec_batches_ = 0;
    std::uint64_t compact_right_panel_matvec_products_ = 0;
    std::size_t raw_input_superchannel_tiles_ = 0;
    std::size_t raw_input_superchannel_batches_ = 0;
    std::size_t peak_raw_input_superchannel_batches_ = 0;
    std::size_t peak_raw_channel_unique_left_count_ = 0;
    std::size_t peak_raw_channel_left_occurrence_count_ = 0;
    std::size_t peak_raw_shared_left_panel_count_ = 0;
    std::size_t peak_raw_shared_left_occurrence_count_ = 0;
    std::size_t peak_raw_output_fusion_waves_ = 0;
    std::size_t peak_raw_output_fusion_groups_ = 0;
    std::size_t peak_raw_output_fusion_tiles_ = 0;
    std::size_t peak_raw_output_fusion_workspace_elements_ = 0;
    std::uint64_t raw_output_fusion_gemm_calls_ = 0;
    std::uint64_t raw_output_fusion_copied_elements_ = 0;
    std::size_t peak_persistent_output_batches_ = 0;
    std::size_t peak_persistent_output_tasks_ = 0;
    std::size_t peak_persistent_output_groups_ = 0;
    std::uint64_t private_output_executor_calls_ = 0;
    std::uint64_t private_output_executor_fallbacks_ = 0;
    std::size_t peak_private_output_tasks_ = 0;
    std::size_t peak_private_output_workspace_bytes_ = 0;
    std::uint64_t private_output_reduced_elements_ = 0;
    std::size_t grouped_output_product_groups_ = 0;
    std::size_t grouped_output_product_bindings_ = 0;
    std::size_t peak_grouped_output_candidate_bindings_ = 0;
    std::array<std::uint64_t, 10>
        peak_grouped_output_candidate_work_counts_{};
    std::uint64_t grouped_output_product_batch_calls_ = 0;
    std::uint64_t grouped_output_products_ = 0;
    std::size_t peak_raw_shared_right_panels_ = 0;
    std::size_t peak_raw_shared_right_bindings_ = 0;
    std::size_t peak_raw_shared_right_workspace_elements_ = 0;
    std::uint64_t raw_shared_right_gemm_calls_ = 0;
    std::uint64_t raw_shared_right_copied_elements_ = 0;
    std::vector<double> complementary_execution_slab_;
    std::size_t complementary_execution_slab_required_elements_ = 0;
    std::size_t peak_complementary_execution_slab_required_elements_ = 0;
    std::size_t peak_complementary_left_required_elements_ = 0;
    std::size_t peak_complementary_right_required_elements_ = 0;
    std::size_t peak_complementary_left_cached_elements_ = 0;
    std::size_t peak_complementary_right_cached_elements_ = 0;
    std::uint64_t complementary_execution_slab_full_prepares_ = 0;
    std::uint64_t complementary_execution_slab_partial_prepares_ = 0;
    std::uint64_t complementary_execution_slab_matvec_repacks_ = 0;
    std::vector<DensePairKernel> dense_pair_kernels_;
    std::vector<DensePairExecution> dense_pair_executions_;
    std::vector<std::size_t> dense_pair_wave_offsets_;
    std::vector<std::size_t> dense_pair_wave_indices_;
    std::size_t dense_pair_max_wave_width_ = 0;
    std::vector<std::vector<double>> dense_pair_thread_inputs_;
    std::vector<std::vector<double>> dense_pair_thread_outputs_;
    std::size_t dense_pair_kernels_built_ = 0;
    std::size_t dense_pair_kernel_elements_ = 0;
    std::size_t dense_pair_routes_ = 0;
    std::vector<double> dense_factor_pack_arena_;
    std::vector<std::int64_t> dense_left_factor_pack_starts_;
    std::vector<std::int64_t> dense_right_factor_pack_starts_;
    std::size_t dense_factor_pack_elements_ = 0;
    std::uint64_t dense_factor_pack_builds_ = 0;
    std::uint64_t dense_factor_pack_reuses_ = 0;
    std::vector<double> raw_batch_left_;
    std::vector<double> raw_batch_right_;
    std::vector<double> raw_unique_temporary_real_;
    std::vector<double> raw_batch_temporary_real_;
    std::vector<double> raw_batch_temporary_imag_;
    std::vector<double> raw_right_first_product_real_;
    std::vector<double> raw_persistent_output_right_;
    std::vector<double> raw_output_fusion_temporary_real_;
    std::vector<double> raw_shared_right_input_real_;
    std::vector<double> raw_shared_right_output_real_;
    std::vector<double> raw_shared_right_deferred_output_real_;
    std::vector<double> raw_shared_right_tile_output_real_;
    std::vector<double> raw_shared_left_output_real_;
    std::vector<std::size_t> raw_channel_temporary_offsets_;
    std::vector<double> raw_channel_temporary_real_;
    std::vector<std::int32_t> grouped_output_transpose_left_;
    std::vector<std::int32_t> grouped_output_transpose_right_;
    std::vector<std::int32_t> grouped_output_rows_;
    std::vector<std::int32_t> grouped_output_cols_;
    std::vector<std::int32_t> grouped_output_inner_;
    std::vector<std::int32_t> grouped_output_left_strides_;
    std::vector<std::int32_t> grouped_output_right_strides_;
    std::vector<std::int32_t> grouped_output_strides_;
    std::vector<std::int32_t> grouped_output_group_sizes_;
    std::vector<double> grouped_output_alpha_;
    std::vector<double> grouped_output_beta_;
    std::vector<const double*> grouped_output_left_;
    std::vector<const double*> grouped_output_right_;
    std::vector<double*> grouped_output_values_;
    std::vector<std::uint8_t> grouped_output_executed_;
    std::vector<std::uint32_t> grouped_output_executed_bindings_;
    std::size_t fused_raw_route_groups_ = 0;
    std::size_t fused_raw_routes_ = 0;
    std::uint64_t raw_factor_cache_hits_ = 0;
    std::uint64_t raw_factor_cache_misses_ = 0;
    std::uint64_t raw_factor_gemm_calls_ = 0;
    std::uint64_t raw_output_product_calls_ = 0;
    std::uint64_t direct_source_factor_loads_ = 0;
    std::array<std::uint64_t, 6> family_route_counts_{};
    std::uint64_t unlabeled_family_route_count_ = 0;
    double raw_factor_build_seconds_ = 0.0;
    double raw_factor_matvec_seconds_ = 0.0;
    double raw_input_pack_seconds_ = 0.0;
    double dense_pair_matvec_seconds_ = 0.0;
    double raw_execution_matvec_seconds_ = 0.0;
    double raw_execution_pack_seconds_ = 0.0;
    double raw_batch_expand_seconds_ = 0.0;
    double raw_batch_right_prepare_seconds_ = 0.0;
    double raw_batch_fallback_prepare_seconds_ = 0.0;
    double raw_channel_first_stage_seconds_ = 0.0;
    double raw_wave_batch_seconds_ = 0.0;
    double raw_shared_left_output_seconds_ = 0.0;
    double raw_grouped_output_seconds_ = 0.0;
    double raw_binding_output_seconds_ = 0.0;
    double raw_fusion_finalize_seconds_ = 0.0;
    double raw_pointer_execution_matvec_seconds_ = 0.0;
    std::uint64_t raw_pointer_execution_matvec_calls_ = 0;
    double direct_complementary_action_seconds_ = 0.0;
    std::uint64_t direct_complementary_action_calls_ = 0;
    std::uint64_t direct_complementary_actions_ = 0;
    double factorized_metric_matvec_seconds_ = 0.0;
    std::size_t peak_raw_factor_cache_bytes_ = 0;

    std::string projection_key_;
    std::string projection_factor_route_key_;
    std::vector<ProjectionComponent> projection_components_;
    std::size_t projection_parent_dimension_ = 0;
    std::size_t projection_dimension_ = 0;
    std::uint64_t projection_topology_revision_ = 0;
    std::uint64_t projection_numeric_revision_ = 0;
    std::vector<Complex> projection_parent_input_;
    std::vector<Complex> projection_parent_output_;
    std::vector<Complex> projection_component_work_;
    std::vector<double> projection_real_input_;
    std::vector<double> projection_imag_input_;
    std::vector<double> projection_real_output_;
    std::vector<double> projection_imag_output_;
    std::vector<double> projection_parent_input_real_;
    std::vector<double> projection_parent_output_real_;
    bool canonical_projection_ready_ = false;
    std::string canonical_projection_metric_key_;
    std::uint64_t canonical_projection_metric_topology_revision_ = 0;
    std::uint64_t canonical_projection_metric_numeric_revision_ = 0;
    std::uint64_t canonical_projection_factor_topology_revision_ = 0;
    std::uint64_t canonical_projection_factor_numeric_revision_ = 0;
    std::size_t canonical_projection_transform_elements_ = 0;
    std::size_t canonical_projection_max_component_dimension_ = 0;
    double canonical_projection_whitening_residual_ = 0.0;
    double canonical_projection_last_build_seconds_ = 0.0;
    std::unordered_map<std::string, CanonicalProjectionCacheEntry>
        canonical_projection_cache_;
    std::string active_canonical_projection_cache_key_;
    std::size_t canonical_projection_cache_transform_elements_ = 0;
    std::size_t canonical_projection_cache_limit_elements_ = 0;
    std::uint64_t canonical_projection_cache_clock_ = 0;
    std::uint64_t canonical_projection_cache_evictions_ = 0;

    pyqed::dmrg::DavidsonWorkspace davidson_workspace_;
    pyqed::dmrg::RealDavidsonWorkspace real_davidson_workspace_;
    pyqed::dmrg::RealBlockDavidsonWorkspace
        real_block_davidson_workspace_;
    pyqed::dmrg::GeneralizedDavidsonWorkspace
        generalized_davidson_workspace_;
    pyqed::dmrg::RealGeneralizedDavidsonWorkspace
        real_generalized_davidson_workspace_;
    pyqed::dmrg::ComplexThinSVDWorkspace block_svd_workspace_;
    std::string metric_key_;
    std::vector<FactorizedMetricRoute> metric_routes_;
    std::size_t metric_dimension_ = 0;
    std::uint64_t metric_topology_revision_ = 0;
    std::uint64_t metric_numeric_revision_ = 0;
    std::uint64_t metric_value_fingerprint_ = 0;
    std::vector<double> metric_input_real_;
    std::vector<double> metric_input_imag_;
    std::vector<double> metric_temporary_real_;
    std::vector<double> metric_temporary_imag_;
    std::vector<double> metric_output_real_;
    std::vector<double> metric_output_imag_;
    std::string direction_;
    std::string lifecycle_phase_ = "idle";
    std::int64_t n_sites_ = 0;
    std::int64_t active_bond_ = -1;
    std::int64_t next_bond_ = -1;
    std::int64_t canonical_center_ = -1;
    bool active_half_sweep_orthonormal_ = false;
    std::uint64_t pending_kept_states_ = 0;
    double pending_truncation_seconds_ = 0.0;
    double current_half_sweep_energy_ =
        std::numeric_limits<double>::quiet_NaN();
    double last_half_sweep_energy_ =
        std::numeric_limits<double>::quiet_NaN();

    std::uint64_t boundary_topology_builds_ = 0;
    std::uint64_t boundary_numeric_refreshes_ = 0;
    std::uint64_t boundary_reallocations_ = 0;
    std::uint64_t boundary_update_topology_builds_ = 0;
    std::uint64_t boundary_update_calls_ = 0;
    std::uint64_t boundary_update_routes_ = 0;
    double boundary_update_seconds_ = 0.0;
    std::vector<double> boundary_update_temporary_;
    std::vector<double> boundary_update_product_;
    std::uint64_t local_topology_builds_ = 0;
    std::uint64_t local_numeric_refreshes_ = 0;
    std::uint64_t local_matvec_calls_ = 0;
    std::uint64_t local_davidson_calls_ = 0;
    std::uint64_t local_davidson_workspace_reuses_ = 0;
    std::uint64_t factor_route_topology_builds_ = 0;
    std::uint64_t factor_route_numeric_refreshes_ = 0;
    std::uint64_t factor_route_matvec_calls_ = 0;
    std::uint64_t real_factor_route_matvec_calls_ = 0;
    std::uint64_t factor_route_diagonal_calls_ = 0;
    std::uint64_t factor_route_davidson_calls_ = 0;
    std::uint64_t factor_route_scratch_growths_ = 0;
    std::uint64_t projection_topology_builds_ = 0;
    std::uint64_t projection_numeric_refreshes_ = 0;
    std::uint64_t projected_matvec_calls_ = 0;
    std::uint64_t projected_davidson_calls_ = 0;
    std::uint64_t generalized_davidson_calls_ = 0;
    std::uint64_t real_generalized_davidson_calls_ = 0;
    std::uint64_t metric_matvec_calls_ = 0;
    std::uint64_t real_metric_matvec_calls_ = 0;
    std::uint64_t canonical_projection_builds_ = 0;
    std::uint64_t canonical_projection_reuses_ = 0;
    std::uint64_t canonical_projection_davidson_calls_ = 0;
    double canonical_projection_build_seconds_ = 0.0;
    std::uint64_t block_svd_calls_ = 0;
    std::uint64_t block_svd_blocks_ = 0;
    double block_svd_seconds_ = 0.0;
    std::size_t peak_borrowed_local_operator_bytes_ = 0;
    std::size_t peak_factor_route_table_bytes_ = 0;
    std::size_t peak_borrowed_factor_pool_bytes_ = 0;
    std::size_t peak_factor_route_scratch_bytes_ = 0;
    std::size_t peak_projection_index_bytes_ = 0;
    std::size_t peak_borrowed_transform_bytes_ = 0;
    std::size_t peak_projection_scratch_bytes_ = 0;
    std::uint64_t half_sweeps_ = 0;
    std::uint64_t half_sweep_executor_calls_ = 0;
    std::uint64_t half_sweep_executor_bonds_ = 0;
    std::uint64_t half_sweep_python_bond_callbacks_ = 0;
    double half_sweep_executor_seconds_ = 0.0;
    std::uint64_t owned_half_sweep_calls_ = 0;
    std::uint64_t owned_half_sweep_bonds_ = 0;
    double owned_half_sweep_seconds_ = 0.0;
    std::uint64_t aborted_half_sweeps_ = 0;
    std::uint64_t bond_steps_ = 0;
    std::uint64_t bond_prepares_ = 0;
    std::uint64_t bond_solves_ = 0;
    std::uint64_t bond_splits_ = 0;
    std::uint64_t bond_advances_ = 0;
    std::uint64_t staged_bond_updates_ = 0;
    std::uint64_t committed_bond_updates_ = 0;
    std::uint64_t matvec_calls_ = 0;
    std::uint64_t davidson_iterations_ = 0;
    std::uint64_t kept_states_ = 0;
    double matvec_seconds_ = 0.0;
    double davidson_seconds_ = 0.0;
    double truncation_seconds_ = 0.0;
};

}  // namespace pyqed::su2
