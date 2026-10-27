#pragma once

#include <cstdint>
#include <complex>
#include "orbital_rank.hpp"
#include <vector>

namespace pyqed::su2 {

// A mode is signed (2*orbital+spin+1): creation positive, destruction
// negative. Normal channels own <=2 consumed modes; complementary channels
// own <=2 outstanding modes. Complete shell component groups are retained.
struct SpatialNCChannel {
    std::int32_t kind = 0; // H, I, normal, complementary
    std::int64_t first = 0;
    std::int64_t second = 0;
};

struct SpatialNCTransition {
    std::int64_t source = 0;
    std::int64_t target = 0;
    std::int64_t primitive = 0;
    double coefficient = 0.;
    std::uint64_t family_mask = 0;
};

struct SpatialNCPrimitive {
    std::vector<std::int64_t> word;
    bool right_parity = false;
    std::vector<double> values;
};

struct SpatialNCQuantumNumber {
    std::int64_t charge = 0;
    std::int32_t two_j = 0;
    OrbitalRank orbital;
};

struct SpatialNCChannelExpansion {
    std::int64_t rank = 0, copy = 0, row = 0;
    double value = 0.;
};

struct SpatialNCPhysicalExpansion {
    std::int64_t bra = 0, ket = 0, rank = 0;
    std::int64_t bra_copy = 0, ket_copy = 0, row = 0;
    std::complex<double> value{};
};

struct SpatialNCCoupling {
    std::int64_t left = 0, local = 0, right = 0;
    std::vector<double> values;
};

struct SpatialNCReducedTransition {
    std::int64_t left = 0, right = 0, bra = 0, ket = 0, rank = 0;
    std::int64_t left_copy = 0, right_copy = 0, bra_copy = 0, ket_copy = 0;
    std::complex<double> value{};
    std::uint64_t family_mask = 0;
};

struct SpatialNCSitePlan {
    std::int64_t physical_dimension = 0;
    std::vector<SpatialNCPrimitive> primitives;
    std::vector<SpatialNCTransition> transitions;
    std::vector<SpatialNCReducedTransition> reduced_transitions;
};

struct SpatialNCSchedule {
    std::vector<std::int64_t> shell_offsets;
    std::vector<std::vector<SpatialNCChannel>> channels;
    std::vector<SpatialNCSitePlan> sites;
    std::uint64_t integral_revision = 0;
    OrbitalGroup group = OrbitalGroup::Scalar;
    std::vector<SpatialNCQuantumNumber> ranks;
    std::vector<std::vector<std::vector<SpatialNCChannelExpansion>>> rank_channels;
};

} // namespace pyqed::su2
