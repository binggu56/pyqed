#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pyqed::su2 {

// Whole orbital irreps. Rows belong to the structural CG maps, never to
// truncation labels. Integer L is independent of the doubled electron spin.
enum class OrbitalGroup : std::int32_t { Scalar = 0, Coov = 1, Dooh = 2, SO3 = 3, Finite = 4 };

struct OrbitalRank {
    std::int32_t order = 0;
    std::int32_t reflection = 1;
    std::int32_t parity = 1;

    bool operator==(const OrbitalRank& other) const noexcept {
        return order == other.order && reflection == other.reflection
            && parity == other.parity;
    }
};

// Finite real irreps use order as an index, never as angular momentum.
// The table is the exact multiplicity-free representation ring supplied by
// PointGroupSymmetry; all rows of each irrep remain structural indices.
struct FiniteOrbitalAlgebra {
    std::vector<std::int64_t> dimensions, products;
    FiniteOrbitalAlgebra(const std::vector<std::int64_t>& dims,
                         const std::vector<std::int64_t>& fusion)
        : dimensions(dims), products(fusion) {
        const auto n = dimensions.size();
        if (!n || products.size() != n*n*n)
            throw std::invalid_argument("Invalid finite orbital fusion table.");
        for (auto d : dimensions) if (d < 1)
            throw std::invalid_argument("Invalid finite orbital dimension.");
        for (std::size_t a=0; a<n; ++a) for (std::size_t b=0; b<n; ++b) {
            std::int64_t dim=0;
            for (std::size_t c=0; c<n; ++c) {
                auto count=products[(a*n+b)*n+c];
                if (count < 0 || count > 1 || count != products[(b*n+a)*n+c])
                    throw std::invalid_argument("Finite NC requires multiplicity-free commutative fusion.");
                dim += count*dimensions[c];
            }
            if (dim != dimensions[a]*dimensions[b])
                throw std::invalid_argument("Finite fusion does not preserve partner dimensions.");
        }
    }
    std::size_t index(const OrbitalRank& rank) const {
        if (rank.order < 0 || static_cast<std::size_t>(rank.order) >= dimensions.size()
            || rank.reflection != 1 || rank.parity != 1)
            throw std::invalid_argument("Invalid finite orbital irrep index.");
        return static_cast<std::size_t>(rank.order);
    }
    std::int64_t dimension(const OrbitalRank& rank) const { return dimensions[index(rank)]; }
    bool allows(const OrbitalRank& bra,const OrbitalRank& op,const OrbitalRank& ket) const {
        auto n=dimensions.size();
        return products[(index(ket)*n+index(op))*n+index(bra)] != 0;
    }
};

inline void validate_orbital_rank(OrbitalGroup group, const OrbitalRank& rank) {
    if (group < OrbitalGroup::Scalar || group > OrbitalGroup::SO3
        || rank.order < 0 || (rank.reflection != 1 && rank.reflection != -1)
        || (rank.parity != 1 && rank.parity != -1)
        || (group != OrbitalGroup::Dooh && rank.parity != 1)
        || ((group == OrbitalGroup::SO3 || rank.order != 0) && rank.reflection != 1)
        || (group == OrbitalGroup::Scalar && !(rank == OrbitalRank{}))) {
        throw std::invalid_argument("Invalid complete orbital irrep.");
    }
}

inline std::int64_t orbital_dimension(OrbitalGroup group, const OrbitalRank& rank) {
    validate_orbital_rank(group, rank);
    return group == OrbitalGroup::SO3 ? 2 * std::int64_t(rank.order) + 1
                                     : (rank.order == 0 ? 1 : 2);
}

// Exact integer angular momentum / cylindrical fusion; no angular cutoff.
// Condon--Shortley angular coupling: NIST DLMF section 34.1.
inline std::vector<OrbitalRank> fuse_orbital_ranks(
    OrbitalGroup group, const OrbitalRank& a, const OrbitalRank& b
) {
    validate_orbital_rank(group, a);
    validate_orbital_rank(group, b);
    if (a.order > std::numeric_limits<std::int32_t>::max() - b.order)
        throw std::overflow_error("Orbital fusion order exceeds int32.");
    const auto sum = a.order + b.order;
    const auto difference = a.order > b.order ? a.order - b.order : b.order - a.order;
    if (group == OrbitalGroup::Scalar) return {OrbitalRank{}};
    std::vector<OrbitalRank> out;
    if (group == OrbitalGroup::SO3) {
        for (std::int64_t c = difference; c <= sum; ++c)
            out.push_back({static_cast<std::int32_t>(c), 1, 1});
    } else if (a.order == 0 || b.order == 0) {
        out.push_back({sum, sum == 0 ? a.reflection * b.reflection : 1,
                       a.parity * b.parity});
    } else {
        if (difference == 0) {
            out.push_back({0, 1, a.parity * b.parity});
            out.push_back({0, -1, a.parity * b.parity});
        } else out.push_back({difference, 1, a.parity * b.parity});
        out.push_back({sum, 1, a.parity * b.parity});
    }
    return out;
}

inline bool orbital_rank_allows(
    OrbitalGroup group, const OrbitalRank& bra, const OrbitalRank& op,
    const OrbitalRank& ket
) {
    validate_orbital_rank(group, bra);
    for (const auto& rank : fuse_orbital_ranks(group, ket, op))
        if (rank == bra) return true;
    return false;
}

} // namespace pyqed::su2
