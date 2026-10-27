#include "su2_dmrg_engine.hpp"

#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>

namespace pyqed::su2 {

// Direct normal/complementary split: integral coefficients enter once when
// a two-/four-fermion term crosses its first/third factor. Structural normal
// growth and complementary closure are shared, not summed once per integral.
// This is an adaptation to complete shell sites of the complementary-operator
// formulation of Chan & Head-Gordon, JCP 116, 4462 (2002), DOI 1.1449459.
void System::configure_spatial_shells(const std::vector<std::int64_t>& offsets) {
    if (offsets.size() < 2 || offsets.front() != 0
        || offsets.back() != static_cast<std::int64_t>(n_sites_))
        throw std::invalid_argument("NC shells must partition all active orbitals.");
    for (std::size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] <= offsets[i-1])
            throw std::invalid_argument("NC shells must be nonempty and ordered.");
    auto schedule = std::make_shared<SpatialNCSchedule>();
    schedule->shell_offsets = offsets;
    schedule->integral_revision = revision_;
    const std::size_t count = offsets.size()-1;
    schedule->channels.resize(count+1);
    schedule->sites.resize(count);
    using ChannelKey = std::array<std::int64_t,3>;
    using RouteKey = std::array<std::int64_t,3>;
    std::vector<std::map<ChannelKey,std::int64_t>> channel_ids(count+1);
    std::vector<std::map<std::vector<std::int64_t>,std::int64_t>> primitive_ids(count);
    std::vector<std::map<RouteKey,SpatialNCTransition>> routes(count);
    std::vector<std::int64_t> shell(n_sites_);
    for (std::size_t i = 0; i < count; ++i)
        for (auto p = offsets[i]; p < offsets[i+1]; ++p) shell[p] = i;
    for (std::size_t i = 0; i <= count; ++i) {
        schedule->channels[i] = {{0,0,0},{1,0,0}};
        channel_ids[i][{0,0,0}] = 0;
        channel_ids[i][{1,0,0}] = 1;
    }
    const auto orbital = [](std::int64_t mode) { return (std::abs(mode)-1)/2; };
    const auto channel = [&](std::size_t cut, std::int32_t kind,
                             const std::vector<std::int64_t>& word) {
        if (word.empty()) return std::int64_t(kind == 2 ? 1 : 0);
        if (word.size() > 2) throw std::logic_error("NC channel exceeds two open modes.");
        const ChannelKey key{kind,word[0],word.size() == 2 ? word[1] : 0};
        auto& ids = channel_ids[cut];
        if (ids.find(key) == ids.end()) {
            // Complete the entire orbital x spin component group, including
            // exact Pauli-null channels; no partner is selected by screening.
            const auto a = shell[orbital(word[0])];
            const auto b = word.size() == 2 ? shell[orbital(word[1])] : a;
            for (auto x = 2*offsets[a]; x < 2*offsets[a+1]; ++x) {
                const auto first = (word[0] > 0 ? 1 : -1)*(x+1);
                const auto begin = word.size() == 2 ? 2*offsets[b] : 0;
                const auto end = word.size() == 2 ? 2*offsets[b+1] : 1;
                for (auto y = begin; y < end; ++y) {
                    const auto second = word.size() == 2 ? (word[1] > 0 ? 1 : -1)*(y+1) : 0;
                    ChannelKey entry{kind,first,second};
                    if (ids.find(entry) == ids.end()) {
                        ids[entry] = schedule->channels[cut].size();
                        schedule->channels[cut].push_back({kind,first,second});
                    }
                }
            }
        }
        return ids.at(key);
    };
    std::size_t primitive_elements = 0;
    constexpr std::size_t max_primitive_elements = 128*1024*1024/sizeof(double);
    const auto primitive = [&](std::size_t site,const std::vector<std::int64_t>& word,bool parity) {
        auto key = word;
        key.push_back(parity ? 1 : 0);
        auto& ids = primitive_ids[site];
        auto found = ids.find(key);
        if (found != ids.end()) return found->second;
        const auto index = static_cast<std::int64_t>(schedule->sites[site].primitives.size());
        auto values = local_fermion_operator(offsets[site],offsets[site+1],word,parity);
        if (values.size() > max_primitive_elements-primitive_elements)
            throw std::length_error("NC local primitive arena exceeds 128 MiB.");
        primitive_elements += values.size();
        schedule->sites[site].physical_dimension = static_cast<std::int64_t>(std::sqrt(values.size()));
        schedule->sites[site].primitives.push_back({word,parity,std::move(values)});
        ids.emplace(std::move(key),index);
        return index;
    };
    const auto emit = [&](std::size_t site,std::int64_t source,std::int64_t target,
                          std::int64_t op,double coefficient,bool integral,std::uint64_t mask) {
        const RouteKey key{source,target,op};
        auto& table = routes[site];
        auto found = table.find(key);
        if (found == table.end()) table.emplace(key,SpatialNCTransition{source,target,op,coefficient,mask});
        else {
            if (integral) found->second.coefficient += coefficient;
            else if (found->second.coefficient != coefficient)
                throw std::logic_error("Inconsistent NC structural route.");
            found->second.family_mask |= mask;
        }
    };
    for (std::size_t site = 0; site < count; ++site) {
        auto op = primitive(site,{},false);
        emit(site,0,0,op,1.,false,0);
        emit(site,1,1,op,1.,false,0);
    }
    const auto add_term = [&](std::vector<std::int64_t> word,double coefficient,std::uint64_t mask) {
        if (std::abs(coefficient) <= cutoff_) return;
        const auto factor_count = word.size();
        const auto midpoint = factor_count-1;
        for (std::size_t i = 0; i < factor_count; ++i)
            for (std::size_t j = i+1; j < factor_count; ++j)
                if (shell[orbital(word[i])] > shell[orbital(word[j])]) coefficient = -coefficient;
        std::stable_sort(word.begin(),word.end(),[&](auto a,auto b) {
            return shell[orbital(a)] < shell[orbital(b)];
        });
        const auto first_site = static_cast<std::size_t>(shell[orbital(word.front())]);
        const auto last_site = static_cast<std::size_t>(shell[orbital(word.back())]);
        std::size_t consumed = 0;
        for (auto site = first_site; site <= last_site; ++site) {
            const auto old = consumed;
            std::vector<std::int64_t> local;
            while (consumed < factor_count && shell[orbital(word[consumed])] == static_cast<std::int64_t>(site))
                local.push_back(word[consumed++]);
            const bool cross = old < midpoint && consumed >= midpoint;
            const bool was_normal = old < midpoint;
            const bool is_normal = consumed < midpoint;
            const auto source = was_normal
                ? channel(site,2,{word.begin(),word.begin()+old})
                : channel(site,3,{word.begin()+old,word.end()});
            const auto target = is_normal
                ? channel(site+1,2,{word.begin(),word.begin()+consumed})
                : channel(site+1,3,{word.begin()+consumed,word.end()});
            const auto op = primitive(site,local,(factor_count-consumed)%2 != 0);
            emit(site,source,target,op,cross ? coefficient : 1.,cross,mask);
        }
    };
    // The R and Q families encode the one-body correction separately. Use
    // their common h arena here, and the existing P integral coefficients in
    // normal order, so their delta correction is included exactly once.
    for (std::size_t p = 0; p < n_sites_; ++p)
        for (std::size_t q = 0; q < n_sites_; ++q)
            for (std::int64_t spin = 0; spin < 2; ++spin)
                add_term({2*static_cast<std::int64_t>(p)+spin+1,
                         -(2*static_cast<std::int64_t>(q)+spin+1)},
                         h1_[p*n_sites_+q],std::uint64_t{1}<<static_cast<int>(Family::R));
    const auto& pairs = family(static_cast<std::int32_t>(Family::P));
    for (std::size_t entry = 0; entry < pairs.values.size(); ++entry) {
        const auto* idx = pairs.indices.data()+4*entry;
        for (std::int64_t spin = 0; spin < 2; ++spin)
            for (std::int64_t other = 0; other < 2; ++other)
                add_term({2*idx[0]+spin+1,2*idx[2]+other+1,
                          -(2*idx[3]+other+1),-(2*idx[1]+spin+1)},pairs.values[entry],
                          (std::uint64_t{1}<<static_cast<int>(Family::P))
                          | (std::uint64_t{1}<<static_cast<int>(Family::A))
                          | (std::uint64_t{1}<<static_cast<int>(Family::B)));
    }
    // Completing channel partners also requires completing their structural
    // recursions. Integral screening may zero an incoming coefficient; it
    // must not delete the corresponding operator from a complete tensor.
    for (std::size_t stop = count; stop > 0; --stop) {
        const auto site = stop-1;
        for (const auto& descriptor : schedule->channels[stop]) {
            if (descriptor.kind != 2) continue;
            std::vector<std::int64_t> word{descriptor.first};
            if (descriptor.second) word.push_back(descriptor.second);
            std::vector<std::int64_t> previous,local;
            for (auto mode : word)
                (shell[orbital(mode)] == static_cast<std::int64_t>(site) ? local : previous).push_back(mode);
            auto source = channel(site,2,previous);
            auto target = channel(stop,2,word);
            emit(site,source,target,primitive(site,local,word.size()%2 != 0),1.,false,
                 std::uint64_t{1}<<static_cast<int>(Family::S));
        }
    }
    for (std::size_t site = 0; site < count; ++site) {
        for (const auto& descriptor : schedule->channels[site]) {
            if (descriptor.kind != 3) continue;
            std::vector<std::int64_t> word{descriptor.first};
            if (descriptor.second) word.push_back(descriptor.second);
            std::vector<std::int64_t> following,local;
            for (auto mode : word)
                (shell[orbital(mode)] == static_cast<std::int64_t>(site) ? local : following).push_back(mode);
            auto source = channel(site,3,word);
            auto target = channel(site+1,3,following);
            emit(site,source,target,primitive(site,local,following.size()%2 != 0),1.,false,
                 std::uint64_t{1}<<static_cast<int>(Family::R));
        }
    }
    if (std::abs(ecore_) > cutoff_) emit(0,1,0,primitive(0,{},false),ecore_,true,0);
    for (std::size_t site = 0; site < count; ++site)
        for (const auto& [key,transition] : routes[site])
            if (std::abs(transition.coefficient) > cutoff_)
                schedule->sites[site].transitions.push_back(transition);
    spatial_schedule_ = std::move(schedule);
    ++spatial_revision_;
}

const SpatialNCSchedule& System::spatial_normal_complementary_schedule() const {
    if (!spatial_schedule_ || spatial_schedule_->integral_revision != revision_)
        throw std::logic_error("Spatial NC schedule is absent or stale after an integral update.");
    return *spatial_schedule_;
}

} // namespace pyqed::su2

namespace pyqed::su2 {

void System::reduce_spatial_normal_complementary(
    OrbitalGroup group, const std::vector<SpatialNCQuantumNumber>& ranks,
    const std::vector<std::vector<std::vector<SpatialNCChannelExpansion>>>& channels,
    const std::vector<std::vector<std::vector<SpatialNCPhysicalExpansion>>>& physical,
    const std::vector<SpatialNCCoupling>& couplings,
        const std::vector<std::int64_t>& finite_dimensions,
        const std::vector<std::int64_t>& finite_products
) {
    const auto& raw = spatial_normal_complementary_schedule();
    if (channels.size() != raw.channels.size() || physical.size() != raw.sites.size())
        throw std::invalid_argument("Spatial NC rank schedule has inconsistent site counts.");
    std::vector<std::int64_t> dimensions;
    dimensions.reserve(ranks.size());
    const FiniteOrbitalAlgebra finite(group == OrbitalGroup::Finite ? finite_dimensions : std::vector<std::int64_t>{1},
                                    group == OrbitalGroup::Finite ? finite_products : std::vector<std::int64_t>{1});
    for (const auto& q : ranks) {
        if (q.two_j < 0) throw std::invalid_argument("Negative NC spin rank.");
        dimensions.push_back((std::int64_t(q.two_j)+1)*(group == OrbitalGroup::Finite ? finite.dimension(q.orbital) : orbital_dimension(group,q.orbital)));
    }
    const auto check_rank = [&](std::int64_t id,std::int64_t row) {
        if (id < 0 || static_cast<std::size_t>(id) >= ranks.size()
            || row < 0 || row >= dimensions[id])
            throw std::invalid_argument("Spatial NC structural row is out of range.");
    };
    for (std::size_t cut = 0; cut < channels.size(); ++cut) {
        if (channels[cut].size() != raw.channels[cut].size())
            throw std::invalid_argument("Spatial NC channel expansion count differs.");
        for (const auto& entries : channels[cut]) for (const auto& entry : entries) {
            check_rank(entry.rank,entry.row);
            if (entry.copy < 0 || !std::isfinite(entry.value))
                throw std::invalid_argument("Invalid NC channel multiplicity or value.");
        }
    }
    std::map<std::array<std::int64_t,3>,const SpatialNCCoupling*> cg;
    for (const auto& entry : couplings) {
        check_rank(entry.left,0);check_rank(entry.local,0);check_rank(entry.right,0);
        const auto& a = ranks[entry.left];const auto& k = ranks[entry.local];const auto& b = ranks[entry.right];
        if (b.charge != a.charge+k.charge
            || b.two_j < std::abs(a.two_j-k.two_j) || b.two_j > a.two_j+k.two_j
            || ((a.two_j+k.two_j-b.two_j)&1)
            || !(group == OrbitalGroup::Finite ? finite.allows(b.orbital,k.orbital,a.orbital) : orbital_rank_allows(group,b.orbital,k.orbital,a.orbital))
            || entry.values.size() != static_cast<std::size_t>(dimensions[entry.left]*dimensions[entry.local]*dimensions[entry.right]))
            throw std::invalid_argument("Forbidden or malformed NC rank coupling.");
        if (!cg.emplace(std::array<std::int64_t,3>{entry.left,entry.local,entry.right},&entry).second)
            throw std::invalid_argument("Duplicate NC structural coupling.");
    }
    // Numerical tensors have multiplicity indices only. The operator CG
    // projection below acts on one bounded local primitive and the <=2-mode
    // channel representation; it never projects a molecular state space.
    std::vector<std::vector<SpatialNCReducedTransition>> output(raw.sites.size());
    for (std::size_t site = 0; site < raw.sites.size(); ++site) {
        if (physical[site].size() != raw.sites[site].primitives.size())
            throw std::invalid_argument("Spatial NC primitive expansion count differs.");
        for (const auto& primitive : physical[site]) for (const auto& entry : primitive) {
            check_rank(entry.bra,0);check_rank(entry.ket,0);check_rank(entry.rank,entry.row);
            if (entry.bra_copy < 0 || entry.ket_copy < 0
                || !std::isfinite(entry.value.real()) || !std::isfinite(entry.value.imag()))
                throw std::invalid_argument("Invalid NC physical multiplicity or value.");
        }
        using Key = std::array<std::int64_t,9>;
        struct Value { std::complex<double> number{}; std::uint64_t mask=0; };
        std::map<Key,Value> accumulated;
        for (const auto& route : raw.sites[site].transitions) {
            for (const auto& l : channels[site][route.source]) {
                for (const auto& r : channels[site+1][route.target]) {
                    for (const auto& p : physical[site][route.primitive]) {
                        const auto found = cg.find({l.rank,p.rank,r.rank});
                        if (found == cg.end()) continue;
                        const auto coefficient = found->second->values[
                            (l.row*dimensions[p.rank]+p.row)*dimensions[r.rank]+r.row];
                        if (coefficient == 0.) continue;
                        const Key key{l.rank,r.rank,p.bra,p.ket,p.rank,l.copy,r.copy,p.bra_copy,p.ket_copy};
                        auto& value = accumulated[key];
                        value.number += route.coefficient*l.value*r.value*p.value*coefficient
                            / static_cast<double>(dimensions[r.rank]);
                        value.mask |= route.family_mask;
                    }
                }
            }
        }
        output[site].reserve(accumulated.size());
        for (const auto& [key,value] : accumulated)
            if (std::abs(value.number) > cutoff_)
                output[site].push_back({key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],key[8],value.number,value.mask});
    }
    ++spatial_revision_;
    // Commit only after the complete new representation has been validated.
    spatial_schedule_->group = group;
    spatial_schedule_->ranks = ranks;
    spatial_schedule_->rank_channels = channels;
    for (std::size_t site = 0; site < output.size(); ++site)
        spatial_schedule_->sites[site].reduced_transitions = std::move(output[site]);
}

} // namespace pyqed::su2

namespace pyqed::su2 {
std::uint64_t System::spatial_normal_complementary_revision() const noexcept {
    return spatial_revision_;
}
}
