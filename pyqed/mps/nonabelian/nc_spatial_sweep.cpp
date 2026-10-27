#include "su2_dmrg_engine.hpp"
#include <stdexcept>

namespace pyqed::su2 {
namespace {
void validate_nc_pool(std::size_t size,const std::vector<std::int64_t>& offsets,
                      const std::vector<std::int64_t>& shapes) {
    if (offsets.empty() || offsets.front() != 0 || offsets.back() < 0
        || static_cast<std::size_t>(offsets.back()) != size
        || shapes.size() != 3*(offsets.size()-1))
        throw std::invalid_argument("Spatial NC reduced arena has invalid offsets.");
    for (std::size_t i = 0; i+1 < offsets.size(); ++i) {
        std::int64_t count = 1;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto d = shapes[3*i+axis];
            if (d <= 0 || count > std::numeric_limits<std::int64_t>::max()/d)
                throw std::invalid_argument("Spatial NC reduced arena has invalid dimensions.");
            count *= d;
        }
        if (offsets[i+1]-offsets[i] != count)
            throw std::invalid_argument("Spatial NC reduced arena shape does not match offsets.");
    }
}
void validate_nc_block(std::int64_t index,const std::vector<std::int64_t>& offsets) {
    if (index < 0 || static_cast<std::size_t>(index)+1 >= offsets.size())
        throw std::out_of_range("Spatial NC route refers to an absent reduced block.");
}
ComplexPackedArena nc_arena(std::vector<Complex> values,
    const std::vector<std::int64_t>& offsets,const std::vector<std::int64_t>& shapes,
    const std::vector<std::int64_t>& labels) {
    validate_nc_pool(values.size(),offsets,shapes);
    if (labels.size() != shapes.size())
        throw std::invalid_argument("Spatial NC reduced labels do not match block count.");
    ComplexPackedArena arena;
    arena.offsets = offsets;arena.shapes = shapes;arena.labels = labels;
    arena.shape_offsets.resize(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) arena.shape_offsets[i] = 3*i;
    arena.owned_values = std::make_shared<std::vector<Complex>>(std::move(values));
    return arena;
}
}

void MovingEnvironment::install_spatial_nc_boundary(
    const std::string& side,std::int64_t bond,const std::vector<Complex>& values,
    const std::vector<std::int64_t>& offsets,const std::vector<std::int64_t>& shapes,
    const std::vector<std::int64_t>& labels
) {
    if ((side != "left" && side != "right") || bond < 0)
        throw std::invalid_argument("Invalid spatial NC boundary location.");
    complex_boundaries_[boundary_key(side,bond)] = nc_arena(values,offsets,shapes,labels);
}

void MovingEnvironment::advance_spatial_nc_boundary(
    const std::string& side,std::int64_t parent_bond,std::int64_t child_bond,
    std::int64_t site,const std::vector<std::vector<std::int64_t>>& routes,
    const std::vector<double>& weights,
    const std::vector<Complex>& bra,const std::vector<std::int64_t>& bra_offsets,
    const std::vector<std::int64_t>& bra_shapes,
    const std::vector<Complex>& ket,const std::vector<std::int64_t>& ket_offsets,
    const std::vector<std::int64_t>& ket_shapes,
    const std::vector<std::int64_t>& output_offsets,
    const std::vector<std::int64_t>& output_shapes,
    const std::vector<std::int64_t>& output_labels
) {
    const bool left = side == "left";
    if ((!left && side != "right") || parent_bond < 0 || child_bond < 0
        || child_bond != parent_bond+(left ? 1 : -1)
        || site != (left ? parent_bond : child_bond))
        throw std::invalid_argument("Spatial NC boundary movement is inconsistent.");
    const auto& schedule = system_->spatial_normal_complementary_schedule();
    if (site < 0 || static_cast<std::size_t>(site) >= schedule.sites.size()
        || routes.size() != weights.size())
        throw std::invalid_argument("Spatial NC boundary site or route count is invalid.");
    const auto& entries = schedule.sites[site].reduced_transitions;
    const auto& parent = complex_boundaries_.at(boundary_key(side,parent_bond));
    validate_nc_pool(bra.size(),bra_offsets,bra_shapes);
    validate_nc_pool(ket.size(),ket_offsets,ket_shapes);
    if (output_offsets.empty() || output_offsets.back() < 0)
        throw std::invalid_argument("Spatial NC output arena has no offsets.");
    std::vector<Complex> output(static_cast<std::size_t>(output_offsets.back()));
    validate_nc_pool(output.size(),output_offsets,output_shapes);
    std::vector<std::vector<std::size_t>> grouped(output_offsets.size()-1);
    for (std::size_t i = 0; i < routes.size(); ++i) {
        const auto& r = routes[i];
        if (r.size() != 5 || r[3] < 0 || static_cast<std::size_t>(r[3]) >= entries.size())
            throw std::invalid_argument("Malformed spatial NC boundary route.");
        validate_nc_block(r[0],parent.offsets);validate_nc_block(r[1],bra_offsets);
        validate_nc_block(r[2],ket_offsets);validate_nc_block(r[4],output_offsets);
        const auto& e = entries[r[3]];
        const auto* a = bra_shapes.data()+3*r[1];const auto* b = ket_shapes.data()+3*r[2];
        const auto* v = parent.shapes.data()+3*r[0];const auto* out = output_shapes.data()+3*r[4];
        if (e.bra_copy >= a[1] || e.ket_copy >= b[1]
            || (left ? (v[0] != a[0] || v[2] != b[0] || out[0] != a[2] || out[2] != b[2]
                        || e.left_copy >= v[1] || e.right_copy >= out[1])
                     : (v[0] != a[2] || v[2] != b[2] || out[0] != a[0] || out[2] != b[0]
                        || e.right_copy >= v[1] || e.left_copy >= out[1])))
            throw std::invalid_argument("Incompatible spatial NC boundary multiplicities.");
        grouped[r[4]].push_back(i);
    }
    const auto blocks = static_cast<std::int64_t>(grouped.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(n_threads_) if(n_threads_ > 1 && blocks > 1)
#endif
    for (std::int64_t out_id = 0; out_id < blocks; ++out_id) {
        std::vector<Complex> tmp;
        for (auto route_id : grouped[out_id]) {
            const auto& r = routes[route_id];const auto& e = entries[r[3]];
            const auto* a_shape = bra_shapes.data()+3*r[1];const auto* b_shape = ket_shapes.data()+3*r[2];
            const auto A=a_shape[0],P=a_shape[1],C=a_shape[2];
            const auto B=b_shape[0],Q=b_shape[1],D=b_shape[2];
            const auto* a=bra.data()+bra_offsets[r[1]];const auto* b=ket.data()+ket_offsets[r[2]];
            const auto* v=parent.owned_values->data()+parent.offsets[r[0]];
            auto* out=output.data()+output_offsets[r[4]];
            const auto V=parent.shapes[3*r[0]+1],Y=output_shapes[3*r[4]+1];
            const auto factor=weights[route_id]*e.value;
            tmp.assign(static_cast<std::size_t>(A*D),Complex{});
            if (left) {
                for (std::int64_t a_id=0;a_id<A;++a_id) for (std::int64_t d=0;d<D;++d)
                    for (std::int64_t b_id=0;b_id<B;++b_id)
                        tmp[a_id*D+d] += v[(a_id*V+e.left_copy)*B+b_id]*b[(b_id*Q+e.ket_copy)*D+d];
                for (std::int64_t c=0;c<C;++c) for (std::int64_t d=0;d<D;++d) {
                    Complex value{};
                    for (std::int64_t a_id=0;a_id<A;++a_id)
                        value += std::conj(a[(a_id*P+e.bra_copy)*C+c])*tmp[a_id*D+d];
                    out[(c*Y+e.right_copy)*D+d] += factor*value;
                }
            } else {
                for (std::int64_t a_id=0;a_id<A;++a_id) for (std::int64_t d=0;d<D;++d)
                    for (std::int64_t c=0;c<C;++c)
                        tmp[a_id*D+d] += std::conj(a[(a_id*P+e.bra_copy)*C+c])*v[(c*V+e.right_copy)*D+d];
                for (std::int64_t a_id=0;a_id<A;++a_id) for (std::int64_t b_id=0;b_id<B;++b_id) {
                    Complex value{};
                    for (std::int64_t d=0;d<D;++d) value += tmp[a_id*D+d]*b[(b_id*Q+e.ket_copy)*D+d];
                    out[(a_id*Y+e.left_copy)*B+b_id] += factor*value;
                }
            }
        }
    }
    complex_boundaries_[boundary_key(side,child_bond)] = nc_arena(std::move(output),output_offsets,output_shapes,output_labels);
}

void MovingEnvironment::install_spatial_nc_local_plan(
    const std::string& key,std::int64_t site,
    const std::vector<std::vector<std::int64_t>>& routes,const std::vector<double>& weights,
    const std::vector<std::int64_t>& offsets,const std::vector<std::int64_t>& shapes
) {
    if (offsets.empty() || offsets.back() < 0 || routes.size() != weights.size())
        throw std::invalid_argument("Invalid spatial NC local layout.");
    validate_nc_pool(static_cast<std::size_t>(offsets.back()),offsets,shapes);
    const auto& schedule=system_->spatial_normal_complementary_schedule();
    if (site < 0 || static_cast<std::size_t>(site) >= schedule.sites.size())
        throw std::out_of_range("Spatial NC local site is outside the schedule.");
    SpatialNCLocalPlan plan;
    plan.site=site;plan.dimension=offsets.back();plan.offsets=offsets;plan.shapes=shapes;
    plan.revision=system_->spatial_normal_complementary_revision();
    plan.left=complex_boundaries_.at(boundary_key("left",site));
    plan.right=complex_boundaries_.at(boundary_key("right",site+1));
    plan.output_routes.resize(offsets.size()-1);plan.weights=weights;
    const auto& entries=schedule.sites[site].reduced_transitions;
    for (std::size_t i=0;i<routes.size();++i) {
        const auto& r=routes[i];
        if (r.size()!=5 || r[4]<0 || static_cast<std::size_t>(r[4])>=entries.size())
            throw std::invalid_argument("Malformed spatial NC local route.");
        validate_nc_block(r[0],offsets);validate_nc_block(r[1],offsets);
        validate_nc_block(r[2],plan.left.offsets);validate_nc_block(r[3],plan.right.offsets);
        const auto& e=entries[r[4]];
        const auto* a=shapes.data()+3*r[0];const auto* b=shapes.data()+3*r[1];
        const auto* l=plan.left.shapes.data()+3*r[2];const auto* rr=plan.right.shapes.data()+3*r[3];
        if (a[0]!=l[0] || b[0]!=l[2] || a[2]!=rr[0] || b[2]!=rr[2]
            || e.bra_copy>=a[1] || e.ket_copy>=b[1] || e.left_copy>=l[1] || e.right_copy>=rr[1])
            throw std::invalid_argument("Incompatible spatial NC local multiplicities.");
        plan.routes.push_back({r[0],r[1],r[2],r[3],r[4]});
        plan.output_routes[r[0]].push_back(i);
    }
    spatial_nc_local_plans_[key]=std::move(plan);
}

void MovingEnvironment::release_spatial_nc_local_plan(const std::string& key) noexcept {
    spatial_nc_local_plans_.erase(key);
}

void MovingEnvironment::spatial_nc_local_matvec(
    const std::string& key,const Complex* input,Complex* output,std::size_t dimension
) const {
    const auto& plan=spatial_nc_local_plans_.at(key);
    if (dimension!=plan.dimension || plan.revision!=system_->spatial_normal_complementary_revision())
        throw std::invalid_argument("Spatial NC local action is stale or has the wrong dimension.");
    const auto& entries=system_->spatial_normal_complementary_schedule().sites[plan.site].reduced_transitions;
    std::fill(output,output+dimension,Complex{});
    const auto blocks=static_cast<std::int64_t>(plan.output_routes.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(n_threads_) if(n_threads_ > 1 && blocks > 1)
#endif
    for (std::int64_t out_id=0;out_id<blocks;++out_id) {
        std::vector<Complex> tmp;
        for (const auto route_id:plan.output_routes[out_id]) {
            const auto& r=plan.routes[route_id];const auto& e=entries[r[4]];
            const auto* a_shape=plan.shapes.data()+3*r[0];const auto* b_shape=plan.shapes.data()+3*r[1];
            const auto A=a_shape[0],P=a_shape[1],C=a_shape[2];
            const auto B=b_shape[0],Q=b_shape[1],D=b_shape[2];
            const auto X=plan.left.shapes[3*r[2]+1],Y=plan.right.shapes[3*r[3]+1];
            const auto* l=plan.left.owned_values->data()+plan.left.offsets[r[2]];
            const auto* rr=plan.right.owned_values->data()+plan.right.offsets[r[3]];
            const auto* v=input+plan.offsets[r[1]];auto* out=output+plan.offsets[r[0]];
            const auto factor=plan.weights[route_id]*e.value;
            tmp.assign(static_cast<std::size_t>(B*C),Complex{});
            for (std::int64_t b=0;b<B;++b) for (std::int64_t c=0;c<C;++c)
                for (std::int64_t d=0;d<D;++d)
                    tmp[b*C+c] += v[(b*Q+e.ket_copy)*D+d]*rr[(c*Y+e.right_copy)*D+d];
            for (std::int64_t a=0;a<A;++a) for (std::int64_t c=0;c<C;++c) {
                Complex value{};
                for (std::int64_t b=0;b<B;++b) value += l[(a*X+e.left_copy)*B+b]*tmp[b*C+c];
                out[(a*P+e.bra_copy)*C+c] += factor*value;
            }
        }
    }
}

void MovingEnvironment::spatial_nc_local_diagonal(
    const std::string& key,Complex* output,std::size_t dimension
) const {
    const auto& plan=spatial_nc_local_plans_.at(key);
    if (dimension!=plan.dimension || plan.revision!=system_->spatial_normal_complementary_revision())
        throw std::invalid_argument("Spatial NC local diagonal is stale or has the wrong dimension.");
    const auto& entries=system_->spatial_normal_complementary_schedule().sites[plan.site].reduced_transitions;
    std::fill(output,output+dimension,Complex{});
    for (std::size_t i=0;i<plan.routes.size();++i) {
        const auto& r=plan.routes[i];const auto& e=entries[r[4]];
        if (r[0]!=r[1] || e.bra_copy!=e.ket_copy) continue;
        const auto A=plan.shapes[3*r[0]],P=plan.shapes[3*r[0]+1],C=plan.shapes[3*r[0]+2];
        const auto X=plan.left.shapes[3*r[2]+1],Y=plan.right.shapes[3*r[3]+1];
        const auto* l=plan.left.owned_values->data()+plan.left.offsets[r[2]];
        const auto* rr=plan.right.owned_values->data()+plan.right.offsets[r[3]];
        const auto factor=plan.weights[i]*e.value;
        auto* out=output+plan.offsets[r[0]];
        for (std::int64_t a=0;a<A;++a) for (std::int64_t c=0;c<C;++c)
            out[(a*P+e.bra_copy)*C+c] += factor*l[(a*X+e.left_copy)*A+a]*rr[(c*Y+e.right_copy)*C+c];
    }
}

pyqed::linalg::DavidsonResult MovingEnvironment::spatial_nc_local_solve(
    const std::string& key,const std::vector<std::int64_t>& indices,
    const std::vector<double>& scales,const std::vector<Complex>& transform,
    const std::vector<Complex>& guess,double tolerance) {
    const auto dimension=spatial_nc_local_plans_.at(key).dimension;
    const auto kept=indices.size(),rank=guess.size();
    if (!rank || scales.size()!=kept || (!transform.empty() && transform.size()!=kept*rank)
        || (transform.empty() && kept!=rank) || !std::isfinite(tolerance) || tolerance<=0)
        throw std::invalid_argument("Invalid NC local variational coordinates.");
    for (std::size_t i=0;i<kept;++i)
        if (indices[i]<0 || static_cast<std::size_t>(indices[i])>=dimension || !std::isfinite(scales[i]))
            throw std::invalid_argument("Invalid NC metric coordinate.");
    const auto lift=[&](const std::vector<Complex>& x) {
        std::vector<Complex> out(dimension);
        for (std::size_t i=0;i<kept;++i) {
            Complex value{};
            if (transform.empty()) value=x[i];
            else for (std::size_t j=0;j<rank;++j) value+=transform[i*rank+j]*x[j];
            out[indices[i]]=scales[i]*value;
        }
        return out;
    };
    const auto pull=[&](const std::vector<Complex>& x) {
        std::vector<Complex> out(rank);
        for (std::size_t i=0;i<kept;++i) {
            const auto value=scales[i]*x[indices[i]];
            if (transform.empty()) out[i]=value;
            else for (std::size_t j=0;j<rank;++j) out[j]+=std::conj(transform[i*rank+j])*value;
        }
        return out;
    };
    const auto apply=[&](const std::vector<Complex>& x) {
        auto full=lift(x);std::vector<Complex> output(dimension);
        spatial_nc_local_matvec(key,full.data(),output.data(),dimension);
        return pull(output);
    };
    std::vector<Complex> full_diagonal(dimension),diagonal(rank);
    spatial_nc_local_diagonal(key,full_diagonal.data(),dimension);
    for (std::size_t i=0;i<kept;++i) {
        const auto value=scales[i]*scales[i]*full_diagonal[indices[i]];
        if (transform.empty()) diagonal[i]=value;
        else for (std::size_t j=0;j<rank;++j) diagonal[j]+=std::norm(transform[i*rank+j])*value;
    }
    auto initial=guess;
    if (pyqed::linalg::norm2(initial)<1e-14) initial.assign(rank,1.);
    pyqed::linalg::DavidsonResult result;
    if (rank<=2) {
        std::vector<Complex> matrix(rank*rank),unit(rank);
        for (std::size_t j=0;j<rank;++j) {
            unit[j]=1.;const auto column=apply(unit);unit[j]=0.;
            for (std::size_t i=0;i<rank;++i) matrix[i*rank+j]=column[i];
        }
        auto pair=pyqed::linalg::lowest_projected_eigenpair(matrix,rank);
        result.energy=pair.value;result.vector=std::move(pair.vector);
        auto residual=apply(result.vector);
        for (std::size_t i=0;i<rank;++i) residual[i]-=result.energy*result.vector[i];
        result.residual_norm=pyqed::linalg::norm2(residual);
        result.accepted=result.converged=result.residual_norm<=tolerance;
        result.iterations=1;result.matvec_calls=rank+1;
    } else {
        result=pyqed::linalg::generalized_davidson(diagonal,std::vector<Complex>(rank,1.),
            std::move(initial),std::max(1e-14,tolerance*tolerance),tolerance,1e-14,
            std::max(300,static_cast<int>(10*rank)),std::min<int>(rank,48),false,
            generalized_davidson_workspace_,apply,[](const std::vector<Complex>& x){return x;});
    }
    if (!result.converged) throw std::runtime_error("Spatial NC local Davidson did not converge: residual="
        +std::to_string(result.residual_norm)+", iterations="+std::to_string(result.iterations));
    result.vector=lift(result.vector);
    return result;
}

} // namespace pyqed::su2
