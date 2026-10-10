// Included inside the integral module after RI input validation.
struct RIBasisArrays {
    ArrayRef shells, origins, exps, weights, nprim;
    explicit RIBasisArrays(PyObject* tuple)
        : shells(PyTuple_GET_ITEM(tuple, 0), NPY_INT64, NPY_ARRAY_IN_ARRAY),
          origins(PyTuple_GET_ITEM(tuple, 1), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
          exps(PyTuple_GET_ITEM(tuple, 2), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
          weights(PyTuple_GET_ITEM(tuple, 3), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
          nprim(PyTuple_GET_ITEM(tuple, 4), NPY_INT64, NPY_ARRAY_IN_ARRAY) {}
    bool valid() const { return shells && origins && exps && weights && nprim; }
};

template<class Emit>
void metric_derivative_shells(const std::int64_t* shells,const double* origins,
    const double* exps,const double* weights,const std::int64_t* nprim,
    const std::int64_t* owners,npy_intp naux,npy_intp stride,int order,
    int target,const double* adjoint,npy_intp nobs,Emit emit) {
    std::vector<ShellBlock> blocks;
    if (!try_build_shell_blocks(shells,origins,exps,nprim,naux,stride,blocks))
        throw std::invalid_argument("Metric derivatives require complete shells.");
    struct Term { std::size_t index; double coefficient; int power; };
    struct Plan {
        DerivativeRecurrencePlan recurrence;
        std::vector<Term> terms;
        std::vector<std::size_t> offsets;
    };
    std::map<std::pair<int,int>,Plan> plans;
    std::vector<double> table,values;
    const double zero[3]={};
    const int components=order==1 ? 3 : 6;
    for (std::size_t ia=0;ia<blocks.size();++ia) for (std::size_t ib=0;ib<ia;++ib) {
        const auto& a=blocks[ia]; const auto& b=blocks[ib];
        if (owners[a.start]==owners[b.start]) continue;
        bool needed=target>=0 && (owners[a.start]==target || owners[b.start]==target);
        if (target<0) for (auto i=a.start;i<a.stop && !needed;++i)
            for (auto j=b.start;j<b.stop && !needed;++j) for (npy_intp o=0;o<nobs;++o)
                needed |= adjoint[(o*naux+i)*naux+j]+adjoint[(o*naux+j)*naux+i]!=0.;
        if (!needed) continue;
        const int max_a=a.l+order,max_m=max_a+b.l;
        auto found=plans.find({a.l,b.l});
        if (found==plans.end()) {
            Plan plan;
            plan.offsets.push_back(0);
            std::vector<std::vector<HrrExpansionTerm>> targets(1);
            for (auto i=a.start;i<a.stop;++i) for (auto j=b.start;j<b.stop;++j) {
                int p[3]={int(shells[3*i]),int(shells[3*i+1]),int(shells[3*i+2])};
                auto add=[&](double coefficient,int power) {
                    if (coefficient==0. || p[0]<0 || p[1]<0 || p[2]<0) return;
                    const auto index=os_vrr_idx(p[0],p[1],p[2],shells[3*j],shells[3*j+1],shells[3*j+2],
                        0,max_a+1,b.l+1,max_m+1);
                    plan.terms.push_back({index,coefficient,power});
                    targets[0].push_back({index,1.});
                };
                for (int x=0;x<3;++x) {
                    if (order==1) {
                        const int px=p[x];
                        ++p[x];add(2.,1);p[x]-=2;add(-px,0);++p[x];
                        plan.offsets.push_back(plan.terms.size());
                    } else for (int y=x;y<3;++y) {
                        for (int sx : {1,-1}) for (int sy : {1,-1}) {
                            const int px=p[x]; const double cx=sx==1 ? 2. : -px;
                            p[x]+=sx;
                            const int py=p[y]; const double cy=sy==1 ? 2. : -py;
                            p[y]+=sy;
                            if (px+sx>=0 && py+sy>=0) add(cx*cy,int(sx==1)+int(sy==1));
                            p[y]-=sy;p[x]-=sx;
                        }
                        plan.offsets.push_back(plan.terms.size());
                    }
                }
            }
            plan.recurrence.build(max_a,b.l,max_m,targets,{},true);
            std::map<std::size_t,std::size_t> slots;
            for (std::size_t k=0;k<plan.recurrence.targets.size();++k)
                slots[plan.recurrence.targets[k]]=plan.recurrence.target_slots[k];
            for (auto& term : plan.terms) term.index=slots.at(term.index);
            found=plans.emplace(std::make_pair(a.l,b.l),std::move(plan)).first;
        }
        const auto& plan=found->second;
        table.resize(plan.recurrence.workspace_size);
        values.assign((a.stop-a.start)*(b.stop-b.start)*components,0.);
        double delta[3],distance2=0.;
        for (int k=0;k<3;++k) {delta[k]=origins[3*a.start+k]-origins[3*b.start+k];distance2+=delta[k]*delta[k];}
        for (int ip=0;ip<nprim[a.start];++ip) for (int jp=0;jp<nprim[b.start];++jp) {
            const double alpha=exps[a.start*stride+ip],beta=exps[b.start*stride+jp],z=alpha+beta;
            plan.recurrence.evaluate(table.data(),max_m,alpha,beta,z,alpha*beta/z*distance2,
                ERI_PREFAC/(alpha*beta*std::sqrt(z)),zero,zero,delta);
            const double powers[3]={1.,alpha,alpha*alpha};
            std::size_t recipe=0;
            for (auto i=a.start;i<a.stop;++i) for (auto j=b.start;j<b.stop;++j)
                for (int c=0;c<components;++c,++recipe) {
                    double v=0.;
                    for (auto k=plan.offsets[recipe];k<plan.offsets[recipe+1];++k) {
                        const auto& term=plan.terms[k];
                        v+=term.coefficient*powers[term.power]*table[term.index];
                    }
                    values[recipe]+=weights[i*stride+ip]*weights[j*stride+jp]*v;
                }
        }
        std::size_t recipe=0;
        for (auto i=a.start;i<a.stop;++i) for (auto j=b.start;j<b.stop;++j)
            for (int x=0;x<3;++x) {
                if (order==1) emit(i,j,x,x,values[recipe++]);
                else for (int y=x;y<3;++y) emit(i,j,x,y,values[recipe++]);
            }
    }
}

PyObject* ri_derivative_evaluate(PyObject* args, bool columns) {
    DerivativeProfile profile;
    const bool profiling=derivative_profile_enabled.load();
    const auto total_start=profiling ? DerivativeClock::now() : DerivativeClock::time_point{};
    PyObject *primary, *auxiliary, *owners, *aux_owners, *three=nullptr, *metric=nullptr;
    int natom, order = 1, direction=-1, count=1;
    PyObject *occupied=Py_None,*rho=Py_None,*primary_map=Py_None,*auxiliary_map=Py_None;
    if (columns) {
        if (!PyArg_ParseTuple(args,"OOOOii|iOOOO",&primary,&auxiliary,&owners,&aux_owners,&natom,&direction,&count,&occupied,&rho,&primary_map,&auxiliary_map)) return nullptr;
        if (direction<0 || direction>=3*natom || (count!=1 && count!=3) || (count==3 && direction%3!=0)) {
            PyErr_SetString(PyExc_ValueError,"RI derivative direction must index a Cartesian coordinate.");
            return nullptr;
        }
    } else if (!PyArg_ParseTuple(args, "OOOOOOi|i", &primary, &auxiliary, &owners,
                          &aux_owners, &three, &metric, &natom, &order)) return nullptr;
    if (order != 1 && order != 2) {
        PyErr_SetString(PyExc_ValueError,"RI contraction order must be 1 or 2.");
        return nullptr;
    }
    if (!PyTuple_Check(primary) || PyTuple_GET_SIZE(primary) != 5 ||
        !PyTuple_Check(auxiliary) || PyTuple_GET_SIZE(auxiliary) != 5 || natom <= 0) {
        PyErr_SetString(PyExc_ValueError, "RI derivatives need two five-array basis tuples and natom > 0.");
        return nullptr;
    }
    RIBasisArrays p(primary), a(auxiliary);
    ArrayRef atoms(owners, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_atoms(aux_owners, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef jbar, mbar;
    if (!columns) {
        jbar.obj=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(three,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
        mbar.obj=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(metric,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    }
    if (!p.valid() || !a.valid() || !atoms || !aux_atoms || (!columns && (!jbar || !mbar))) return nullptr;
    if (PyArray_NDIM(p.shells.obj) != 2 || PyArray_NDIM(a.shells.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "RI shell arrays must be two-dimensional.");
        return nullptr;
    }
    const npy_intp nao = PyArray_DIM(p.shells.obj, 0), naux = PyArray_DIM(a.shells.obj, 0);
    npy_intp bounds_shape[2] = {nao, nao};
    ArrayRef bounds;
    bounds.obj = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, bounds_shape, NPY_DOUBLE, 0));
    if (!bounds || !validate_ri_inputs(p.shells.obj, p.origins.obj, p.exps.obj,
        p.weights.obj, p.nprim.obj, a.shells.obj, a.origins.obj, a.exps.obj,
        a.weights.obj, a.nprim.obj, bounds.obj)) return nullptr;
    const npy_intp npair = nao*(nao+1)/2;
    npy_intp output_nao=nao,output_naux=naux;
    std::vector<RIGradientContraction::OutputRow> pt,at;
    if (primary_map!=Py_None || auxiliary_map!=Py_None) {
        ArrayRef t(primary_map,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY),u(auxiliary_map,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY);
        if (!t || !u) return nullptr;
        if (occupied==Py_None || PyArray_NDIM(t.obj)!=2 || PyArray_NDIM(u.obj)!=2 ||
            PyArray_DIM(t.obj,0)!=nao || PyArray_DIM(u.obj,0)!=naux) {
            PyErr_SetString(PyExc_ValueError,"Spherical derivative output requires occupied contraction and matching transforms.");
            return nullptr;
        }
        output_nao=PyArray_DIM(t.obj,1);output_naux=PyArray_DIM(u.obj,1);
        auto rows=[](PyArrayObject* array) {
            const auto n=PyArray_DIM(array,0),m=PyArray_DIM(array,1);
            const auto* data=static_cast<const double*>(PyArray_DATA(array));
            std::vector<RIGradientContraction::OutputRow> result(n);
            for (npy_intp i=0;i<n;++i) for (npy_intp j=0;j<m;++j)
                if (data[i*m+j]!=0.) result[i].emplace_back(j,data[i*m+j]);
            return result;
        };
        pt=rows(t.obj);at=rows(u.obj);
    }
    ArrayRef orbitals, density, coulomb;
    if (occupied!=Py_None || rho!=Py_None) {
        orbitals.obj=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(occupied,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
        density.obj=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(rho,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
        if (!orbitals || !density) return nullptr;
        if (count!=3 || PyArray_NDIM(orbitals.obj)!=2 || PyArray_DIM(orbitals.obj,0)!=nao ||
            PyArray_NDIM(density.obj)!=1 || PyArray_DIM(density.obj,0)!=naux) {
            PyErr_SetString(PyExc_ValueError,"Occupied RI derivatives need three directions, Cartesian orbitals and auxiliary density.");
            return nullptr;
        }
        npy_intp dims[3]={3,output_nao,output_nao};
        coulomb.obj=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(3,dims,NPY_DOUBLE,0));
        if (!coulomb) return nullptr;
    }
    if (PyArray_NDIM(atoms.obj) != 1 || PyArray_NDIM(aux_atoms.obj) != 1 ||
        PyArray_DIM(atoms.obj, 0) != nao || PyArray_DIM(aux_atoms.obj, 0) != naux ||
        (!columns && (PyArray_NDIM(jbar.obj) != 3 || PyArray_NDIM(mbar.obj) != 3 ||
        PyArray_DIM(jbar.obj, 1) != naux || PyArray_DIM(jbar.obj, 2) != npair ||
        PyArray_DIM(mbar.obj, 0) != PyArray_DIM(jbar.obj, 0) ||
        PyArray_DIM(mbar.obj, 1) != naux || PyArray_DIM(mbar.obj, 2) != naux))) {
        PyErr_SetString(PyExc_ValueError, "Inconsistent RI derivative adjoint/owner shapes.");
        return nullptr;
    }
    const auto* po = static_cast<const std::int64_t*>(PyArray_DATA(atoms.obj));
    const auto* ao = static_cast<const std::int64_t*>(PyArray_DATA(aux_atoms.obj));
    for (int side = 0; side < 2; ++side) {
        const auto* ids = side ? ao : po;
        for (npy_intp i = 0; i < (side ? naux : nao); ++i) {
            if (ids[i] < 0 || ids[i] >= natom) {
                PyErr_SetString(PyExc_ValueError, "RI basis owners must index an atom.");
                return nullptr;
            }
        }
    }
    const auto* ps = static_cast<const std::int64_t*>(PyArray_DATA(p.shells.obj));
    const auto* as = static_cast<const std::int64_t*>(PyArray_DATA(a.shells.obj));
    const auto* pn = static_cast<const std::int64_t*>(PyArray_DATA(p.nprim.obj));
    const auto* an = static_cast<const std::int64_t*>(PyArray_DATA(a.nprim.obj));
    const auto* px = static_cast<const double*>(PyArray_DATA(p.origins.obj));
    const auto* ax = static_cast<const double*>(PyArray_DATA(a.origins.obj));
    const auto* pe = static_cast<const double*>(PyArray_DATA(p.exps.obj));
    const auto* ae = static_cast<const double*>(PyArray_DATA(a.exps.obj));
    const auto* pw = static_cast<const double*>(PyArray_DATA(p.weights.obj));
    const auto* aw = static_cast<const double*>(PyArray_DATA(a.weights.obj));
    const auto* mw = columns ? nullptr : static_cast<const double*>(PyArray_DATA(mbar.obj));
    const npy_intp pm = PyArray_DIM(p.exps.obj, 1), am = PyArray_DIM(a.exps.obj, 1);
    const npy_intp nobs = columns ? 1 : PyArray_DIM(jbar.obj, 0);
    npy_intp shape[3] = {nobs, order == 1 ? natom : 3*natom, order == 1 ? 3 : 3*natom};
    npy_intp three_shape[3]={count,naux,npair}, metric_shape[3]={count,output_naux,output_naux};
    ArrayRef metric_output;
    if (columns) metric_output.obj=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(count==1?2:3,metric_shape+(count==1),NPY_DOUBLE,0));
    if (columns && !metric_output) return nullptr;
    npy_intp projected_shape[4]={3,output_naux,output_nao,orbitals ? PyArray_DIM(orbitals.obj,1) : 0};
    PyObject* result = orbitals ? PyArray_ZEROS(4,projected_shape,NPY_DOUBLE,0) :
        (columns ? PyArray_ZEROS(count==1?2:3,three_shape+(count==1),NPY_DOUBLE,0) : PyArray_ZEROS(3, shape, NPY_DOUBLE, 0));
    if (!result) return nullptr;
    auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)));
    RIGradientContraction response{po, ao,
        columns ? nullptr : static_cast<const double*>(PyArray_DATA(jbar.obj)), out, nobs, natom};
    response.order = order;
    response.profile=profiling ? &profile : nullptr;
    if (columns) { response.first_three=out;response.direction=direction;response.first_count=count; }
    if (orbitals) {
        response.occupied=static_cast<const double*>(PyArray_DATA(orbitals.obj));
        response.fitted_density=static_cast<const double*>(PyArray_DATA(density.obj));
        response.coulomb=static_cast<double*>(PyArray_DATA(coulomb.obj));
        response.noccupied=projected_shape[3];
        if (!pt.empty()) {
            response.primary_output=&pt;response.auxiliary_output=&at;
            response.output_nao=output_nao;response.output_naux=output_naux;
        }
    }
    try {
        if (!compute_ri_j3_shell_blocked(ps, px, pe, pw, pn, pm, as, ax, ae, aw, an, am,
            static_cast<const double*>(PyArray_DATA(bounds.obj)), nullptr, nullptr,
            nao, naux, 0., &response)) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_NotImplementedError,
                "RI gradients require complete shells with primary pair l <= 6 and auxiliary l <= 6.");
            return nullptr;
        }
        const auto metric_start=profiling ? DerivativeClock::now() : DerivativeClock::time_point{};
        metric_derivative_shells(as,ax,ae,aw,an,ao,naux,am,order,
            columns ? direction/3 : -1,mw,nobs,
            [&](npy_intp i,npy_intp j,int x,int y,double derivative) {
                if (columns) {
                    if (count==1 && x!=direction%3) return;
                    auto* data=static_cast<double*>(PyArray_DATA(metric_output.obj));
                    if (!at.empty()) {
                        data+=x*output_naux*output_naux;
                        const double v=derivative*(int(ao[i]==direction/3)-int(ao[j]==direction/3));
                        for (const auto& [p,cp]:at[i]) for (const auto& [q,cq]:at[j]) {
                            data[p*output_naux+q]+=v*cp*cq;
                            if (i!=j) data[q*output_naux+p]+=v*cp*cq;
                        }
                        return;
                    }
                    if (count==3) data+=x*naux*naux;
                    data[i*naux+j]=data[j*naux+i]=derivative*(int(ao[i]==direction/3)-int(ao[j]==direction/3));
                    return;
                }
                for (npy_intp o=0;o<nobs;++o) {
                    const double value=derivative*(mw[(o*naux+i)*naux+j]+mw[(o*naux+j)*naux+i]);
                    if (order==1) {
                        out[(o*natom+ao[i])*3+x]+=value;
                        out[(o*natom+ao[j])*3+x]-=value;
                    } else {
                        const npy_intp atoms[2]={ao[i],ao[j]},ncoord=3*natom;
                        for (int a=0;a<2;++a) for (int b=0;b<2;++b) {
                            const double sign=a==b ? 1. : -1.;
                            out[(o*ncoord+3*atoms[a]+x)*ncoord+3*atoms[b]+y]+=sign*value;
                            if (x!=y) out[(o*ncoord+3*atoms[b]+y)*ncoord+3*atoms[a]+x]+=sign*value;
                        }
                    }
                }
            });
        if (profiling) profile.seconds[3+order]+=derivative_elapsed(metric_start);
    } catch (const std::exception& error) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    if (profiling) {
        profile.seconds[7+order]+=derivative_elapsed(total_start);
        merge_derivative_profile(profile);
    }
    if (orbitals) return Py_BuildValue("NOO",result,metric_output.obj,coulomb.obj);
    if (columns) return Py_BuildValue("NO",result,metric_output.obj);
    return result;
}

PyObject* contract_ri_derivatives(PyObject*,PyObject* args) {
    return ri_derivative_evaluate(args,false);
}

PyObject* ri_derivative_columns(PyObject*,PyObject* args) {
    return ri_derivative_evaluate(args,true);
}
