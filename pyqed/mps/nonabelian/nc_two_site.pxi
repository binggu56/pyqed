"""Direct NC two-site optimization with complete intermediate multiplets."""
cdef extern from "nc_pair_action.hpp" namespace "pyqed::su2" nogil:
    cdef cppclass NCPairOperator:
        vector[int64_t] offsets,columns
        vector[cpp_complex[double]] values
    cdef cppclass NCPairBuilder:
        NCPairBuilder(int64_t,int64_t) except +
        void add_block(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
                       const cpp_complex[double]*,cpp_complex[double]) except +
        size_t size()
        NCPairOperator compressed() except +
    cdef cppclass NCPairAction:
        NCPairAction(const vector[int64_t]&,const vector[int64_t]&,
                     const vector[int64_t]&,const vector[int64_t]&,
                     const vector[NCPairOperator] &,
                     const vector[NCPairOperator] &,int) except +
        int threads_used()
        uint64_t parallel_calls()
        vector[cpp_complex[double]] apply(const vector[cpp_complex[double]]&) except +
        vector[cpp_complex[double]] diagonal() except +
        CppDavidsonResult solve(vector[cpp_complex[double]],double) except +


cdef class NCOperatorHalf:
    cdef NCPairBuilder* builder
    cdef int64_t rows,cols

    def __cinit__(self,int64_t rows,int64_t cols):
        self.rows,self.cols=rows,cols
        self.builder=new NCPairBuilder(rows,cols)

    def __dealloc__(self):
        del self.builder

    def tocsr(self):
        from scipy.sparse import csr_matrix
        cdef NCPairOperator result=self.builder.compressed()
        return csr_matrix((np.asarray(result.values,dtype=complex),
            np.asarray(result.columns,dtype=np.int64),np.asarray(result.offsets,dtype=np.int64)),
            shape=(self.rows,self.cols))

    def __bool__(self):
        return self.builder.size()!=0


cdef class NCPairHamiltonian:
    """Compiled NC channel action; no two-site Hamiltonian matrix is formed."""
    cdef NCPairAction* action

    def __cinit__(self,matrices,halves,int n_threads=1):
        cdef vector[int64_t] rows,cols,outputs,inputs
        from scipy.sparse import csr_matrix
        cdef vector[NCPairOperator] left,right
        cdef NCPairOperator entry
        positions={q:i for i,q in enumerate(matrices)}
        for matrix in matrices.values():
            rows.push_back(matrix.shape[0]);cols.push_back(matrix.shape[1])
        for (oq,iq,_),(l,r) in halves.items():
            outputs.push_back(positions[oq]);inputs.push_back(positions[iq])
            l=csr_matrix(l,dtype=complex);r=csr_matrix(r,dtype=complex)
            entry.offsets=l.indptr;entry.columns=l.indices;entry.values=l.data
            left.push_back(entry)
            entry.offsets=r.indptr;entry.columns=r.indices;entry.values=r.data
            right.push_back(entry)
        self.action=new NCPairAction(rows,cols,outputs,inputs,left,right,n_threads)

    def __dealloc__(self):
        del self.action

    def __call__(self,values):
        cdef vector[cpp_complex[double]] x=np.asarray(values,dtype=complex).ravel()
        cdef vector[cpp_complex[double]] y
        with nogil: y=self.action.apply(x)
        return np.asarray(y,dtype=complex)

    def diagonal(self):
        cdef vector[cpp_complex[double]] y
        with nogil: y=self.action.diagonal()
        return np.asarray(y,dtype=complex)

    def solve(self,guess,double tolerance):
        cdef vector[cpp_complex[double]] x=np.asarray(guess,dtype=complex).ravel()
        cdef CppDavidsonResult result
        with nogil: result=self.action.solve(x,tolerance)
        return result.energy,np.asarray(result.vector,dtype=complex),dict(
            iterations=result.iterations,residual_norm=result.residual_norm,
            matvec_calls=result.matvec_calls,converged=bool(result.converged),local_solver='davidson',
            pair_threads_used=self.action.threads_used(),pair_parallel_calls=self.action.parallel_calls())


def spatial_nc_pair_problem(solver,int bond,left,right,norm_left,norm_right):
    from .multiplet_operator import dimension
    matrices,layouts,lm,rm=multiplet_pair(solver,bond)
    weights=spatial_nc_pair_weights(solver,matrices,layouts,norm_left,norm_right)
    halves=spatial_nc_cbe_halves(solver,bond,matrices,layouts,lm,rm,left,right,sparse=True)
    scales={q:(w[:,0],w[0]/w[0,0]) for q,w in weights.items()}
    halves={key:(l.multiply(1/scales[key[0]][0][:,None]).multiply(1/scales[key[1]][0][None,:]).tocsr(),
                 r.multiply(1/scales[key[0]][1][:,None]).multiply(1/scales[key[1]][1][None,:]).tocsr()) for key,(l,r) in halves.items()}
    return NCPairHamiltonian(matrices,halves,solver.n_threads),matrices,layouts,weights


def optimize_spatial_nc_pair(solver,int bond,direction,left,right,norm_left,norm_right,double local_tol):
    from .multiplet_operator import dimension
    action,matrices,layouts,weights=spatial_nc_pair_problem(solver,bond,left,right,norm_left,norm_right)
    guess=np.concatenate([(m*weights[q]).ravel() for q,m in matrices.items()])
    energy,vector,info=action.solve(guess,local_tol)
    left_values,right_values={},{};offset=0;discarded=0.
    for q,m in matrices.items():
        pair=vector[offset:offset+m.size].reshape(m.shape)/weights[q];offset+=m.size
        column_weights=np.empty(m.shape[1])
        for key,(span,_) in layouts[q][1].items():
            column_weights[span]=np.sqrt(dimension(solver.sites[0].symmetry,key[2])/dimension(solver.sites[0].symmetry,q))
        u,s,vh=_multiplet_svd(pair*column_weights[None,:])
        keep=min(solver.D,int(np.count_nonzero(s>1e-12*max(s.max(initial=0),1e-30))))
        discarded+=dimension(solver.sites[0].symmetry,q)*float(np.sum(s[keep:]**2))
        u,vh=u[:,:keep],vh[:keep]/column_weights[None,:]
        if direction=='lr': left_values[q],right_values[q]=u,s[:keep,None]*vh
        else: left_values[q],right_values[q]=u*s[:keep],vh
    install_multiplet_pair(solver,bond,layouts,left_values,right_values)
    return dict(info,site=bond,parameters=len(guess),energy=energy+solver.ecore,discarded_weight=discarded)


def run_spatial_nc_two_site(solver,int nsweeps,double tol,double local_tol,int verbose):
    """Two-site finite-system sweeps; SVD keeps whole product-group irreps."""
    cdef int cycle,i,step,direction_id
    fixed=None
    for cycle in solver._sweep_control.steps:
        before=solver.energy;records=[]
        for direction_id in range(2):
            direction='lr' if direction_id==0 else 'rl'
            fixed_h,fixed_n=solver._boundaries(direction) if fixed is None else fixed
            next_h,next_n=[None]*(solver.nsites+1),[None]*(solver.nsites+1)
            moving_h,moving_n=solver._edge(direction_id==1)
            next_h[0 if direction_id==0 else solver.nsites]=moving_h
            next_n[0 if direction_id==0 else solver.nsites]=moving_n
            for step in range(solver.nsites-1):
                i=step if direction_id==0 else solver.nsites-2-step
                if direction_id==0:
                    record=optimize_spatial_nc_pair(solver,i,direction,moving_h,fixed_h[i+2],moving_n,fixed_n[i+2],local_tol)
                    moving_h,moving_n=solver._advance(i,moving_h,moving_n,direction)
                    next_h[i+1],next_n[i+1]=moving_h,moving_n
                else:
                    record=optimize_spatial_nc_pair(solver,i,direction,fixed_h[i],moving_h,fixed_n[i],moving_n,local_tol)
                    moving_h,moving_n=solver._advance(i+1,moving_h,moving_n,direction)
                    next_h[i+1],next_n[i+1]=moving_h,moving_n
                records.append(record)
            i=solver.nsites-1 if direction_id==0 else 0
            moving_h,moving_n=solver._advance(i,moving_h,moving_n,direction)
            next_h[solver.nsites if direction_id==0 else 0]=moving_h
            next_n[solver.nsites if direction_id==0 else 0]=moving_n
            fixed=next_h,next_n
        solver._finish_cycle(moving_h,moving_n)
        done=solver._complete_cycle(cycle,before,records)
        if verbose: print(f'Multiplet NC {cycle+1}: {solver.energy:.12f}',flush=True)
        if done:
            break
    solver.diagnostics['two_site_driver']='compiled_nc'
