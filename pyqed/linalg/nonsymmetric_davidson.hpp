#pragma once
#include "davidson.hpp"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace pyqed::linalg {
extern "C" void zgeev_(char*,char*,int*,Complex*,int*,Complex*,Complex*,int*,
                       Complex*,int*,Complex*,int*,double*,int*);
extern "C" void zgeqrf_(int*,int*,Complex*,int*,Complex*,Complex*,int*,int*);
extern "C" void zunmqr_(char*,char*,int*,int*,int*,Complex*,int*,Complex*,
                        Complex*,int*,Complex*,int*,int*);
extern "C" void ztrtrs_(char*,char*,char*,int*,int*,Complex*,int*,Complex*,int*,int*);

// Right-eigenvector, restarted Davidson. All subspace algebra is complex;
// no Hermitian symmetrization or indefinite-metric projection is applied.
inline pybind11::tuple nonsymmetric_davidson(
    pybind11::object action, pybind11::array_t<Complex,pybind11::array::f_style> diagonal,
    int roots, double tolerance, int iterations, int cap, std::size_t memory_limit,
    pybind11::object guess, int selection, Complex target, double imaginary_tolerance,
    double dependence_tolerance) {
    namespace py=pybind11;
    const auto size=diagonal.size();
    if (diagonal.ndim()!=1 || size<1 || size>std::numeric_limits<int>::max()
        || roots<1 || roots>size || cap<roots || cap>size || (cap==roots && cap<size)
        || iterations<1 || !(tolerance>0) || !std::isfinite(tolerance)
        || !(dependence_tolerance>0 && dependence_tolerance<1)
        || !(imaginary_tolerance>=0) || !std::isfinite(imaginary_tolerance)
        || selection<0 || selection>3 || !PyCallable_Check(action.ptr()))
        throw std::invalid_argument("Invalid nonsymmetric Davidson parameters");
    const int n=size;
    auto finite=[](Complex z){return std::isfinite(z.real())&&std::isfinite(z.imag());};
    if (!finite(target)) throw std::invalid_argument("Nonfinite spectral target");
    std::vector<Complex> d(diagonal.data(),diagonal.data()+n);
    for (auto z:d) if (!finite(z)) throw std::invalid_argument("Nonfinite diagonal");
    const long double estimate=sizeof(Complex)*(6.L*n*cap+12.L*n*roots+12.L*cap*cap+10.L*n);
    if (estimate>memory_limit) throw std::runtime_error("Nonsymmetric Davidson workspace exceeds memory limit");
    auto score=[&](Complex z) {
        if (!finite(z)) return std::numeric_limits<double>::infinity();
        if (selection==1) return std::abs(z);
        if (selection==3) return std::abs(z-target);
        return z.real();
    };
    auto eligible=[&](Complex z){return finite(z) && (selection!=2 || (z.real()>tolerance && std::abs(z.imag())<=imaginary_tolerance));};
    std::vector<int> seeds(n); std::iota(seeds.begin(),seeds.end(),0);
    std::stable_sort(seeds.begin(),seeds.end(),[&](int a,int b){
        if (eligible(d[a])!=eligible(d[b])) return eligible(d[a]);
        return score(d[a])<score(d[b]);
    });
    std::vector<Complex> basis, applied;
    basis.reserve(static_cast<std::size_t>(n)*cap); applied.reserve(static_cast<std::size_t>(n)*cap);
    // Relative rank test on the candidate, separate from absolute residual convergence.
    auto append=[&](std::vector<Complex>& q,std::vector<Complex> x) {
        double original=davidson_norm(x.data(),n);
        if (!(original>0) || !std::isfinite(original)) return false;
        for (int pass=0;pass<2;++pass)
            for (std::size_t start=0;start<q.size();start+=n) {
                Complex overlap=davidson_dot(q.data()+start,x.data(),n);
                for (int i=0;i<n;++i) x[i]-=q[start+i]*overlap;
            }
        double norm=davidson_norm(x.data(),n);
        if (norm<=dependence_tolerance*original) return false;
        for (auto z:x) q.push_back(z/norm);
        return true;
    };
    if (!guess.is_none()) {
        auto array=py::array_t<Complex,py::array::f_style|py::array::forcecast>::ensure(guess);
        if (!array || array.ndim()!=2 || array.shape(0)!=n || array.shape(1)<1 || array.shape(1)>cap)
            throw std::invalid_argument("Invalid initial subspace");
        for (py::ssize_t j=0;j<array.shape(1);++j) {
            std::vector<Complex> x(array.data()+j*n,array.data()+(j+1)*n);
            for (auto z:x) if (!finite(z)) throw std::invalid_argument("Nonfinite initial subspace");
            append(basis,std::move(x));
        }
    }
    int initial=std::min(cap,std::max(roots+2,2*roots));
    for (int seed:seeds) {
        if (basis.size()/n>=static_cast<std::size_t>(initial)) break;
        std::vector<Complex> x(n); x[seed]=1.; append(basis,std::move(x));
    }
    std::vector<Complex> energies, states;
    std::vector<double> residuals;
    int completed=0,restarts=0,calls=0; std::size_t columns=0;
    bool converged=false; double action_seconds=0.;
    {
        py::gil_scoped_release release;
        for (int cycle=0;cycle<iterations;++cycle) {
            completed=cycle+1;
            int m=basis.size()/n, old=applied.size()/n, added=m-old;
            if (added) {
                auto started=std::chrono::steady_clock::now();
                py::gil_scoped_acquire acquire;
                py::array_t<Complex> input({n,added},{sizeof(Complex),static_cast<std::size_t>(n)*sizeof(Complex)});
                std::copy(basis.begin()+static_cast<std::size_t>(old)*n,basis.end(),input.mutable_data());
                auto output=py::array_t<Complex,py::array::f_style|py::array::forcecast>::ensure(action(input));
                if (!output || output.ndim()!=2 || output.shape(0)!=n || output.shape(1)!=added)
                    throw std::invalid_argument("Nonsymmetric Davidson callback returned wrong shape");
                for (int j=0;j<added;++j) for (int i=0;i<n;++i) {
                    Complex z=output.data()[static_cast<std::size_t>(j)*n+i];
                    if (!finite(z)) throw std::invalid_argument("Nonfinite operator action");
                    applied.push_back(z);
                }
                ++calls; columns+=added;
                action_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            }
            std::vector<Complex> h(static_cast<std::size_t>(m)*m);
            std::vector<Complex> w(m),vr(static_cast<std::size_t>(m)*m),work;
            std::vector<double> rwork(8*m);
            Complex dummy,query; int one=1,lwork=-1,info=0; char no='N',yes='V';
            if (selection==0 || m==n) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
                davidson_gemm(102,113,111,m,m,n,1.,basis.data(),n,applied.data(),n,0.,h.data(),m);
#else
                for (int j=0;j<m;++j) for (int i=0;i<m;++i)
                    h[j*m+i]=davidson_dot(basis.data()+static_cast<std::size_t>(i)*n,applied.data()+static_cast<std::size_t>(j)*n,n);
#endif
                zgeev_(&no,&yes,&m,h.data(),&m,w.data(),&dummy,&one,vr.data(),&m,&query,&lwork,rwork.data(),&info);
                if (info) throw std::runtime_error("LAPACK workspace query failed");
                lwork=std::max(2*m,static_cast<int>(query.real())); work.resize(lwork);
                zgeev_(&no,&yes,&m,h.data(),&m,w.data(),&dummy,&one,vr.data(),&m,work.data(),&lwork,rwork.data(),&info);
            } else {
                // F=QR gives R y=(theta-shift) Q^H V y. Solve the
                // reciprocal problem R^{-1} Q^H V y=mu y, then theta=shift+1/mu.
                // This avoids normal equations and QZ on clustered pencils.
                Complex shift=(selection==3?target:Complex(0.));
                shift+=1e-6*(1.+std::abs(shift));
                std::vector<Complex> f(applied), transformed(basis), tau(m);
                for (std::size_t i=0;i<f.size();++i) f[i]-=shift*basis[i];
                int rows=n;
                zgeqrf_(&rows,&m,f.data(),&rows,tau.data(),&query,&lwork,&info);
                if (info) throw std::runtime_error("Harmonic QR workspace query failed");
                lwork=std::max(m,static_cast<int>(query.real())); work.resize(lwork);
                zgeqrf_(&rows,&m,f.data(),&rows,tau.data(),work.data(),&lwork,&info);
                if (info) throw std::runtime_error("Harmonic QR failed");
                char left='L',adjoint='C'; lwork=-1;
                zunmqr_(&left,&adjoint,&rows,&m,&m,f.data(),&rows,tau.data(),
                    transformed.data(),&rows,&query,&lwork,&info);
                if (info) throw std::runtime_error("Harmonic projection workspace query failed");
                lwork=std::max(m,static_cast<int>(query.real())); work.resize(lwork);
                zunmqr_(&left,&adjoint,&rows,&m,&m,f.data(),&rows,tau.data(),
                    transformed.data(),&rows,work.data(),&lwork,&info);
                if (info) throw std::runtime_error("Harmonic projection failed");
                char upper='U',normal='N';
                ztrtrs_(&upper,&normal,&normal,&m,&m,f.data(),&rows,transformed.data(),&rows,&info);
                if (info) throw std::runtime_error("Harmonic shifted action is rank deficient");
                for (int j=0;j<m;++j) for (int i=0;i<m;++i)
                    h[j*m+i]=transformed[j*n+i];
                lwork=-1;
                zgeev_(&no,&yes,&m,h.data(),&m,w.data(),&dummy,&one,vr.data(),&m,&query,&lwork,rwork.data(),&info);
                if (info) throw std::runtime_error("Harmonic reciprocal workspace query failed");
                lwork=std::max(2*m,static_cast<int>(query.real())); work.resize(lwork);
                zgeev_(&no,&yes,&m,h.data(),&m,w.data(),&dummy,&one,vr.data(),&m,work.data(),&lwork,rwork.data(),&info);
                for (int i=0;i<m;++i) w[i]=std::abs(w[i])>0?shift+1./w[i]:Complex(std::numeric_limits<double>::infinity());
            }
            if (info) throw std::runtime_error("Projected nonsymmetric eigensolve failed: LAPACK info="+std::to_string(info)+", iteration="+std::to_string(completed)+", space="+std::to_string(m));
            std::vector<int> order(m); std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){
                if (eligible(w[a])!=eligible(w[b])) return eligible(w[a]);
                return score(w[a])<score(w[b]);
            });
            int available=0; for (int index:order) if (eligible(w[index])) ++available;
            int found=std::min(roots,available);
            int retained=std::min(m,std::min(cap-1,std::max(roots+2,2*roots)));
            int rotate_count=std::max(found,retained);
            std::vector<Complex> coefficients(static_cast<std::size_t>(m)*rotate_count);
            for (int j=0;j<rotate_count;++j) for (int i=0;i<m;++i)
                coefficients[i*rotate_count+j]=vr[order[j]*m+i];
            std::vector<Complex> ritz(static_cast<std::size_t>(n)*rotate_count), aritz(ritz.size());
            davidson_rotate(basis.data(),n,n,m,coefficients.data(),rotate_count,rotate_count,ritz.data());
            davidson_rotate(applied.data(),n,n,m,coefficients.data(),rotate_count,rotate_count,aritz.data());
            energies.resize(found); residuals.resize(found); states.assign(ritz.begin(),ritz.begin()+static_cast<std::size_t>(n)*found);
            std::vector<std::vector<Complex>> corrections;
            converged=found==roots;
            for (int j=0;j<found;++j) {
                energies[j]=w[order[j]];
                auto offset=static_cast<std::size_t>(j)*n;
                double norm=davidson_norm(states.data()+offset,n);
                std::vector<Complex> residual(n);
                for (int i=0;i<n;++i) {
                    states[offset+i]/=norm;
                    residual[i]=aritz[offset+i]/norm-energies[j]*states[offset+i];
                }
                residuals[j]=davidson_norm(residual.data(),n);
                if (residuals[j]>tolerance) {
                    converged=false;
                    for (int i=0;i<n;++i) {
                        Complex denominator=d[i]-energies[j];
                        if (std::abs(denominator)<1e-8)
                            denominator=std::abs(denominator)>0?denominator*(1e-8/std::abs(denominator)):Complex(1e-8);
                        residual[i]=-residual[i]/denominator;
                    }
                    corrections.push_back(std::move(residual));
                }
            }
            if (converged || m==n) break;
            if (m+static_cast<int>(corrections.size())>cap || m==cap) {
                basis.clear(); applied.clear(); ++restarts;
                for (int j=0;j<retained;++j)
                    append(basis,std::vector<Complex>(ritz.begin()+static_cast<std::size_t>(j)*n,ritz.begin()+static_cast<std::size_t>(j+1)*n));
            }
            auto before=basis.size();
            for (auto& x:corrections) {
                if (basis.size()/n==static_cast<std::size_t>(cap)) break;
                append(basis,std::move(x));
            }
            if (basis.size()==before || found<roots) {
                for (int seed:seeds) {
                    if (basis.size()/n==static_cast<std::size_t>(cap)) break;
                    std::vector<Complex> x(n); x[seed]=1.;
                    if (append(basis,std::move(x))) break;
                }
            }
            if (basis.size()==before && !applied.empty()) break;
        }
    }
    int found=energies.size();
    py::array_t<Complex> vectors({n,found},{sizeof(Complex),static_cast<std::size_t>(n)*sizeof(Complex)});
    std::copy(states.begin(),states.end(),vectors.mutable_data());
    py::dict info;
    info["converged"]=converged; info["iterations"]=completed; info["restarts"]=restarts;
    info["matvecs"]=columns; info["matmat_callback_calls"]=calls; info["residual_norms"]=residuals;
    info["action_seconds"]=action_seconds; info["estimated_workspace_bytes"]=static_cast<double>(estimate);
    return py::make_tuple(energies,vectors,info);
}
}
