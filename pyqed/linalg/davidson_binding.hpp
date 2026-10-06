#pragma once
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "davidson.hpp"

namespace pyqed::linalg {
template <typename Scalar>
inline auto davidson_guesses(pybind11::object guess, std::size_t n,
    std::size_t cap, int roots) {
    namespace py = pybind11;
        std::vector<std::vector<Scalar>> guesses;
        if (!guess.is_none()) {
            auto array=py::array_t<Scalar,py::array::c_style>::ensure(guess);
            if (!array || array.ndim()!=2 || array.shape(0)!=static_cast<py::ssize_t>(n)
                || array.shape(1)<1 || array.shape(1)>static_cast<py::ssize_t>(std::min<std::size_t>(cap,2*roots)))
                throw std::invalid_argument("Invalid Davidson guess dimensions or dtype");
            auto input=array.template unchecked<2>();
            guesses.assign(array.shape(1),std::vector<Scalar>(n));
            for (py::ssize_t j=0;j<array.shape(1);++j) for (std::size_t i=0;i<n;++i) {
                if ((!std::isfinite(std::real(input(i,j))) || !std::isfinite(std::imag(input(i,j))))) throw std::invalid_argument("Nonfinite Davidson guess");
                guesses[j][i]=input(i,j);
            }
        }
    return guesses;
}

template <typename Scalar>
inline pybind11::tuple davidson_output(const BlockDavidsonResult& result,
    const BlockDavidsonWorkspace<Scalar>& workspace, int roots, long double estimate) {
    namespace py = pybind11;
    const auto n = workspace.dimension;
        py::array_t<Scalar> vectors({static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(roots)});
        auto out=vectors.template mutable_unchecked<2>();
        for (int j=0;j<roots;++j) for (std::size_t i=0;i<n;++i) {
            if constexpr (std::is_same_v<Scalar,double>) out(i,j)=result.vectors[j][i].real();
            else out(i,j)=result.vectors[j][i];
        }
        py::dict info;
        info["converged"]=result.converged; info["iterations"]=result.iterations;
        info["restarts"]=result.restarts;
        info["locked_roots"]=result.locked_roots;
        info["deflated_iterations"]=result.deflated_iterations;
        info["deflation_unlocks"]=result.deflation_unlocks;
        info["restart_history_vectors"]=result.restart_history_vectors; info["matvecs"]=result.matvec_calls;
        info["residual_norms"]=result.residual_norms;
        info["workspace_bytes"]=workspace.memory_bytes();
        info["estimated_peak_workspace_bytes"]=static_cast<double>(estimate);
        info["action_seconds"]=result.action_seconds;
        info["projection_seconds"]=result.projection_seconds;
        info["diagonalization_seconds"]=result.diagonalization_seconds;
        info["ritz_seconds"]=result.ritz_seconds;
        info["correction_seconds"]=result.correction_seconds;
        info["cholesky_qr_blocks"]=result.cholesky_qr_blocks;
        info["householder_qr_blocks"]=result.householder_qr_blocks;
        info["restart_seconds"]=result.restart_seconds;
        return py::make_tuple(result.energies,vectors,info);
}

template <typename Scalar>
inline void bind_davidson_type(pybind11::module_& module) {
    namespace py = pybind11;
    module.def("davidson", [](py::array_t<Scalar, py::array::c_style> matrix,
        int roots, double tolerance, int iterations, int space, std::size_t memory_limit,
        py::object guess) {
        if (matrix.ndim()!=2 || matrix.shape(0)!=matrix.shape(1) ||
            roots<1 || roots>matrix.shape(0) || space<roots ||
            matrix.shape(0)>std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid Davidson matrix/root/subspace dimensions");
        const std::size_t n=matrix.shape(0), cap=std::min<std::size_t>(n,space);
        // Conservative peak estimate: persistent arenas, projected solve,
        // guesses, returned vectors, orthogonalization and validation blocks.
        const long double estimate=static_cast<long double>(sizeof(Scalar))*(2.L*n*cap+14.L*n*roots+8.L*cap*cap);
        if (estimate>memory_limit) throw std::runtime_error("Davidson workspace exceeds memory limit");
        const Scalar* data=matrix.data();
        std::vector<double> diagonal(n);
        for (std::size_t i=0;i<n;++i) {
            diagonal[i]=std::real(data[i*n+i]);
            for (std::size_t j=0;j<n;++j)
                if ((!std::isfinite(std::real(data[i*n+j])) || !std::isfinite(std::imag(data[i*n+j]))) ||
                    std::abs(data[i*n+j]-davidson_conj(data[j*n+i]))>1.e-12*std::max(1.,std::abs(data[i*n+j])))
                    throw std::invalid_argument("Davidson requires a finite Hermitian matrix");
        }
        auto guesses = davidson_guesses<Scalar>(guess, n, cap, roots);
        BlockDavidsonWorkspace<Scalar> workspace;
        BlockDavidsonResult result;
        {
            py::gil_scoped_release release;
            result=block_davidson(diagonal,guesses,roots,tolerance,iterations,cap,true,workspace,
                [&](const Scalar* x,Scalar* y,std::size_t dim,std::size_t columns,std::size_t stride){
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
                    davidson_gemm(101,111,112,columns,dim,dim,1.,x,stride,data,dim,0.,y,stride);
#else
                    for (std::size_t c=0;c<columns;++c)
                        for (std::size_t i=0;i<dim;++i) {
                            Scalar sum=0.;
                            for (std::size_t j=0;j<dim;++j) sum+=data[i*dim+j]*x[c*stride+j];
                            y[c*stride+i]=sum;
                        }
#endif
                });
        }
        return davidson_output(result, workspace, roots, estimate);
    },py::arg("matrix").noconvert(),py::arg("roots"),py::arg("tolerance")=1.e-10,
      py::arg("iterations")=200,py::arg("space")=64,py::arg("memory_limit")=536870912,py::arg("guess")=py::none(), R"doc(
Lowest eigenpairs of a finite real symmetric or complex Hermitian matrix.

Accepts C-contiguous float64 or complex128 matrices and returns (values,
vectors, info). Check info["converged"]; unconverged Ritz pairs are returned
on iteration exhaustion or stagnation. Tolerance is an absolute residual norm.
Memory limit covers estimated solver scratch space, excluding the input matrix.

Thick-restarted block adaptation of Davidson, J. Comput. Phys. 17, 87-94
(1975), doi:10.1016/0021-9991(75)90065-0, with diagonal preconditioning and
GD+k-inspired history retention (Stathopoulos and McCombs, ACM TOMS 37(2),
Article 21, 2010, doi:10.1145/1731022.1731031). This is not PRIMME or an
exact reproduction: a converged lowest prefix is temporarily frozen at restarts,
with a final coupled Rayleigh-Ritz residual check and unlocking when a lower
active state is discovered. This heuristic is not PRIMME's locking policy.
There is no inner Jacobi-Davidson solve or universal convergence or performance
guarantee. Complex arithmetic
uses conjugate projections and pivoted Householder QR on Accelerate;
portable builds use two-pass Gram-Schmidt and complex Jacobi diagonalization.
)doc");
}
// Callback arrays own their memory: retained Python references cannot alias
// solver storage, and callback writes cannot corrupt the search basis.
template <typename Scalar>
inline pybind11::tuple davidson_operator(pybind11::object action,
    pybind11::array_t<double, pybind11::array::c_style> diagonal,
    int roots, double tolerance, int iterations, int space, std::size_t memory_limit,
    pybind11::object guess, pybind11::object matmat) {
    namespace py = pybind11;
    if ((!action.is_none() && !PyCallable_Check(action.ptr())) ||
        (!matmat.is_none() && !PyCallable_Check(matmat.ptr())) ||
        (action.is_none() && matmat.is_none()))
        throw std::invalid_argument("Davidson requires callable matvec or matmat");
    if (diagonal.ndim()!=1 || roots<1 || roots>diagonal.size() || space<roots ||
        diagonal.size()>std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid Davidson diagonal/root/subspace dimensions");
    const std::size_t n=diagonal.size(), cap=std::min<std::size_t>(n,space);
    const long double estimate=sizeof(Scalar)*(2.L*n*cap+18.L*n*roots+8.L*cap*cap);
    if (estimate>memory_limit) throw std::runtime_error("Davidson workspace exceeds memory limit");
    std::vector<double> d(diagonal.data(),diagonal.data()+n);
    for (double value:d) if (!std::isfinite(value))
        throw std::invalid_argument("Davidson diagonal must be finite");
    auto guesses=davidson_guesses<Scalar>(guess,n,cap,roots);
    BlockDavidsonWorkspace<Scalar> workspace;
    BlockDavidsonResult result;
    std::size_t vector_calls=0, block_calls=0;
    {
        py::gil_scoped_release release;
        result=block_davidson(d,guesses,roots,tolerance,iterations,cap,true,workspace,
            [&](const Scalar* x,Scalar* y,std::size_t dim,std::size_t columns,std::size_t stride) {
                py::gil_scoped_acquire acquire;
                const bool batched=!matmat.is_none();
                const std::size_t count=batched ? columns : 1;
                for (std::size_t first=0;first<columns;first+=count) {
                    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(dim)};
                    std::vector<py::ssize_t> strides{sizeof(Scalar)};
                    if (batched) {
                        shape.push_back(count);
                        strides.push_back(dim*sizeof(Scalar));
                    }
                    py::array_t<Scalar> input(shape,strides);
                    for (std::size_t j=0;j<count;++j)
                        std::copy(x+(first+j)*stride,x+(first+j)*stride+dim,input.mutable_data()+j*dim);
                    py::object value;
                    if (batched) { ++block_calls; value=matmat(input); }
                    else { ++vector_calls; value=action(input); }
                    // No forcecast: in particular, never discard imaginary output.
                    auto output=py::array_t<Scalar,py::array::f_style>::ensure(value);
                    if (!output || output.ndim()!=static_cast<py::ssize_t>(shape.size()) ||
                        output.shape(0)!=static_cast<py::ssize_t>(dim) ||
                        (batched && output.shape(1)!=static_cast<py::ssize_t>(count)))
                        throw std::invalid_argument("Davidson callback returned incompatible shape or dtype");
                    for (std::size_t j=0;j<count;++j) for (std::size_t i=0;i<dim;++i) {
                        const Scalar z=output.data()[j*dim+i];
                        if (!std::isfinite(std::real(z)) || !std::isfinite(std::imag(z)))
                            throw std::invalid_argument("Davidson callback returned nonfinite values");
                        y[(first+j)*stride+i]=z;
                    }
                }
            });
    }
    auto out=davidson_output(result,workspace,roots,estimate);
    auto info=out[2].template cast<py::dict>();
    info["matvec_callback_calls"]=vector_calls;
    info["matmat_callback_calls"]=block_calls;
    return out;
}

inline void bind_davidson(pybind11::module_& module) {
    bind_davidson_type<double>(module);
    bind_davidson_type<Complex>(module);
    namespace py = pybind11;
    module.def("davidson_operator", [](py::object action,
        py::array_t<double,py::array::c_style> diagonal, int roots, double tolerance,
        int iterations, int space, std::size_t memory_limit, py::object guess,
        py::object matmat, bool complex_values) {
        if (complex_values)
            return davidson_operator<Complex>(action,diagonal,roots,tolerance,iterations,
                space,memory_limit,guess,matmat);
        return davidson_operator<double>(action,diagonal,roots,tolerance,iterations,
            space,memory_limit,guess,matmat);
    },py::arg("action"),py::arg("diagonal").noconvert(),py::arg("roots"),
      py::arg("tolerance")=1.e-10,py::arg("iterations")=200,py::arg("space")=64,
      py::arg("memory_limit")=536870912,py::arg("guess")=py::none(),
      py::arg("matmat")=py::none(),py::arg("complex_values")=true,
      "Matrix-free adapter to the same block algorithm documented by davidson. "
      "Requires a Hermitian action; returns partial Ritz pairs and diagnostics. "
      "Callbacks receive owned arrays of shape (n,) or (n, columns). "
      "The GIL is released for solver work and acquired for Python callbacks.");
}

}
