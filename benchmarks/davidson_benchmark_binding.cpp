// Benchmark-only adapter to the existing multi-root solver; no new algorithm.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "davidson.hpp"
#include "davidson_binding.hpp"
#include "../pyqed/narg/conditional_action.hpp"
namespace py = pybind11;

template <typename Scalar>
void bind_solve(py::module_& module) {
    module.def("solve", [](py::array_t<Scalar, py::array::c_style | py::array::forcecast> matrix,
                           int roots, double tolerance, int iterations, int space,
                           bool batched, bool reuse_workspace) {
        const auto n = matrix.shape(0);
        if (matrix.ndim() != 2 || matrix.shape(1) != n || roots < 1 || roots > n)
            throw std::invalid_argument("Expected a square matrix and valid root count");
        const Scalar* data = matrix.data();
        std::vector<double> diagonal(n);
        for (int i=0; i<n; ++i) diagonal[i] = std::real(data[i*n+i]);
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {return diagonal[a]<diagonal[b];});
        std::vector<std::vector<Scalar>> guesses(roots, std::vector<Scalar>(n, 0.));
        for (int i=0; i<roots; ++i) guesses[i][order[i]]=1.;
        pyqed::linalg::BlockDavidsonWorkspace<Scalar> workspace;
        if (reuse_workspace) workspace.ensure(n+7, n, roots);
        pyqed::linalg::BlockDavidsonResult result;
        {
            py::gil_scoped_release release;
            auto action = [&](const Scalar* x, Scalar* y, std::size_t dim,
                                                       std::size_t columns, std::size_t stride) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
                    pyqed::linalg::davidson_gemm(101, 111, 112, columns, dim, dim,
                        1., x, stride, data, dim, 0., y, stride);
#else
                    for (std::size_t c=0;c<columns;++c) for (std::size_t i=0;i<dim;++i) {
                        Scalar sum=0.;
                        for (std::size_t j=0;j<dim;++j) sum+=data[i*dim+j]*x[c*stride+j];
                        y[c*stride+i]=sum;
                    }
#endif
                };
            if (batched)
                result = pyqed::linalg::block_davidson(diagonal, guesses, roots, tolerance,
                    iterations, space, true, workspace, action);
            else
                result = pyqed::linalg::block_davidson(diagonal, guesses, roots, tolerance,
                    iterations, space, true, workspace, [&](const Scalar* x, Scalar* y, std::size_t dim) {
                        action(x,y,dim,1,dim);
                    });
        }
        py::array_t<Scalar> vectors({n, static_cast<py::ssize_t>(roots)});
        auto out = vectors.template mutable_unchecked<2>();
        for (int j=0; j<roots; ++j)
            for (int i=0; i<n; ++i) {
                if constexpr (std::is_same_v<Scalar,double>) out(i,j)=result.vectors[j][i].real();
                else out(i,j)=result.vectors[j][i];
            }
        py::dict info;
        info["iterations"]=result.iterations;
        info["converged"]=result.converged;
        info["restarts"]=result.restarts;
        info["restart_history_vectors"]=result.restart_history_vectors;
        info["basis_size"]=result.basis_size;
        info["matvec_calls"]=result.matvec_calls;
        info["workspace_reused"]=result.workspace_reused;
        info["action_seconds"]=result.action_seconds;
        info["projection_seconds"]=result.projection_seconds;
        info["diagonalization_seconds"]=result.diagonalization_seconds;
        info["ritz_seconds"]=result.ritz_seconds;
        info["correction_seconds"]=result.correction_seconds;
        info["restart_seconds"]=result.restart_seconds;
        return py::make_tuple(result.energies, vectors, info);
    }, py::arg("matrix"), py::arg("roots"), py::arg("tolerance"), py::arg("iterations"),
       py::arg("space"), py::arg("batched")=true, py::arg("reuse_workspace")=false);
}

PYBIND11_MODULE(davidson_benchmark_binding, module) {
    pyqed::linalg::bind_davidson(module);
    pyqed::linalg::bind_conditional_action(module);
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
    module.attr("cholesky_available")=true;
#else
    module.attr("cholesky_available")=false;
#endif
    module.def("orthogonalize", [](py::array_t<double,py::array::c_style> matrix) {
        if (matrix.ndim()!=2) throw std::invalid_argument("Expected matrix");
        const std::size_t n=matrix.shape(0), k=matrix.shape(1), stride=n+7;
        std::vector<double> columns(stride*k,123.);
        auto input=matrix.unchecked<2>();
        for (std::size_t j=0;j<k;++j) for (std::size_t i=0;i<n;++i) columns[j*stride+i]=input(i,j);
        bool accepted=pyqed::linalg::davidson_cholesky_qr(columns.data(),n,k,stride);
        py::array_t<double> output({matrix.shape(0),matrix.shape(1)});
        auto out=output.mutable_unchecked<2>();
        for (std::size_t j=0;j<k;++j) {
            for (std::size_t i=0;i<n;++i) out(i,j)=columns[j*stride+i];
            for (std::size_t i=n;i<stride;++i)
                if (columns[j*stride+i]!=123.) throw std::runtime_error("QR touched padding");
        }
        return py::make_tuple(output,accepted);
    });
    bind_solve<double>(module);
    bind_solve<pyqed::linalg::Complex>(module);
}
