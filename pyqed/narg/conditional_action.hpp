#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include "../linalg/davidson.hpp"

namespace pyqed::linalg {
// Persistent references to reduced terms; no total Hamiltonian is formed.
template <typename Scalar>
class ConditionalAction {
    using Array = pybind11::array_t<Scalar, pybind11::array::c_style>;
    std::vector<Array> matrices_;
    Array blocks_;
    std::size_t dimension_, block_size_ = 0, count_ = 0;
public:
    ConditionalAction(pybind11::list matrices, pybind11::object blocks,
                      std::size_t dimension): dimension_(dimension) {
        if (!dimension || dimension>std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid conditional action dimension");
        for (auto value: matrices) {
            auto matrix = Array::ensure(value);
            if (!matrix || matrix.ndim()!=2 || matrix.shape(0)!=dimension || matrix.shape(1)!=dimension)
                throw std::invalid_argument("Invalid reduced matrix shape or dtype");
            matrices_.push_back(std::move(matrix));
        }
        if (!blocks.is_none()) {
            blocks_ = Array::ensure(blocks);
            if (!blocks_ || blocks_.ndim()!=3 || blocks_.shape(1)!=blocks_.shape(2)
                || blocks_.shape(0)*blocks_.shape(1)!=dimension)
                throw std::invalid_argument("Invalid conditional block shape or dtype");
            count_=blocks_.shape(0); block_size_=blocks_.shape(1);
        }
    }

    pybind11::array_t<Scalar> matmat(
            pybind11::array_t<Scalar,pybind11::array::f_style> input) const {
        namespace py = pybind11;
        if (input.ndim()!=2 || input.shape(0)!=dimension_
                || input.shape(1)>std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid conditional trial block");
        const std::size_t columns=input.shape(1), n=dimension_;
        py::array_t<Scalar> output(
            std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(columns)},
            std::vector<py::ssize_t>{sizeof(Scalar),static_cast<py::ssize_t>(n*sizeof(Scalar))});
        auto* y=output.mutable_data(); const auto* x=input.data();
        py::gil_scoped_release release;
        std::fill(y,y+n*columns,Scalar{});
        auto apply = [&](const Scalar* matrix, std::size_t size, std::size_t offset) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
            // Row-major views of X.T and Y.T avoid transposing trial blocks.
            davidson_gemm(101,111,112,columns,size,size,1.,x+offset,n,
                matrix,size,1.,y+offset,n);
#else
            for (std::size_t c=0;c<columns;++c)
                for (std::size_t i=0;i<size;++i) {
                    Scalar value{};
                    for (std::size_t j=0;j<size;++j)
                        value+=matrix[i*size+j]*x[c*n+offset+j];
                    y[c*n+offset+i]+=value;
                }
#endif
        };
        if (columns) {
            for (const auto& matrix:matrices_) apply(matrix.data(),n,0);
            for (std::size_t branch=0;branch<count_;++branch)
                apply(blocks_.data()+branch*block_size_*block_size_,block_size_,branch*block_size_);
        }
        return output;
    }
};

// Real cross terms sharing U: lift once, accumulate in the incoming basis,
// restrict once. Parent operators remain branch diagonal, not expanded.
class FactorizedAction {
    using Array = pybind11::array_t<double, pybind11::array::c_style>;
    Array bases_;
    struct Term { double coefficient; Array blocks, local; int size, count; };
    std::vector<Term> terms_;
    int branches_, incoming_, keep_, batch_;
    std::vector<double> lifted_, mixed_, accumulated_;
    std::mutex mutex_;
    static void gemm(bool ta, bool tb, int m, int n, int k, double alpha,
                     const double* a, int lda, const double* b, int ldb,
                     double beta, double* c, int ldc) {
#if defined(__APPLE__) && !defined(PYQED_DAVIDSON_PORTABLE)
        davidson_gemm(101,ta?112:111,tb?112:111,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
#else
        for (int i=0;i<m;++i) for (int j=0;j<n;++j) {
            double sum=0;
            for (int l=0;l<k;++l) sum+=(ta?a[l*lda+i]:a[i*lda+l])*(tb?b[j*ldb+l]:b[l*ldb+j]);
            c[i*ldc+j]=alpha*sum+(beta?beta*c[i*ldc+j]:0.);
        }
#endif
    }
public:
    FactorizedAction(Array bases, pybind11::list terms, int batch): bases_(bases), batch_(batch) {
        if (bases.ndim()!=3 || batch<1) throw std::invalid_argument("Invalid factorized bases or batch");
        for (int axis=0;axis<3;++axis)
            if (bases.shape(axis)<1 || bases.shape(axis)>std::numeric_limits<int>::max())
                throw std::invalid_argument("Invalid factorized dimension");
        branches_=bases.shape(0); incoming_=bases.shape(1); keep_=bases.shape(2);
        if (static_cast<long long>(branches_)*keep_>std::numeric_limits<int>::max()
            || incoming_>std::numeric_limits<int>::max()/batch_/branches_)
            throw std::invalid_argument("Factorized action exceeds BLAS dimensions");
        for (auto value:terms) {
            auto tuple=pybind11::cast<pybind11::tuple>(value);
            if (tuple.size()!=3) throw std::invalid_argument("Invalid factorized term");
            auto blocks=Array::ensure(tuple[1]), local=Array::ensure(tuple[2]);
            double coefficient=pybind11::cast<double>(tuple[0]);
            if (!std::isfinite(coefficient) || !blocks || !local || blocks.ndim()!=3 || local.ndim()!=2
                || blocks.shape(1)<1 || blocks.shape(1)!=blocks.shape(2)
                || blocks.shape(0)*blocks.shape(1)!=incoming_
                || local.shape(0)!=branches_ || local.shape(1)!=branches_)
                throw std::invalid_argument("Invalid factorized term dimensions");
            terms_.push_back({coefficient,blocks,local,static_cast<int>(blocks.shape(1)),static_cast<int>(blocks.shape(0))});
        }
        auto size=static_cast<std::size_t>(branches_)*incoming_*batch_;
        lifted_.resize(size); mixed_.resize(size); accumulated_.resize(size);
    }
    pybind11::array_t<double> matmat(Array input) {
        int dimension=branches_*keep_;
        if (input.ndim()!=2 || input.shape(0)!=dimension || input.shape(1)>std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid factorized trial block");
        int columns=input.shape(1);
        pybind11::array_t<double> output({dimension,columns});
        const auto* x=input.data(); auto* y=output.mutable_data();
        pybind11::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        for (int start=0;start<columns;) {
            int width=std::min(batch_,columns-start), stride=incoming_*width;
            for (int a=0;a<branches_;++a)
                gemm(false,false,incoming_,width,keep_,1.,bases_.data()+static_cast<std::size_t>(a)*incoming_*keep_,keep_,
                     x+static_cast<std::size_t>(a)*keep_*columns+start,columns,0.,lifted_.data()+a*stride,width);
            std::fill(accumulated_.begin(),accumulated_.begin()+static_cast<std::size_t>(branches_)*stride,0.);
            for (const auto& term:terms_) for (int adjoint=0;adjoint<2;++adjoint) {
                gemm(adjoint,false,branches_,stride,branches_,1.,term.local.data(),branches_,
                     lifted_.data(),stride,0.,mixed_.data(),stride);
                for (int a=0;a<branches_;++a) for (int b=0;b<term.count;++b) {
                    auto offset=static_cast<std::size_t>(a)*stride+b*term.size*width;
                    gemm(adjoint,false,term.size,width,term.size,.5*term.coefficient,
                         term.blocks.data()+static_cast<std::size_t>(b)*term.size*term.size,term.size,
                         mixed_.data()+offset,width,1.,accumulated_.data()+offset,width);
                }
            }
            for (int a=0;a<branches_;++a)
                gemm(true,false,keep_,width,incoming_,1.,bases_.data()+static_cast<std::size_t>(a)*incoming_*keep_,keep_,
                     accumulated_.data()+a*stride,width,0.,y+static_cast<std::size_t>(a)*keep_*columns+start,columns);
            start+=width;
        }
        return output;
    }
};

inline void bind_conditional_action(pybind11::module_& module) {
    namespace py = pybind11;
    py::class_<FactorizedAction>(module,"FactorizedAction",py::module_local())
        .def(py::init<pybind11::array_t<double,pybind11::array::c_style>,py::list,int>())
        .def("matmat",&FactorizedAction::matmat);
    py::class_<ConditionalAction<double>>(module,"RealConditionalAction",py::module_local())
        .def(py::init<py::list,py::object,std::size_t>())
        .def("matmat",&ConditionalAction<double>::matmat);
    py::class_<ConditionalAction<Complex>>(module,"ComplexConditionalAction",py::module_local())
        .def(py::init<py::list,py::object,std::size_t>())
        .def("matmat",&ConditionalAction<Complex>::matmat);
}
}
