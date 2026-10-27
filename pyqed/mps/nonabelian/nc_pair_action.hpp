#pragma once
#include "../../linalg/davidson.hpp"
#include <climits>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyqed::su2 {
struct NCPairOperator {
    std::vector<std::int64_t> offsets, columns;
    std::vector<std::complex<double>> values;
    void validate(std::int64_t rows,std::int64_t cols) const {
        if (offsets.size()!=rows+1 || offsets.front()!=0 || offsets.back()!=values.size()
            || columns.size()!=values.size()) throw std::invalid_argument("Invalid sparse NC half");
        for (std::size_t i=1;i<offsets.size();++i)
            if (offsets[i]<offsets[i-1]) throw std::invalid_argument("Invalid sparse NC row offsets");
        for (auto j:columns) if (j<0 || j>=cols) throw std::invalid_argument("Invalid sparse NC column");
    }
};
// Retain references to small boundary blocks, then compress one operator
// half at a time. This avoids simultaneous hash tables for every channel.
class NCPairBuilder {
    using Complex=std::complex<double>;
    struct Block {
        std::int64_t row,col,m,n,row_stride,col_stride,leading;
        const Complex* values;
        Complex coefficient;
    };
    std::int64_t rows_,cols_;
    std::vector<Block> blocks_;
public:
    NCPairBuilder(std::int64_t rows,std::int64_t cols):rows_(rows),cols_(cols) {
        if (rows<1 || cols<1) throw std::invalid_argument("Invalid NC half dimensions");
    }
    void add_block(std::int64_t row,std::int64_t col,std::int64_t m,std::int64_t n,
                   std::int64_t row_stride,std::int64_t col_stride,std::int64_t leading,
                   const Complex* values,Complex coefficient) {
        if (row<0 || col<0 || m<1 || n<1 || row_stride<1 || col_stride<1
            || row+(m-1)*row_stride>=rows_ || col+(n-1)*col_stride>=cols_)
            throw std::invalid_argument("NC half block exceeds operator dimensions");
        blocks_.push_back({row,col,m,n,row_stride,col_stride,leading,values,coefficient});
    }
    std::size_t size() const {return blocks_.size();}
    NCPairOperator compressed() {
        std::unordered_map<std::int64_t,Complex> entries;
        for (const auto& block:blocks_)
            for (std::int64_t i=0;i<block.m;++i) for (std::int64_t j=0;j<block.n;++j) {
                auto value=block.coefficient*block.values[i*block.leading+j];
                if (value!=Complex{}) entries[(block.row+i*block.row_stride)*cols_+block.col+j*block.col_stride]+=value;
            }
        std::vector<Block>().swap(blocks_);
        NCPairOperator result;
        std::vector<std::pair<std::int64_t,Complex>> ordered;
        ordered.reserve(entries.size());
        for (const auto& entry:entries) if (entry.second!=Complex{}) ordered.push_back(entry);
        std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        result.offsets.assign(rows_+1,0);
        for (const auto& entry:ordered) {
            ++result.offsets[entry.first/cols_+1];
            result.columns.push_back(entry.first%cols_);result.values.push_back(entry.second);
        }
        for (std::int64_t i=0;i<rows_;++i) result.offsets[i+1]+=result.offsets[i];
        return result;
    }
};
// Matrix-free sum of direct NC channel halves in orthonormal pair coordinates.
class NCPairAction {
    using Complex = std::complex<double>;
    std::vector<std::int64_t> offsets_, rows_, cols_, outputs_, inputs_;
    std::vector<NCPairOperator> left_, right_;
    pyqed::linalg::GeneralizedDavidsonWorkspace workspace_;
    std::vector<std::vector<std::size_t>> output_routes_;
    int threads_;
    mutable std::uint64_t parallel_calls_ = 0;
    mutable int threads_used_ = 1;
public:
    NCPairAction(const std::vector<std::int64_t>& rows,
                 const std::vector<std::int64_t>& cols,
                 const std::vector<std::int64_t>& outputs,
                 const std::vector<std::int64_t>& inputs,
                 const std::vector<NCPairOperator>& left,
                 const std::vector<NCPairOperator>& right, int threads)
        : rows_(rows), cols_(cols), outputs_(outputs), inputs_(inputs), left_(left), right_(right), threads_(threads) {
        if (threads < 1) throw std::invalid_argument("NC pair threads must be positive");
        if (rows.empty() || rows.size()!=cols.size() || outputs.size()!=inputs.size()
            || outputs.size()!=left.size() || outputs.size()!=right.size())
            throw std::invalid_argument("Invalid NC pair layout");
        offsets_.push_back(0);
        for (std::size_t i=0;i<rows.size();++i) {
            if (rows[i]<=0 || cols[i]<=0 || rows[i]>INT_MAX || cols[i]>INT_MAX)
                throw std::invalid_argument("Invalid NC pair shape");
            offsets_.push_back(offsets_.back()+rows[i]*cols[i]);
        }
        output_routes_.resize(rows.size());
        for (std::size_t t=0;t<outputs.size();++t) {
            auto a=outputs[t],b=inputs[t];
            if (a<0 || b<0 || a>=rows.size() || b>=rows.size())
                throw std::invalid_argument("Invalid NC pair channel");
            left[t].validate(rows[a],rows[b]);right[t].validate(cols[a],cols[b]);
            output_routes_[a].push_back(t);
        }
    }
    std::vector<Complex> apply(const std::vector<Complex>& x) const {
        if (x.size()!=offsets_.back()) throw std::invalid_argument("NC pair vector size");
        std::vector<Complex> y(x.size());
        // Separate output sectors never share storage. Preserve the original
        // summation order within each sector for reproducible thread counts.
        int workers = 1;
#ifdef _OPENMP
        if (!omp_in_parallel()) workers = std::min<int>(threads_, rows_.size());
#endif
        std::size_t scratch = 0;
        for (std::size_t t=0;t<inputs_.size();++t)
            scratch=std::max(scratch,static_cast<std::size_t>(rows_[outputs_[t]]*cols_[inputs_[t]]));
        std::vector<std::vector<Complex>> workspaces(workers,std::vector<Complex>(scratch));
        if (workers>1) ++parallel_calls_;
#pragma omp parallel num_threads(workers) if(workers>1)
        {
            int worker = 0;
#ifdef _OPENMP
            worker=omp_get_thread_num();
#pragma omp single
            threads_used_=omp_get_num_threads();
#endif
            auto& work=workspaces[worker];
#pragma omp for schedule(static)
            for (std::int64_t a=0;a<static_cast<std::int64_t>(rows_.size());++a) {
                for (auto t : output_routes_[a]) {
                    auto b=inputs_[t];
                    const auto m=rows_[a],n=cols_[b],p=cols_[a];
                    const auto& l=left_[t];const auto& r=right_[t];
                    for (std::int64_t i=0;i<m;++i) {
                        if (l.offsets[i]==l.offsets[i+1]) continue;
                        std::fill(work.begin()+i*n,work.begin()+(i+1)*n,Complex{});
                        for (auto k=l.offsets[i];k<l.offsets[i+1];++k) {
                            const auto* input=x.data()+offsets_[b]+l.columns[k]*n;
                            for (std::int64_t j=0;j<n;++j) work[i*n+j]+=l.values[k]*input[j];
                        }
                    }
                    for (std::int64_t i=0;i<m;++i) {
                        if (l.offsets[i]==l.offsets[i+1]) continue;
                        for (std::int64_t j=0;j<p;++j) {
                        if (r.offsets[j]==r.offsets[j+1]) continue;
                        Complex value{};
                        for (auto k=r.offsets[j];k<r.offsets[j+1];++k)
                            value+=work[i*n+r.columns[k]]*r.values[k];
                        y[offsets_[a]+i*p+j]+=value;
                        }
                    }
                }
            }
        }
        return y;
    }
    int threads_used() const { return threads_used_; }
    std::uint64_t parallel_calls() const { return parallel_calls_; }
    std::vector<Complex> diagonal() const {
        std::vector<Complex> d(offsets_.back());
        for (std::size_t t=0;t<inputs_.size();++t) if (inputs_[t]==outputs_[t]) {
            auto a=inputs_[t];
            const auto& l=left_[t];const auto& r=right_[t];
            for (std::int64_t i=0;i<rows_[a];++i) {
                Complex dl{};
                for (auto k=l.offsets[i];k<l.offsets[i+1];++k) if (l.columns[k]==i) dl+=l.values[k];
                for (std::int64_t j=0;j<cols_[a];++j) {
                    Complex dr{};
                    for (auto k=r.offsets[j];k<r.offsets[j+1];++k) if (r.columns[k]==j) dr+=r.values[k];
                    d[offsets_[a]+i*cols_[a]+j]+=dl*dr;
                }
            }
        }
        return d;
    }
    pyqed::linalg::DavidsonResult solve(std::vector<Complex> guess,double tolerance) {
        const auto n=offsets_.back();
        if (guess.size()!=n || !std::isfinite(tolerance) || tolerance<=0)
            throw std::invalid_argument("Invalid NC pair solve");
        if (pyqed::linalg::norm2(guess)<1e-14) guess.assign(n,1.);
        pyqed::linalg::DavidsonResult result;
        if (n<=2) {
            std::vector<Complex> matrix(n*n),unit(n);
            for (std::size_t j=0;j<n;++j) {
                unit[j]=1.;auto column=apply(unit);unit[j]=0.;
                for (std::size_t i=0;i<n;++i) matrix[i*n+j]=column[i];
            }
            auto pair=pyqed::linalg::lowest_projected_eigenpair(matrix,n);
            result.energy=pair.value;result.vector=std::move(pair.vector);
            auto residual=apply(result.vector);
            for (std::size_t i=0;i<n;++i) residual[i]-=result.energy*result.vector[i];
            result.residual_norm=pyqed::linalg::norm2(residual);
            result.converged=result.accepted=result.residual_norm<=tolerance;
            result.iterations=1;result.matvec_calls=n+1;
        } else {
            result=pyqed::linalg::generalized_davidson(diagonal(),std::vector<Complex>(n,1.),
                std::move(guess),std::max(1e-14,tolerance*tolerance),tolerance,1e-14,
                std::max(300,static_cast<int>(10*n)),std::min<int>(n,48),false,workspace_,
                [this](const std::vector<Complex>& x){return apply(x);},
                [](const std::vector<Complex>& x){return x;});
        }
        if (!result.converged) throw std::runtime_error("NC pair Davidson did not converge");
        return result;
    }
};
}
