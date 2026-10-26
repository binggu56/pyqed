// Diagonal Dyson pole selection and continuation references; orbital OpenMP.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static bool openmp_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

struct Equation {
    const double *poles, *residue;
    py::ssize_t count;
    double energy, shift, eta, coupling;
    double residual(double w) const {
        double value = shift;
        const double eta2 = eta*eta;
        for (py::ssize_t k = 0; k < count; ++k) {
            if (residue[k] == 0.) continue;
            const double x = w-poles[k];
            value += eta == 0. ? residue[k]/x : residue[k]*x/(x*x+eta2);
        }
        return w-energy-coupling*value;
    }
    void evaluate(double w, double &residual, double &slope) const {
        double value = shift, derivative = 0.;
        const double eta2 = eta*eta;
        for (py::ssize_t k = 0; k < count; ++k) {
            if (residue[k] == 0.) continue;
            const double x = w-poles[k], denominator = x*x+eta2;
            if(eta==0.) {
                value += residue[k]/x;
                derivative -= residue[k]/x/x;
            } else {
                value += residue[k]*x/denominator;
                derivative += residue[k]*(eta2-x*x)/(denominator*denominator);
            }
        }
        residual = w-energy-coupling*value;
        slope = 1.-coupling*derivative;
    }
};

static bool bracket_root(const Equation &eq, double left, double right, double &root) {
    double x = .5*(left+right);
    for (int iteration = 0; iteration < 120; ++iteration) {
        double f, slope;
        eq.evaluate(x, f, slope);
        if (!std::isfinite(f)) return false;
        if (std::abs(f) <= 1e-11) {
            root = x;
            return slope > 0.;
        }
        if (f < 0.) left = x; else right = x;
        const double trial = x-f/slope;
        // A bracket safeguards Newton; a central step guarantees contraction.
        const double margin = .1*(right-left);
        x = slope > 0. && trial > left+margin && trial < right-margin
            ? trial : .5*(left+right);
    }
    return false;
}

static bool continue_root(Equation &eq, double &root, double &weight, double &error) {
    root = eq.energy;
    double coupling = 0., increment = .125;
    while (coupling < 1.) {
        eq.coupling = std::min(1., coupling+increment);
        double f, slope;
        eq.evaluate(root, f, slope);
        bool valid = std::abs(f) <= 1e-9 && slope > 0.;
        double candidate = root, left = root, right = root;
        double fl = f, fr = f;
        const double scale = std::log(2.*std::max(1., std::abs(root))/1e-7);
        for (int shell = 0; shell < 120 && !valid; ++shell) {
            const double radius = 1e-7*std::exp(scale*shell/119.);
            const double far_left = eq.residual(root-radius), far_right = eq.residual(root+radius);
            double a, b;
            const bool found_left = far_left <= 0. && fl >= 0. && bracket_root(eq, root-radius, left, a);
            const bool found_right = fr <= 0. && far_right >= 0. && bracket_root(eq, right, root+radius, b);
            if (found_left || found_right) {
                candidate = found_left && (!found_right || std::abs(a-root) <= std::abs(b-root)) ? a : b;
                valid = true;
            }
            left = root-radius; right = root+radius;
            fl = far_left; fr = far_right;
        }
        if (valid) {
            root = candidate; coupling = eq.coupling;
            increment = std::min(.125, 2.*increment);
        } else {
            increment *= .5;
            if (increment < 1e-4) return false;
        }
    }
    double f, slope;
    eq.evaluate(root, f, slope);
    weight = 1./slope; error = std::abs(f);
    return std::isfinite(weight) && weight > 0. && error <= 1e-9;
}

static py::tuple solve(Array poles, Array residues, Array energies, Array shifts,
                       double eta, int threads) {
    if (poles.ndim() != 1 || residues.ndim() != 2 || energies.ndim() != 1 || shifts.ndim() != 1
        || residues.shape(1) != poles.size() || residues.shape(0) != energies.size()
        || shifts.size() != energies.size() || threads < 1 || !std::isfinite(eta) || eta < 0.)
        throw std::invalid_argument("Invalid Dyson batch dimensions, eta, or threads");
    if (threads > 1 && !openmp_available())
        throw std::runtime_error("GW Dyson extension was built without OpenMP");
    const auto n = energies.size(), count = poles.size();
    const double *p = poles.data(), *r = residues.data(), *e = energies.data(), *s = shifts.data();
    for (py::ssize_t k = 0; k < count; ++k)
        if (!std::isfinite(p[k])) throw std::invalid_argument("Nonfinite Dyson pole");
    for (py::ssize_t k = 0; k < residues.size(); ++k)
        if (!std::isfinite(r[k]) || r[k] < 0.) throw std::invalid_argument("Dyson residues must be finite and nonnegative");
    for (py::ssize_t k = 0; k < n; ++k)
        if (!std::isfinite(e[k]) || !std::isfinite(s[k])) throw std::invalid_argument("Nonfinite Dyson energy or shift");
    Array roots(n), weights(n), residuals(n);
    double *out = roots.mutable_data(), *z = weights.mutable_data(), *err = residuals.mutable_data();
    std::vector<int> success(n);
    int used = 1;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(std::min<int>(threads, std::max<py::ssize_t>(1, n))) if(threads > 1 && n > 1)
        {
#pragma omp single
            used = omp_get_num_threads();
#pragma omp for schedule(dynamic, 1)
#endif
            for (py::ssize_t orbital = 0; orbital < n; ++orbital) {
                Equation equation{p, r+orbital*count, count, e[orbital], s[orbital], eta, 0.};
                success[orbital] = continue_root(equation, out[orbital], z[orbital], err[orbital]);
            }
#ifdef _OPENMP
        }
#endif
    }
    for (py::ssize_t orbital = 0; orbital < n; ++orbital)
        if (!success[orbital]) throw std::runtime_error("Could not track a positive-weight quasiparticle root in batch orbital " + std::to_string(orbital));
    return py::make_tuple(roots, weights, residuals, used);
}

// Largest-weight pole of a diagonal arrowhead Green function, eta -> 0.
// Positive residues make the secular equation monotone between its poles.
struct Interval { double bound, lo, hi; size_t index; };

static double weight_bound(const std::vector<double>& p, const std::vector<double>& r,
                           const std::vector<long double>& prefix, size_t i,
                           double lo, double hi, double center) {
    const size_t n=p.size();
    const double left=i ? r[i-1] : 0., right=i<n ? r[i] : 0.;
    const double width=hi-lo;
    if (width<=0.) return 0.;
    long double lower=0., upper=0., derivative=std::pow(std::cbrt(left)+std::cbrt(right),3)/(width*width);
    // Conservative prefix-summation error; bounds must never prune a possible winner.
    const long double error=4*std::numeric_limits<long double>::epsilon()*n*prefix[n];
    auto group=[&](size_t first,size_t end) {
        const long double mass=prefix[end]-prefix[first];
        const long double low=std::max(0.L,mass-error), high=mass+error;
        const double dl=hi-p[first], du=lo-p[end-1];
        lower+=(dl>0 ? low : high)/dl;
        upper+=(du>0 ? high : low)/du;
        const double far=std::max(std::abs(lo-p[end-1]),std::abs(hi-p[first]));
        derivative+=low/(far*far);
    };
    size_t end=i ? i-1 : 0, step=1;
    while(end) { size_t first=end>step ? end-step : 0; group(first,end);end=first;step=std::max(step+1,size_t(step*1.4)); }
    size_t first=std::min(n,i+1);step=1;
    while(first<n) {end=std::min(n,first+step);group(first,end);first=end;step=std::max(step+1,size_t(step*1.4));}
    double bound=1./(1.+double(derivative));
    const double fmin=lo-center-double(upper), fmax=hi-center-double(lower);
    if(fmin>0.) bound=std::min(bound,left/(left+fmin*fmin));
    if(fmax<0.) bound=std::min(bound,right/(right+fmax*fmax));
    return std::min(1.,bound*(1.+1e-10)+1e-14);
}

static bool dominant_root(const double* poles,const double* residues,size_t count,
                          double energy,double shift,double tol,double& root,double& weight,
                          double& error,double& excluded,int& searched,double& bracket_lo,double& bracket_hi) {
    std::vector<std::pair<double,double>> entries;
    for(size_t k=0;k<count;++k) if(residues[k]>tol*tol/(100.*std::max<size_t>(1,count))) entries.emplace_back(poles[k],residues[k]);
    std::sort(entries.begin(),entries.end());
    std::vector<double> p,r;
    for(auto entry:entries) {
        if(!p.empty() && entry.first==p.back()) r.back()+=entry.second;
        else {p.push_back(entry.first);r.push_back(entry.second);}
    }
    searched=0;excluded=0.;
    if(p.empty()) {root=energy+shift;weight=1.;error=0.;bracket_lo=-INFINITY;bracket_hi=INFINITY;return true;}
    const size_t n=p.size();
    std::vector<long double> prefix(n+1,0.);
    for(size_t k=0;k<n;++k) prefix[k+1]=prefix[k]+r[k];
    const double center=energy+shift, radius=std::sqrt(double(prefix[n]))+1.;
    const double low=std::min(center,p.front())-radius, high=std::max(center,p.back())+radius;
    std::vector<Interval> intervals;
    for(size_t i=0;i<=n;++i) {
        double lo=i ? p[i-1] : low, hi=i<n ? p[i] : high;
        intervals.push_back({weight_bound(p,r,prefix,i,lo,hi,center),lo,hi,i});
    }
    std::sort(intervals.begin(),intervals.end(),[](const Interval&a,const Interval&b){return a.bound>b.bound;});
    Equation eq{p.data(),r.data(),py::ssize_t(n),energy,shift,0.,1.};
    weight=0.;root=NAN;error=INFINITY;
    double unresolved=0.;
    for(const auto& interval:intervals) {
        if(interval.bound<=weight) {excluded=interval.bound;break;}
        double lo=std::nextafter(interval.lo,interval.hi), hi=std::nextafter(interval.hi,interval.lo);
        double fl=eq.residual(lo), fh=eq.residual(hi);
        if(!(fl<=0. && fh>=0.)) {
            // An unresolved root closer to a pole than one ulp has vanishing weight.
            const size_t i=interval.index;
            double upper=1.;
            if(fl>0. && i>0) upper=std::pow(lo-interval.lo,2)/r[i-1];
            if(fh<0. && i<n) upper=std::pow(hi-interval.hi,2)/r[i];
            unresolved=std::max(unresolved,std::min(upper,interval.bound));
            continue;
        }
        double x=std::clamp(center,lo,hi), f=0.,slope=0.;
        bool converged=false;
        for(int iteration=0;iteration<180;++iteration) {
            eq.evaluate(x,f,slope);
            if(std::abs(f)<=tol) {converged=true;break;}
            if(f<0.)lo=x;else hi=x;
            double trial=x-f/slope, margin=.05*(hi-lo);
            double next=trial>lo+margin && trial<hi-margin ? trial : .5*(lo+hi);
            if(next==x) break;
            x=next;
        }
        ++searched;
        const double z=1./slope;
        if(!converged) {
            long double lower=0.;
            for(size_t k=0;k<n;++k) {
                const double d=std::max(std::abs(lo-p[k]),std::abs(hi-p[k]));
                lower+=r[k]/(d*d);
            }
            const double upper=std::min(interval.bound,1./(1.+double(lower))*(1.+1e-10));
            unresolved=std::max(unresolved,upper);
        }
        if(converged && z>weight) {
            root=x;weight=z;error=std::abs(f);bracket_lo=interval.lo;bracket_hi=interval.hi;
        }
    }
    excluded=std::max(excluded,unresolved);
    return std::isfinite(root) && weight>0. && unresolved<=weight;
}

static py::tuple dominant(Array poles,Array residues,Array energies,Array shifts,int threads,double tol) {
    if(poles.ndim()!=1 || residues.ndim()!=2 || energies.ndim()!=1 || shifts.ndim()!=1
       || residues.shape(1)!=poles.size() || residues.shape(0)!=energies.size()
       || shifts.size()!=energies.size() || threads<1 || !std::isfinite(tol) || tol<=0.)
        throw std::invalid_argument("Invalid dominant-root inputs");
    if(threads>1 && !openmp_available()) throw std::runtime_error("GW Dyson extension was built without OpenMP");
    for(auto array:{poles,residues,energies,shifts}) for(py::ssize_t k=0;k<array.size();++k)
        if(!std::isfinite(array.data()[k])) throw std::invalid_argument("Nonfinite dominant-root input");
    for(py::ssize_t k=0;k<residues.size();++k) if(residues.data()[k]<0.) throw std::invalid_argument("Negative Dyson residue");
    auto n=energies.size();Array roots(n),weights(n),errors(n),excluded(n),brackets({n,py::ssize_t(2)});
    py::array_t<int> searched(n);std::vector<int> success(n);int used=1;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(std::min<int>(threads,std::max<py::ssize_t>(1,n))) if(threads>1 && n>1)
        {
#pragma omp single
            used=omp_get_num_threads();
#pragma omp for schedule(dynamic,1)
#endif
            for(py::ssize_t i=0;i<n;++i) success[i]=dominant_root(poles.data(),residues.data()+i*poles.size(),
                poles.size(),energies.data()[i],shifts.data()[i],tol,roots.mutable_data()[i],weights.mutable_data()[i],
                errors.mutable_data()[i],excluded.mutable_data()[i],searched.mutable_data()[i],
                brackets.mutable_data()[2*i],brackets.mutable_data()[2*i+1]);
#ifdef _OPENMP
        }
#endif
    }
    for(py::ssize_t i=0;i<n;++i) if(!success[i]) throw std::runtime_error("Dominant Dyson root unresolved in batch orbital "+std::to_string(i)+"; selected weight="+std::to_string(weights.data()[i])+" unresolved bound="+std::to_string(excluded.data()[i])+" intervals="+std::to_string(searched.data()[i]));
    return py::make_tuple(roots,weights,errors,used,excluded,searched,brackets);
}

PYBIND11_MODULE(_dyson, module) {
    module.def("dominant", &dominant, py::arg("poles"), py::arg("residues"), py::arg("energies"),
               py::arg("shifts"), py::arg("threads")=1, py::arg("tol")=1e-9);
    module.doc() = "Dominant spectral Dyson poles and continuation references with orbital OpenMP";
    module.def("openmp_available", &openmp_available);
    module.def("solve", &solve, py::arg("poles"), py::arg("residues"), py::arg("energies"),
               py::arg("shifts"), py::arg("eta"), py::arg("threads") = 1);
}
