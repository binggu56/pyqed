#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

#include "_rys3_cheb.hpp"

constexpr double PI = 3.141592653589793238462643383279502884;
const double ERI_PREFAC = 2.0 * PI * PI * std::sqrt(PI);
constexpr int OS_VRR_PAIR_MAX_L = 6;
constexpr int OS_VRR_MAX_CART = 28;
constexpr int CARTESIAN_SCALAR_MAX_L = 6;
constexpr std::size_t DEFAULT_RYS_RECURRENCE_CACHE_MAX_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t DEFAULT_RYS_SPHERICAL_CACHE_MAX_BYTES = 0;

class ReusableWorkerPool {
public:
    ~ReusableWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    void run(int nthread, const std::function<void(int)>& job) {
        if (nthread <= 1) {
            job(0);
            return;
        }
        std::lock_guard<std::mutex> run_lock(run_mutex_);
        ensure_capacity(nthread - 1);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            job_ = job;
            worker_error_ = nullptr;
            active_workers_ = nthread - 1;
            remaining_workers_ = active_workers_;
            ++generation_;
        }
        start_cv_.notify_all();
        std::exception_ptr main_error;
        try {
            job(0);
        } catch (...) {
            main_error = std::current_exception();
        }
        std::exception_ptr worker_error;
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            done_cv_.wait(lock, [&] { return remaining_workers_ == 0; });
            worker_error = worker_error_;
            job_ = nullptr;
        }
        if (main_error) std::rethrow_exception(main_error);
        if (worker_error) std::rethrow_exception(worker_error);
    }

private:
    void ensure_capacity(int count) {
        while (static_cast<int>(workers_.size()) < count) {
            const int worker_id = static_cast<int>(workers_.size()) + 1;
            const std::size_t initial_generation = generation_;
            workers_.emplace_back([this, worker_id, initial_generation] {
                worker_loop(worker_id, initial_generation);
            });
        }
    }

    void worker_loop(int worker_id, std::size_t seen_generation) {
        while (true) {
            std::function<void(int)> job;
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                start_cv_.wait(lock, [&] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) return;
                seen_generation = generation_;
                if (worker_id > active_workers_) continue;
                job = job_;
            }
            try {
                job(worker_id);
            } catch (...) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (!worker_error_) worker_error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                --remaining_workers_;
                if (remaining_workers_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::mutex run_mutex_;
    std::mutex state_mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::function<void(int)> job_;
    std::exception_ptr worker_error_;
    std::size_t generation_ = 0;
    int active_workers_ = 0;
    int remaining_workers_ = 0;
    bool stopping_ = false;
};

ReusableWorkerPool& native_integral_worker_pool() {
    static ReusableWorkerPool pool;
    return pool;
}

constexpr double BOYS0_MID_CHEB[3][17] = {
    {3.69260892593684109e-01, -6.28317191907165201e-02, 7.78147807755485196e-03, -1.01824922931021138e-03, 1.30508527081563824e-04, -1.58459023097007741e-05, 1.79378499513903238e-06, -1.88060978338879438e-07, 1.82309276136738246e-08, -1.63602207972775239e-09, 1.36246942215674343e-10, -1.05627704833339070e-11, 7.64920512606895655e-13, -5.19020980460245112e-14, 3.10927261055600145e-15, -2.16717292100600925e-16, -1.91834944614729702e-17},
    {2.82392962887930754e-01, -2.85562567876372145e-02, 2.16121319327928184e-03, -1.80974042560373282e-04, 1.57609803527738631e-05, -1.38719093334998265e-06, 1.21011061292869219e-07, -1.03077047929483520e-08, 8.47686398613829223e-10, -6.67675966377318255e-11, 5.01093146865571330e-12, -3.57236219061710461e-13, 2.43404256846936807e-14, -1.57750782602794538e-15, -4.30489072102299092e-17, -8.38146365290142540e-18, 1.80395600448499878e-16},
    {2.37770624901000233e-01, -1.70820914178422857e-02, 9.19985072678147688e-04, -5.50362676255404418e-05, 3.45464478138237826e-06, -2.22644470115022074e-07, 1.45569210486204908e-08, -9.56969674338625626e-10, 6.27465433914607864e-11, -4.07166502220521936e-12, 2.59717592016691244e-13, -1.61581557658677476e-14, 1.13365440188229370e-15, -6.79821661128639535e-17, -7.01062679852013498e-17, 1.59364317989742982e-18, 1.86498993798541971e-16},
};

constexpr double BOYS0_SMALL_CHEB[19] = {
    6.57186257657690676e-01, -2.66817291710163607e-01,
    6.11450766743829172e-02, -1.23329742967011637e-02,
    2.14537771790367093e-03, -3.24238500012818173e-04,
    4.30892255459287019e-05, -5.09514596310239750e-06,
    5.41772106229890224e-07, -5.22794262476461671e-08,
    4.61459185483047427e-09, -3.75137727347970943e-10,
    2.82540313420055434e-11, -1.98183387666075457e-12,
    1.30438207295877616e-13, -8.22014661804220726e-15,
    9.20457102772300290e-16, -1.65470754326506321e-16,
    2.87011308337681401e-16,
};

inline double boys0_small_chebyshev(double T) {
    const double x = 0.5 * T - 1.0;
    double next = 0.0;
    double next_next = 0.0;
    for (int degree = 18; degree >= 1; --degree) {
        const double current =
            2.0 * x * next - next_next + BOYS0_SMALL_CHEB[degree];
        next_next = next;
        next = current;
    }
    return x * next - next_next + BOYS0_SMALL_CHEB[0];
}

inline double boys0_mid_chebyshev(double T) {
    const int interval = T < 8.0 ? 0 : (T < 12.0 ? 1 : 2);
    const double center = interval == 0 ? 6.0 : (interval == 1 ? 10.0 : 14.0);
    const double x = 0.5 * (T - center);
    const double* coefficients = BOYS0_MID_CHEB[interval];
    double next = 0.0;
    double next_next = 0.0;
    for (int degree = 16; degree >= 1; --degree) {
        const double current = 2.0 * x * next - next_next + coefficients[degree];
        next_next = next;
        next = current;
    }
    return x * next - next_next + coefficients[0];
}

struct ArrayRef {
    PyArrayObject* obj = nullptr;

    ArrayRef() = default;
    ArrayRef(PyObject* value, int typenum, int flags)
        : obj(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(value, typenum, flags))) {}

    ~ArrayRef() {
        Py_XDECREF(obj);
    }

    ArrayRef(const ArrayRef&) = delete;
    ArrayRef& operator=(const ArrayRef&) = delete;

    explicit operator bool() const {
        return obj != nullptr;
    }
};

struct PrimitivePair {
    double a = 0.0;
    double b = 0.0;
    double p = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double weight = 0.0;
};

double boys0(double T) {
    if (T < 1.0e-14) {
        return 1.0;
    }
    if (T < 4.0) {
        return boys0_small_chebyshev(T);
    }
    if (T >= 4.0 && T < 16.0) {
        return boys0_mid_chebyshev(T);
    }
    const double root = std::sqrt(T);
    if (T >= 16.0) {
        double series = 1.0;
        double term = 1.0;
        double previous = std::numeric_limits<double>::infinity();
        for (int order = 1; order < 40; ++order) {
            term *= -(2.0 * order - 1.0) / (2.0 * T);
            if (std::abs(term) >= previous) break;
            series += term;
            previous = std::abs(term);
        }
        const double erfc = std::exp(-T) * series / (std::sqrt(PI) * root);
        return 0.5 * std::sqrt(PI / T) * (1.0 - erfc);
    }
    return 0.5 * std::sqrt(PI / T) * std::erf(root);
}

double primitive_eri_ssss(
    double a,
    const double* A,
    double b,
    const double* B,
    double c,
    const double* C,
    double d,
    const double* D
) {
    const double p = a + b;
    const double q = c + d;
    const double alpha = p * q / (p + q);

    const double Px = (a * A[0] + b * B[0]) / p;
    const double Py = (a * A[1] + b * B[1]) / p;
    const double Pz = (a * A[2] + b * B[2]) / p;
    const double Qx = (c * C[0] + d * D[0]) / q;
    const double Qy = (c * C[1] + d * D[1]) / q;
    const double Qz = (c * C[2] + d * D[2]) / q;

    const double ABx = A[0] - B[0];
    const double ABy = A[1] - B[1];
    const double ABz = A[2] - B[2];
    const double CDx = C[0] - D[0];
    const double CDy = C[1] - D[1];
    const double CDz = C[2] - D[2];
    const double PQx = Px - Qx;
    const double PQy = Py - Qy;
    const double PQz = Pz - Qz;

    const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;
    const double CD2 = CDx * CDx + CDy * CDy + CDz * CDz;
    const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;

    const double Kab = std::exp(-(a * b / p) * AB2);
    const double Kcd = std::exp(-(c * d / q) * CD2);
    return ERI_PREFAC * Kab * Kcd * boys0(alpha * PQ2) / (p * q * std::sqrt(p + q));
}

inline npy_intp dense_index(npy_intp nao, npy_intp p, npy_intp q, npy_intp r, npy_intp s) {
    return ((p * nao + q) * nao + r) * nao + s;
}

inline int ncart_for_l(int l) {
    return (l + 1) * (l + 2) / 2;
}

void fill_cartesian_components(int l, int* lx, int* ly, int* lz) {
    int idx = 0;
    for (int ix = l; ix >= 0; --ix) {
        const int rem = l - ix;
        for (int iy = rem; iy >= 0; --iy) {
            lx[idx] = ix;
            ly[idx] = iy;
            lz[idx] = rem - iy;
            ++idx;
        }
    }
}

constexpr unsigned char CARTESIAN_POWERS_L2[3][6][3] = {
    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}},
};

struct RysFixedDirectOutput {
    unsigned char local_index[4]{};
    unsigned char axis_index[3]{};
};

template<std::size_t NOUT>
struct RysFixedDirectPlan {
    std::array<RysFixedDirectOutput, NOUT> outputs{};
    std::array<std::uint16_t, 9> class_offsets{};
    std::uint16_t size = 0;
};

template<int LP, int LQ, int LR, int LS,
         bool PQ_SAME_SHELL, bool RS_SAME_SHELL, bool SAME_PAIR>
constexpr auto make_rys_fixed_direct_plan() {
    constexpr int NP = (LP + 1) * (LP + 2) / 2;
    constexpr int NQ = (LQ + 1) * (LQ + 2) / 2;
    constexpr int NR = (LR + 1) * (LR + 2) / 2;
    constexpr int NS = (LS + 1) * (LS + 2) / 2;
    constexpr std::size_t NOUT =
        static_cast<std::size_t>(NP) * NQ * NR * NS;
    RysFixedDirectPlan<NOUT> plan{};
    std::array<std::uint16_t, 8> counts{};
    for (int ip = 0; ip < NP; ++ip) {
        for (int iq = 0; iq < NQ; ++iq) {
            for (int ir = 0; ir < NR; ++ir) {
                for (int is = 0; is < NS; ++is) {
                    if constexpr (PQ_SAME_SHELL) {
                        if (ip < iq) continue;
                    }
                    if constexpr (RS_SAME_SHELL) {
                        if (ir < is) continue;
                    }
                    if constexpr (SAME_PAIR) {
                        if (ip < ir || (ip == ir && iq < is)) continue;
                    }
                    const bool pq_same = PQ_SAME_SHELL && ip == iq;
                    const bool rs_same = RS_SAME_SHELL && ir == is;
                    const bool same_ao_pair = SAME_PAIR && ip == ir && iq == is;
                    const unsigned int symmetry =
                        static_cast<unsigned int>(pq_same) |
                        (static_cast<unsigned int>(rs_same) << 1) |
                        (static_cast<unsigned int>(same_ao_pair) << 2);
                    ++counts[symmetry];
                }
            }
        }
    }
    std::array<std::uint16_t, 8> cursor{};
    for (std::size_t symmetry = 0; symmetry < 8; ++symmetry) {
        plan.class_offsets[symmetry + 1] =
            plan.class_offsets[symmetry] + counts[symmetry];
        cursor[symmetry] = plan.class_offsets[symmetry];
    }
    plan.size = plan.class_offsets[8];
    for (int ip = 0; ip < NP; ++ip) {
        for (int iq = 0; iq < NQ; ++iq) {
            for (int ir = 0; ir < NR; ++ir) {
                for (int is = 0; is < NS; ++is) {
                    if constexpr (PQ_SAME_SHELL) {
                        if (ip < iq) continue;
                    }
                    if constexpr (RS_SAME_SHELL) {
                        if (ir < is) continue;
                    }
                    if constexpr (SAME_PAIR) {
                        if (ip < ir || (ip == ir && iq < is)) continue;
                    }
                    const bool pq_same = PQ_SAME_SHELL && ip == iq;
                    const bool rs_same = RS_SAME_SHELL && ir == is;
                    const bool same_ao_pair = SAME_PAIR && ip == ir && iq == is;
                    const unsigned int symmetry =
                        static_cast<unsigned int>(pq_same) |
                        (static_cast<unsigned int>(rs_same) << 1) |
                        (static_cast<unsigned int>(same_ao_pair) << 2);
                    RysFixedDirectOutput& output = plan.outputs[cursor[symmetry]++];
                    output.local_index[0] = static_cast<unsigned char>(ip);
                    output.local_index[1] = static_cast<unsigned char>(iq);
                    output.local_index[2] = static_cast<unsigned char>(ir);
                    output.local_index[3] = static_cast<unsigned char>(is);
                    for (int axis = 0; axis < 3; ++axis) {
                        output.axis_index[axis] = static_cast<unsigned char>(
                            CARTESIAN_POWERS_L2[LP][ip][axis] |
                            (CARTESIAN_POWERS_L2[LQ][iq][axis] << 2) |
                            (CARTESIAN_POWERS_L2[LR][ir][axis] << 4) |
                            (CARTESIAN_POWERS_L2[LS][is][axis] << 6)
                        );
                    }
                }
            }
        }
    }
    return plan;
}

template<int LP, int LQ, int LR, int LS,
         bool PQ_SAME_SHELL, bool RS_SAME_SHELL, bool SAME_PAIR>
inline const auto& rys_fixed_direct_plan() {
    static constexpr auto plan = make_rys_fixed_direct_plan<
        LP, LQ, LR, LS, PQ_SAME_SHELL, RS_SAME_SHELL, SAME_PAIR
    >();
    return plan;
}

struct ShellBlock {
    npy_intp start;
    npy_intp stop;
    int l;
};

bool same_shell_block_member(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp ref,
    npy_intp cur
) {
    const int ref_l = static_cast<int>(shells[3 * ref] + shells[3 * ref + 1] + shells[3 * ref + 2]);
    const int cur_l = static_cast<int>(shells[3 * cur] + shells[3 * cur + 1] + shells[3 * cur + 2]);
    if (ref_l != cur_l || nprim[ref] != nprim[cur]) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (origins[3 * ref + axis] != origins[3 * cur + axis]) {
            return false;
        }
    }
    for (std::int64_t ip = 0; ip < nprim[ref]; ++ip) {
        if (exps[ref * max_prim + ip] != exps[cur * max_prim + ip]) {
            return false;
        }
    }
    return true;
}

bool try_build_shell_blocks(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp nao,
    npy_intp max_prim,
    std::vector<ShellBlock>& blocks
) {
    blocks.clear();
    npy_intp start = 0;
    while (start < nao) {
        const int l = static_cast<int>(shells[3 * start] + shells[3 * start + 1] + shells[3 * start + 2]);
        if (l < 0 || l >= OS_VRR_MAX_CART) {
            return false;
        }
        const int ncart = ncart_for_l(l);
        if (ncart > OS_VRR_MAX_CART) {
            return false;
        }
        const npy_intp stop = start + ncart;
        if (stop > nao) {
            return false;
        }

        int lx[OS_VRR_MAX_CART];
        int ly[OS_VRR_MAX_CART];
        int lz[OS_VRR_MAX_CART];
        fill_cartesian_components(l, lx, ly, lz);
        for (int local = 0; local < ncart; ++local) {
            const npy_intp cur = start + local;
            if (
                shells[3 * cur] != lx[local] ||
                shells[3 * cur + 1] != ly[local] ||
                shells[3 * cur + 2] != lz[local] ||
                !same_shell_block_member(shells, origins, exps, nprim, max_prim, start, cur)
            ) {
                return false;
            }
        }
        blocks.push_back({start, stop, l});
        start = stop;
    }
    return true;
}

int precompute_primitive_pair_geom(
    npy_intp p,
    npy_intp q,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp max_prim,
    double* a_out,
    double* b_out,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out,
    double* k_out,
    const double* weights = nullptr,
    double* weight_out = nullptr
);

struct ShellPairGeomData {
    npy_intp nshell = 0;
    npy_intp pair_cap = 0;
    std::vector<int> n;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> p;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<double> pz;
    std::vector<double> k;
    std::vector<double> weight;
};

struct PrimaryShellPairGeomCache {
    std::uint64_t key = 0;
    npy_intp nao = 0;
    npy_intp max_prim = 0;
    ShellPairGeomData data;
};

PrimaryShellPairGeomCache primary_shell_pair_geom_cache;

inline std::uint64_t fnv1a_mix(std::uint64_t hash, const void* ptr, std::size_t nbytes) {
    const auto* bytes = static_cast<const unsigned char*>(ptr);
    for (std::size_t i = 0; i < nbytes; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t primary_shell_pair_geom_key(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp nao,
    npy_intp max_prim
) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_mix(hash, &nao, sizeof(nao));
    hash = fnv1a_mix(hash, &max_prim, sizeof(max_prim));
    hash = fnv1a_mix(hash, shells, static_cast<std::size_t>(nao) * 3U * sizeof(std::int64_t));
    hash = fnv1a_mix(hash, origins, static_cast<std::size_t>(nao) * 3U * sizeof(double));
    hash = fnv1a_mix(hash, exps, static_cast<std::size_t>(nao) * static_cast<std::size_t>(max_prim) * sizeof(double));
    hash = fnv1a_mix(hash, weights, static_cast<std::size_t>(nao) * static_cast<std::size_t>(max_prim) * sizeof(double));
    hash = fnv1a_mix(hash, nprim, static_cast<std::size_t>(nao) * sizeof(std::int64_t));
    return hash;
}

const ShellPairGeomData& get_primary_shell_pair_geom(
    const std::vector<ShellBlock>& shell_blocks,
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp nao,
    npy_intp max_prim
) {
    const std::uint64_t key = primary_shell_pair_geom_key(
        shells, origins, exps, weights, nprim, nao, max_prim
    );
    if (
        primary_shell_pair_geom_cache.key == key &&
        primary_shell_pair_geom_cache.nao == nao &&
        primary_shell_pair_geom_cache.max_prim == max_prim
    ) {
        return primary_shell_pair_geom_cache.data;
    }

    const npy_intp nshell = static_cast<npy_intp>(shell_blocks.size());
    const npy_intp pair_cap = max_prim * max_prim;
    const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
    const std::size_t pair_storage_size = shell_pair_count * static_cast<std::size_t>(pair_cap);
    ShellPairGeomData data;
    data.nshell = nshell;
    data.pair_cap = pair_cap;
    data.n.assign(shell_pair_count, 0);
    data.a.assign(pair_storage_size, 0.0);
    data.b.assign(pair_storage_size, 0.0);
    data.p.assign(pair_storage_size, 0.0);
    data.px.assign(pair_storage_size, 0.0);
    data.py.assign(pair_storage_size, 0.0);
    data.pz.assign(pair_storage_size, 0.0);
    data.k.assign(pair_storage_size, 0.0);
    data.weight.assign(pair_storage_size, 0.0);

    for (npy_intp ish = 0; ish < nshell; ++ish) {
        for (npy_intp jsh = 0; jsh < nshell; ++jsh) {
            const npy_intp idx = ish * nshell + jsh;
            const std::size_t off = static_cast<std::size_t>(idx) * static_cast<std::size_t>(pair_cap);
            data.n[static_cast<std::size_t>(idx)] = precompute_primitive_pair_geom(
                shell_blocks[static_cast<std::size_t>(ish)].start,
                shell_blocks[static_cast<std::size_t>(jsh)].start,
                origins,
                exps,
                nprim,
                max_prim,
                data.a.data() + off,
                data.b.data() + off,
                data.p.data() + off,
                data.px.data() + off,
                data.py.data() + off,
                data.pz.data() + off,
                data.k.data() + off,
                weights,
                data.weight.data() + off
            );
        }
    }

    primary_shell_pair_geom_cache.key = key;
    primary_shell_pair_geom_cache.nao = nao;
    primary_shell_pair_geom_cache.max_prim = max_prim;
    primary_shell_pair_geom_cache.data = std::move(data);
    return primary_shell_pair_geom_cache.data;
}

inline int pair_index(int i, int j) {
    if (i < j) {
        std::swap(i, j);
    }
    return i * (i + 1) / 2 + j;
}

inline npy_intp pair_pair_index(npy_intp ij, npy_intp kl) {
    if (ij < kl) {
        std::swap(ij, kl);
    }
    return ij * (ij + 1) / 2 + kl;
}


struct S8IndexCache {
    npy_intp nao = -1;
    npy_intp npair = 0;
    bool has_pair_pair = false;
    std::vector<npy_intp> pair_i;
    std::vector<npy_intp> pair_j;
    std::vector<npy_intp> ao_pair_index;
    std::vector<npy_intp> pair_pair_lookup;
};

S8IndexCache s8_index_cache;

inline npy_intp s8_lookup_index(const S8IndexCache& cache, npy_intp npair, npy_intp ij, npy_intp kl) {
    return cache.has_pair_pair
        ? cache.pair_pair_lookup[static_cast<std::size_t>(ij) * static_cast<std::size_t>(npair) + static_cast<std::size_t>(kl)]
        : pair_pair_index(ij, kl);
}

const S8IndexCache& get_s8_index_cache(npy_intp nao, npy_intp npair) {
    if (s8_index_cache.nao == nao && s8_index_cache.npair == npair) {
        return s8_index_cache;
    }
    s8_index_cache.nao = nao;
    s8_index_cache.npair = npair;
    s8_index_cache.pair_i.assign(static_cast<std::size_t>(npair), 0);
    s8_index_cache.pair_j.assign(static_cast<std::size_t>(npair), 0);
    s8_index_cache.ao_pair_index.assign(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0);
    npy_intp pair = 0;
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j <= i; ++j) {
            s8_index_cache.pair_i[static_cast<std::size_t>(pair)] = i;
            s8_index_cache.pair_j[static_cast<std::size_t>(pair)] = j;
            ++pair;
        }
    }
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j < nao; ++j) {
            s8_index_cache.ao_pair_index[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] =
                pair_index(static_cast<int>(i), static_cast<int>(j));
        }
    }
    const std::size_t pair_pair_count = static_cast<std::size_t>(npair) * static_cast<std::size_t>(npair);
    s8_index_cache.has_pair_pair = pair_pair_count <= 4000000ULL;
    if (s8_index_cache.has_pair_pair) {
        s8_index_cache.pair_pair_lookup.resize(pair_pair_count);
        for (npy_intp ij = 0; ij < npair; ++ij) {
            for (npy_intp kl = 0; kl < npair; ++kl) {
                s8_index_cache.pair_pair_lookup[static_cast<std::size_t>(ij) * static_cast<std::size_t>(npair) + static_cast<std::size_t>(kl)] =
                    pair_pair_index(ij, kl);
            }
        }
    } else {
        s8_index_cache.pair_pair_lookup.clear();
    }
    return s8_index_cache;
}

long long shell_unique_quartet_count(
    const ShellBlock& a,
    const ShellBlock& b,
    const ShellBlock& c,
    const ShellBlock& d,
    int ish,
    int jsh,
    int ksh,
    int lsh
) {
    const long long np = a.stop - a.start;
    const long long nq = b.stop - b.start;
    const long long nr = c.stop - c.start;
    const long long ns = d.stop - d.start;
    const long long npair_pq = (ish == jsh) ? (np * (np + 1)) / 2 : np * nq;
    const long long npair_rs = (ksh == lsh) ? (nr * (nr + 1)) / 2 : nr * ns;
    return (ish == ksh && jsh == lsh) ? (npair_pq * (npair_pq + 1)) / 2 : npair_pq * npair_rs;
}

double contracted_eri_ssss(
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    const double* C = origins + 3 * r;
    const double* D = origins + 3 * s;

    double value = 0.0;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double ap = exps[p * max_prim + ip];
        const double wp = weights[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double aq = exps[q * max_prim + iq];
            const double wq = weights[q * max_prim + iq];
            for (std::int64_t ir = 0; ir < nprim[r]; ++ir) {
                const double ar = exps[r * max_prim + ir];
                const double wr = weights[r * max_prim + ir];
                for (std::int64_t is = 0; is < nprim[s]; ++is) {
                    const double as = exps[s * max_prim + is];
                    const double ws = weights[s * max_prim + is];
                    value += wp * wq * wr * ws * primitive_eri_ssss(ap, A, aq, B, ar, C, as, D);
                }
            }
        }
    }
    return value;
}

class HermiteEMemo {
public:
    HermiteEMemo(int i_max, int j_max, double qx, double a, double b)
        : j_max_(j_max),
          t_max_(i_max + j_max),
          qx_(qx),
          a_(a),
          b_(b),
          values_((i_max + 1) * (j_max + 1) * (t_max_ + 1), std::numeric_limits<double>::quiet_NaN()) {}

    double value(int i, int j, int t) {
        if (t < 0 || t > i + j || i < 0 || j < 0) {
            return 0.0;
        }
        double& cached = values_[index(i, j, t)];
        if (!std::isnan(cached)) {
            return cached;
        }

        const double p = a_ + b_;
        const double q = a_ * b_ / p;
        if (i == 0 && j == 0 && t == 0) {
            cached = std::exp(-q * qx_ * qx_);
        } else if (j == 0) {
            cached = (
                (1.0 / (2.0 * p)) * value(i - 1, j, t - 1)
                - (q * qx_ / a_) * value(i - 1, j, t)
                + (t + 1.0) * value(i - 1, j, t + 1)
            );
        } else {
            cached = (
                (1.0 / (2.0 * p)) * value(i, j - 1, t - 1)
                + (q * qx_ / b_) * value(i, j - 1, t)
                + (t + 1.0) * value(i, j - 1, t + 1)
            );
        }
        return cached;
    }

private:
    int index(int i, int j, int t) const {
        return ((i * (j_max_ + 1) + j) * (t_max_ + 1) + t);
    }

    int j_max_;
    int t_max_;
    double qx_;
    double a_;
    double b_;
    std::vector<double> values_;
};

double boys(int n, double T) {
    if (T < 1.0e-14) {
        return 1.0 / (2.0 * n + 1.0);
    }

    // The alternating series loses digits for moderate T; use it only in the
    // small-T region and switch to stable upward recurrence from F0 otherwise.
    if (T < 4.0) {
        long double value = 1.0L / (2.0L * n + 1.0L);
        long double term = 1.0L;
        const long double t_ld = static_cast<long double>(T);
        for (int k = 1; k < 300; ++k) {
            term *= -t_ld / k;
            const long double add = term / (2.0L * n + 2.0L * k + 1.0L);
            value += add;
            if (std::abs(static_cast<double>(add)) < 1.0e-18) {
                break;
            }
        }
        return static_cast<double>(value);
    }

    double value = boys0(T);
    if (n == 0) {
        return value;
    }
    const double exp_T = std::exp(-T);
    for (int m = 0; m < n; ++m) {
        value = ((2.0 * m + 1.0) * value - exp_T) / (2.0 * T);
    }
    return value;
}

double boys_series_value(int n, double T) {
    double value = 1.0 / (2.0 * n + 1.0);
    double term = 1.0;
    for (int k = 1; k < 300; ++k) {
        term *= -T / k;
        const double add = term / (2.0 * n + 2.0 * k + 1.0);
        value += add;
        if (std::abs(add) < 1.0e-19) break;
    }
    return value;
}

const std::array<std::array<double, 32>, 41>& boys_small_t_table() {
    static const auto table = [] {
        std::array<std::array<double, 32>, 41> result{};
        for (std::size_t point = 0; point < result.size(); ++point) {
            const double center = 0.1 * static_cast<double>(point);
            for (std::size_t order = 0; order < result[point].size(); ++order) {
                result[point][order] = boys_series_value(
                    static_cast<int>(order), center
                );
            }
        }
        return result;
    }();
    return table;
}

void fill_boys_values(int max_n, double T, double* values) {
    if (T < 1.0e-14) {
        for (int n = 0; n <= max_n; ++n) {
            values[n] = 1.0 / (2.0 * n + 1.0);
        }
        return;
    }

    const double exp_T = std::exp(-T);
    if (T < 4.0) {
        const int point = std::min(40, static_cast<int>(T * 10.0 + 0.5));
        const double delta = T - 0.1 * point;
        const auto& table = boys_small_t_table();
        double value = table[static_cast<std::size_t>(point)][static_cast<std::size_t>(max_n)];
        double factor = 1.0;
        for (int order = 1; order <= 7; ++order) {
            factor *= -delta / order;
            value += factor * table[static_cast<std::size_t>(point)][static_cast<std::size_t>(max_n + order)];
        }
        values[max_n] = value;
        for (int n = max_n - 1; n >= 0; --n) {
            values[n] = (2.0 * T * values[n + 1] + exp_T) / (2.0 * n + 1.0);
        }
        return;
    }

    values[0] = boys0(T);
    for (int n = 0; n < max_n; ++n) {
        values[n + 1] = ((2.0 * n + 1.0) * values[n] - exp_T) / (2.0 * T);
    }
}


// Degree-14 Chebyshev fits for the two-root quadrature. These cover the
// numerically active range without exp, erf, or sqrt in the hot path.
static constexpr double RYS2_INTERVALS[6][2] = {
    {0.0, 1.0},
    {1.0, 3.0},
    {3.0, 5.0},
    {5.0, 10.0},
    {10.0, 20.0},
    {20.0, 40.0},
};
static constexpr double RYS2_CHEB[6][4][15] = {
    {
        {1.17685292413195461e-01, -1.25284285840017318e-02, 4.65643419132346411e-04, -1.39147213280513220e-05, 3.22279394285110572e-07, -4.81028744700100081e-09, 1.17116491378668417e-11, 1.89281953441553775e-12, -7.15574564866951229e-14, 1.12341968499947627e-15, 1.90712225179514386e-16, -1.93127814824838360e-16, 1.40982247270516285e-16, -8.60656720439008160e-17, 3.25106188003669105e-17},
        {2.57041233575314676e+00, -2.92269197822894955e-01, 6.61437643928532160e-03, -1.18417376872243202e-05, -1.38657578401120119e-06, -2.73929861482499345e-08, 1.21028145452918560e-09, 1.68235882755828885e-11, -7.51184776189402585e-13, -1.43065879432997957e-14, 2.95162787063119462e-15, -2.47050558902206096e-15, 1.69767764190919943e-15, -7.03408040893526855e-16, 3.00261228238453579e-16},
        {5.98692458689695783e-01, -5.09999114387107016e-02, 2.34938406395306890e-03, -9.94739460001650179e-05, 3.79107760698093730e-06, -1.31316958587190361e-07, 4.20124534696713020e-09, -1.24826652014038628e-10, 3.45834625573800525e-12, -8.99745935487362467e-14, 2.41046160967333214e-15, -3.21341731579670243e-16, 2.19624919546785835e-16, -1.03134503355128625e-16, 7.06020639602301879e-17},
        {2.65801470149460151e-01, -7.50757989715576252e-02, 6.54441908329200274e-03, -4.11769514353342830e-04, 2.05092037627641922e-05, -8.47358989133296434e-07, 2.99085261019214904e-08, -9.21681727961398925e-10, 2.51971330002489751e-11, -6.18175268395277844e-13, 1.35124361333566732e-14, -4.71574931279160676e-17, -1.53561843720028796e-16, 1.18099382503746644e-16, -2.42677127378209878e-17},
    },
    {
        {8.79137309504673531e-02, -1.65208560013249406e-02, 1.10826517042891706e-03, -6.28337601923809947e-05, 3.10566744326911796e-06, -1.15846425255728085e-07, 2.61481808431385526e-09, -3.03076666002335068e-12, -4.14956143301802602e-12, 3.30987998491241387e-13, -1.23409650612337082e-14, 6.54634510050050243e-17, 3.42460321857740846e-17, -1.14122296430095123e-17, 1.56990028267367104e-17},
        {1.82823441602516135e+00, -4.31941054516220446e-01, 2.42008736458263138e-02, -3.92163851755044756e-04, -2.37277862173895949e-05, 5.91904775522412291e-07, 6.90693739070672654e-08, -2.62380987015565089e-09, -1.43952889907275592e-10, 8.09044222795859289e-12, 2.84556443083460718e-13, -2.27700694707397896e-14, -5.87119502428281454e-16, 3.41062663420470542e-16, 2.22318299040248395e-16},
        {4.82276007055860989e-01, -6.26054709912140378e-02, 4.69873226126522360e-03, -3.41338935100339016e-04, 2.30592027526127174e-05, -1.42552358507778330e-06, 8.24852685555470790e-08, -4.51137847152921231e-09, 2.31182884275220756e-10, -1.12254159104480615e-11, 5.21757666557593084e-13, -2.26090853831556503e-14, 7.92731910472103291e-16, 6.51518289057135619e-17, 2.85476362975410015e-17},
        {1.29465623912643490e-01, -5.72316891116598989e-02, 9.02056391749204989e-03, -1.05243171380860526e-03, 9.92344361828986977e-05, -7.89280413184708113e-06, 5.41262026470284771e-07, -3.26032494641710383e-08, 1.75299013673699765e-09, -8.49685362191686247e-11, 3.74122325759672006e-12, -1.51125700696318585e-13, 5.68033553598351582e-15, -1.90428035250562072e-16, 1.29460981110900383e-17},
    },
    {
        {6.18441113641011592e-02, -9.98079296532775571e-03, 5.86506588598080959e-04, -2.84575986646103155e-05, 1.36719035953449397e-06, -6.11957026226608862e-08, 1.85777712456110384e-09, -2.72554519405633504e-11, 5.55400791327319545e-13, -1.60698360103705350e-14, -4.09562523950600181e-15, 2.62534628480800578e-16, 1.07836059679410105e-17, -9.87122173123813482e-19, 5.40543983239681737e-19},
        {1.13937247497265526e+00, -2.62275839644722675e-01, 1.78169782824611558e-02, -6.17559462850494965e-04, -2.68667121528597984e-06, 1.14545886694121513e-06, -2.09418155571078470e-08, -2.32147610877853338e-09, 1.21746887584630822e-10, 2.12536339424779015e-12, -3.52747072499377689e-13, 4.47130815252804993e-15, 8.90821238590833035e-16, -7.09378920943147799e-17, 6.96688184615379938e-17},
        {3.85021827107549897e-01, -3.67610460586081586e-02, 2.13460674788581127e-03, -1.24497620820496815e-04, 7.25117589227422388e-06, -3.92271849044253966e-07, 1.93022845527395828e-08, -9.32684768729467619e-10, 4.47517347593208863e-11, -1.92062477350034203e-12, 7.57142450857998562e-14, -3.35666692544861488e-15, 1.66171451815539196e-16, -7.87746391551202688e-18, 1.34113577558288605e-17},
        {6.04842013759411476e-02, -1.71819258019047412e-02, 2.35889093706491892e-03, -2.46192925660813303e-04, 2.10351375509621918e-05, -1.56371970859716996e-06, 1.02781928527460747e-07, -5.96256022925806296e-09, 3.09406409264734149e-10, -1.47163599715123336e-11, 6.43152039289586985e-13, -2.53699678881347269e-14, 9.24569993943097722e-16, -2.22594392740536910e-17, 4.16776494443202811e-18},
    },
    {
        {3.87643643518078979e-02, -1.19626856075316437e-02, 1.53032468013476557e-03, -1.52163973517473530e-04, 1.19549411891877905e-05, -1.02861286560165509e-06, 1.36509812250090715e-07, -1.59746842489233237e-08, 6.70135761110688554e-10, 9.88496932236182332e-11, -1.02288203319566388e-11, -1.53218663446338790e-12, 3.03886953445797634e-13, 8.66869257630951737e-15, -6.52843383039496487e-15},
        {5.88460027552511589e-01, -2.55449202156373789e-01, 4.41815531303328679e-02, -5.71331089850572130e-03, 4.83235481438785692e-04, -1.12920220326500588e-05, -3.56025878625989875e-06, 5.52027668225482035e-07, -2.37386334307180905e-08, -3.81119946660035691e-09, 7.33015362686167635e-10, -4.00230406527014784e-11, -3.96133752310607381e-12, 9.05142755776151721e-13, -5.63139290386019883e-14},
        {2.97222224079370878e-01, -4.74804012761249616e-02, 5.00810200512586453e-03, -5.09035486019915307e-04, 5.14400621604662093e-05, -5.80995722074131529e-06, 6.83887888947631589e-07, -6.59539175147263573e-08, 4.41054148478121099e-09, -3.07645760020585734e-10, 5.72906170981712144e-11, -8.89564525444036495e-12, 4.21680464807315751e-13, 6.67091029683127996e-14, -5.84453136034053161e-15},
        {3.34143926833101804e-02, -9.24079051958252157e-03, 2.19463075957873665e-03, -4.83202531381911359e-04, 8.65978985496435564e-05, -1.29188521377594188e-05, 1.74416823382828496e-06, -2.31269552654330168e-07, 2.97313655608508460e-08, -3.36327893004932600e-09, 3.12061652571535639e-10, -2.59192538594485611e-11, 2.65839653414541354e-12, -3.23025666144596309e-13, 2.59703414560038772e-14},
    },
    {
        {1.98513448259728595e-02, -6.91522750109742606e-03, 1.19004264114257446e-03, -1.98527385572575812e-04, 3.09983647985449589e-05, -4.26672150099780292e-06, 4.59325750848296574e-07, -2.46798511766401040e-08, -3.34152130102558015e-09, 1.01975659905543890e-09, -6.05998773770632540e-11, -3.26925548314967569e-11, 1.26931939314729595e-11, -2.47922035942579882e-12, 2.41195988187191934e-13},
        {2.42551803692848650e-01, -1.02611706790747659e-01, 2.14223536421467541e-02, -4.34740881578883891e-03, 8.38518911858850699e-04, -1.49253801638168297e-04, 2.35676927820395632e-05, -3.08940749925060042e-06, 2.81283682175293525e-07, -6.54499696381928101e-10, -6.65284390775497040e-09, 1.73473033271025344e-09, -2.81007204159246597e-10, 2.89341416877326760e-11, -4.52817927814340667e-13},
        {2.12391195114796399e-01, -3.64962264839996248e-02, 4.65466212565236043e-03, -6.43815361824266134e-04, 8.85714888314273794e-05, -1.13158628010313610e-05, 1.23977602072822833e-06, -1.05545667286954229e-07, 7.47899427694000629e-09, -1.47831291695374425e-09, 5.90561314231266564e-10, -1.69682353449687341e-10, 3.26104990450334201e-11, -3.59571589626661240e-12, -1.19010139406225598e-13},
        {2.15122301895915857e-02, -3.78452098047883743e-03, 5.34696017176789049e-04, -9.83688477054177199e-05, 2.27750259782236900e-05, -5.83342952483749607e-06, 1.44017161124135381e-06, -3.15789073853008099e-07, 5.86662184274046798e-08, -8.81715652744521487e-09, 9.87590913177589734e-10, -6.71137426754612157e-11, 1.99059925704720399e-12, -1.30785930327158701e-12, 7.90927188141794650e-13},
    },
    {
        {9.83329494619508519e-03, -3.40741999236046601e-03, 5.90365056587662863e-04, -1.02284299603914850e-04, 1.77203856259697685e-05, -3.06943866650690990e-06, 5.31393276378255342e-07, -9.18694518259231873e-08, 1.58302565492233386e-08, -2.70809258680772065e-09, 4.56559436709321885e-10, -7.48659255386266585e-11, 1.16647182502369014e-11, -1.65020142536125719e-12, 1.89127731945637515e-13},
        {1.07374997441595052e-01, -4.07872302375530890e-02, 7.74664313811899291e-03, -1.47128396803991308e-03, 2.79420518194601817e-04, -5.30587816073272876e-05, 1.00713713933809799e-05, -1.90992259677677870e-06, 3.61458338336659771e-07, -6.81291419478327228e-08, 1.27455996459213389e-08, -2.35426068165789269e-09, 4.26085360982084744e-10, -7.47509174818572959e-11, 1.25183834053001515e-11},
        {1.50219689441885323e-01, -2.58698732293489095e-02, 3.33306934727949609e-03, -4.76844879142597348e-04, 7.16096388749137044e-05, -1.10579994099031418e-05, 1.73820336100340989e-06, -2.76372827399929693e-07, 4.42068351237026276e-08, -7.06581584007637977e-09, 1.11691228641886370e-09, -1.71598004296555797e-10, 2.48400684473056085e-11, -3.17754870445923777e-12, 2.96678759177967481e-13},
        {1.51752777979091511e-02, -2.61340203063279829e-03, 3.36720232776451965e-04, -4.81799832962912403e-05, 7.23998183531710838e-06, -1.12066066251779140e-06, 1.77538766793407567e-07, -2.88789656330917829e-08, 4.89751952609482122e-09, -8.91590386114555853e-10, 1.80112412299465963e-10, -4.08038909625429882e-11, 1.00749550505511637e-11, -2.57912173566835658e-12, 6.54611511159409464e-13},
    },
};

inline void evaluate_rys2_chebyshev(
    const double* coefficients,
    double x,
    double& value
) {
    double next = 0.0;
    double next_next = 0.0;
    for (int degree = 14; degree >= 1; --degree) {
        const double current = 2.0 * x * next - next_next + coefficients[degree];
        next_next = next;
        next = current;
    }
    value = x * next - next_next + coefficients[0];
}

inline bool rys_roots_weights_two_fast(double T, double* roots, double* weights) {
    if (T < 0.0) return false;
    if (T >= 40.0) {
        constexpr double first_node = 0.275255128608411;
        constexpr double second_node = 2.72474487139158;
        constexpr double second_weight_fraction = 0.0917517095361369;
        const double total_weight = std::sqrt((PI * 0.25) / T);
        roots[0] = first_node / (T - first_node);
        roots[1] = second_node / (T - second_node);
        weights[1] = second_weight_fraction * total_weight;
        weights[0] = total_weight - weights[1];
        return true;
    }

    int interval = 0;
    if (T >= 20.0) {
        interval = 5;
    } else if (T >= 10.0) {
        interval = 4;
    } else if (T >= 5.0) {
        interval = 3;
    } else if (T >= 3.0) {
        interval = 2;
    } else if (T >= 1.0) {
        interval = 1;
    }
    const double lower = RYS2_INTERVALS[interval][0];
    const double upper = RYS2_INTERVALS[interval][1];
    const double x = (2.0 * T - lower - upper) / (upper - lower);
    evaluate_rys2_chebyshev(RYS2_CHEB[interval][0], x, roots[0]);
    evaluate_rys2_chebyshev(RYS2_CHEB[interval][1], x, roots[1]);
    evaluate_rys2_chebyshev(RYS2_CHEB[interval][2], x, weights[0]);
    evaluate_rys2_chebyshev(RYS2_CHEB[interval][3], x, weights[1]);
    return true;
}

inline double evaluate_rys3_chebyshev(
    const double* coefficients,
    double x,
    int degree
) {
    double next = 0.0;
    double next_next = 0.0;
    for (int order = degree; order >= 1; --order) {
        const double current = 2.0 * x * next - next_next + coefficients[order];
        next_next = next;
        next = current;
    }
    return x * next - next_next + coefficients[0];
}

inline bool rys_roots_weights_three_fast(double T, double* roots, double* weights) {
    if (T < 0.0) return false;
    if (T >= 40.0) {
        constexpr double nodes[3] = {
            1.90163509193488151e-01,
            1.78449274854325135e+00,
            5.52534374226326008e+00,
        };
        constexpr double weight_fractions[3] = {
            8.17656939112058390e-01,
            1.77231492083829101e-01,
            5.11156880411249223e-03,
        };
        const double total_weight = std::sqrt((PI * 0.25) / T);
        for (int root = 0; root < 3; ++root) {
            roots[root] = nodes[root] / (T - nodes[root]);
            weights[root] = weight_fractions[root] * total_weight;
        }
        return true;
    }
    int interval = 0;
    if (T >= 20.0) {
        interval = 5;
    } else if (T >= 10.0) {
        interval = 4;
    } else if (T >= 5.0) {
        interval = 3;
    } else if (T >= 3.0) {
        interval = 2;
    } else if (T >= 1.0) {
        interval = 1;
    }
    const double lower = RYS2_INTERVALS[interval][0];
    const double upper = RYS2_INTERVALS[interval][1];
    const double x = (2.0 * T - lower - upper) / (upper - lower);
    constexpr int degrees[6] = {10, 12, 12, 14, 16, 18};
    const int degree = degrees[interval];
    for (int root = 0; root < 3; ++root) {
        roots[root] = evaluate_rys3_chebyshev(
            RYS3_CHEB[interval][root], x, degree
        );
        weights[root] = evaluate_rys3_chebyshev(
            RYS3_CHEB[interval][root + 3], x, degree
        );
    }
    return true;
}


inline bool rys_roots_weights_low(int nroots, double T, double* roots, double* weights) {
    if (nroots < 1 || nroots > 3) return false;
    if (nroots == 1) {
        double boys_values[2]{};
        fill_boys_values(1, T, boys_values);
        const double node = boys_values[1] / boys_values[0];
        if (node < 0.0 || node >= 1.0) return false;
        roots[0] = node / (1.0 - node);
        weights[0] = boys_values[0];
        return true;
    }
    if (nroots == 2) {
        return rys_roots_weights_two_fast(T, roots, weights);
    }
    return rys_roots_weights_three_fast(T, roots, weights);
}

bool rys_roots_weights_general(int nroots, double T, double* roots, double* weights) {
    if (nroots <= 3) return rys_roots_weights_low(nroots, T, roots, weights);
    if (nroots > 7 || T < 0.0) return false;

    double boys_values[14]{};
    double moments[14]{};
    fill_boys_values(2 * nroots - 1, T, boys_values);
    if (!(boys_values[0] > 0.0)) return false;
    const double scale = std::max(T, 1.0);
    double scale_power = 1.0;
    for (int order = 0; order < 2 * nroots; ++order) {
        moments[order] = scale_power * boys_values[order] / boys_values[0];
        scale_power *= scale;
    }

    double hankel[7][7]{};
    double shifted[7][7]{};
    double lower[7][7]{};
    for (int i = 0; i < nroots; ++i) {
        for (int j = 0; j < nroots; ++j) {
            hankel[i][j] = moments[i + j];
            shifted[i][j] = moments[i + j + 1];
        }
    }
    for (int i = 0; i < nroots; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = hankel[i][j];
            for (int k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
            if (i == j) {
                if (!(value > 1.0e-18)) return false;
                lower[i][j] = std::sqrt(value);
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }

    double inverse_lower[7][7]{};
    for (int column = 0; column < nroots; ++column) {
        for (int row = 0; row < nroots; ++row) {
            double value = row == column ? 1.0 : 0.0;
            for (int k = 0; k < row; ++k) {
                value -= lower[row][k] * inverse_lower[k][column];
            }
            inverse_lower[row][column] = value / lower[row][row];
        }
    }
    double transformed_shifted[7][7]{};
    for (int i = 0; i < nroots; ++i) {
        for (int j = 0; j < nroots; ++j) {
            double value = 0.0;
            for (int k = 0; k < nroots; ++k) {
                value += inverse_lower[i][k] * shifted[k][j];
            }
            transformed_shifted[i][j] = value;
        }
    }
    double jacobi[7][7]{};
    for (int i = 0; i < nroots; ++i) {
        for (int j = 0; j < nroots; ++j) {
            double value = 0.0;
            for (int k = 0; k < nroots; ++k) {
                value += transformed_shifted[i][k] * inverse_lower[j][k];
            }
            jacobi[i][j] = value;
        }
    }
    double eigenvectors[7][7]{};
    for (int i = 0; i < nroots; ++i) eigenvectors[i][i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        int p = 0;
        int q = 1;
        double largest = 0.0;
        for (int i = 0; i < nroots; ++i) {
            for (int j = i + 1; j < nroots; ++j) {
                if (std::abs(jacobi[i][j]) > largest) {
                    largest = std::abs(jacobi[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (largest < 1.0e-15) break;
        const double tau = (jacobi[q][q] - jacobi[p][p]) / (2.0 * jacobi[p][q]);
        const double tangent = (tau >= 0.0 ? 1.0 : -1.0) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        const double app = jacobi[p][p];
        const double aqq = jacobi[q][q];
        const double apq = jacobi[p][q];
        jacobi[p][p] = app - tangent * apq;
        jacobi[q][q] = aqq + tangent * apq;
        jacobi[p][q] = jacobi[q][p] = 0.0;
        for (int k = 0; k < nroots; ++k) {
            if (k != p && k != q) {
                const double akp = jacobi[k][p];
                const double akq = jacobi[k][q];
                jacobi[k][p] = jacobi[p][k] = cosine * akp - sine * akq;
                jacobi[k][q] = jacobi[q][k] = sine * akp + cosine * akq;
            }
            const double vkp = eigenvectors[k][p];
            const double vkq = eigenvectors[k][q];
            eigenvectors[k][p] = cosine * vkp - sine * vkq;
            eigenvectors[k][q] = sine * vkp + cosine * vkq;
        }
    }

    std::array<int, 7> order{};
    for (int i = 0; i < nroots; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.begin() + nroots, [&](int left, int right) {
        return jacobi[left][left] < jacobi[right][right];
    });
    for (int root = 0; root < nroots; ++root) {
        const int column = order[static_cast<std::size_t>(root)];
        double node = jacobi[column][column] / scale;
        if (node < 0.0 && node > -1.0e-13) node = 0.0;
        if (!(node >= 0.0 && node < 1.0)) return false;
        const double weight = boys_values[0] *
            eigenvectors[0][column] * eigenvectors[0][column];
        if (!(weight > 0.0)) return false;
        roots[root] = node / (1.0 - node);
        weights[root] = weight;
    }
    return true;
}

bool rys_roots_weights_general(int nroots, double T, double* roots, double* weights);

struct RysQuadraturePointData {
    double u2 = 0.0;
    double root_prefactor = 0.0;
};

struct RysQuadratureCache {
    const RysQuadraturePointData* points = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
};

bool prepare_rys_quadrature_cache(
    RysQuadratureCache& cache,
    std::vector<RysQuadraturePointData>& storage,
    int nroots,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    int nrs
) {
    const std::size_t count = static_cast<std::size_t>(npq) * nrs * nroots;
    if (cache.size != count || cache.offset + count > storage.size()) return false;
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const double zeta = pq_p[idx_pq] + rs_p[idx_rs];
            const double alpha = pq_p[idx_pq] * rs_p[idx_rs] / zeta;
            const double dx = pq_px[idx_pq] - rs_px[idx_rs];
            const double dy = pq_py[idx_pq] - rs_py[idx_rs];
            const double dz = pq_pz[idx_pq] - rs_pz[idx_rs];
            const double T = alpha * (dx * dx + dy * dy + dz * dz);
            double roots[7]{};
            double weights[7]{};
            if (!rys_roots_weights_general(nroots, T, roots, weights)) {
                return false;
            }
            const double primitive_prefactor = ERI_PREFAC *
                pq_k[idx_pq] * rs_k[idx_rs] /
                (pq_p[idx_pq] * rs_p[idx_rs] * std::sqrt(zeta));
            const std::size_t offset =
                (static_cast<std::size_t>(idx_pq) * nrs + idx_rs) * nroots;
            for (int root = 0; root < nroots; ++root) {
                const double u2 = alpha * roots[root];
                RysQuadraturePointData& item = storage[cache.offset + offset + root];
                item.u2 = u2;
                item.root_prefactor = primitive_prefactor * weights[root];
            }
        }
    }
    return true;
}

std::uint64_t rys_pair_geometry_key(
    const double* p,
    const double* px,
    const double* py,
    const double* pz,
    const double* k,
    int npair
) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_mix(hash, &npair, sizeof(npair));
    const std::size_t bytes = static_cast<std::size_t>(npair) * sizeof(double);
    hash = fnv1a_mix(hash, p, bytes);
    hash = fnv1a_mix(hash, px, bytes);
    hash = fnv1a_mix(hash, py, bytes);
    hash = fnv1a_mix(hash, pz, bytes);
    hash = fnv1a_mix(hash, k, bytes);
    return hash;
}

inline std::uint64_t rys_quadrature_geometry_key(
    int nroots,
    std::uint64_t pq_key,
    std::uint64_t rs_key
) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_mix(hash, &nroots, sizeof(nroots));
    hash = fnv1a_mix(hash, &pq_key, sizeof(pq_key));
    return fnv1a_mix(hash, &rs_key, sizeof(rs_key));
}

inline void rys_build_1d_sp(
    double c00,
    double c0p,
    double b10,
    double b01,
    double b00,
    double out[3][3]
) {
    out[0][0] = 1.0;
    out[1][0] = c00;
    out[2][0] = c00 * c00 + b10;
    out[0][1] = c0p;
    out[0][2] = c0p * c0p + b01;
    out[1][1] = c00 * c0p + b00;
    out[2][1] = c00 * out[1][1] + b10 * out[0][1] + b00 * out[1][0];
    out[1][2] = c0p * out[1][1] + b01 * out[1][0] + b00 * out[0][1];
    out[2][2] = c00 * out[1][2] + b10 * out[0][2] + 2.0 * b00 * out[1][1];
}

inline double rys_distribute_sp(
    const double g[3][3],
    int a,
    int b,
    int c,
    int d,
    double AB,
    double CD
) {
    const int bra_order = a + b;
    const int ket_order = c + d;
    double value = g[bra_order][ket_order];
    if (b) value += AB * g[bra_order - 1][ket_order];
    if (d) value += CD * g[bra_order][ket_order - 1];
    if (b && d) value += AB * CD * g[bra_order - 1][ket_order - 1];
    return value;
}

template<unsigned int ACTIVE_CENTER_MASK>
inline void rys_distribute_sp_active(
    const double g[3][3],
    double AB,
    double CD,
    double* distributed
) {
#define PYQED_RYS_DISTRIBUTE_SP(code) \
    if constexpr (((code) & ~ACTIVE_CENTER_MASK) == 0) { \
        distributed[code] = rys_distribute_sp( \
            g, (code) & 1, ((code) >> 1) & 1, \
            ((code) >> 2) & 1, ((code) >> 3) & 1, AB, CD \
        ); \
    }
    PYQED_RYS_DISTRIBUTE_SP(0)
    PYQED_RYS_DISTRIBUTE_SP(1)
    PYQED_RYS_DISTRIBUTE_SP(2)
    PYQED_RYS_DISTRIBUTE_SP(3)
    PYQED_RYS_DISTRIBUTE_SP(4)
    PYQED_RYS_DISTRIBUTE_SP(5)
    PYQED_RYS_DISTRIBUTE_SP(6)
    PYQED_RYS_DISTRIBUTE_SP(7)
    PYQED_RYS_DISTRIBUTE_SP(8)
    PYQED_RYS_DISTRIBUTE_SP(9)
    PYQED_RYS_DISTRIBUTE_SP(10)
    PYQED_RYS_DISTRIBUTE_SP(11)
    PYQED_RYS_DISTRIBUTE_SP(12)
    PYQED_RYS_DISTRIBUTE_SP(13)
    PYQED_RYS_DISTRIBUTE_SP(14)
    PYQED_RYS_DISTRIBUTE_SP(15)
#undef PYQED_RYS_DISTRIBUTE_SP
}

inline void direct_jk_add_unique_permutations(
    double*, double*, const double*, npy_intp, double,
    npy_intp, npy_intp, npy_intp, npy_intp
);
inline void direct_jk_add_unique_permutations_symmetric_density(
    double*, double*, const double*, npy_intp, double,
    npy_intp, npy_intp, npy_intp, npy_intp
);
inline void direct_jk_add_symmetric_class_code(
    double*, double*, const double*, npy_intp, double,
    npy_intp, npy_intp, npy_intp, npy_intp, unsigned char
);
template<bool PQ_SAME, bool RS_SAME, bool SAME_PAIR>
inline void direct_jk_add_symmetric_class(
    double*, double*, const double*, npy_intp, double,
    npy_intp, npy_intp, npy_intp, npy_intp
);

struct SPDirectJKConsumer {
    const double* dm = nullptr;
    double* vj = nullptr;
    double* vk = nullptr;
    npy_intp nao = 0;
    npy_intp starts[4]{};
    double integral_bound = 0.0;
    double screen_tol = 0.0;
    bool symmetric_density = false;
    bool same_pair = false;
    long long* computed = nullptr;
    long long* skipped = nullptr;
};

struct CartesianDirectJKConsumer {
    const double* dm = nullptr;
    double* vj = nullptr;
    double* vk = nullptr;
    npy_intp nao = 0;
    npy_intp starts[4]{};
    bool symmetric_density = false;
    bool pq_same_shell = false;
    bool rs_same_shell = false;
    bool same_pair = false;
    long long* computed = nullptr;
};

template<unsigned char SYMMETRY_CLASS, std::size_t NOUT, typename ValueAt>
inline void consume_rys_fixed_direct_class(
    const RysFixedDirectPlan<NOUT>& plan,
    const CartesianDirectJKConsumer& direct,
    ValueAt value_at
) {
    constexpr bool pq_same = (SYMMETRY_CLASS & 1) != 0;
    constexpr bool rs_same = (SYMMETRY_CLASS & 2) != 0;
    constexpr bool same_pair = (SYMMETRY_CLASS & 4) != 0;
    for (
        std::size_t output = plan.class_offsets[SYMMETRY_CLASS];
        output < plan.class_offsets[SYMMETRY_CLASS + 1];
        ++output
    ) {
        const RysFixedDirectOutput& item = plan.outputs[output];
        direct_jk_add_symmetric_class<pq_same, rs_same, same_pair>(
            direct.vj, direct.vk, direct.dm, direct.nao,
            value_at(output, item),
            direct.starts[0] + item.local_index[0],
            direct.starts[1] + item.local_index[1],
            direct.starts[2] + item.local_index[2],
            direct.starts[3] + item.local_index[3]
        );
        ++*direct.computed;
    }
}

template<int NP, int NQ, int NR, int NS, typename Consumer, typename ValueAt>
inline void direct_jk_accumulate_distinct_shell_tiles(
    const Consumer& direct,
    ValueAt value_at
) {
    std::array<double, NP * NQ> jpq{};
    std::array<double, NR * NS> jrs{};
    std::array<double, NP * NS> kps{};
    std::array<double, NQ * NS> kqs{};
    std::array<double, NP * NR> kpr{};
    std::array<double, NQ * NR> kqr{};
    auto dm_at = [&](npy_intp a, npy_intp b) {
        return direct.dm[static_cast<std::size_t>(a) * direct.nao + b];
    };
    std::size_t index = 0;
    for (int ip = 0; ip < NP; ++ip) {
        const npy_intp p = direct.starts[0] + ip;
        for (int iq = 0; iq < NQ; ++iq) {
            const npy_intp q = direct.starts[1] + iq;
            for (int ir = 0; ir < NR; ++ir) {
                const npy_intp r = direct.starts[2] + ir;
                for (int is = 0; is < NS; ++is, ++index) {
                    const npy_intp s = direct.starts[3] + is;
                    const double value = value_at(index, ip, iq, ir, is);
                    jpq[static_cast<std::size_t>(ip) * NQ + iq] +=
                        2.0 * dm_at(r, s) * value;
                    jrs[static_cast<std::size_t>(ir) * NS + is] +=
                        2.0 * dm_at(p, q) * value;
                    if (direct.vk != nullptr) {
                        kps[static_cast<std::size_t>(ip) * NS + is] += dm_at(q, r) * value;
                        kqs[static_cast<std::size_t>(iq) * NS + is] += dm_at(p, r) * value;
                        kpr[static_cast<std::size_t>(ip) * NR + ir] += dm_at(q, s) * value;
                        kqr[static_cast<std::size_t>(iq) * NR + ir] += dm_at(p, s) * value;
                    }
                }
            }
        }
    }
    auto flush = [&](double* matrix, const auto& tile, int nleft, int nright,
                     npy_intp left0, npy_intp right0) {
        for (int left = 0; left < nleft; ++left) {
            for (int right = 0; right < nright; ++right) {
                const double value = tile[static_cast<std::size_t>(left) * nright + right];
                matrix[static_cast<std::size_t>(left0 + left) * direct.nao + right0 + right] += value;
                matrix[static_cast<std::size_t>(right0 + right) * direct.nao + left0 + left] += value;
            }
        }
    };
    flush(direct.vj, jpq, NP, NQ, direct.starts[0], direct.starts[1]);
    flush(direct.vj, jrs, NR, NS, direct.starts[2], direct.starts[3]);
    if (direct.vk != nullptr) {
        flush(direct.vk, kps, NP, NS, direct.starts[0], direct.starts[3]);
        flush(direct.vk, kqs, NQ, NS, direct.starts[1], direct.starts[3]);
        flush(direct.vk, kpr, NP, NR, direct.starts[0], direct.starts[2]);
        flush(direct.vk, kqr, NQ, NR, direct.starts[1], direct.starts[2]);
    }
    *direct.computed += static_cast<long long>(NP) * NQ * NR * NS;
}

struct SPActiveOutput {
    unsigned char ip, iq, ir, is, index;
    unsigned char axis_codes[3];
};

template<int LP, int LQ, int LR, int LS,
         bool PQ_SAME_SHELL, bool RS_SAME_SHELL, bool SAME_PAIR>
inline std::size_t build_sp_active_outputs_fixed(
    const SPDirectJKConsumer* direct,
    SPActiveOutput* outputs
) {
    constexpr int NP = LP ? 3 : 1;
    constexpr int NQ = LQ ? 3 : 1;
    constexpr int NR = LR ? 3 : 1;
    constexpr int NS = LS ? 3 : 1;
    std::size_t nactive = 0;
    std::size_t index = 0;
    for (int ip = 0; ip < NP; ++ip) {
        for (int iq = 0; iq < NQ; ++iq) {
            if constexpr (PQ_SAME_SHELL) {
                if (ip < iq) {
                    index += static_cast<std::size_t>(NR) * NS;
                    continue;
                }
            }
            for (int ir = 0; ir < NR; ++ir) {
                for (int is = 0; is < NS; ++is, ++index) {
                    if constexpr (RS_SAME_SHELL) {
                        if (ir < is) continue;
                    }
                    if constexpr (SAME_PAIR) {
                        if (ip < ir || (ip == ir && iq < is)) continue;
                    }
                    if (direct != nullptr) {
                        const npy_intp gp = direct->starts[0] + ip;
                        const npy_intp gq = direct->starts[1] + iq;
                        const npy_intp gr = direct->starts[2] + ir;
                        const npy_intp gs = direct->starts[3] + is;
                        if (direct->screen_tol > 0.0) {
                            auto abs_dm = [&](npy_intp a, npy_intp b) {
                                return std::abs(direct->dm[
                                    static_cast<std::size_t>(a) * direct->nao + b
                                ]);
                            };
                            const double dm_bound = std::max({
                                abs_dm(gp, gq), abs_dm(gq, gp),
                                abs_dm(gr, gs), abs_dm(gs, gr),
                                abs_dm(gp, gr), abs_dm(gr, gp),
                                abs_dm(gp, gs), abs_dm(gs, gp),
                                abs_dm(gq, gr), abs_dm(gr, gq),
                                abs_dm(gq, gs), abs_dm(gs, gq),
                            });
                            if (direct->integral_bound * dm_bound < direct->screen_tol) {
                                ++*direct->skipped;
                                continue;
                            }
                        }
                    }
                    SPActiveOutput& output = outputs[nactive++];
                    output = {
                        static_cast<unsigned char>(ip),
                        static_cast<unsigned char>(iq),
                        static_cast<unsigned char>(ir),
                        static_cast<unsigned char>(is),
                        static_cast<unsigned char>(index),
                    };
                    for (int axis = 0; axis < 3; ++axis) {
                        output.axis_codes[axis] = static_cast<unsigned char>(
                            (LP && ip == axis ? 1 : 0) |
                            (LQ && iq == axis ? 2 : 0) |
                            (LR && ir == axis ? 4 : 0) |
                            (LS && is == axis ? 8 : 0)
                        );
                    }
                }
            }
        }
    }
    return nactive;
}

template<int LP, int LQ, int LR, int LS, bool CACHED>
bool compute_sp_shell_quartet_rys_values_fixed(
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp q0,
    npy_intp r0,
    npy_intp s0,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    const double* pq_weight,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    const double* rs_weight,
    int nrs,
    const RysQuadratureCache* quadrature_cache,
    bool unique_direct,
    bool pq_same_shell,
    bool rs_same_shell,
    bool same_pair,
    const SPDirectJKConsumer* direct,
    std::vector<double>& values
) {
    constexpr int NP = LP ? 3 : 1;
    constexpr int NQ = LQ ? 3 : 1;
    constexpr int NR = LR ? 3 : 1;
    constexpr int NS = LS ? 3 : 1;
    constexpr int NROOTS = (LP + LQ + LR + LS) / 2 + 1;
    constexpr std::size_t NOUT = NP * NQ * NR * NS;
    constexpr unsigned int ACTIVE_CENTER_MASK =
        LP | (LQ << 1) | (LR << 2) | (LS << 3);
    std::array<double, NOUT> contracted_values{};
    SPActiveOutput active_outputs[NOUT]{};
    std::size_t nactive = 0;
    const bool use_active_outputs = unique_direct || (
        direct != nullptr && (
            pq_same_shell || rs_same_shell || same_pair || direct->screen_tol > 0.0
        )
    );
    if (use_active_outputs) {
        if (same_pair) {
            nactive = pq_same_shell
                ? build_sp_active_outputs_fixed<LP, LQ, LR, LS, true, true, true>(
                    direct, active_outputs
                )
                : build_sp_active_outputs_fixed<LP, LQ, LR, LS, false, false, true>(
                    direct, active_outputs
                );
        } else if (pq_same_shell) {
            nactive = rs_same_shell
                ? build_sp_active_outputs_fixed<LP, LQ, LR, LS, true, true, false>(
                    direct, active_outputs
                )
                : build_sp_active_outputs_fixed<LP, LQ, LR, LS, true, false, false>(
                    direct, active_outputs
                );
        } else {
            nactive = rs_same_shell
                ? build_sp_active_outputs_fixed<LP, LQ, LR, LS, false, true, false>(
                    direct, active_outputs
                )
                : build_sp_active_outputs_fixed<LP, LQ, LR, LS, false, false, false>(
                    direct, active_outputs
                );
        }
    }

    if constexpr (LP + LQ + LR + LS == 0) {
        double value_even = 0.0;
        double value_odd = 0.0;
        for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
            auto primitive_value = [&](int idx_rs) {
                const std::size_t quadrature_offset =
                    static_cast<std::size_t>(idx_pq) * nrs + idx_rs;
                if constexpr (CACHED) {
                    return pq_weight[idx_pq] * rs_weight[idx_rs] *
                        quadrature_cache->points[quadrature_offset].root_prefactor;
                }
                const double pq_exponent = pq_p[idx_pq];
                const double rs_exponent = rs_p[idx_rs];
                const double zeta = pq_exponent + rs_exponent;
                const double alpha = pq_exponent * rs_exponent / zeta;
                const double dx = pq_px[idx_pq] - rs_px[idx_rs];
                const double dy = pq_py[idx_pq] - rs_py[idx_rs];
                const double dz = pq_pz[idx_pq] - rs_pz[idx_rs];
                const double primitive_prefactor =
                    pq_weight[idx_pq] * rs_weight[idx_rs] * ERI_PREFAC *
                    pq_k[idx_pq] * rs_k[idx_rs] /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
                return primitive_prefactor * boys0(
                    alpha * (dx * dx + dy * dy + dz * dz)
                );
            };
            int idx_rs = 0;
            for (; idx_rs + 1 < nrs; idx_rs += 2) {
                value_even += primitive_value(idx_rs);
                value_odd += primitive_value(idx_rs + 1);
            }
            if (idx_rs < nrs) value_even += primitive_value(idx_rs);
        }
        contracted_values[0] = value_even + value_odd;
    } else {
    const double AB[3] = {
        origins[3 * p0] - origins[3 * q0],
        origins[3 * p0 + 1] - origins[3 * q0 + 1],
        origins[3 * p0 + 2] - origins[3 * q0 + 2],
    };
    const double CD[3] = {
        origins[3 * r0] - origins[3 * s0],
        origins[3 * r0 + 1] - origins[3 * s0 + 1],
        origins[3 * r0 + 2] - origins[3 * s0 + 2],
    };
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const double pq_exponent = pq_p[idx_pq];
            const double rs_exponent = rs_p[idx_rs];
            const double zeta = pq_exponent + rs_exponent;
            const double alpha = pq_exponent * rs_exponent / zeta;
            const double PQ[3] = {
                pq_px[idx_pq] - rs_px[idx_rs],
                pq_py[idx_pq] - rs_py[idx_rs],
                pq_pz[idx_pq] - rs_pz[idx_rs],
            };
            double local_roots[3]{};
            double local_rys_weights[3]{};
            double primitive_prefactor = 0.0;
            const std::size_t quadrature_offset =
                (static_cast<std::size_t>(idx_pq) * nrs + idx_rs) * NROOTS;
            if constexpr (!CACHED) {
                const double T = alpha * (
                    PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]
                );
                if (!rys_roots_weights_low(
                        NROOTS, T, local_roots, local_rys_weights
                    )) return false;
                primitive_prefactor = pq_weight[idx_pq] * rs_weight[idx_rs] *
                    ERI_PREFAC * pq_k[idx_pq] * rs_k[idx_rs] /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
            }
            #pragma clang loop unroll(full)
            for (int root = 0; root < NROOTS; ++root) {
                double c00[3]{};
                double c0p[3]{};
                double b00 = 0.0;
                double b10 = 0.0;
                double b01 = 0.0;
                double root_prefactor = 0.0;
                double u2 = 0.0;
                if constexpr (CACHED) {
                    const RysQuadraturePointData& item =
                        quadrature_cache->points[quadrature_offset + root];
                    u2 = item.u2;
                    root_prefactor = pq_weight[idx_pq] * rs_weight[idx_rs] *
                        item.root_prefactor;
                } else {
                    u2 = alpha * local_roots[root];
                    root_prefactor = primitive_prefactor * local_rys_weights[root];
                }
                const double tmp4 = 0.5 /
                    (u2 * zeta + pq_exponent * rs_exponent);
                const double tmp5 = u2 * tmp4;
                const double tmp2 = 2.0 * tmp5 * rs_exponent;
                const double tmp3 = 2.0 * tmp5 * pq_exponent;
                b00 = tmp5;
                b10 = tmp5 + tmp4 * rs_exponent;
                b01 = tmp5 + tmp4 * pq_exponent;
                const double pq_center[3] = {
                    pq_px[idx_pq], pq_py[idx_pq], pq_pz[idx_pq]
                };
                const double rs_center[3] = {
                    rs_px[idx_rs], rs_py[idx_rs], rs_pz[idx_rs]
                };
                for (int axis = 0; axis < 3; ++axis) {
                    c00[axis] = pq_center[axis] - origins[3 * p0 + axis] -
                        tmp2 * PQ[axis];
                    c0p[axis] = rs_center[axis] - origins[3 * r0 + axis] +
                        tmp3 * PQ[axis];
                }
                double recurrence[3][3][3];
                for (int axis = 0; axis < 3; ++axis) {
                    rys_build_1d_sp(
                        c00[axis],
                        c0p[axis],
                        b10, b01, b00, recurrence[axis]
                    );
                }
                double distributed[3][16];
                for (int axis = 0; axis < 3; ++axis) {
                    rys_distribute_sp_active<ACTIVE_CENTER_MASK>(
                        recurrence[axis], AB[axis], CD[axis], distributed[axis]
                    );
                }
                if (!use_active_outputs) {
                    std::size_t index = 0;
                    for (int ip = 0; ip < NP; ++ip) {
                        for (int iq = 0; iq < NQ; ++iq) {
                            for (int ir = 0; ir < NR; ++ir) {
                                for (int is = 0; is < NS; ++is, ++index) {
                                    const unsigned int code_x =
                                        (LP && ip == 0 ? 1 : 0) |
                                        (LQ && iq == 0 ? 2 : 0) |
                                        (LR && ir == 0 ? 4 : 0) |
                                        (LS && is == 0 ? 8 : 0);
                                    const unsigned int code_y =
                                        (LP && ip == 1 ? 1 : 0) |
                                        (LQ && iq == 1 ? 2 : 0) |
                                        (LR && ir == 1 ? 4 : 0) |
                                        (LS && is == 1 ? 8 : 0);
                                    const unsigned int code_z =
                                        (LP && ip == 2 ? 1 : 0) |
                                        (LQ && iq == 2 ? 2 : 0) |
                                        (LR && ir == 2 ? 4 : 0) |
                                        (LS && is == 2 ? 8 : 0);
                                    const double product =
                                        distributed[0][code_x] *
                                        distributed[1][code_y] *
                                        distributed[2][code_z];
                                    contracted_values[index] += root_prefactor * product;
                                }
                            }
                        }
                    }
                } else {
                    for (std::size_t active = 0; active < nactive; ++active) {
                        const SPActiveOutput& item = active_outputs[active];
                        const double product =
                            distributed[0][item.axis_codes[0]] *
                            distributed[1][item.axis_codes[1]] *
                            distributed[2][item.axis_codes[2]];
                        contracted_values[item.index] += root_prefactor * product;
                    }
                }
            }
        }
    }
    }
    if (direct != nullptr) {
        if (
            direct->symmetric_density && direct->screen_tol <= 0.0 &&
            !pq_same_shell && !rs_same_shell && !same_pair
        ) {
            const bool all_shells_distinct =
                direct->starts[0] != direct->starts[1] &&
                direct->starts[0] != direct->starts[2] &&
                direct->starts[0] != direct->starts[3] &&
                direct->starts[1] != direct->starts[2] &&
                direct->starts[1] != direct->starts[3] &&
                direct->starts[2] != direct->starts[3];
            if (all_shells_distinct) {
                direct_jk_accumulate_distinct_shell_tiles<NP, NQ, NR, NS>(
                    *direct,
                    [&](std::size_t index, int, int, int, int) {
                        return contracted_values[index];
                    }
                );
                return true;
            }
            std::size_t index = 0;
            for (int ip = 0; ip < NP; ++ip) {
                const npy_intp gp = direct->starts[0] + ip;
                for (int iq = 0; iq < NQ; ++iq) {
                    const npy_intp gq = direct->starts[1] + iq;
                    for (int ir = 0; ir < NR; ++ir) {
                        const npy_intp gr = direct->starts[2] + ir;
                        for (int is = 0; is < NS; ++is, ++index) {
                            const npy_intp gs = direct->starts[3] + is;
                            direct_jk_add_symmetric_class<false, false, false>(
                                direct->vj,
                                direct->vk,
                                direct->dm,
                                direct->nao,
                                contracted_values[index],
                                gp,
                                gq,
                                gr,
                                gs
                            );
                            ++*direct->computed;
                        }
                    }
                }
            }
            return true;
        }
        auto scatter = [&](int ip, int iq, int ir, int is, std::size_t index) {
            const npy_intp gp = direct->starts[0] + ip;
            const npy_intp gq = direct->starts[1] + iq;
            const npy_intp gr = direct->starts[2] + ir;
            const npy_intp gs = direct->starts[3] + is;
            if (gp < gq || gr < gs) return;
            if (
                direct->same_pair &&
                pair_index(static_cast<int>(gp), static_cast<int>(gq)) <
                    pair_index(static_cast<int>(gr), static_cast<int>(gs))
            ) return;
            const double value = contracted_values[index];
            if (direct->screen_tol > 0.0 && std::abs(value) < direct->screen_tol) {
                ++*direct->skipped;
                return;
            }
            ++*direct->computed;
            if (direct->symmetric_density) {
                direct_jk_add_unique_permutations_symmetric_density(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, gp, gq, gr, gs
                );
            } else {
                direct_jk_add_unique_permutations(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, gp, gq, gr, gs
                );
            }
        };
        if (use_active_outputs) {
            for (std::size_t active = 0; active < nactive; ++active) {
                const SPActiveOutput& item = active_outputs[active];
                scatter(item.ip, item.iq, item.ir, item.is, item.index);
            }
        } else {
            std::size_t index = 0;
            for (int ip = 0; ip < NP; ++ip) {
                for (int iq = 0; iq < NQ; ++iq) {
                    for (int ir = 0; ir < NR; ++ir) {
                        for (int is = 0; is < NS; ++is, ++index) {
                            scatter(ip, iq, ir, is, index);
                        }
                    }
                }
            }
        }
    } else {
        values.assign(contracted_values.begin(), contracted_values.end());
    }
    return true;
}

bool compute_sp_shell_quartet_rys_values_specialized(
    int shell_mask,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp q0,
    npy_intp r0,
    npy_intp s0,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    const double* pq_weight,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    const double* rs_weight,
    int nrs,
    const RysQuadratureCache* quadrature_cache,
    bool unique_direct,
    bool pq_same_shell,
    bool rs_same_shell,
    bool same_pair,
    const SPDirectJKConsumer* direct,
    std::vector<double>& values
) {
#define PYQED_SP_RYS_CASE(mask, lp, lq, lr, ls, cached) \
    case mask: return compute_sp_shell_quartet_rys_values_fixed<lp, lq, lr, ls, cached>( \
        origins, weights, nprim, max_prim, p0, q0, r0, s0, \
        pq_p, pq_px, pq_py, pq_pz, pq_k, pq_weight, npq, \
        rs_p, rs_px, rs_py, rs_pz, rs_k, rs_weight, nrs, \
        quadrature_cache, \
        unique_direct, pq_same_shell, rs_same_shell, same_pair, direct, values)
#define PYQED_SP_RYS_SWITCH(cached) \
    switch (shell_mask) { \
        PYQED_SP_RYS_CASE(0, 0, 0, 0, 0, cached); \
        PYQED_SP_RYS_CASE(1, 1, 0, 0, 0, cached); \
        PYQED_SP_RYS_CASE(2, 0, 1, 0, 0, cached); \
        PYQED_SP_RYS_CASE(3, 1, 1, 0, 0, cached); \
        PYQED_SP_RYS_CASE(4, 0, 0, 1, 0, cached); \
        PYQED_SP_RYS_CASE(5, 1, 0, 1, 0, cached); \
        PYQED_SP_RYS_CASE(6, 0, 1, 1, 0, cached); \
        PYQED_SP_RYS_CASE(7, 1, 1, 1, 0, cached); \
        PYQED_SP_RYS_CASE(8, 0, 0, 0, 1, cached); \
        PYQED_SP_RYS_CASE(9, 1, 0, 0, 1, cached); \
        PYQED_SP_RYS_CASE(10, 0, 1, 0, 1, cached); \
        PYQED_SP_RYS_CASE(11, 1, 1, 0, 1, cached); \
        PYQED_SP_RYS_CASE(12, 0, 0, 1, 1, cached); \
        PYQED_SP_RYS_CASE(13, 1, 0, 1, 1, cached); \
        PYQED_SP_RYS_CASE(14, 0, 1, 1, 1, cached); \
        PYQED_SP_RYS_CASE(15, 1, 1, 1, 1, cached); \
        default: return false; \
    }
    if (quadrature_cache != nullptr) {
        PYQED_SP_RYS_SWITCH(true);
    }
    PYQED_SP_RYS_SWITCH(false);
#undef PYQED_SP_RYS_SWITCH
#undef PYQED_SP_RYS_CASE
}

inline void rys_build_1d_general(
    int bra_order,
    int ket_order,
    double c00,
    double c0p,
    double b10,
    double b01,
    double b00,
    double out[7][7]
) {
    out[0][0] = 1.0;
    for (int i = 1; i <= bra_order; ++i) {
        out[i][0] = c00 * out[i - 1][0];
        if (i > 1) out[i][0] += (i - 1) * b10 * out[i - 2][0];
    }
    for (int j = 1; j <= ket_order; ++j) {
        out[0][j] = c0p * out[0][j - 1];
        if (j > 1) out[0][j] += (j - 1) * b01 * out[0][j - 2];
        for (int i = 1; i <= bra_order; ++i) {
            out[i][j] = c00 * out[i - 1][j] + j * b00 * out[i - 1][j - 1];
            if (i > 1) out[i][j] += (i - 1) * b10 * out[i - 2][j];
        }
    }
}

struct RysCartesianOutputAngular {
    unsigned char axis_index[3]{};
    unsigned char local_index[4]{};
    unsigned char symmetry_class = 0;
    double weight_scale = 1.0;
};

struct RysCartesianOutputPlan {
    std::vector<RysCartesianOutputAngular> outputs;
    std::vector<std::size_t> dense_to_output;
};

inline int rys_cartesian_plan_key(
    int lp,
    int lq,
    int lr,
    int ls,
    bool pq_same_shell,
    bool rs_same_shell,
    bool same_pair
) {
    const int shape = lp | (lq << 2) | (lr << 4) | (ls << 6);
    const int symmetry = static_cast<int>(pq_same_shell) |
        (static_cast<int>(rs_same_shell) << 1) |
        (static_cast<int>(same_pair) << 2);
    return shape | (symmetry << 8);
}

struct RysSphericalTerm {
    unsigned char axis_slot[3]{};
    double coefficient = 0.0;
    double weight_scale = 1.0;
};

struct RysAxisExpansionTerm {
    unsigned char bra_order = 0;
    unsigned char ket_order = 0;
    double coefficient = 0.0;
};

struct RysSphericalOutput {
    npy_intp p = 0;
    npy_intp q = 0;
    npy_intp r = 0;
    npy_intp s = 0;
    unsigned char symmetry_class = 0;
};

struct RysSphericalOutputPlan {
    std::vector<RysSphericalOutput> outputs;
    std::vector<std::size_t> offsets;
    std::vector<RysSphericalTerm> terms;
    std::array<std::vector<unsigned char>, 3> axis_codes;
    std::array<std::vector<std::size_t>, 3> axis_offsets;
    std::array<std::vector<RysAxisExpansionTerm>, 3> axis_terms;
};

std::size_t rys_spherical_output_plan_bytes(const RysSphericalOutputPlan& plan) {
    std::size_t bytes = sizeof(RysSphericalOutputPlan) +
        plan.outputs.size() * sizeof(RysSphericalOutput) +
        plan.offsets.size() * sizeof(std::size_t) +
        plan.terms.size() * sizeof(RysSphericalTerm);
    for (int axis = 0; axis < 3; ++axis) {
        bytes += plan.axis_codes[axis].size() * sizeof(unsigned char);
        bytes += plan.axis_offsets[axis].size() * sizeof(std::size_t);
        bytes += plan.axis_terms[axis].size() * sizeof(RysAxisExpansionTerm);
    }
    return bytes;
}

bool build_rys_cartesian_output_plan(
    const std::int64_t* shells,
    const double* weights,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp r0,
    npy_intp r1,
    npy_intp s0,
    npy_intp s1,
    bool pack_s8,
    RysCartesianOutputPlan& plan
) {
    const npy_intp counts[4] = {p1 - p0, q1 - q0, r1 - r0, s1 - s0};
    const std::size_t nout = static_cast<std::size_t>(counts[0]) * counts[1] *
        counts[2] * counts[3];
    if (nout == 0 || nout > 10000) return false;
    plan.outputs.clear();
    plan.outputs.reserve(nout);
    plan.dense_to_output.resize(nout);
    std::unordered_map<npy_intp, std::size_t> s8_to_output;
    if (pack_s8) s8_to_output.reserve(nout);
    std::size_t dense_output = 0;
    for (npy_intp ip = 0; ip < counts[0]; ++ip) {
        for (npy_intp iq = 0; iq < counts[1]; ++iq) {
            for (npy_intp ir = 0; ir < counts[2]; ++ir) {
                for (npy_intp is = 0; is < counts[3]; ++is, ++dense_output) {
                    const npy_intp aos[4] = {p0 + ip, q0 + iq, r0 + ir, s0 + is};
                    std::size_t output = plan.outputs.size();
                    if (pack_s8) {
                        const npy_intp s8_index = pair_pair_index(
                            pair_index(static_cast<int>(aos[0]), static_cast<int>(aos[1])),
                            pair_index(static_cast<int>(aos[2]), static_cast<int>(aos[3]))
                        );
                        const auto inserted = s8_to_output.emplace(s8_index, output);
                        if (!inserted.second) {
                            plan.dense_to_output[dense_output] = inserted.first->second;
                            continue;
                        }
                    }
                    plan.dense_to_output[dense_output] = output;
                    plan.outputs.emplace_back();
                    RysCartesianOutputAngular& item = plan.outputs.back();
                    item.local_index[0] = static_cast<unsigned char>(ip);
                    item.local_index[1] = static_cast<unsigned char>(iq);
                    item.local_index[2] = static_cast<unsigned char>(ir);
                    item.local_index[3] = static_cast<unsigned char>(is);
                    item.symmetry_class = static_cast<unsigned char>(
                        (aos[0] == aos[1]) |
                        ((aos[2] == aos[3]) << 1) |
                        ((aos[0] == aos[2] && aos[1] == aos[3]) << 2)
                    );
                    item.weight_scale =
                        weights[aos[0] * max_prim] / weights[p0 * max_prim] *
                        weights[aos[1] * max_prim] / weights[q0 * max_prim] *
                        weights[aos[2] * max_prim] / weights[r0 * max_prim] *
                        weights[aos[3] * max_prim] / weights[s0 * max_prim];
                    for (int axis = 0; axis < 3; ++axis) {
                        item.axis_index[axis] = static_cast<unsigned char>(
                            shells[3 * aos[0] + axis] |
                            (shells[3 * aos[1] + axis] << 2) |
                            (shells[3 * aos[2] + axis] << 4) |
                            (shells[3 * aos[3] + axis] << 6)
                        );
                    }
                }
            }
        }
    }
    return true;
}

void rys_build_distributed_axis_tables(
    const double recurrence[3][7][7],
    const int shell_l[4],
    const double AB[3],
    const double CD[3],
    double distributed[3][256]
) {
    const int lab = shell_l[0] + shell_l[1];
    const int lcd = shell_l[2] + shell_l[3];
    for (int axis = 0; axis < 3; ++axis) {
        double hrr[7][4][7][4];
        for (int a = 0; a <= lab; ++a) {
            for (int c = 0; c <= lcd; ++c) {
                hrr[a][0][c][0] = recurrence[axis][a][c];
            }
        }
        for (int b = 1; b <= shell_l[1]; ++b) {
            for (int a = 0; a <= lab - b; ++a) {
                for (int c = 0; c <= lcd; ++c) {
                    hrr[a][b][c][0] = hrr[a + 1][b - 1][c][0] +
                        AB[axis] * hrr[a][b - 1][c][0];
                }
            }
        }
        for (int d = 1; d <= shell_l[3]; ++d) {
            for (int a = 0; a <= shell_l[0]; ++a) {
                for (int b = 0; b <= shell_l[1]; ++b) {
                    for (int c = 0; c <= lcd - d; ++c) {
                        hrr[a][b][c][d] = hrr[a][b][c + 1][d - 1] +
                            CD[axis] * hrr[a][b][c][d - 1];
                    }
                }
            }
        }
        for (int a = 0; a <= shell_l[0]; ++a) {
            for (int b = 0; b <= shell_l[1]; ++b) {
                for (int c = 0; c <= shell_l[2]; ++c) {
                    for (int d = 0; d <= shell_l[3]; ++d) {
                        const unsigned int index = static_cast<unsigned int>(
                            a | (b << 2) | (c << 4) | (d << 6)
                        );
                        distributed[axis][index] = hrr[a][b][c][d];
                    }
                }
            }
        }
    }
}

template<int BRA_ORDER, int KET_ORDER>
inline void rys_build_1d_general_fixed(
    double c00,
    double c0p,
    double b10,
    double b01,
    double b00,
    double (&out)[BRA_ORDER + 1][KET_ORDER + 1]
) {
    out[0][0] = 1.0;
    for (int i = 1; i <= BRA_ORDER; ++i) {
        out[i][0] = c00 * out[i - 1][0];
        if (i > 1) out[i][0] += (i - 1) * b10 * out[i - 2][0];
    }
    for (int j = 1; j <= KET_ORDER; ++j) {
        out[0][j] = c0p * out[0][j - 1];
        if (j > 1) out[0][j] += (j - 1) * b01 * out[0][j - 2];
        for (int i = 1; i <= BRA_ORDER; ++i) {
            out[i][j] = c00 * out[i - 1][j] + j * b00 * out[i - 1][j - 1];
            if (i > 1) out[i][j] += (i - 1) * b10 * out[i - 2][j];
        }
    }
}

template<int LP, int LQ, int LR, int LS, std::size_t NDISTRIBUTED>
inline void rys_build_shell_shape_tables_fixed(
    const double c00[3],
    const double c0p[3],
    double b10,
    double b01,
    double b00,
    const double AB[3],
    const double CD[3],
    double (&distributed)[3][NDISTRIBUTED]
) {
    constexpr int LAB = LP + LQ;
    constexpr int LCD = LR + LS;
    constexpr std::size_t LAST_DISTRIBUTED = static_cast<std::size_t>(
        LP | (LQ << 2) | (LR << 4) | (LS << 6)
    );
    static_assert(NDISTRIBUTED > LAST_DISTRIBUTED);
    for (int axis = 0; axis < 3; ++axis) {
        double recurrence[LAB + 1][LCD + 1];
        rys_build_1d_general_fixed<LAB, LCD>(
            c00[axis], c0p[axis], b10, b01, b00, recurrence
        );
        double hrr[LAB + 1][LQ + 1][LCD + 1][LS + 1];
        for (int a = 0; a <= LAB; ++a) {
            for (int c = 0; c <= LCD; ++c) {
                hrr[a][0][c][0] = recurrence[a][c];
            }
        }
        for (int b = 1; b <= LQ; ++b) {
            for (int a = 0; a <= LAB - b; ++a) {
                for (int c = 0; c <= LCD; ++c) {
                    hrr[a][b][c][0] = hrr[a + 1][b - 1][c][0] +
                        AB[axis] * hrr[a][b - 1][c][0];
                }
            }
        }
        for (int d = 1; d <= LS; ++d) {
            for (int a = 0; a <= LP; ++a) {
                for (int b = 0; b <= LQ; ++b) {
                    for (int c = 0; c <= LCD - d; ++c) {
                        hrr[a][b][c][d] = hrr[a][b][c + 1][d - 1] +
                            CD[axis] * hrr[a][b][c][d - 1];
                    }
                }
            }
        }
        for (int a = 0; a <= LP; ++a) {
            for (int b = 0; b <= LQ; ++b) {
                for (int c = 0; c <= LR; ++c) {
                    for (int d = 0; d <= LS; ++d) {
                        const unsigned int index = static_cast<unsigned int>(
                            a | (b << 2) | (c << 4) | (d << 6)
                        );
                        distributed[axis][index] = hrr[a][b][c][d];
                    }
                }
            }
        }
    }
}

inline void rys_build_shell_shape_tables(
    const int shell_l[4],
    const double c00[3],
    const double c0p[3],
    double b10,
    double b01,
    double b00,
    const double AB[3],
    const double CD[3],
    double (&distributed)[3][256]
) {
    const int shape = shell_l[0] | (shell_l[1] << 2) |
        (shell_l[2] << 4) | (shell_l[3] << 6);
#define PYQED_RYS_D_CASE(lp, lq, lr, ls) \
    case ((lp) | ((lq) << 2) | ((lr) << 4) | ((ls) << 6)): \
        rys_build_shell_shape_tables_fixed<lp, lq, lr, ls>( \
            c00, c0p, b10, b01, b00, AB, CD, distributed \
        ); \
        return
#define PYQED_RYS_D_LS(lp, lq, lr) \
    PYQED_RYS_D_CASE(lp, lq, lr, 0); \
    PYQED_RYS_D_CASE(lp, lq, lr, 1); \
    PYQED_RYS_D_CASE(lp, lq, lr, 2)
#define PYQED_RYS_D_LR(lp, lq) \
    PYQED_RYS_D_LS(lp, lq, 0); \
    PYQED_RYS_D_LS(lp, lq, 1); \
    PYQED_RYS_D_LS(lp, lq, 2)
#define PYQED_RYS_D_LQ(lp) \
    PYQED_RYS_D_LR(lp, 0); \
    PYQED_RYS_D_LR(lp, 1); \
    PYQED_RYS_D_LR(lp, 2)
    switch (shape) {
        PYQED_RYS_D_LQ(0);
        PYQED_RYS_D_LQ(1);
        PYQED_RYS_D_LQ(2);
        default:
            break;
    }
#undef PYQED_RYS_D_LQ
#undef PYQED_RYS_D_LR
#undef PYQED_RYS_D_LS
#undef PYQED_RYS_D_CASE
    double recurrence[3][7][7];
    for (int axis = 0; axis < 3; ++axis) {
        rys_build_1d_general(
            shell_l[0] + shell_l[1],
            shell_l[2] + shell_l[3],
            c00[axis], c0p[axis], b10, b01, b00,
            recurrence[axis]
        );
    }
    rys_build_distributed_axis_tables(recurrence, shell_l, AB, CD, distributed);
}

template<int LP, int LQ, int LR, int LS, bool CACHED>
bool compute_l3_shell_quartet_rys_values_fixed(
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp r0,
    npy_intp r1,
    npy_intp s0,
    npy_intp s1,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    const double* pq_weight,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    const double* rs_weight,
    int nrs,
    std::vector<double>& values,
    const RysCartesianOutputPlan* reusable_output_plan = nullptr,
    const RysQuadratureCache* quadrature_cache = nullptr,
    const CartesianDirectJKConsumer* direct = nullptr
) {
    constexpr bool fixed_shape = LP >= 0;
    static_assert(
        (fixed_shape && LQ >= 0 && LR >= 0 && LS >= 0) ||
        (!fixed_shape && LQ < 0 && LR < 0 && LS < 0)
    );
    const int shell_l[4] = {
        fixed_shape ? LP : static_cast<int>(
            shells[3 * p0] + shells[3 * p0 + 1] + shells[3 * p0 + 2]
        ),
        fixed_shape ? LQ : static_cast<int>(
            shells[3 * q0] + shells[3 * q0 + 1] + shells[3 * q0 + 2]
        ),
        fixed_shape ? LR : static_cast<int>(
            shells[3 * r0] + shells[3 * r0 + 1] + shells[3 * r0 + 2]
        ),
        fixed_shape ? LS : static_cast<int>(
            shells[3 * s0] + shells[3 * s0 + 1] + shells[3 * s0 + 2]
        ),
    };
    int rank = 0;
    for (int center = 0; center < 4; ++center) {
        if (shell_l[center] < 0 || shell_l[center] > 3) return false;
        rank += shell_l[center];
    }
    const int nroots = rank / 2 + 1;
    const npy_intp counts[4] = {p1 - p0, q1 - q0, r1 - r0, s1 - s0};
    const std::size_t nout = static_cast<std::size_t>(counts[0]) * counts[1] * counts[2] * counts[3];
    if (nout == 0 || nout > 10000) return false;
    constexpr std::size_t fixed_nout = fixed_shape
        ? static_cast<std::size_t>((LP + 1) * (LP + 2) / 2) *
            static_cast<std::size_t>((LQ + 1) * (LQ + 2) / 2) *
            static_cast<std::size_t>((LR + 1) * (LR + 2) / 2) *
            static_cast<std::size_t>((LS + 1) * (LS + 2) / 2)
        : 1;
    const bool use_fixed_direct = fixed_shape && direct != nullptr;
    RysCartesianOutputPlan local_output_plan;
    if (!use_fixed_direct && (
        reusable_output_plan == nullptr ||
        reusable_output_plan->outputs.empty() ||
        reusable_output_plan->dense_to_output.size() != nout
    )) {
        if (!build_rys_cartesian_output_plan(
                shells, weights, max_prim,
                p0, p1, q0, q1, r0, r1, s0, s1,
                false,
                local_output_plan
            )) return false;
        reusable_output_plan = &local_output_plan;
    }
    const std::vector<RysCartesianOutputAngular>& output_angular =
        use_fixed_direct ? local_output_plan.outputs : reusable_output_plan->outputs;
    std::array<double, fixed_nout> fixed_direct_values{};
    const RysFixedDirectPlan<fixed_nout>* fixed_symmetry_plan = nullptr;
    if constexpr (fixed_shape) {
        if (direct != nullptr && direct->same_pair) {
            if (direct->pq_same_shell) {
                fixed_symmetry_plan = &rys_fixed_direct_plan<
                    LP, LQ, LR, LS, true, true, true
                >();
            } else {
                fixed_symmetry_plan = &rys_fixed_direct_plan<
                    LP, LQ, LR, LS, false, false, true
                >();
            }
        } else if (direct != nullptr && direct->pq_same_shell) {
            if (direct->rs_same_shell) {
                fixed_symmetry_plan = &rys_fixed_direct_plan<
                    LP, LQ, LR, LS, true, true, false
                >();
            } else {
                fixed_symmetry_plan = &rys_fixed_direct_plan<
                    LP, LQ, LR, LS, true, false, false
                >();
            }
        } else if (direct != nullptr && direct->rs_same_shell) {
            fixed_symmetry_plan = &rys_fixed_direct_plan<
                LP, LQ, LR, LS, false, true, false
            >();
        }
    }
    if (!use_fixed_direct) values.assign(output_angular.size(), 0.0);

    const double AB[3] = {
        origins[3 * p0] - origins[3 * q0],
        origins[3 * p0 + 1] - origins[3 * q0 + 1],
        origins[3 * p0 + 2] - origins[3 * q0 + 2],
    };
    const double CD[3] = {
        origins[3 * r0] - origins[3 * s0],
        origins[3 * r0 + 1] - origins[3 * s0 + 1],
        origins[3 * r0 + 2] - origins[3 * s0 + 2],
    };
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        const double left_weight = pq_weight[idx_pq];
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const double pq_exponent = pq_p[idx_pq];
            const double rs_exponent = rs_p[idx_rs];
            const double zeta = pq_exponent + rs_exponent;
            const double alpha = pq_exponent * rs_exponent / zeta;
            const double PQ[3] = {
                pq_px[idx_pq] - rs_px[idx_rs],
                pq_py[idx_pq] - rs_py[idx_rs],
                pq_pz[idx_pq] - rs_pz[idx_rs],
            };
            double local_roots[7]{};
            double local_rys_weights[7]{};
            const std::size_t quadrature_offset =
                (static_cast<std::size_t>(idx_pq) * nrs + idx_rs) * nroots;
            double primitive_prefactor = 0.0;
            if constexpr (!CACHED) {
                const double T = alpha * (
                    PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]
                );
                if (!rys_roots_weights_general(
                        nroots, T, local_roots, local_rys_weights
                    )) return false;
                primitive_prefactor = left_weight * rs_weight[idx_rs] * ERI_PREFAC *
                    pq_k[idx_pq] * rs_k[idx_rs] /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
            }
            #pragma clang loop unroll_count(2)
            for (int root = 0; root < nroots; ++root) {
                double c00[3]{};
                double c0p[3]{};
                double b00 = 0.0;
                double b10 = 0.0;
                double b01 = 0.0;
                double root_prefactor = 0.0;
                double u2 = 0.0;
                if constexpr (CACHED) {
                    const RysQuadraturePointData& item =
                        quadrature_cache->points[quadrature_offset + root];
                    u2 = item.u2;
                    root_prefactor = pq_weight[idx_pq] * rs_weight[idx_rs] *
                        item.root_prefactor;
                } else {
                    u2 = alpha * local_roots[root];
                    root_prefactor = primitive_prefactor * local_rys_weights[root];
                }
                const double tmp4 = 0.5 /
                    (u2 * zeta + pq_exponent * rs_exponent);
                const double tmp5 = u2 * tmp4;
                const double tmp2 = 2.0 * tmp5 * rs_exponent;
                const double tmp3 = 2.0 * tmp5 * pq_exponent;
                b00 = tmp5;
                b10 = tmp5 + tmp4 * rs_exponent;
                b01 = tmp5 + tmp4 * pq_exponent;
                const double pq_center[3] = {
                    pq_px[idx_pq], pq_py[idx_pq], pq_pz[idx_pq]
                };
                const double rs_center[3] = {
                    rs_px[idx_rs], rs_py[idx_rs], rs_pz[idx_rs]
                };
                for (int axis = 0; axis < 3; ++axis) {
                    c00[axis] = pq_center[axis] - origins[3 * p0 + axis] -
                        tmp2 * PQ[axis];
                    c0p[axis] = rs_center[axis] - origins[3 * r0 + axis] +
                        tmp3 * PQ[axis];
                }
                constexpr std::size_t distributed_size = fixed_shape
                    ? static_cast<std::size_t>(
                        LP | (LQ << 2) | (LR << 4) | (LS << 6)
                    ) + 1
                    : 256;
                double distributed[3][distributed_size];
                if constexpr (fixed_shape) {
                    rys_build_shell_shape_tables_fixed<LP, LQ, LR, LS>(
                        c00, c0p, b10, b01, b00, AB, CD, distributed
                    );
                } else {
                    rys_build_shell_shape_tables(
                        shell_l, c00, c0p, b10, b01, b00, AB, CD, distributed
                    );
                }
                if (use_fixed_direct) {
                    if (fixed_symmetry_plan != nullptr) {
                        for (
                            std::size_t output = 0;
                            output < fixed_symmetry_plan->size;
                            ++output
                        ) {
                            const RysFixedDirectOutput& item =
                                fixed_symmetry_plan->outputs[output];
                            fixed_direct_values[output] += root_prefactor *
                                distributed[0][item.axis_index[0]] *
                                distributed[1][item.axis_index[1]] *
                                distributed[2][item.axis_index[2]];
                        }
                    } else {
                        constexpr int NP = (LP + 1) * (LP + 2) / 2;
                        constexpr int NQ = (LQ + 1) * (LQ + 2) / 2;
                        constexpr int NR = (LR + 1) * (LR + 2) / 2;
                        constexpr int NS = (LS + 1) * (LS + 2) / 2;
                        std::size_t index = 0;
                        for (int ip = 0; ip < NP; ++ip) {
                            for (int iq = 0; iq < NQ; ++iq) {
                                for (int ir = 0; ir < NR; ++ir) {
                                    for (int is = 0; is < NS; ++is, ++index) {
                                    const unsigned int code_x =
                                        CARTESIAN_POWERS_L2[LP][ip][0] |
                                        (CARTESIAN_POWERS_L2[LQ][iq][0] << 2) |
                                        (CARTESIAN_POWERS_L2[LR][ir][0] << 4) |
                                        (CARTESIAN_POWERS_L2[LS][is][0] << 6);
                                    const unsigned int code_y =
                                        CARTESIAN_POWERS_L2[LP][ip][1] |
                                        (CARTESIAN_POWERS_L2[LQ][iq][1] << 2) |
                                        (CARTESIAN_POWERS_L2[LR][ir][1] << 4) |
                                        (CARTESIAN_POWERS_L2[LS][is][1] << 6);
                                    const unsigned int code_z =
                                        CARTESIAN_POWERS_L2[LP][ip][2] |
                                        (CARTESIAN_POWERS_L2[LQ][iq][2] << 2) |
                                        (CARTESIAN_POWERS_L2[LR][ir][2] << 4) |
                                        (CARTESIAN_POWERS_L2[LS][is][2] << 6);
                                    fixed_direct_values[index] += root_prefactor *
                                        distributed[0][code_x] *
                                        distributed[1][code_y] *
                                        distributed[2][code_z];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    std::size_t index = 0;
                    for (; index + 3 < output_angular.size(); index += 4) {
                        const auto& item0 = output_angular[index];
                        const auto& item1 = output_angular[index + 1];
                        const auto& item2 = output_angular[index + 2];
                        const auto& item3 = output_angular[index + 3];
                        const double product0 =
                            distributed[0][item0.axis_index[0]] *
                            distributed[1][item0.axis_index[1]] *
                            distributed[2][item0.axis_index[2]];
                        const double product1 =
                            distributed[0][item1.axis_index[0]] *
                            distributed[1][item1.axis_index[1]] *
                            distributed[2][item1.axis_index[2]];
                        const double product2 =
                            distributed[0][item2.axis_index[0]] *
                            distributed[1][item2.axis_index[1]] *
                            distributed[2][item2.axis_index[2]];
                        const double product3 =
                            distributed[0][item3.axis_index[0]] *
                            distributed[1][item3.axis_index[1]] *
                            distributed[2][item3.axis_index[2]];
                        values[index] += root_prefactor * product0;
                        values[index + 1] += root_prefactor * product1;
                        values[index + 2] += root_prefactor * product2;
                        values[index + 3] += root_prefactor * product3;
                    }
                    for (; index < output_angular.size(); ++index) {
                        const auto& item = output_angular[index];
                        values[index] += root_prefactor *
                            distributed[0][item.axis_index[0]] *
                            distributed[1][item.axis_index[1]] *
                            distributed[2][item.axis_index[2]];
                    }
                }
            }
        }
    }
    if (use_fixed_direct) {
        constexpr int NP = (LP + 1) * (LP + 2) / 2;
        constexpr int NQ = (LQ + 1) * (LQ + 2) / 2;
        constexpr int NR = (LR + 1) * (LR + 2) / 2;
        constexpr int NS = (LS + 1) * (LS + 2) / 2;
        const double p_weight = weights[p0 * max_prim];
        const double q_weight = weights[q0 * max_prim];
        const double r_weight = weights[r0 * max_prim];
        const double s_weight = weights[s0 * max_prim];
        double component_scale[4][6]{};
        for (int ip = 0; ip < NP; ++ip) {
            component_scale[0][ip] = weights[(p0 + ip) * max_prim] / p_weight;
        }
        for (int iq = 0; iq < NQ; ++iq) {
            component_scale[1][iq] = weights[(q0 + iq) * max_prim] / q_weight;
        }
        for (int ir = 0; ir < NR; ++ir) {
            component_scale[2][ir] = weights[(r0 + ir) * max_prim] / r_weight;
        }
        for (int is = 0; is < NS; ++is) {
            component_scale[3][is] = weights[(s0 + is) * max_prim] / s_weight;
        }
        auto fixed_value_at = [&](
            std::size_t output,
            const RysFixedDirectOutput& item
        ) {
            return fixed_direct_values[output] *
                component_scale[0][item.local_index[0]] *
                component_scale[1][item.local_index[1]] *
                component_scale[2][item.local_index[2]] *
                component_scale[3][item.local_index[3]];
        };
        if (fixed_symmetry_plan != nullptr) {
            if (direct->symmetric_density) {
                consume_rys_fixed_direct_class<0>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
                consume_rys_fixed_direct_class<1>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
                consume_rys_fixed_direct_class<2>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
                consume_rys_fixed_direct_class<3>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
                consume_rys_fixed_direct_class<4>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
                consume_rys_fixed_direct_class<7>(
                    *fixed_symmetry_plan, *direct, fixed_value_at
                );
            } else {
                for (
                    std::size_t output = 0;
                    output < fixed_symmetry_plan->size;
                    ++output
                ) {
                    const RysFixedDirectOutput& item =
                        fixed_symmetry_plan->outputs[output];
                    direct_jk_add_unique_permutations(
                        direct->vj, direct->vk, direct->dm, direct->nao,
                        fixed_value_at(output, item),
                        direct->starts[0] + item.local_index[0],
                        direct->starts[1] + item.local_index[1],
                        direct->starts[2] + item.local_index[2],
                        direct->starts[3] + item.local_index[3]
                    );
                    ++*direct->computed;
                }
            }
            return true;
        }
        const bool all_shells_distinct =
            direct->starts[0] != direct->starts[1] &&
            direct->starts[0] != direct->starts[2] &&
            direct->starts[0] != direct->starts[3] &&
            direct->starts[1] != direct->starts[2] &&
            direct->starts[1] != direct->starts[3] &&
            direct->starts[2] != direct->starts[3];
        if (direct->symmetric_density && all_shells_distinct) {
            direct_jk_accumulate_distinct_shell_tiles<NP, NQ, NR, NS>(
                *direct,
                [&](std::size_t index, int ip, int iq, int ir, int is) {
                    return fixed_direct_values[index] *
                        component_scale[0][ip] * component_scale[1][iq] *
                        component_scale[2][ir] * component_scale[3][is];
                }
            );
            return true;
        }
        std::size_t index = 0;
        for (int ip = 0; ip < NP; ++ip) {
            const npy_intp p = direct->starts[0] + ip;
            const double scale_p = component_scale[0][ip];
            for (int iq = 0; iq < NQ; ++iq) {
                const npy_intp q = direct->starts[1] + iq;
                const double scale_q = component_scale[1][iq];
                for (int ir = 0; ir < NR; ++ir) {
                    const npy_intp r = direct->starts[2] + ir;
                    const double scale_r = component_scale[2][ir];
                    for (int is = 0; is < NS; ++is, ++index) {
                        const npy_intp s = direct->starts[3] + is;
                        const double value = fixed_direct_values[index] *
                            scale_p * scale_q * scale_r *
                            component_scale[3][is];
                        if (direct->symmetric_density) {
                            direct_jk_add_symmetric_class<false, false, false>(
                                direct->vj, direct->vk, direct->dm, direct->nao,
                                value, p, q, r, s
                            );
                        } else {
                            direct_jk_add_unique_permutations(
                                direct->vj, direct->vk, direct->dm, direct->nao,
                                value, p, q, r, s
                            );
                        }
                        ++*direct->computed;
                    }
                }
            }
        }
    } else if (direct != nullptr) {
        for (std::size_t index = 0; index < output_angular.size(); ++index) {
            const RysCartesianOutputAngular& item = output_angular[index];
            const double value = values[index] * item.weight_scale;
            const npy_intp p = direct->starts[0] + item.local_index[0];
            const npy_intp q = direct->starts[1] + item.local_index[1];
            const npy_intp r = direct->starts[2] + item.local_index[2];
            const npy_intp s = direct->starts[3] + item.local_index[3];
            if (direct->symmetric_density) {
                direct_jk_add_symmetric_class_code(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, p, q, r, s, item.symmetry_class
                );
            } else {
                direct_jk_add_unique_permutations(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, p, q, r, s
                );
            }
            ++*direct->computed;
        }
    } else {
        for (std::size_t index = 0; index < output_angular.size(); ++index) {
            values[index] *= output_angular[index].weight_scale;
        }
    }
    return true;
}

bool compute_l3_shell_quartet_rys_values(
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp r0,
    npy_intp r1,
    npy_intp s0,
    npy_intp s1,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    const double* pq_weight,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    const double* rs_weight,
    int nrs,
    std::vector<double>& values,
    const RysCartesianOutputPlan* reusable_output_plan = nullptr,
    const RysQuadratureCache* quadrature_cache = nullptr,
    const CartesianDirectJKConsumer* direct = nullptr
) {
    const int lp = static_cast<int>(
        shells[3 * p0] + shells[3 * p0 + 1] + shells[3 * p0 + 2]
    );
    const int lq = static_cast<int>(
        shells[3 * q0] + shells[3 * q0 + 1] + shells[3 * q0 + 2]
    );
    const int lr = static_cast<int>(
        shells[3 * r0] + shells[3 * r0 + 1] + shells[3 * r0 + 2]
    );
    const int ls = static_cast<int>(
        shells[3 * s0] + shells[3 * s0 + 1] + shells[3 * s0 + 2]
    );
    const int shape = lp | (lq << 2) | (lr << 4) | (ls << 6);
#define PYQED_RYS_FIXED_QUARTET_CALL(lp_, lq_, lr_, ls_, cached_) \
    return compute_l3_shell_quartet_rys_values_fixed<lp_, lq_, lr_, ls_, cached_>( \
        shells, origins, weights, nprim, max_prim, \
        p0, p1, q0, q1, r0, r1, s0, s1, \
        pq_p, pq_px, pq_py, pq_pz, pq_k, pq_weight, npq, \
        rs_p, rs_px, rs_py, rs_pz, rs_k, rs_weight, nrs, \
        values, reusable_output_plan, quadrature_cache, direct \
    )
#define PYQED_RYS_FIXED_QUARTET_CASE(lp_, lq_, lr_, ls_, cached_) \
    case ((lp_) | ((lq_) << 2) | ((lr_) << 4) | ((ls_) << 6)): \
        PYQED_RYS_FIXED_QUARTET_CALL(lp_, lq_, lr_, ls_, cached_)
#define PYQED_RYS_FIXED_QUARTET_LS(lp_, lq_, lr_, cached_) \
    PYQED_RYS_FIXED_QUARTET_CASE(lp_, lq_, lr_, 0, cached_); \
    PYQED_RYS_FIXED_QUARTET_CASE(lp_, lq_, lr_, 1, cached_); \
    PYQED_RYS_FIXED_QUARTET_CASE(lp_, lq_, lr_, 2, cached_)
#define PYQED_RYS_FIXED_QUARTET_LR(lp_, lq_, cached_) \
    PYQED_RYS_FIXED_QUARTET_LS(lp_, lq_, 0, cached_); \
    PYQED_RYS_FIXED_QUARTET_LS(lp_, lq_, 1, cached_); \
    PYQED_RYS_FIXED_QUARTET_LS(lp_, lq_, 2, cached_)
#define PYQED_RYS_FIXED_QUARTET_LQ(lp_, cached_) \
    PYQED_RYS_FIXED_QUARTET_LR(lp_, 0, cached_); \
    PYQED_RYS_FIXED_QUARTET_LR(lp_, 1, cached_); \
    PYQED_RYS_FIXED_QUARTET_LR(lp_, 2, cached_)
#define PYQED_RYS_FIXED_QUARTET_SWITCH(cached_) \
    switch (shape) { \
        PYQED_RYS_FIXED_QUARTET_LQ(0, cached_); \
        PYQED_RYS_FIXED_QUARTET_LQ(1, cached_); \
        PYQED_RYS_FIXED_QUARTET_LQ(2, cached_); \
        default: break; \
    }
    if (quadrature_cache != nullptr) {
        PYQED_RYS_FIXED_QUARTET_SWITCH(true);
    } else {
        PYQED_RYS_FIXED_QUARTET_SWITCH(false);
    }
#undef PYQED_RYS_FIXED_QUARTET_SWITCH
#undef PYQED_RYS_FIXED_QUARTET_LQ
#undef PYQED_RYS_FIXED_QUARTET_LR
#undef PYQED_RYS_FIXED_QUARTET_LS
#undef PYQED_RYS_FIXED_QUARTET_CASE
#undef PYQED_RYS_FIXED_QUARTET_CALL
#define PYQED_RYS_GENERAL_QUARTET_CALL(cached_) \
    return compute_l3_shell_quartet_rys_values_fixed< \
        -1, -1, -1, -1, cached_ \
    >( \
        shells, origins, weights, nprim, max_prim, \
        p0, p1, q0, q1, r0, r1, s0, s1, \
        pq_p, pq_px, pq_py, pq_pz, pq_k, pq_weight, npq, \
        rs_p, rs_px, rs_py, rs_pz, rs_k, rs_weight, nrs, \
        values, reusable_output_plan, quadrature_cache, direct \
    )
    if (quadrature_cache != nullptr) {
        PYQED_RYS_GENERAL_QUARTET_CALL(true);
    }
    PYQED_RYS_GENERAL_QUARTET_CALL(false);
#undef PYQED_RYS_GENERAL_QUARTET_CALL
}

bool compute_l3_spherical_rys_values(
    const RysSphericalOutputPlan& plan,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p0,
    npy_intp q0,
    npy_intp r0,
    npy_intp s0,
    int lp,
    int lq,
    int lr,
    int ls,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    const double* pq_weight,
    int npq,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    const double* rs_weight,
    int nrs,
    const SPDirectJKConsumer* direct,
    std::vector<double>& values,
    const RysQuadratureCache* quadrature_cache = nullptr
) {
    const int nroots = (lp + lq + lr + ls) / 2 + 1;
    values.assign(plan.outputs.size(), 0.0);
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        const double left_weight = pq_weight[idx_pq];
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const double pq_exponent = pq_p[idx_pq];
            const double rs_exponent = rs_p[idx_rs];
            const double zeta = pq_exponent + rs_exponent;
            const double alpha = pq_exponent * rs_exponent / zeta;
            const double PQ[3] = {
                pq_px[idx_pq] - rs_px[idx_rs],
                pq_py[idx_pq] - rs_py[idx_rs],
                pq_pz[idx_pq] - rs_pz[idx_rs],
            };
            double local_roots[7]{};
            double local_rys_weights[7]{};
            const std::size_t quadrature_offset =
                (static_cast<std::size_t>(idx_pq) * nrs + idx_rs) * nroots;
            double primitive_prefactor = 0.0;
            if (quadrature_cache == nullptr) {
                const double T = alpha * (
                    PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]
                );
                if (!rys_roots_weights_general(
                        nroots, T, local_roots, local_rys_weights
                    )) return false;
                primitive_prefactor = left_weight * rs_weight[idx_rs] * ERI_PREFAC *
                    pq_k[idx_pq] * rs_k[idx_rs] /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
            }
            for (int root = 0; root < nroots; ++root) {
                double c00[3]{};
                double c0p[3]{};
                double b00 = 0.0;
                double b10 = 0.0;
                double b01 = 0.0;
                double root_prefactor = 0.0;
                double u2 = 0.0;
                if (quadrature_cache != nullptr) {
                    const RysQuadraturePointData& item =
                        quadrature_cache->points[quadrature_offset + root];
                    u2 = item.u2;
                    root_prefactor = pq_weight[idx_pq] * rs_weight[idx_rs] *
                        item.root_prefactor;
                } else {
                    u2 = alpha * local_roots[root];
                    root_prefactor = primitive_prefactor * local_rys_weights[root];
                }
                const double tmp4 = 0.5 /
                    (u2 * zeta + pq_exponent * rs_exponent);
                const double tmp5 = u2 * tmp4;
                const double tmp2 = 2.0 * tmp5 * rs_exponent;
                const double tmp3 = 2.0 * tmp5 * pq_exponent;
                b00 = tmp5;
                b10 = tmp5 + tmp4 * rs_exponent;
                b01 = tmp5 + tmp4 * pq_exponent;
                const double pq_center[3] = {
                    pq_px[idx_pq], pq_py[idx_pq], pq_pz[idx_pq]
                };
                const double rs_center[3] = {
                    rs_px[idx_rs], rs_py[idx_rs], rs_pz[idx_rs]
                };
                for (int axis = 0; axis < 3; ++axis) {
                    c00[axis] = pq_center[axis] - origins[3 * p0 + axis] -
                        tmp2 * PQ[axis];
                    c0p[axis] = rs_center[axis] - origins[3 * r0 + axis] +
                        tmp3 * PQ[axis];
                }
                double recurrence[3][7][7];
                for (int axis = 0; axis < 3; ++axis) {
                    rys_build_1d_general(
                        lp + lq,
                        lr + ls,
                        c00[axis], c0p[axis], b10, b01, b00,
                        recurrence[axis]
                    );
                }
                double axis_values[3][256];
                for (int axis = 0; axis < 3; ++axis) {
                    const auto& expansion_offsets = plan.axis_offsets[axis];
                    const auto& expansion_terms = plan.axis_terms[axis];
                    for (std::size_t slot = 0; slot < plan.axis_codes[axis].size(); ++slot) {
                        double axis_value = 0.0;
                        for (
                            std::size_t it = expansion_offsets[slot];
                            it < expansion_offsets[slot + 1];
                            ++it
                        ) {
                            const RysAxisExpansionTerm& term = expansion_terms[it];
                            axis_value += term.coefficient *
                                recurrence[axis][term.bra_order][term.ket_order];
                        }
                        axis_values[axis][slot] = axis_value;
                    }
                }
                for (std::size_t output = 0; output < plan.outputs.size(); ++output) {
                    double transformed = 0.0;
                    for (std::size_t it = plan.offsets[output]; it < plan.offsets[output + 1]; ++it) {
                        const RysSphericalTerm& term = plan.terms[it];
                        const double product = term.coefficient * term.weight_scale *
                            axis_values[0][term.axis_slot[0]] *
                            axis_values[1][term.axis_slot[1]] *
                            axis_values[2][term.axis_slot[2]];
                        transformed += product;
                    }
                    values[output] += root_prefactor * transformed;
                }
            }
        }
    }
    if (direct != nullptr) {
        for (std::size_t output = 0; output < plan.outputs.size(); ++output) {
            const double value = values[output];
            if (direct->screen_tol > 0.0 && std::abs(value) < direct->screen_tol) {
                ++*direct->skipped;
                continue;
            }
            ++*direct->computed;
            const RysSphericalOutput& item = plan.outputs[output];
            if (direct->symmetric_density) {
                direct_jk_add_symmetric_class_code(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, item.p, item.q, item.r, item.s,
                    item.symmetry_class
                );
            } else {
                direct_jk_add_unique_permutations(
                    direct->vj, direct->vk, direct->dm, direct->nao,
                    value, item.p, item.q, item.r, item.s
                );
            }
        }
    }
    return true;
}

enum OneElectronKernel {
    ONE_OVERLAP = 0,
    ONE_KINETIC = 1,
    ONE_NUCLEAR = 2,
    ONE_HCORE = 3,
};

double primitive_overlap_cartesian(
    double a,
    const int* angular_a,
    const double* A,
    double b,
    const int* angular_b,
    const double* B
) {
    for (int axis = 0; axis < 3; ++axis) {
        if (angular_a[axis] < 0 || angular_b[axis] < 0) {
            return 0.0;
        }
    }
    HermiteEMemo ex(angular_a[0], angular_b[0], A[0] - B[0], a, b);
    HermiteEMemo ey(angular_a[1], angular_b[1], A[1] - B[1], a, b);
    HermiteEMemo ez(angular_a[2], angular_b[2], A[2] - B[2], a, b);
    return ex.value(angular_a[0], angular_b[0], 0)
        * ey.value(angular_a[1], angular_b[1], 0)
        * ez.value(angular_a[2], angular_b[2], 0)
        * std::pow(PI / (a + b), 1.5);
}

double primitive_kinetic_cartesian(
    double a,
    const int* angular_a,
    const double* A,
    double b,
    const int* angular_b,
    const double* B
) {
    for (int axis = 0; axis < 3; ++axis) {
        if (angular_a[axis] < 0 || angular_b[axis] < 0) {
            return 0.0;
        }
    }
    HermiteEMemo ex(angular_a[0], angular_b[0] + 2, A[0] - B[0], a, b);
    HermiteEMemo ey(angular_a[1], angular_b[1] + 2, A[1] - B[1], a, b);
    HermiteEMemo ez(angular_a[2], angular_b[2] + 2, A[2] - B[2], a, b);
    const double scale = std::sqrt(PI / (a + b));
    auto sx = [&](int j) {
        return j < 0 ? 0.0 : scale * ex.value(angular_a[0], j, 0);
    };
    auto sy = [&](int j) {
        return j < 0 ? 0.0 : scale * ey.value(angular_a[1], j, 0);
    };
    auto sz = [&](int j) {
        return j < 0 ? 0.0 : scale * ez.value(angular_a[2], j, 0);
    };

    const int lx = angular_b[0];
    const int ly = angular_b[1];
    const int lz = angular_b[2];
    const double base = sx(lx) * sy(ly) * sz(lz);
    const double raised =
        sx(lx + 2) * sy(ly) * sz(lz)
        + sx(lx) * sy(ly + 2) * sz(lz)
        + sx(lx) * sy(ly) * sz(lz + 2);
    const double lowered =
        lx * (lx - 1.0) * sx(lx - 2) * sy(ly) * sz(lz)
        + ly * (ly - 1.0) * sx(lx) * sy(ly - 2) * sz(lz)
        + lz * (lz - 1.0) * sx(lx) * sy(ly) * sz(lz - 2);
    return b * (2.0 * (lx + ly + lz) + 3.0) * base
        - 2.0 * b * b * raised
        - 0.5 * lowered;
}

class OneCoulombRMemo {
public:
    OneCoulombRMemo(int tx, int uy, int vz, double p, const double* PC)
        : uy_(uy),
          vz_(vz),
          nmax_(tx + uy + vz),
          p_(p),
          pc_{PC[0], PC[1], PC[2]},
          T_(p * (PC[0] * PC[0] + PC[1] * PC[1] + PC[2] * PC[2])),
          values_(
              static_cast<std::size_t>(tx + 1)
                  * static_cast<std::size_t>(uy + 1)
                  * static_cast<std::size_t>(vz + 1)
                  * static_cast<std::size_t>(nmax_ + 1),
              std::numeric_limits<double>::quiet_NaN()
          ) {}

    double value(int t, int u, int v, int n) {
        if (t < 0 || u < 0 || v < 0 || n < 0 || n > nmax_) {
            return 0.0;
        }
        double& cached = values_[index(t, u, v, n)];
        if (!std::isnan(cached)) {
            return cached;
        }
        if (t == 0 && u == 0 && v == 0) {
            cached = std::pow(-2.0 * p_, n) * boys(n, T_);
        } else if (t == 0 && u == 0) {
            cached = pc_[2] * value(t, u, v - 1, n + 1);
            if (v > 1) {
                cached += (v - 1.0) * value(t, u, v - 2, n + 1);
            }
        } else if (t == 0) {
            cached = pc_[1] * value(t, u - 1, v, n + 1);
            if (u > 1) {
                cached += (u - 1.0) * value(t, u - 2, v, n + 1);
            }
        } else {
            cached = pc_[0] * value(t - 1, u, v, n + 1);
            if (t > 1) {
                cached += (t - 1.0) * value(t - 2, u, v, n + 1);
            }
        }
        return cached;
    }

private:
    std::size_t index(int t, int u, int v, int n) const {
        return (((static_cast<std::size_t>(t) * (uy_ + 1) + u) * (vz_ + 1) + v)
            * (nmax_ + 1) + n);
    }

    int uy_;
    int vz_;
    int nmax_;
    double p_;
    std::array<double, 3> pc_;
    double T_;
    std::vector<double> values_;
};

double primitive_nuclear_cartesian(
    double a,
    const int* angular_a,
    const double* A,
    double b,
    const int* angular_b,
    const double* B,
    const double* C
) {
    for (int axis = 0; axis < 3; ++axis) {
        if (angular_a[axis] < 0 || angular_b[axis] < 0) {
            return 0.0;
        }
    }
    const double p = a + b;
    const double P[3] = {
        (a * A[0] + b * B[0]) / p,
        (a * A[1] + b * B[1]) / p,
        (a * A[2] + b * B[2]) / p,
    };
    const double PC[3] = {P[0] - C[0], P[1] - C[1], P[2] - C[2]};
    const int tx = angular_a[0] + angular_b[0];
    const int uy = angular_a[1] + angular_b[1];
    const int vz = angular_a[2] + angular_b[2];
    HermiteEMemo ex(angular_a[0], angular_b[0], A[0] - B[0], a, b);
    HermiteEMemo ey(angular_a[1], angular_b[1], A[1] - B[1], a, b);
    HermiteEMemo ez(angular_a[2], angular_b[2], A[2] - B[2], a, b);
    OneCoulombRMemo r(tx, uy, vz, p, PC);
    double value = 0.0;
    for (int t = 0; t <= tx; ++t) {
        const double vx = ex.value(angular_a[0], angular_b[0], t);
        for (int u = 0; u <= uy; ++u) {
            const double vxy = vx * ey.value(angular_a[1], angular_b[1], u);
            for (int v = 0; v <= vz; ++v) {
                value += vxy * ez.value(angular_a[2], angular_b[2], v)
                    * r.value(t, u, v, 0);
            }
        }
    }
    return value * (2.0 * PI / p);
}

struct OneElectronIntegralEntry {
    std::array<int, 6> angular;
    double value;
};

class OneElectronPrimitiveCache {
public:
    OneElectronPrimitiveCache(
        int kernel,
        double a,
        const int* angular_a,
        const double* A,
        double b,
        const int* angular_b,
        const double* B,
        const double* C = nullptr
    ) : kernel_(kernel), a_(a), b_(b), A_(A), B_(B), C_(C) {
        for (int axis = 0; axis < 3; ++axis) {
            base_[axis] = angular_a[axis];
            base_[3 + axis] = angular_b[axis];
        }
        entries_.reserve(80);
    }

    const std::array<int, 6>& base() const {
        return base_;
    }

    double value(const std::array<int, 6>& angular) {
        for (const OneElectronIntegralEntry& entry : entries_) {
            if (entry.angular == angular) {
                return entry.value;
            }
        }
        const double evaluated = evaluate(angular);
        entries_.push_back({angular, evaluated});
        return evaluated;
    }

private:
    double evaluate(const std::array<int, 6>& angular) const {
        if (kernel_ == ONE_OVERLAP) {
            return primitive_overlap_cartesian(a_, angular.data(), A_, b_, angular.data() + 3, B_);
        }
        if (kernel_ == ONE_KINETIC) {
            return primitive_kinetic_cartesian(a_, angular.data(), A_, b_, angular.data() + 3, B_);
        }
        return primitive_nuclear_cartesian(
            a_, angular.data(), A_, b_, angular.data() + 3, B_, C_
        );
    }

    int kernel_;
    double a_;
    double b_;
    const double* A_;
    const double* B_;
    const double* C_;
    std::array<int, 6> base_;
    std::vector<OneElectronIntegralEntry> entries_;
};

double one_center_first_derivative(
    OneElectronPrimitiveCache& cache,
    int slot,
    int axis,
    double exponent
) {
    std::array<int, 6> angular = cache.base();
    const int idx = 3 * slot + axis;
    const int power = angular[idx];
    angular[idx] += 1;
    double value = 2.0 * exponent * cache.value(angular);
    if (power > 0) {
        angular[idx] -= 2;
        value -= power * cache.value(angular);
    }
    return value;
}

double one_center_second_derivative(
    OneElectronPrimitiveCache& cache,
    int slot_a,
    int axis_a,
    double exponent_a,
    int slot_b,
    int axis_b,
    double exponent_b
) {
    const int idx_a = 3 * slot_a + axis_a;
    const int idx_b = 3 * slot_b + axis_b;
    const int power_a = cache.base()[idx_a];
    const int power_b = cache.base()[idx_b];
    std::array<int, 6> angular = cache.base();
    if (idx_a == idx_b) {
        angular[idx_a] += 2;
        double value = 4.0 * exponent_a * exponent_a * cache.value(angular);
        angular[idx_a] -= 2;
        value -= 2.0 * exponent_a * (2.0 * power_a + 1.0) * cache.value(angular);
        if (power_a > 1) {
            angular[idx_a] -= 2;
            value += power_a * (power_a - 1.0) * cache.value(angular);
        }
        return value;
    }

    angular[idx_a] += 1;
    angular[idx_b] += 1;
    double value = 4.0 * exponent_a * exponent_b * cache.value(angular);
    if (power_b > 0) {
        angular[idx_b] -= 2;
        value -= 2.0 * exponent_a * power_b * cache.value(angular);
        angular[idx_b] += 2;
    }
    if (power_a > 0) {
        angular[idx_a] -= 2;
        value -= 2.0 * exponent_b * power_a * cache.value(angular);
        angular[idx_a] += 2;
    }
    if (power_a > 0 && power_b > 0) {
        angular[idx_a] -= 2;
        angular[idx_b] -= 2;
        value += power_a * power_b * cache.value(angular);
    }
    return value;
}

inline double one_direction_component(
    const double* directions,
    npy_intp natm,
    npy_intp mode,
    npy_intp atom,
    int axis
) {
    return directions[(mode * natm + atom) * 3 + axis];
}

void accumulate_directional_one_primitive(
    OneElectronPrimitiveCache& cache,
    double exponent_a,
    double exponent_b,
    npy_intp atom_a,
    npy_intp atom_b,
    npy_intp charge_atom,
    bool relative_to_charge,
    const double* directions,
    npy_intp natm,
    npy_intp nmodes,
    int order,
    double scale,
    double* values
) {
    const npy_intp atoms[2] = {atom_a, atom_b};
    auto coefficient = [&](int slot, npy_intp mode, int axis) {
        double value = one_direction_component(
            directions, natm, mode, atoms[slot], axis
        );
        if (relative_to_charge) {
            value -= one_direction_component(
                directions, natm, mode, charge_atom, axis
            );
        }
        return value;
    };

    if (order == 1) {
        double gradient[2][3];
        for (int slot = 0; slot < 2; ++slot) {
            const double exponent = slot == 0 ? exponent_a : exponent_b;
            for (int axis = 0; axis < 3; ++axis) {
                gradient[slot][axis] = one_center_first_derivative(
                    cache, slot, axis, exponent
                );
            }
        }
        for (npy_intp mode = 0; mode < nmodes; ++mode) {
            double contracted = 0.0;
            for (int slot = 0; slot < 2; ++slot) {
                for (int axis = 0; axis < 3; ++axis) {
                    contracted += coefficient(slot, mode, axis) * gradient[slot][axis];
                }
            }
            values[mode] += scale * contracted;
        }
        return;
    }

    double hessian[2][3][2][3];
    for (int slot_a = 0; slot_a < 2; ++slot_a) {
        const double exp_a = slot_a == 0 ? exponent_a : exponent_b;
        for (int axis_a = 0; axis_a < 3; ++axis_a) {
            for (int slot_b = 0; slot_b < 2; ++slot_b) {
                const double exp_b = slot_b == 0 ? exponent_a : exponent_b;
                for (int axis_b = 0; axis_b < 3; ++axis_b) {
                    hessian[slot_a][axis_a][slot_b][axis_b] =
                        one_center_second_derivative(
                            cache,
                            slot_a,
                            axis_a,
                            exp_a,
                            slot_b,
                            axis_b,
                            exp_b
                        );
                }
            }
        }
    }
    for (npy_intp mode_a = 0; mode_a < nmodes; ++mode_a) {
        for (npy_intp mode_b = 0; mode_b <= mode_a; ++mode_b) {
            double contracted = 0.0;
            for (int slot_a = 0; slot_a < 2; ++slot_a) {
                for (int axis_a = 0; axis_a < 3; ++axis_a) {
                    const double ca = coefficient(slot_a, mode_a, axis_a);
                    for (int slot_b = 0; slot_b < 2; ++slot_b) {
                        for (int axis_b = 0; axis_b < 3; ++axis_b) {
                            contracted += ca
                                * coefficient(slot_b, mode_b, axis_b)
                                * hessian[slot_a][axis_a][slot_b][axis_b];
                        }
                    }
                }
            }
            values[mode_a * nmodes + mode_b] += scale * contracted;
            if (mode_a != mode_b) {
                values[mode_b * nmodes + mode_a] += scale * contracted;
            }
        }
    }
}

bool compute_directional_one_electron_derivatives_native(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const std::int64_t* atom_ids,
    const double* atom_coords,
    const double* charges,
    const double* directions,
    npy_intp natm,
    npy_intp nmodes,
    npy_intp nao,
    npy_intp max_prim,
    int kernel,
    int order,
    int workers,
    double* out
) {
    try {
        std::vector<std::pair<npy_intp, npy_intp>> pairs;
        pairs.reserve(static_cast<std::size_t>(nao * (nao + 1) / 2));
        for (npy_intp p = 0; p < nao; ++p) {
            for (npy_intp q = 0; q <= p; ++q) {
                pairs.emplace_back(p, q);
            }
        }

        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(pairs.size()))
        );
        std::atomic<std::size_t> next_pair{0};
        std::atomic<bool> failed{false};
        auto run_worker = [&]() {
            try {
                std::vector<double> pair_values(
                    order == 1
                        ? static_cast<std::size_t>(nmodes)
                        : static_cast<std::size_t>(nmodes) * nmodes,
                    0.0
                );
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t pair_index = next_pair.fetch_add(
                        1, std::memory_order_relaxed
                    );
                    if (pair_index >= pairs.size()) {
                        break;
                    }
                    const npy_intp p = pairs[pair_index].first;
                    const npy_intp q = pairs[pair_index].second;
                    std::fill(pair_values.begin(), pair_values.end(), 0.0);
                    const int angular_p[3] = {
                        static_cast<int>(shells[3 * p]),
                        static_cast<int>(shells[3 * p + 1]),
                        static_cast<int>(shells[3 * p + 2]),
                    };
                    const int angular_q[3] = {
                        static_cast<int>(shells[3 * q]),
                        static_cast<int>(shells[3 * q + 1]),
                        static_cast<int>(shells[3 * q + 2]),
                    };
                    const double* A = origins + 3 * p;
                    const double* B = origins + 3 * q;
                    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
                        const double a = exps[p * max_prim + ip];
                        const double wa = weights[p * max_prim + ip];
                        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
                            const double b = exps[q * max_prim + iq];
                            const double prefactor = wa * weights[q * max_prim + iq];
                            if (kernel == ONE_OVERLAP || kernel == ONE_KINETIC || kernel == ONE_HCORE) {
                                const int primitive_kernel = kernel == ONE_OVERLAP
                                    ? ONE_OVERLAP
                                    : ONE_KINETIC;
                                OneElectronPrimitiveCache cache(
                                    primitive_kernel,
                                    a,
                                    angular_p,
                                    A,
                                    b,
                                    angular_q,
                                    B
                                );
                                accumulate_directional_one_primitive(
                                    cache,
                                    a,
                                    b,
                                    atom_ids[p],
                                    atom_ids[q],
                                    0,
                                    false,
                                    directions,
                                    natm,
                                    nmodes,
                                    order,
                                    prefactor,
                                    pair_values.data()
                                );
                            }
                            if (kernel == ONE_NUCLEAR || kernel == ONE_HCORE) {
                                for (npy_intp charge_atom = 0; charge_atom < natm; ++charge_atom) {
                                    if (charges[charge_atom] == 0.0) {
                                        continue;
                                    }
                                    OneElectronPrimitiveCache cache(
                                        ONE_NUCLEAR,
                                        a,
                                        angular_p,
                                        A,
                                        b,
                                        angular_q,
                                        B,
                                        atom_coords + 3 * charge_atom
                                    );
                                    accumulate_directional_one_primitive(
                                        cache,
                                        a,
                                        b,
                                        atom_ids[p],
                                        atom_ids[q],
                                        charge_atom,
                                        true,
                                        directions,
                                        natm,
                                        nmodes,
                                        order,
                                        -charges[charge_atom] * prefactor,
                                        pair_values.data()
                                    );
                                }
                            }
                        }
                    }

                    if (order == 1) {
                        for (npy_intp mode = 0; mode < nmodes; ++mode) {
                            const double value = pair_values[mode];
                            out[(mode * nao + p) * nao + q] = value;
                            out[(mode * nao + q) * nao + p] = value;
                        }
                    } else {
                        for (npy_intp mode_a = 0; mode_a < nmodes; ++mode_a) {
                            for (npy_intp mode_b = 0; mode_b < nmodes; ++mode_b) {
                                const double value = pair_values[mode_a * nmodes + mode_b];
                                const std::size_t offset =
                                    (static_cast<std::size_t>(mode_a) * nmodes + mode_b)
                                    * static_cast<std::size_t>(nao) * nao;
                                out[offset + p * nao + q] = value;
                                out[offset + q * nao + p] = value;
                            }
                        }
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        };

        if (nthread == 1) {
            run_worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(nthread));
            try {
                for (int tid = 0; tid < nthread; ++tid) {
                    threads.emplace_back(run_worker);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        return !failed.load(std::memory_order_relaxed);
    } catch (...) {
        return false;
    }
}

bool compute_one_index_one_electron_derivatives_native(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const std::int64_t* atom_ids,
    npy_intp natm,
    npy_intp nao,
    npy_intp max_prim,
    int kernel,
    int index_slot,
    int order,
    int workers,
    double* out
) {
    try {
        const std::size_t npair = static_cast<std::size_t>(nao) * nao;
        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(npair))
        );
        std::atomic<std::size_t> next_pair{0};
        std::atomic<bool> failed{false};
        auto run_worker = [&]() {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t pair = next_pair.fetch_add(1, std::memory_order_relaxed);
                    if (pair >= npair) {
                        break;
                    }
                    const npy_intp p = static_cast<npy_intp>(pair / nao);
                    const npy_intp q = static_cast<npy_intp>(pair % nao);
                    const int angular_p[3] = {
                        static_cast<int>(shells[3 * p]),
                        static_cast<int>(shells[3 * p + 1]),
                        static_cast<int>(shells[3 * p + 2]),
                    };
                    const int angular_q[3] = {
                        static_cast<int>(shells[3 * q]),
                        static_cast<int>(shells[3 * q + 1]),
                        static_cast<int>(shells[3 * q + 2]),
                    };
                    const double* A = origins + 3 * p;
                    const double* B = origins + 3 * q;
                    double gradient[3] = {0.0, 0.0, 0.0};
                    double hessian[3][3] = {};
                    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
                        const double a = exps[p * max_prim + ip];
                        const double wa = weights[p * max_prim + ip];
                        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
                            const double b = exps[q * max_prim + iq];
                            const double prefactor = wa * weights[q * max_prim + iq];
                            OneElectronPrimitiveCache cache(
                                kernel,
                                a,
                                angular_p,
                                A,
                                b,
                                angular_q,
                                B
                            );
                            const double exponent = index_slot == 0 ? a : b;
                            if (order == 1) {
                                for (int axis = 0; axis < 3; ++axis) {
                                    gradient[axis] += prefactor * one_center_first_derivative(
                                        cache, index_slot, axis, exponent
                                    );
                                }
                            } else {
                                for (int axis_a = 0; axis_a < 3; ++axis_a) {
                                    for (int axis_b = 0; axis_b < 3; ++axis_b) {
                                        hessian[axis_a][axis_b] += prefactor
                                            * one_center_second_derivative(
                                                cache,
                                                index_slot,
                                                axis_a,
                                                exponent,
                                                index_slot,
                                                axis_b,
                                                exponent
                                            );
                                    }
                                }
                            }
                        }
                    }

                    const npy_intp atom = atom_ids[index_slot == 0 ? p : q];
                    if (order == 1) {
                        for (int axis = 0; axis < 3; ++axis) {
                            const std::size_t offset =
                                (static_cast<std::size_t>(atom) * 3 + axis)
                                * static_cast<std::size_t>(nao) * nao;
                            out[offset + p * nao + q] = gradient[axis];
                        }
                    } else {
                        const std::size_t ncart = static_cast<std::size_t>(natm) * 3;
                        for (int axis_a = 0; axis_a < 3; ++axis_a) {
                            const std::size_t cart_a = static_cast<std::size_t>(atom) * 3 + axis_a;
                            for (int axis_b = 0; axis_b < 3; ++axis_b) {
                                const std::size_t cart_b = static_cast<std::size_t>(atom) * 3 + axis_b;
                                const std::size_t offset =
                                    (cart_a * ncart + cart_b)
                                    * static_cast<std::size_t>(nao) * nao;
                                out[offset + p * nao + q] = hessian[axis_a][axis_b];
                            }
                        }
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        };

        if (nthread == 1) {
            run_worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(nthread));
            try {
                for (int tid = 0; tid < nthread; ++tid) {
                    threads.emplace_back(run_worker);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        return !failed.load(std::memory_order_relaxed);
    } catch (...) {
        return false;
    }
}

inline int os_axis_of_max3(int a, int b, int c) {
    if (a >= b && a >= c) {
        return 0;
    }
    if (b >= c) {
        return 1;
    }
    return 2;
}

inline std::size_t os_vrr_idx(
    int ax,
    int ay,
    int az,
    int cx,
    int cy,
    int cz,
    int m,
    int adim,
    int cdim,
    int mdim
) {
    (void)adim;
    static constexpr int lut_dim = OS_VRR_PAIR_MAX_L + 1;
    static const std::array<unsigned char, lut_dim * lut_dim * lut_dim> cart_index = [] {
        std::array<unsigned char, lut_dim * lut_dim * lut_dim> result{};
        for (int x = 0; x < lut_dim; ++x) {
            for (int y = 0; y < lut_dim; ++y) {
                for (int z = 0; z < lut_dim; ++z) {
                    const int total = x + y + z;
                    const int rem = y + z;
                    result[(x * lut_dim + y) * lut_dim + z] = static_cast<unsigned char>(
                        total * (total + 1) * (total + 2) / 6 + rem * (rem + 1) / 2 + z
                    );
                }
            }
        }
        return result;
    }();
    static constexpr std::array<unsigned char, OS_VRR_PAIR_MAX_L + 1> cumulative_count = {
        1, 4, 10, 20, 35, 56, 84
    };
    const std::size_t aidx = cart_index[(ax * lut_dim + ay) * lut_dim + az];
    const std::size_t cidx = cart_index[(cx * lut_dim + cy) * lut_dim + cz];
    const int max_c = cdim - 1;
    const std::size_t ccount = cumulative_count[max_c];
    return (aidx * ccount + cidx) * static_cast<std::size_t>(mdim) + m;
}

inline std::size_t os_vrr_table_size(int max_a, int max_c, int max_m) {
    const std::size_t acount =
        static_cast<std::size_t>(max_a + 1) * (max_a + 2) * (max_a + 3) / 6;
    const std::size_t ccount =
        static_cast<std::size_t>(max_c + 1) * (max_c + 2) * (max_c + 3) / 6;
    return acount * ccount * static_cast<std::size_t>(max_m + 1);
}

inline double os_vrr_get_raw(
    const double* table,
    int ax,
    int ay,
    int az,
    int cx,
    int cy,
    int cz,
    int m,
    int adim,
    int cdim,
    int mdim
) {
    return table[os_vrr_idx(ax, ay, az, cx, cy, cz, m, adim, cdim, mdim)];
}

inline void os_vrr_set(
    double* table,
    int ax,
    int ay,
    int az,
    int cx,
    int cy,
    int cz,
    int m,
    int adim,
    int cdim,
    int mdim,
    double value
) {
    table[os_vrr_idx(ax, ay, az, cx, cy, cz, m, adim, cdim, mdim)] = value;
}

template <int FixedA, int FixedC>
void os_fill_vrr_table_impl(
    double* table,
    int max_a,
    int max_c,
    int max_m,
    double p,
    double q,
    double z,
    double T,
    double base_pref,
    const double* PA,
    const double* QC,
    const double* PQ
) {
    const int active_max_a = FixedA >= 0 ? FixedA : max_a;
    const int active_max_c = FixedC >= 0 ? FixedC : max_c;
    const int active_max_m = FixedA >= 0 && FixedC >= 0
        ? FixedA + FixedC
        : max_m;
    const int adim = active_max_a + 1;
    const int cdim = active_max_c + 1;
    const int mdim = active_max_m + 1;
    double boys_values[2 * OS_VRR_PAIR_MAX_L + 1];
    fill_boys_values(active_max_m, T, boys_values);
    for (int m = 0; m <= active_max_m; ++m) {
        os_vrr_set(table, 0, 0, 0, 0, 0, 0, m, adim, cdim, mdim, base_pref * boys_values[m]);
    }
    const double q_over_z = q / z;
    const double p_over_z = p / z;
    const double inv_2p = 0.5 / p;
    const double inv_2q = 0.5 / q;
    const double inv_2z = 0.5 / z;
    for (int total = 1; total <= active_max_m; ++total) {
        const int asum_min = std::max(0, total - active_max_c);
        const int asum_max = std::min(active_max_a, total);
        for (int asum = asum_min; asum <= asum_max; ++asum) {
            const int csum = total - asum;
            for (int ax = 0; ax <= asum; ++ax) {
                for (int ay = 0; ay <= asum - ax; ++ay) {
                    const int az = asum - ax - ay;
                    for (int cx = 0; cx <= csum; ++cx) {
                        for (int cy = 0; cy <= csum - cx; ++cy) {
                            const int cz = csum - cx - cy;
                            const std::size_t output_base = os_vrr_idx(
                                ax, ay, az, cx, cy, cz, 0, adim, cdim, mdim
                            );
                            const int mcount = active_max_m - total;
                            if (asum > 0) {
                                const int axis = os_axis_of_max3(ax, ay, az);
                                if (axis == 0) {
                                    const std::size_t previous_base = os_vrr_idx(ax - 1, ay, az, cx, cy, cz, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = ax > 1 ? os_vrr_idx(ax - 2, ay, az, cx, cy, cz, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = cx > 0 ? os_vrr_idx(ax - 1, ay, az, cx - 1, cy, cz, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = PA[0] * table[previous_base + m] - q_over_z * PQ[0] * table[previous_base + m + 1];
                                        if (ax > 1) value += (ax - 1) * inv_2p * (table[lower_base + m] - q_over_z * table[lower_base + m + 1]);
                                        if (cx > 0) value += cx * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                } else if (axis == 1) {
                                    const std::size_t previous_base = os_vrr_idx(ax, ay - 1, az, cx, cy, cz, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = ay > 1 ? os_vrr_idx(ax, ay - 2, az, cx, cy, cz, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = cy > 0 ? os_vrr_idx(ax, ay - 1, az, cx, cy - 1, cz, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = PA[1] * table[previous_base + m] - q_over_z * PQ[1] * table[previous_base + m + 1];
                                        if (ay > 1) value += (ay - 1) * inv_2p * (table[lower_base + m] - q_over_z * table[lower_base + m + 1]);
                                        if (cy > 0) value += cy * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                } else {
                                    const std::size_t previous_base = os_vrr_idx(ax, ay, az - 1, cx, cy, cz, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = az > 1 ? os_vrr_idx(ax, ay, az - 2, cx, cy, cz, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = cz > 0 ? os_vrr_idx(ax, ay, az - 1, cx, cy, cz - 1, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = PA[2] * table[previous_base + m] - q_over_z * PQ[2] * table[previous_base + m + 1];
                                        if (az > 1) value += (az - 1) * inv_2p * (table[lower_base + m] - q_over_z * table[lower_base + m + 1]);
                                        if (cz > 0) value += cz * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                }
                            } else {
                                const int axis = os_axis_of_max3(cx, cy, cz);
                                if (axis == 0) {
                                    const std::size_t previous_base = os_vrr_idx(ax, ay, az, cx - 1, cy, cz, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = cx > 1 ? os_vrr_idx(ax, ay, az, cx - 2, cy, cz, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = ax > 0 ? os_vrr_idx(ax - 1, ay, az, cx - 1, cy, cz, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = QC[0] * table[previous_base + m] + p_over_z * PQ[0] * table[previous_base + m + 1];
                                        if (cx > 1) value += (cx - 1) * inv_2q * (table[lower_base + m] - p_over_z * table[lower_base + m + 1]);
                                        if (ax > 0) value += ax * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                } else if (axis == 1) {
                                    const std::size_t previous_base = os_vrr_idx(ax, ay, az, cx, cy - 1, cz, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = cy > 1 ? os_vrr_idx(ax, ay, az, cx, cy - 2, cz, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = ay > 0 ? os_vrr_idx(ax, ay - 1, az, cx, cy - 1, cz, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = QC[1] * table[previous_base + m] + p_over_z * PQ[1] * table[previous_base + m + 1];
                                        if (cy > 1) value += (cy - 1) * inv_2q * (table[lower_base + m] - p_over_z * table[lower_base + m + 1]);
                                        if (ay > 0) value += ay * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                } else {
                                    const std::size_t previous_base = os_vrr_idx(ax, ay, az, cx, cy, cz - 1, 0, adim, cdim, mdim);
                                    const std::size_t lower_base = cz > 1 ? os_vrr_idx(ax, ay, az, cx, cy, cz - 2, 0, adim, cdim, mdim) : 0;
                                    const std::size_t cross_base = az > 0 ? os_vrr_idx(ax, ay, az - 1, cx, cy, cz - 1, 0, adim, cdim, mdim) : 0;
                                    for (int m = 0; m <= mcount; ++m) {
                                        double value = QC[2] * table[previous_base + m] + p_over_z * PQ[2] * table[previous_base + m + 1];
                                        if (cz > 1) value += (cz - 1) * inv_2q * (table[lower_base + m] - p_over_z * table[lower_base + m + 1]);
                                        if (az > 0) value += az * inv_2z * table[cross_base + m + 1];
                                        table[output_base + m] = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

using OSVRRFillFunction = void (*)(
    double*, int, int, int,
    double, double, double, double, double,
    const double*, const double*, const double*
);

void os_fill_vrr_table(
    double* table,
    int max_a,
    int max_c,
    int max_m,
    double p,
    double q,
    double z,
    double T,
    double base_pref,
    const double* PA,
    const double* QC,
    const double* PQ
) {
    static constexpr std::array<std::array<OSVRRFillFunction, 5>, 5> specialized = {{
        {{&os_fill_vrr_table_impl<0, 0>, &os_fill_vrr_table_impl<0, 1>, &os_fill_vrr_table_impl<0, 2>, &os_fill_vrr_table_impl<0, 3>, &os_fill_vrr_table_impl<0, 4>}},
        {{&os_fill_vrr_table_impl<1, 0>, &os_fill_vrr_table_impl<1, 1>, &os_fill_vrr_table_impl<1, 2>, &os_fill_vrr_table_impl<1, 3>, &os_fill_vrr_table_impl<1, 4>}},
        {{&os_fill_vrr_table_impl<2, 0>, &os_fill_vrr_table_impl<2, 1>, &os_fill_vrr_table_impl<2, 2>, &os_fill_vrr_table_impl<2, 3>, &os_fill_vrr_table_impl<2, 4>}},
        {{&os_fill_vrr_table_impl<3, 0>, &os_fill_vrr_table_impl<3, 1>, &os_fill_vrr_table_impl<3, 2>, &os_fill_vrr_table_impl<3, 3>, &os_fill_vrr_table_impl<3, 4>}},
        {{&os_fill_vrr_table_impl<4, 0>, &os_fill_vrr_table_impl<4, 1>, &os_fill_vrr_table_impl<4, 2>, &os_fill_vrr_table_impl<4, 3>, &os_fill_vrr_table_impl<4, 4>}},
    }};
    if (max_a <= 4 && max_c <= 4) {
        specialized[static_cast<std::size_t>(max_a)][static_cast<std::size_t>(max_c)](
            table, max_a, max_c, max_m, p, q, z, T, base_pref, PA, QC, PQ
        );
        return;
    }
    os_fill_vrr_table_impl<-1, -1>(
        table, max_a, max_c, max_m, p, q, z, T, base_pref, PA, QC, PQ
    );
}

inline double os_pow_small(double x, int n) {
    double value = 1.0;
    for (int power = 0; power < n; ++power) {
        value *= x;
    }
    return value;
}

inline double os_binom_small(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    k = std::min(k, n - k);
    double value = 1.0;
    for (int i = 1; i <= k; ++i) {
        value *= static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return value;
}

double os_vrr_hrr_eval_expanded(
    const double* table,
    int ax,
    int ay,
    int az,
    int bx,
    int by,
    int bz,
    int cx,
    int cy,
    int cz,
    int dxx,
    int dyy,
    int dzz,
    int m,
    int max_a,
    int max_c,
    int max_m,
    const double* AB,
    const double* CD
) {
    const int adim = max_a + 1;
    const int cdim = max_c + 1;
    const int mdim = max_m + 1;
    double value = 0.0;

    for (int ix = 0; ix <= bx; ++ix) {
        for (int iy = 0; iy <= by; ++iy) {
            for (int iz = 0; iz <= bz; ++iz) {
                const double coeff_b =
                    os_binom_small(bx, ix) * os_pow_small(AB[0], bx - ix) *
                    os_binom_small(by, iy) * os_pow_small(AB[1], by - iy) *
                    os_binom_small(bz, iz) * os_pow_small(AB[2], bz - iz);
                for (int jx = 0; jx <= dxx; ++jx) {
                    for (int jy = 0; jy <= dyy; ++jy) {
                        for (int jz = 0; jz <= dzz; ++jz) {
                            const double coeff_d =
                                os_binom_small(dxx, jx) * os_pow_small(CD[0], dxx - jx) *
                                os_binom_small(dyy, jy) * os_pow_small(CD[1], dyy - jy) *
                                os_binom_small(dzz, jz) * os_pow_small(CD[2], dzz - jz);
                            value += coeff_b * coeff_d * os_vrr_get_raw(
                                table,
                                ax + ix,
                                ay + iy,
                                az + iz,
                                cx + jx,
                                cy + jy,
                                cz + jz,
                                m,
                                adim,
                                cdim,
                                mdim
                            );
                        }
                    }
                }
            }
        }
    }
    return value;
}

struct OSHRRExpansionTerm {
    std::size_t table_index;
    double coefficient;
};

struct OSHRRExpansionPlan {
    std::vector<std::size_t> offsets;
    std::vector<OSHRRExpansionTerm> terms;
};

struct OSPrimitiveWeightPlan {
    int npq = 0;
    int nrs = 0;
    std::size_t ntarget = 0;
    std::vector<double> left;
    std::vector<double> right;
};

void os_build_hrr_expansion(
    int ax,
    int ay,
    int az,
    int bx,
    int by,
    int bz,
    int cx,
    int cy,
    int cz,
    int dxx,
    int dyy,
    int dzz,
    int max_a,
    int max_c,
    int max_m,
    const double* AB,
    const double* CD,
    std::vector<OSHRRExpansionTerm>& terms
) {
    const int adim = max_a + 1;
    const int cdim = max_c + 1;
    const int mdim = max_m + 1;
    for (int ix = 0; ix <= bx; ++ix) {
        for (int iy = 0; iy <= by; ++iy) {
            for (int iz = 0; iz <= bz; ++iz) {
                const double coeff_b =
                    os_binom_small(bx, ix) * os_pow_small(AB[0], bx - ix) *
                    os_binom_small(by, iy) * os_pow_small(AB[1], by - iy) *
                    os_binom_small(bz, iz) * os_pow_small(AB[2], bz - iz);
                for (int jx = 0; jx <= dxx; ++jx) {
                    for (int jy = 0; jy <= dyy; ++jy) {
                        for (int jz = 0; jz <= dzz; ++jz) {
                            const double coefficient = coeff_b *
                                os_binom_small(dxx, jx) * os_pow_small(CD[0], dxx - jx) *
                                os_binom_small(dyy, jy) * os_pow_small(CD[1], dyy - jy) *
                                os_binom_small(dzz, jz) * os_pow_small(CD[2], dzz - jz);
                            if (coefficient == 0.0) {
                                continue;
                            }
                            terms.push_back({
                                os_vrr_idx(
                                    ax + ix,
                                    ay + iy,
                                    az + iz,
                                    cx + jx,
                                    cy + jy,
                                    cz + jz,
                                    0,
                                    adim,
                                    cdim,
                                    mdim
                                ),
                                coefficient,
                            });
                        }
                    }
                }
            }
        }
    }
}

int precompute_primitive_pair_geom(
    npy_intp p,
    npy_intp q,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp max_prim,
    double* a_out,
    double* b_out,
    double* p_out,
    double* px_out,
    double* py_out,
    double* pz_out,
    double* k_out,
    const double* weights,
    double* weight_out
) {
    int idx = 0;
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    const double abx = A[0] - B[0];
    const double aby = A[1] - B[1];
    const double abz = A[2] - B[2];
    const double ab2 = abx * abx + aby * aby + abz * abz;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double a = exps[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double b = exps[q * max_prim + iq];
            const double pexp = a + b;
            const double pair_decay = std::exp(-(a * b / pexp) * ab2);
            const double pair_weight = weight_out != nullptr
                ? weights[p * max_prim + ip] * weights[q * max_prim + iq]
                : 1.0;
            if (pair_decay == 0.0 || pair_weight == 0.0) {
                continue;
            }
            a_out[idx] = a;
            b_out[idx] = b;
            p_out[idx] = pexp;
            px_out[idx] = (a * A[0] + b * B[0]) / pexp;
            py_out[idx] = (a * A[1] + b * B[1]) / pexp;
            pz_out[idx] = (a * A[2] + b * B[2]) / pexp;
            k_out[idx] = pair_decay;
            if (weight_out != nullptr) {
                weight_out[idx] = pair_weight;
            }
            ++idx;
        }
    }
    return idx;
}

void add_eri_symmetries_unique(
    double* eri,
    npy_intp nao,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s,
    double value
) {
    std::array<npy_intp, 8> indices = {
        dense_index(nao, p, q, r, s),
        dense_index(nao, q, p, r, s),
        dense_index(nao, p, q, s, r),
        dense_index(nao, q, p, s, r),
        dense_index(nao, r, s, p, q),
        dense_index(nao, s, r, p, q),
        dense_index(nao, r, s, q, p),
        dense_index(nao, s, r, q, p),
    };

    int nseen = 0;
    std::array<npy_intp, 8> seen{};
    for (npy_intp idx : indices) {
        bool duplicate = false;
        for (int i = 0; i < nseen; ++i) {
            if (seen[i] == idx) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            eri[idx] += value;
            seen[nseen++] = idx;
        }
    }
}

struct ShellQuartetTarget {
    int ax;
    int ay;
    int az;
    int bx;
    int by;
    int bz;
    int cx;
    int cy;
    int cz;
    int dx;
    int dy;
    int dz;
    npy_intp ao_p;
    npy_intp ao_q;
    npy_intp ao_r;
    npy_intp ao_s;
    npy_intp weight_p;
    npy_intp weight_q;
    npy_intp weight_r;
    npy_intp weight_s;
    npy_intp s8_index;
};

struct ShellQuartetTask {
    int ish;
    int jsh;
    int ksh;
    int lsh;
    std::uint32_t cache_index;
    std::uint32_t shell_count;
    std::int32_t contraction_batch_next = -1;
    std::uint16_t contraction_batch_size = 1;
};
static_assert(sizeof(ShellQuartetTask) == 32);

inline double direct_jk_density_bound_for_ao_quartet(
    const double* dm,
    npy_intp nao,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    auto abs_dm = [&](npy_intp a, npy_intp b) {
        return std::abs(dm[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(b)]);
    };
    double bound = 0.0;
    bound = std::max(bound, abs_dm(p, q));
    bound = std::max(bound, abs_dm(q, p));
    bound = std::max(bound, abs_dm(r, s));
    bound = std::max(bound, abs_dm(s, r));
    bound = std::max(bound, abs_dm(p, r));
    bound = std::max(bound, abs_dm(r, p));
    bound = std::max(bound, abs_dm(p, s));
    bound = std::max(bound, abs_dm(s, p));
    bound = std::max(bound, abs_dm(q, r));
    bound = std::max(bound, abs_dm(r, q));
    bound = std::max(bound, abs_dm(q, s));
    bound = std::max(bound, abs_dm(s, q));
    return bound;
}

void build_shell_quartet_tasks(
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<double>& shell_pair_bounds,
    int nshell,
    double screen_tol,
    std::vector<ShellQuartetTask>& tasks,
    long long& shell_screened
) {
    tasks.clear();
    shell_screened = 0;
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& pblk = shell_blocks[ish];
        for (int jsh = 0; jsh <= ish; ++jsh) {
            const ShellBlock& qblk = shell_blocks[jsh];
            const double bound_pq = shell_pair_bounds[static_cast<std::size_t>(ish) * nshell + jsh];
            for (int ksh = 0; ksh <= ish; ++ksh) {
                const ShellBlock& rblk = shell_blocks[ksh];
                const int lsh_max = (ksh == ish) ? jsh : ksh;
                for (int lsh = 0; lsh <= lsh_max; ++lsh) {
                    const ShellBlock& sblk = shell_blocks[lsh];
                    const double bound_rs = shell_pair_bounds[static_cast<std::size_t>(ksh) * nshell + lsh];
                    const long long shell_count = shell_unique_quartet_count(
                        pblk,
                        qblk,
                        rblk,
                        sblk,
                        ish,
                        jsh,
                        ksh,
                        lsh
                    );
                    if (screen_tol > 0.0 && bound_pq * bound_rs < screen_tol) {
                        shell_screened += shell_count;
                        continue;
                    }
                    if (
                        tasks.size() >=
                            static_cast<std::size_t>(
                                std::numeric_limits<std::uint32_t>::max()
                            ) ||
                        shell_count >
                            static_cast<long long>(
                                std::numeric_limits<std::uint32_t>::max()
                            )
                    ) {
                        throw std::length_error(
                            "Shell-quartet task plan exceeds compact index limits"
                        );
                    }
                    ShellQuartetTask task;
                    task.cache_index = static_cast<std::uint32_t>(tasks.size());
                    task.shell_count = static_cast<std::uint32_t>(shell_count);
                    task.ish = ish;
                    task.jsh = jsh;
                    task.ksh = ksh;
                    task.lsh = lsh;
                    tasks.push_back(task);
                }
            }
        }
    }
}

bool compute_shell_quartet_vrr_hrr_target_values(
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp nao,
    double screen_tol,
    const double* pair_bounds,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp r0,
    npy_intp r1,
    npy_intp s0,
    npy_intp s1,
    const double* pq_a,
    const double* pq_b,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    int npq,
    const double* rs_a,
    const double* rs_b,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    int nrs,
    double* vrr_table,
    std::size_t vrr_table_cap,
    std::vector<ShellQuartetTarget>& targets,
    std::vector<double>& values,
    long long& target_screened,
    const double* direct_dm = nullptr,
    double direct_screen_tol = 0.0,
    OSHRRExpansionPlan* reusable_hrr_plan = nullptr,
    bool reuse_existing_targets = false,
    OSPrimitiveWeightPlan* reusable_weight_plan = nullptr,
    double* secondary_vrr_table = nullptr
) {
    const bool have_reusable_targets = reuse_existing_targets && !targets.empty();
    if (!have_reusable_targets) {
        targets.clear();
    }
    values.clear();
    target_screened = 0;

    const int np = static_cast<int>(p1 - p0);
    const int nq = static_cast<int>(q1 - q0);
    const int nr = static_cast<int>(r1 - r0);
    const int ns = static_cast<int>(s1 - s0);
    const int lA = static_cast<int>(shells[3 * p0] + shells[3 * p0 + 1] + shells[3 * p0 + 2]);
    const int lB = static_cast<int>(shells[3 * q0] + shells[3 * q0 + 1] + shells[3 * q0 + 2]);
    const int lC = static_cast<int>(shells[3 * r0] + shells[3 * r0 + 1] + shells[3 * r0 + 2]);
    const int lD = static_cast<int>(shells[3 * s0] + shells[3 * s0 + 1] + shells[3 * s0 + 2]);
    const int max_a_l = lA + lB;
    const int max_c_l = lC + lD;
    const int max_m_l = max_a_l + max_c_l;
    if (
        max_a_l > OS_VRR_PAIR_MAX_L ||
        max_c_l > OS_VRR_PAIR_MAX_L ||
        np != ncart_for_l(lA) ||
        nq != ncart_for_l(lB) ||
        nr != ncart_for_l(lC) ||
        ns != ncart_for_l(lD)
    ) {
        return false;
    }

    const std::size_t vrr_table_size = os_vrr_table_size(max_a_l, max_c_l, max_m_l);
    if (vrr_table == nullptr || vrr_table_size > vrr_table_cap) {
        return false;
    }

    int ax[OS_VRR_MAX_CART], ay[OS_VRR_MAX_CART], az[OS_VRR_MAX_CART];
    int bx[OS_VRR_MAX_CART], by[OS_VRR_MAX_CART], bz[OS_VRR_MAX_CART];
    int cx[OS_VRR_MAX_CART], cy[OS_VRR_MAX_CART], cz[OS_VRR_MAX_CART];
    int dx_l[OS_VRR_MAX_CART], dy_l[OS_VRR_MAX_CART], dz_l[OS_VRR_MAX_CART];
    fill_cartesian_components(lA, ax, ay, az);
    fill_cartesian_components(lB, bx, by, bz);
    fill_cartesian_components(lC, cx, cy, cz);
    fill_cartesian_components(lD, dx_l, dy_l, dz_l);

    const bool same_shell_pair = (p0 == r0 && q0 == s0);
    const bool do_screen = screen_tol > 0.0 && pair_bounds != nullptr;
    if (!have_reusable_targets) {
        targets.reserve(static_cast<std::size_t>(np) * nq * nr * ns);
        for (int ia = 0; ia < np; ++ia) {
        const npy_intp ao_p = p0 + ia;
        const npy_intp weight_p = ao_p * max_prim;
        for (int ib = 0; ib < nq; ++ib) {
            const npy_intp ao_q = q0 + ib;
            if (ao_p < ao_q) {
                continue;
            }
            const npy_intp pair_pq = pair_index(static_cast<int>(ao_p), static_cast<int>(ao_q));
            const double bound_pq = do_screen ? pair_bounds[ao_p * nao + ao_q] : 0.0;
            const npy_intp weight_q = ao_q * max_prim;
            for (int ic = 0; ic < nr; ++ic) {
                const npy_intp ao_r = r0 + ic;
                const npy_intp weight_r = ao_r * max_prim;
                for (int id = 0; id < ns; ++id) {
                    const npy_intp ao_s = s0 + id;
                    if (ao_r < ao_s) {
                        continue;
                    }
                    const npy_intp pair_rs = pair_index(static_cast<int>(ao_r), static_cast<int>(ao_s));
                    if (same_shell_pair && pair_pq < pair_rs) {
                        continue;
                    }
                    if (do_screen && bound_pq * pair_bounds[ao_r * nao + ao_s] < screen_tol) {
                        ++target_screened;
                        continue;
                    }
                    if (direct_dm != nullptr && direct_screen_tol > 0.0) {
                        const double eri_bound = do_screen
                            ? bound_pq * pair_bounds[ao_r * nao + ao_s]
                            : std::numeric_limits<double>::infinity();
                        const double dm_bound = direct_jk_density_bound_for_ao_quartet(
                            direct_dm,
                            nao,
                            ao_p,
                            ao_q,
                            ao_r,
                            ao_s
                        );
                        if (eri_bound * dm_bound < direct_screen_tol) {
                            ++target_screened;
                            continue;
                        }
                    }
                    targets.push_back({
                        ax[ia], ay[ia], az[ia],
                        bx[ib], by[ib], bz[ib],
                        cx[ic], cy[ic], cz[ic],
                        dx_l[id], dy_l[id], dz_l[id],
                        ao_p,
                        ao_q,
                        ao_r,
                        ao_s,
                        weight_p,
                        weight_q,
                        weight_r,
                        ao_s * max_prim,
                        pair_pair_index(pair_pq, pair_rs),
                    });
                }
            }
        }
        }
    }
    if (targets.empty()) {
        return true;
    }
    values.assign(targets.size(), 0.0);
    if (
        reusable_weight_plan != nullptr &&
        (
            reusable_weight_plan->npq != npq ||
            reusable_weight_plan->nrs != nrs ||
            reusable_weight_plan->ntarget != targets.size()
        )
    ) {
        reusable_weight_plan->npq = npq;
        reusable_weight_plan->nrs = nrs;
        reusable_weight_plan->ntarget = targets.size();
        reusable_weight_plan->left.resize(targets.size() * static_cast<std::size_t>(npq));
        reusable_weight_plan->right.resize(targets.size() * static_cast<std::size_t>(nrs));
        const int target_nprim_q = static_cast<int>(nprim[q0]);
        const int target_nprim_s = static_cast<int>(nprim[s0]);
        for (std::size_t it = 0; it < targets.size(); ++it) {
            const ShellQuartetTarget& target = targets[it];
            for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
                const int ip = idx_pq / target_nprim_q;
                const int iq = idx_pq - ip * target_nprim_q;
                reusable_weight_plan->left[
                    static_cast<std::size_t>(idx_pq) * targets.size() + it
                ] =
                    weights[target.weight_p + ip] * weights[target.weight_q + iq];
            }
            for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
                const int ir = idx_rs / target_nprim_s;
                const int is = idx_rs - ir * target_nprim_s;
                reusable_weight_plan->right[
                    static_cast<std::size_t>(idx_rs) * targets.size() + it
                ] =
                    weights[target.weight_r + ir] * weights[target.weight_s + is];
            }
        }
    }

    if (max_a_l == 0 && max_c_l == 0) {
        const int nprim_q = static_cast<int>(nprim[q0]);
        const int nprim_s = static_cast<int>(nprim[s0]);
        for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
            const int ip = idx_pq / nprim_q;
            const int iq = idx_pq - ip * nprim_q;
            for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
                const int ir = idx_rs / nprim_s;
                const int is = idx_rs - ir * nprim_s;
                const double zeta = pq_p[idx_pq] + rs_p[idx_rs];
                const double alpha = pq_p[idx_pq] * rs_p[idx_rs] / zeta;
                const double pqx = pq_px[idx_pq] - rs_px[idx_rs];
                const double pqy = pq_py[idx_pq] - rs_py[idx_rs];
                const double pqz = pq_pz[idx_pq] - rs_pz[idx_rs];
                const double pq2 = pqx * pqx + pqy * pqy + pqz * pqz;
                const double primitive =
                    ERI_PREFAC * pq_k[idx_pq] * rs_k[idx_rs] *
                    boys0(alpha * pq2) /
                    (pq_p[idx_pq] * rs_p[idx_rs] * std::sqrt(zeta));
                for (std::size_t it = 0; it < targets.size(); ++it) {
                    const ShellQuartetTarget& target = targets[it];
                    const double contraction_weight = reusable_weight_plan
                        ? reusable_weight_plan->left[
                                static_cast<std::size_t>(idx_pq) * targets.size() + it
                            ] * reusable_weight_plan->right[
                                static_cast<std::size_t>(idx_rs) * targets.size() + it
                            ]
                        : weights[target.weight_p + ip] *
                            weights[target.weight_q + iq] *
                            weights[target.weight_r + ir] *
                            weights[target.weight_s + is];
                    values[it] += contraction_weight * primitive;
                }
            }
        }
        return true;
    }

    const double AB[3] = {
        origins[3 * p0] - origins[3 * q0],
        origins[3 * p0 + 1] - origins[3 * q0 + 1],
        origins[3 * p0 + 2] - origins[3 * q0 + 2],
    };
    const double CD[3] = {
        origins[3 * r0] - origins[3 * s0],
        origins[3 * r0 + 1] - origins[3 * s0 + 1],
        origins[3 * r0 + 2] - origins[3 * s0 + 2],
    };
    const bool use_hrr_plan =
        reusable_hrr_plan != nullptr || static_cast<long long>(npq) * nrs > 1;
    std::vector<std::size_t> local_hrr_offsets;
    std::vector<OSHRRExpansionTerm> local_hrr_terms;
    std::vector<std::size_t>& hrr_offsets = reusable_hrr_plan
        ? reusable_hrr_plan->offsets
        : local_hrr_offsets;
    std::vector<OSHRRExpansionTerm>& hrr_terms = reusable_hrr_plan
        ? reusable_hrr_plan->terms
        : local_hrr_terms;
    if (use_hrr_plan && hrr_offsets.size() != targets.size() + 1) {
        hrr_offsets.clear();
        hrr_terms.clear();
        hrr_offsets.resize(targets.size() + 1, 0);
        hrr_terms.reserve(targets.size() * 4);
        for (std::size_t it = 0; it < targets.size(); ++it) {
            const ShellQuartetTarget& target = targets[it];
            os_build_hrr_expansion(
                target.ax,
                target.ay,
                target.az,
                target.bx,
                target.by,
                target.bz,
                target.cx,
                target.cy,
                target.cz,
                target.dx,
                target.dy,
                target.dz,
                max_a_l,
                max_c_l,
                max_m_l,
                AB,
                CD,
                hrr_terms
            );
            hrr_offsets[it + 1] = hrr_terms.size();
        }
    }
    const int nprim_q = static_cast<int>(nprim[q0]);
    const int nprim_s = static_cast<int>(nprim[s0]);
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        const int ip = idx_pq / nprim_q;
        const int iq = idx_pq - ip * nprim_q;
        const double* left_weights = reusable_weight_plan
            ? reusable_weight_plan->left.data() +
                static_cast<std::size_t>(idx_pq) * targets.size()
            : nullptr;
        const int primitive_batch =
            secondary_vrr_table != nullptr && nrs >= 4 && targets.size() >= 24
                ? 2
                : 1;
        for (int idx_rs = 0; idx_rs < nrs; idx_rs += primitive_batch) {
            const int lane_count = std::min(primitive_batch, nrs - idx_rs);
            double* tables[2] = {vrr_table, secondary_vrr_table};
            int ir[2]{};
            int is[2]{};
            const double* right_weights[2]{};
            for (int lane = 0; lane < lane_count; ++lane) {
                const int rs_index = idx_rs + lane;
                ir[lane] = rs_index / nprim_s;
                is[lane] = rs_index - ir[lane] * nprim_s;
                const double zeta = pq_p[idx_pq] + rs_p[rs_index];
                const double alpha = pq_p[idx_pq] * rs_p[rs_index] / zeta;
                const double pqx = pq_px[idx_pq] - rs_px[rs_index];
                const double pqy = pq_py[idx_pq] - rs_py[rs_index];
                const double pqz = pq_pz[idx_pq] - rs_pz[rs_index];
                const double pq2 = pqx * pqx + pqy * pqy + pqz * pqz;
                const double base_pref =
                    ERI_PREFAC * pq_k[idx_pq] * rs_k[rs_index] /
                    (pq_p[idx_pq] * rs_p[rs_index] * std::sqrt(zeta));
                const double PA[3] = {
                    pq_px[idx_pq] - origins[3 * p0],
                    pq_py[idx_pq] - origins[3 * p0 + 1],
                    pq_pz[idx_pq] - origins[3 * p0 + 2],
                };
                const double QC[3] = {
                    rs_px[rs_index] - origins[3 * r0],
                    rs_py[rs_index] - origins[3 * r0 + 1],
                    rs_pz[rs_index] - origins[3 * r0 + 2],
                };
                const double PQ[3] = {pqx, pqy, pqz};
                os_fill_vrr_table(
                    tables[lane], max_a_l, max_c_l, max_m_l,
                    pq_p[idx_pq], rs_p[rs_index], zeta, alpha * pq2,
                    base_pref, PA, QC, PQ
                );
                right_weights[lane] = reusable_weight_plan
                    ? reusable_weight_plan->right.data() +
                        static_cast<std::size_t>(rs_index) * targets.size()
                    : nullptr;
            }
            for (std::size_t it = 0; it < targets.size(); ++it) {
                const ShellQuartetTarget& target = targets[it];
                double batch_value = 0.0;
                for (int lane = 0; lane < lane_count; ++lane) {
                    const double prefac = reusable_weight_plan
                        ? left_weights[it] * right_weights[lane][it]
                        : weights[target.weight_p + ip] *
                            weights[target.weight_q + iq] *
                            weights[target.weight_r + ir[lane]] *
                            weights[target.weight_s + is[lane]];
                    if (use_hrr_plan) {
                        double hrr_value = 0.0;
                        for (
                            std::size_t term = hrr_offsets[it];
                            term < hrr_offsets[it + 1];
                            ++term
                        ) {
                            const OSHRRExpansionTerm& item = hrr_terms[term];
                            hrr_value += item.coefficient * tables[lane][item.table_index];
                        }
                        batch_value += prefac * hrr_value;
                    } else {
                        batch_value += prefac * os_vrr_hrr_eval_expanded(
                            tables[lane],
                            target.ax, target.ay, target.az,
                            target.bx, target.by, target.bz,
                            target.cx, target.cy, target.cz,
                            target.dx, target.dy, target.dz,
                            0, max_a_l, max_c_l, max_m_l, AB, CD
                        );
                    }
                }
                values[it] += batch_value;
            }
        }
    }
    return true;
}

class CoulombRMemo {
public:
    CoulombRMemo(int t_max, int u_max, int v_max, double p, double pcx, double pcy, double pcz)
        : u_max_(u_max),
          v_max_(v_max),
          n_max_(t_max + u_max + v_max + 1),
          p_(p),
          pcx_(pcx),
          pcy_(pcy),
          pcz_(pcz),
          rpc_(std::sqrt(pcx * pcx + pcy * pcy + pcz * pcz)),
          values_((t_max + 1) * (u_max + 1) * (v_max + 1) * (n_max_ + 1), std::numeric_limits<double>::quiet_NaN()) {}

    double value(int t, int u, int v, int n) {
        if (t < 0 || u < 0 || v < 0 || n < 0 || n > n_max_) {
            return 0.0;
        }
        double& cached = values_[index(t, u, v, n)];
        if (!std::isnan(cached)) {
            return cached;
        }

        if (t == 0 && u == 0 && v == 0) {
            cached = std::pow(-2.0 * p_, n) * boys(n, p_ * rpc_ * rpc_);
        } else if (t == 0 && u == 0) {
            cached = 0.0;
            if (v > 1) {
                cached += (v - 1.0) * value(t, u, v - 2, n + 1);
            }
            cached += pcz_ * value(t, u, v - 1, n + 1);
        } else if (t == 0) {
            cached = 0.0;
            if (u > 1) {
                cached += (u - 1.0) * value(t, u - 2, v, n + 1);
            }
            cached += pcy_ * value(t, u - 1, v, n + 1);
        } else {
            cached = 0.0;
            if (t > 1) {
                cached += (t - 1.0) * value(t - 2, u, v, n + 1);
            }
            cached += pcx_ * value(t - 1, u, v, n + 1);
        }
        return cached;
    }

private:
    int index(int t, int u, int v, int n) const {
        return (((t * (u_max_ + 1) + u) * (v_max_ + 1) + v) * (n_max_ + 1) + n);
    }

    int u_max_;
    int v_max_;
    int n_max_;
    double p_;
    double pcx_;
    double pcy_;
    double pcz_;
    double rpc_;
    std::vector<double> values_;
};

double primitive_eri_cartesian(
    double a,
    int l1,
    int m1,
    int n1,
    const double* A,
    double b,
    int l2,
    int m2,
    int n2,
    const double* B,
    double c,
    int l3,
    int m3,
    int n3,
    const double* C,
    double d,
    int l4,
    int m4,
    int n4,
    const double* D
) {
    const double p = a + b;
    const double q = c + d;
    const double alpha = p * q / (p + q);

    const double px = (a * A[0] + b * B[0]) / p;
    const double py = (a * A[1] + b * B[1]) / p;
    const double pz = (a * A[2] + b * B[2]) / p;
    const double qx = (c * C[0] + d * D[0]) / q;
    const double qy = (c * C[1] + d * D[1]) / q;
    const double qz = (c * C[2] + d * D[2]) / q;

    HermiteEMemo ex_ab(l1, l2, A[0] - B[0], a, b);
    HermiteEMemo ey_ab(m1, m2, A[1] - B[1], a, b);
    HermiteEMemo ez_ab(n1, n2, A[2] - B[2], a, b);
    HermiteEMemo ex_cd(l3, l4, C[0] - D[0], c, d);
    HermiteEMemo ey_cd(m3, m4, C[1] - D[1], c, d);
    HermiteEMemo ez_cd(n3, n4, C[2] - D[2], c, d);
    CoulombRMemo r_memo(
        l1 + l2 + l3 + l4,
        m1 + m2 + m3 + m4,
        n1 + n2 + n3 + n4,
        alpha,
        px - qx,
        py - qy,
        pz - qz
    );

    double value = 0.0;
    for (int t = 0; t <= l1 + l2; ++t) {
        const double exab = ex_ab.value(l1, l2, t);
        for (int u = 0; u <= m1 + m2; ++u) {
            const double exyab = exab * ey_ab.value(m1, m2, u);
            for (int v = 0; v <= n1 + n2; ++v) {
                const double xyzab = exyab * ez_ab.value(n1, n2, v);
                for (int tau = 0; tau <= l3 + l4; ++tau) {
                    const double excd = ex_cd.value(l3, l4, tau);
                    for (int nu = 0; nu <= m3 + m4; ++nu) {
                        const double exycd = excd * ey_cd.value(m3, m4, nu);
                        for (int phi = 0; phi <= n3 + n4; ++phi) {
                            const double sign = ((tau + nu + phi) & 1) ? -1.0 : 1.0;
                            value += (
                                xyzab
                                * exycd
                                * ez_cd.value(n3, n4, phi)
                                * sign
                                * r_memo.value(t + tau, u + nu, v + phi, 0)
                            );
                        }
                    }
                }
            }
        }
    }

    return value * (ERI_PREFAC / (p * q * std::sqrt(p + q)));
}

double contracted_eri_cartesian(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    const double* C = origins + 3 * r;
    const double* D = origins + 3 * s;
    const std::int64_t* shell_p = shells + 3 * p;
    const std::int64_t* shell_q = shells + 3 * q;
    const std::int64_t* shell_r = shells + 3 * r;
    const std::int64_t* shell_s = shells + 3 * s;

    double value = 0.0;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double ap = exps[p * max_prim + ip];
        const double wp = weights[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double aq = exps[q * max_prim + iq];
            const double wq = weights[q * max_prim + iq];
            for (std::int64_t ir = 0; ir < nprim[r]; ++ir) {
                const double ar = exps[r * max_prim + ir];
                const double wr = weights[r * max_prim + ir];
                for (std::int64_t is = 0; is < nprim[s]; ++is) {
                    const double as = exps[s * max_prim + is];
                    const double ws = weights[s * max_prim + is];
                    value += wp * wq * wr * ws * primitive_eri_cartesian(
                        ap,
                        static_cast<int>(shell_p[0]),
                        static_cast<int>(shell_p[1]),
                        static_cast<int>(shell_p[2]),
                        A,
                        aq,
                        static_cast<int>(shell_q[0]),
                        static_cast<int>(shell_q[1]),
                        static_cast<int>(shell_q[2]),
                        B,
                        ar,
                        static_cast<int>(shell_r[0]),
                        static_cast<int>(shell_r[1]),
                        static_cast<int>(shell_r[2]),
                        C,
                        as,
                        static_cast<int>(shell_s[0]),
                        static_cast<int>(shell_s[1]),
                        static_cast<int>(shell_s[2]),
                        D
                    );
                }
            }
        }
    }
    return value;
}

double primitive_two_center_coulomb(
    double a,
    int l1,
    int m1,
    int n1,
    const double* A,
    double b,
    int l2,
    int m2,
    int n2,
    const double* B
) {
    const double p = a;
    const double q = b;
    const double alpha = p * q / (p + q);
    const double dx = A[0] - B[0];
    const double dy = A[1] - B[1];
    const double dz = A[2] - B[2];

    HermiteEMemo ex_a(l1, 0, 0.0, a, 0.0);
    HermiteEMemo ey_a(m1, 0, 0.0, a, 0.0);
    HermiteEMemo ez_a(n1, 0, 0.0, a, 0.0);
    HermiteEMemo ex_b(l2, 0, 0.0, b, 0.0);
    HermiteEMemo ey_b(m2, 0, 0.0, b, 0.0);
    HermiteEMemo ez_b(n2, 0, 0.0, b, 0.0);
    CoulombRMemo r_memo(l1 + l2, m1 + m2, n1 + n2, alpha, dx, dy, dz);

    double value = 0.0;
    for (int t = 0; t <= l1; ++t) {
        const double exa = ex_a.value(l1, 0, t);
        for (int u = 0; u <= m1; ++u) {
            const double exya = exa * ey_a.value(m1, 0, u);
            for (int v = 0; v <= n1; ++v) {
                const double xyza = exya * ez_a.value(n1, 0, v);
                for (int tau = 0; tau <= l2; ++tau) {
                    const double exb = ex_b.value(l2, 0, tau);
                    for (int nu = 0; nu <= m2; ++nu) {
                        const double exyb = exb * ey_b.value(m2, 0, nu);
                        for (int phi = 0; phi <= n2; ++phi) {
                            const double sign = ((tau + nu + phi) & 1) ? -1.0 : 1.0;
                            value += (
                                xyza
                                * exyb
                                * ez_b.value(n2, 0, phi)
                                * sign
                                * r_memo.value(t + tau, u + nu, v + phi, 0)
                            );
                        }
                    }
                }
            }
        }
    }

    return value * (ERI_PREFAC / (p * q * std::sqrt(p + q)));
}


double primitive_three_center_precomputed(
    const PrimitivePair& pq,
    double abx,
    double aby,
    double abz,
    int l1,
    int m1,
    int n1,
    int l2,
    int m2,
    int n2,
    double c,
    int l3,
    int m3,
    int n3,
    const double* C
) {
    const double alpha = pq.p * c / (pq.p + c);
    const double dx = pq.px - C[0];
    const double dy = pq.py - C[1];
    const double dz = pq.pz - C[2];

    HermiteEMemo ex_ab(l1, l2, abx, pq.a, pq.b);
    HermiteEMemo ey_ab(m1, m2, aby, pq.a, pq.b);
    HermiteEMemo ez_ab(n1, n2, abz, pq.a, pq.b);
    HermiteEMemo ex_c(l3, 0, 0.0, c, 0.0);
    HermiteEMemo ey_c(m3, 0, 0.0, c, 0.0);
    HermiteEMemo ez_c(n3, 0, 0.0, c, 0.0);
    CoulombRMemo r_memo(l1 + l2 + l3, m1 + m2 + m3, n1 + n2 + n3, alpha, dx, dy, dz);

    double value = 0.0;
    for (int t = 0; t <= l1 + l2; ++t) {
        const double exab = ex_ab.value(l1, l2, t);
        for (int u = 0; u <= m1 + m2; ++u) {
            const double exyab = exab * ey_ab.value(m1, m2, u);
            for (int v = 0; v <= n1 + n2; ++v) {
                const double xyzab = exyab * ez_ab.value(n1, n2, v);
                for (int tau = 0; tau <= l3; ++tau) {
                    const double exc = ex_c.value(l3, 0, tau);
                    for (int nu = 0; nu <= m3; ++nu) {
                        const double exyc = exc * ey_c.value(m3, 0, nu);
                        for (int phi = 0; phi <= n3; ++phi) {
                            const double sign = ((tau + nu + phi) & 1) ? -1.0 : 1.0;
                            value += (
                                xyzab
                                * exyc
                                * ez_c.value(n3, 0, phi)
                                * sign
                                * r_memo.value(t + tau, u + nu, v + phi, 0)
                            );
                        }
                    }
                }
            }
        }
    }

    return value * (ERI_PREFAC / (pq.p * c * std::sqrt(pq.p + c)));
}

double contracted_two_center_coulomb(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p,
    npy_intp q
) {
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    const std::int64_t* shell_p = shells + 3 * p;
    const std::int64_t* shell_q = shells + 3 * q;

    double value = 0.0;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double ap = exps[p * max_prim + ip];
        const double wp = weights[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double aq = exps[q * max_prim + iq];
            const double wq = weights[q * max_prim + iq];
            value += wp * wq * primitive_two_center_coulomb(
                ap,
                static_cast<int>(shell_p[0]),
                static_cast<int>(shell_p[1]),
                static_cast<int>(shell_p[2]),
                A,
                aq,
                static_cast<int>(shell_q[0]),
                static_cast<int>(shell_q[1]),
                static_cast<int>(shell_q[2]),
                B
            );
        }
    }
    return value;
}

void precompute_primitive_pairs(
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    npy_intp p,
    npy_intp q,
    std::vector<PrimitivePair>& pairs
) {
    pairs.clear();
    pairs.reserve(static_cast<std::size_t>(nprim[p] * nprim[q]));
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double ap = exps[p * max_prim + ip];
        const double wp = weights[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double aq = exps[q * max_prim + iq];
            const double wq = weights[q * max_prim + iq];
            const double pexp = ap + aq;
            pairs.push_back({
                ap,
                aq,
                pexp,
                (ap * A[0] + aq * B[0]) / pexp,
                (ap * A[1] + aq * B[1]) / pexp,
                (ap * A[2] + aq * B[2]) / pexp,
                wp * wq,
            });
        }
    }
}

bool compute_shell_triplet_vrr_hrr_into_j3(
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    const std::int64_t* aux_shells,
    const double* aux_origins,
    const double* aux_exps,
    const double* aux_weights,
    const std::int64_t* aux_nprim,
    npy_intp aux_max_prim,
    const double* pair_bounds,
    const double* aux_diag,
    double* j3,
    npy_intp nao,
    npy_intp npair,
    npy_intp naux,
    double screen_tol,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp a0,
    npy_intp a1,
    const double* pq_a,
    const double* pq_b,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    int npq,
    double* vrr_table,
    std::size_t vrr_table_cap
) {
    const int np = static_cast<int>(p1 - p0);
    const int nq = static_cast<int>(q1 - q0);
    const int na = static_cast<int>(a1 - a0);
    const int lA = static_cast<int>(shells[3 * p0] + shells[3 * p0 + 1] + shells[3 * p0 + 2]);
    const int lB = static_cast<int>(shells[3 * q0] + shells[3 * q0 + 1] + shells[3 * q0 + 2]);
    const int lC = static_cast<int>(aux_shells[3 * a0] + aux_shells[3 * a0 + 1] + aux_shells[3 * a0 + 2]);
    const int max_a_l = lA + lB;
    const int max_c_l = lC;
    const int max_m_l = max_a_l + max_c_l;
    if (
        max_a_l > OS_VRR_PAIR_MAX_L ||
        max_c_l > OS_VRR_PAIR_MAX_L ||
        np != ncart_for_l(lA) ||
        nq != ncart_for_l(lB) ||
        na != ncart_for_l(lC)
    ) {
        return false;
    }

    const std::size_t vrr_table_size = os_vrr_table_size(max_a_l, max_c_l, max_m_l);
    if (vrr_table == nullptr || vrr_table_size > vrr_table_cap) {
        return false;
    }

    int ax[OS_VRR_MAX_CART], ay[OS_VRR_MAX_CART], az[OS_VRR_MAX_CART];
    int bx[OS_VRR_MAX_CART], by[OS_VRR_MAX_CART], bz[OS_VRR_MAX_CART];
    int cx[OS_VRR_MAX_CART], cy[OS_VRR_MAX_CART], cz[OS_VRR_MAX_CART];
    fill_cartesian_components(lA, ax, ay, az);
    fill_cartesian_components(lB, bx, by, bz);
    fill_cartesian_components(lC, cx, cy, cz);

    const double* A = origins + 3 * p0;
    const double* B = origins + 3 * q0;
    const double abx = A[0] - B[0];
    const double aby = A[1] - B[1];
    const double abz = A[2] - B[2];
    const double AB[3] = {abx, aby, abz};
    const double CD[3] = {0.0, 0.0, 0.0};
    const double QC[3] = {0.0, 0.0, 0.0};

    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        const int ip = idx_pq / static_cast<int>(nprim[q0]);
        const int iq = idx_pq - ip * static_cast<int>(nprim[q0]);
        for (std::int64_t iap = 0; iap < aux_nprim[a0]; ++iap) {
            const double cexp = aux_exps[a0 * aux_max_prim + iap];
            const double zeta = pq_p[idx_pq] + cexp;
            const double alpha = pq_p[idx_pq] * cexp / zeta;
            const double dx = pq_px[idx_pq] - aux_origins[3 * a0 + 0];
            const double dy = pq_py[idx_pq] - aux_origins[3 * a0 + 1];
            const double dz = pq_pz[idx_pq] - aux_origins[3 * a0 + 2];
            const double pc2 = dx * dx + dy * dy + dz * dz;
            const double T = alpha * pc2;
            const double base_pref = (
                ERI_PREFAC
                * pq_k[idx_pq]
                / (pq_p[idx_pq] * cexp * std::sqrt(zeta))
            );
            const double PA[3] = {
                pq_px[idx_pq] - A[0],
                pq_py[idx_pq] - A[1],
                pq_pz[idx_pq] - A[2],
            };
            const double PQ[3] = {dx, dy, dz};

            std::fill(vrr_table, vrr_table + vrr_table_size, 0.0);
            os_fill_vrr_table(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                cexp,
                zeta,
                T,
                base_pref,
                PA,
                QC,
                PQ
            );

            for (int ia = 0; ia < np; ++ia) {
                const npy_intp ao_p = p0 + ia;
                for (int ib = 0; ib < nq; ++ib) {
                    const npy_intp ao_q = q0 + ib;
                    if (ao_p < ao_q) {
                        continue;
                    }
                    const npy_intp pair = pair_index(static_cast<int>(ao_p), static_cast<int>(ao_q));
                    const double pair_bound = pair_bounds[ao_p * nao + ao_q];
                    for (int ic = 0; ic < na; ++ic) {
                        const npy_intp aux_i = a0 + ic;
                        if (screen_tol > 0.0 && pair_bound * aux_diag[aux_i] < screen_tol) {
                            continue;
                        }
                        const double prefac =
                            weights[ao_p * max_prim + ip] *
                            weights[ao_q * max_prim + iq] *
                            aux_weights[aux_i * aux_max_prim + iap];
                        const double value = prefac * os_vrr_hrr_eval_expanded(
                            vrr_table,
                            ax[ia], ay[ia], az[ia],
                            bx[ib], by[ib], bz[ib],
                            cx[ic], cy[ic], cz[ic],
                            0, 0, 0,
                            0,
                            max_a_l,
                            max_c_l,
                            max_m_l,
                            AB,
                            CD
                        );
                        j3[aux_i * npair + pair] += value;
                    }
                }
            }
        }
    }
    (void)naux;
    return true;
}

bool compute_ri_j3_shell_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    npy_intp max_prim,
    const std::int64_t* aux_shells,
    const double* aux_origins,
    const double* aux_exps,
    const double* aux_weights,
    const std::int64_t* aux_nprim,
    npy_intp aux_max_prim,
    const double* pair_bounds,
    const double* aux_diag,
    double* j3,
    npy_intp nao,
    npy_intp naux,
    double screen_tol
) {
    std::vector<ShellBlock> shell_blocks;
    std::vector<ShellBlock> aux_shell_blocks;
    if (
        !try_build_shell_blocks(shells, origins, exps, nprim, nao, max_prim, shell_blocks) ||
        !try_build_shell_blocks(aux_shells, aux_origins, aux_exps, aux_nprim, naux, aux_max_prim, aux_shell_blocks)
    ) {
        return false;
    }
    if (shell_blocks.empty() || aux_shell_blocks.empty()) {
        return false;
    }

    const npy_intp nshell = static_cast<npy_intp>(shell_blocks.size());
    const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
        shell_blocks,
        shells,
        origins,
        exps,
        weights,
        nprim,
        nao,
        max_prim
    );
    const npy_intp pair_cap = pair_geom.pair_cap;

    const std::size_t vrr_table_cap = os_vrr_table_size(
        OS_VRR_PAIR_MAX_L,
        OS_VRR_PAIR_MAX_L,
        2 * OS_VRR_PAIR_MAX_L
    );
    std::vector<double> vrr_table(vrr_table_cap, 0.0);
    const npy_intp npair = nao * (nao + 1) / 2;

    for (const ShellBlock& aux_block : aux_shell_blocks) {
        for (npy_intp ish = 0; ish < nshell; ++ish) {
            const ShellBlock& p_block = shell_blocks[ish];
            for (npy_intp jsh = 0; jsh <= ish; ++jsh) {
                const ShellBlock& q_block = shell_blocks[jsh];
                const npy_intp idx = ish * nshell + jsh;
                const std::size_t off = static_cast<std::size_t>(idx) * static_cast<std::size_t>(pair_cap);
                const bool ok = compute_shell_triplet_vrr_hrr_into_j3(
                    shells,
                    origins,
                    weights,
                    nprim,
                    max_prim,
                    aux_shells,
                    aux_origins,
                    aux_exps,
                    aux_weights,
                    aux_nprim,
                    aux_max_prim,
                    pair_bounds,
                    aux_diag,
                    j3,
                    nao,
                    npair,
                    naux,
                    screen_tol,
                    p_block.start,
                    p_block.stop,
                    q_block.start,
                    q_block.stop,
                    aux_block.start,
                    aux_block.stop,
                    pair_geom.a.data() + off,
                    pair_geom.b.data() + off,
                    pair_geom.p.data() + off,
                    pair_geom.px.data() + off,
                    pair_geom.py.data() + off,
                    pair_geom.pz.data() + off,
                    pair_geom.k.data() + off,
                    pair_geom.n[static_cast<std::size_t>(idx)],
                    vrr_table.data(),
                    vrr_table.size()
                );
                if (!ok) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool compute_dense_eri_cartesian_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* pair_bounds,
    npy_intp nao,
    npy_intp max_prim,
    double screen_tol,
    double* eri,
    long long& computed,
    long long& skipped,
    int workers
) {
    std::vector<ShellBlock> shell_blocks;
    if (!try_build_shell_blocks(shells, origins, exps, nprim, nao, max_prim, shell_blocks)) {
        return false;
    }

    const int nshell = static_cast<int>(shell_blocks.size());
    for (int ish = 0; ish < nshell; ++ish) {
        for (int jsh = 0; jsh <= ish; ++jsh) {
            if (shell_blocks[ish].l + shell_blocks[jsh].l > OS_VRR_PAIR_MAX_L) {
                return false;
            }
            for (int ksh = 0; ksh <= ish; ++ksh) {
                const int lsh_max = (ksh == ish) ? jsh : ksh;
                for (int lsh = 0; lsh <= lsh_max; ++lsh) {
                    if (shell_blocks[ksh].l + shell_blocks[lsh].l > OS_VRR_PAIR_MAX_L) {
                        return false;
                    }
                }
            }
        }
    }

    try {
        const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
            shell_blocks,
            shells,
            origins,
            exps,
            weights,
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        for (int ish = 0; ish < nshell; ++ish) {
            const ShellBlock& pblk = shell_blocks[ish];
            for (int jsh = 0; jsh < nshell; ++jsh) {
                const ShellBlock& qblk = shell_blocks[jsh];
                double bound = 0.0;
                for (npy_intp ip = pblk.start; ip < pblk.stop; ++ip) {
                    for (npy_intp iq = qblk.start; iq < qblk.stop; ++iq) {
                        bound = std::max(bound, pair_bounds[ip * nao + iq]);
                    }
                }
                const std::size_t pair_idx = static_cast<std::size_t>(ish) * nshell + jsh;
                shell_pair_bounds[pair_idx] = bound;
            }
        }

        std::vector<ShellQuartetTask> tasks;
        long long shell_screened = 0;
        build_shell_quartet_tasks(
            shell_blocks,
            shell_pair_bounds,
            nshell,
            screen_tol,
            tasks,
            shell_screened
        );

        std::size_t vrr_table_cap = 1;
        for (const ShellQuartetTask& task : tasks) {
            const int lab = shell_blocks[task.ish].l + shell_blocks[task.jsh].l;
            const int lcd = shell_blocks[task.ksh].l + shell_blocks[task.lsh].l;
            const std::size_t task_cap = os_vrr_table_size(lab, lcd, lab + lcd);
            vrr_table_cap = std::max(vrr_table_cap, task_cap);
        }

        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(tasks.size()))
        );
        std::vector<long long> thread_computed(static_cast<std::size_t>(nthread), 0);
        std::vector<long long> thread_skipped(static_cast<std::size_t>(nthread), 0);
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};

        auto run_worker = [&](int tid) {
            long long local_computed = 0;
            long long local_skipped = 0;
            std::size_t serial_task_index = 0;
            std::vector<double> vrr_table(vrr_table_cap, 0.0);
            std::vector<ShellQuartetTarget> targets;
            std::vector<double> target_values;
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t task_index = nthread == 1
                        ? serial_task_index++
                        : next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size()) {
                        break;
                    }
                    const ShellQuartetTask& task = tasks[task_index];
                    const ShellBlock& pblk = shell_blocks[task.ish];
                    const ShellBlock& qblk = shell_blocks[task.jsh];
                    const ShellBlock& rblk = shell_blocks[task.ksh];
                    const ShellBlock& sblk = shell_blocks[task.lsh];
                    const std::size_t pq_pair_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
                    const std::size_t rs_pair_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
                    const std::size_t pq_offset = pq_pair_idx * pair_cap;
                    const std::size_t rs_offset = rs_pair_idx * pair_cap;
                    long long target_screened = 0;
                    if (!compute_shell_quartet_vrr_hrr_target_values(
                            shells,
                            origins,
                            weights,
                            nprim,
                            max_prim,
                            nao,
                            screen_tol,
                            pair_bounds,
                            pblk.start,
                            pblk.stop,
                            qblk.start,
                            qblk.stop,
                            rblk.start,
                            rblk.stop,
                            sblk.start,
                            sblk.stop,
                            pair_geom.a.data() + pq_offset,
                            pair_geom.b.data() + pq_offset,
                            pair_geom.p.data() + pq_offset,
                            pair_geom.px.data() + pq_offset,
                            pair_geom.py.data() + pq_offset,
                            pair_geom.pz.data() + pq_offset,
                            pair_geom.k.data() + pq_offset,
                            pair_geom.n[pq_pair_idx],
                            pair_geom.a.data() + rs_offset,
                            pair_geom.b.data() + rs_offset,
                            pair_geom.p.data() + rs_offset,
                            pair_geom.px.data() + rs_offset,
                            pair_geom.py.data() + rs_offset,
                            pair_geom.pz.data() + rs_offset,
                            pair_geom.k.data() + rs_offset,
                            pair_geom.n[rs_pair_idx],
                            vrr_table.data(),
                            vrr_table_cap,
                            targets,
                            target_values,
                            target_screened
                        )) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                    local_skipped += target_screened;

                    for (std::size_t it = 0; it < targets.size(); ++it) {
                        const double value = target_values[it];
                        if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                            ++local_skipped;
                            continue;
                        }
                        const ShellQuartetTarget& target = targets[it];
                        add_eri_symmetries_unique(
                            eri,
                            nao,
                            target.ao_p,
                            target.ao_q,
                            target.ao_r,
                            target.ao_s,
                            value
                        );
                        ++local_computed;
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            thread_computed[static_cast<std::size_t>(tid)] = local_computed;
            thread_skipped[static_cast<std::size_t>(tid)] = local_skipped;
        };

        if (nthread <= 1) {
            run_worker(0);
        } else {
            try {
                native_integral_worker_pool().run(nthread, run_worker);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        }
        if (failed.load(std::memory_order_relaxed)) {
            return false;
        }

        computed = 0;
        skipped = shell_screened;
        for (int tid = 0; tid < nthread; ++tid) {
            computed += thread_computed[static_cast<std::size_t>(tid)];
            skipped += thread_skipped[static_cast<std::size_t>(tid)];
        }
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

void fill_symmetric_eri(
    double* eri,
    npy_intp nao,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s,
    double value
);

inline void direct_jk_add_unique_permutations(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
);

inline bool direct_density_is_symmetric(const double* dm, npy_intp nao);

template<bool PQ_SAME, bool RS_SAME, bool SAME_PAIR>
inline void direct_jk_add_symmetric_class(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
);

struct SparseShellTransform {
    std::vector<std::size_t> offsets;
    std::vector<npy_intp> cart_indices;
    std::vector<double> coefficients;
};

void build_spherical_shell_pair_bounds(
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<SparseShellTransform>& transforms,
    const double* cart_pair_bounds,
    npy_intp nao,
    std::vector<double>& shell_pair_bounds
) {
    const int nshell = static_cast<int>(shell_blocks.size());
    shell_pair_bounds.assign(static_cast<std::size_t>(nshell) * nshell, 0.0);
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& pblk = shell_blocks[ish];
        const SparseShellTransform& transform_p = transforms[ish];
        const npy_intp np = static_cast<npy_intp>(transform_p.offsets.size()) - 1;
        for (int jsh = 0; jsh <= ish; ++jsh) {
            const ShellBlock& qblk = shell_blocks[jsh];
            const SparseShellTransform& transform_q = transforms[jsh];
            const npy_intp nq = static_cast<npy_intp>(transform_q.offsets.size()) - 1;
            double shell_bound = 0.0;
            for (npy_intp a = 0; a < np; ++a) {
                for (npy_intp b = 0; b < nq; ++b) {
                    double pair_bound = 0.0;
                    for (
                        std::size_t ip = transform_p.offsets[a];
                        ip < transform_p.offsets[a + 1];
                        ++ip
                    ) {
                        const npy_intp p = pblk.start + transform_p.cart_indices[ip];
                        const double cp = std::abs(transform_p.coefficients[ip]);
                        for (
                            std::size_t iq = transform_q.offsets[b];
                            iq < transform_q.offsets[b + 1];
                            ++iq
                        ) {
                            const npy_intp q = qblk.start + transform_q.cart_indices[iq];
                            pair_bound += cp * std::abs(transform_q.coefficients[iq]) *
                                cart_pair_bounds[p * nao + q];
                        }
                    }
                    shell_bound = std::max(shell_bound, pair_bound);
                }
            }
            shell_pair_bounds[static_cast<std::size_t>(ish) * nshell + jsh] = shell_bound;
            shell_pair_bounds[static_cast<std::size_t>(jsh) * nshell + ish] = shell_bound;
        }
    }
}

void refine_spherical_shell_pair_bounds_exact(
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<SparseShellTransform>& transforms,
    const std::vector<npy_intp>& sph_starts,
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* cart_pair_bounds,
    npy_intp nao,
    npy_intp nsph,
    npy_intp max_prim,
    std::vector<double>& shell_pair_bounds,
    std::vector<double>& pair_diagonal
) {
    const int nshell = static_cast<int>(shell_blocks.size());
    pair_diagonal.assign(
        static_cast<std::size_t>(nsph * (nsph + 1) / 2),
        0.0
    );
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& pblk = shell_blocks[ish];
        const SparseShellTransform& transform_p = transforms[ish];
        const npy_intp np = static_cast<npy_intp>(transform_p.offsets.size()) - 1;
        for (int jsh = 0; jsh <= ish; ++jsh) {
            const ShellBlock& qblk = shell_blocks[jsh];
            const SparseShellTransform& transform_q = transforms[jsh];
            const npy_intp nq = static_cast<npy_intp>(transform_q.offsets.size()) - 1;
            const bool identity_transform = pblk.l <= 1 && qblk.l <= 1;
            double shell_bound = 0.0;
            for (npy_intp a = 0; a < np; ++a) {
                for (npy_intp b = 0; b < nq; ++b) {
                    if (ish == jsh && b > a) continue;
                    double diagonal = 0.0;
                    if (identity_transform) {
                        const npy_intp p = pblk.start + a;
                        const npy_intp q = qblk.start + b;
                        const double bound = cart_pair_bounds[p * nao + q];
                        diagonal = bound * bound;
                    } else {
                        for (
                            std::size_t ip = transform_p.offsets[a];
                            ip < transform_p.offsets[a + 1];
                            ++ip
                        ) {
                            const npy_intp p = pblk.start + transform_p.cart_indices[ip];
                            const double cp = transform_p.coefficients[ip];
                            for (
                                std::size_t iq = transform_q.offsets[b];
                                iq < transform_q.offsets[b + 1];
                                ++iq
                            ) {
                                const npy_intp q = qblk.start + transform_q.cart_indices[iq];
                                const double cpq = cp * transform_q.coefficients[iq];
                                for (
                                    std::size_t ir = transform_p.offsets[a];
                                    ir < transform_p.offsets[a + 1];
                                    ++ir
                                ) {
                                    const npy_intp r = pblk.start + transform_p.cart_indices[ir];
                                    const double cpqr = cpq * transform_p.coefficients[ir];
                                    for (
                                        std::size_t is = transform_q.offsets[b];
                                        is < transform_q.offsets[b + 1];
                                        ++is
                                    ) {
                                        const npy_intp s = qblk.start + transform_q.cart_indices[is];
                                        diagonal += cpqr * transform_q.coefficients[is] *
                                            contracted_eri_cartesian(
                                                shells,
                                                origins,
                                                exps,
                                                weights,
                                                nprim,
                                                max_prim,
                                                p,
                                                q,
                                                r,
                                                s
                                            );
                                    }
                                }
                            }
                        }
                    }
                    const npy_intp ga = sph_starts[static_cast<std::size_t>(ish)] + a;
                    const npy_intp gb = sph_starts[static_cast<std::size_t>(jsh)] + b;
                    pair_diagonal[static_cast<std::size_t>(pair_index(
                        static_cast<int>(ga), static_cast<int>(gb)
                    ))] = std::max(std::abs(diagonal), 0.0);
                    shell_bound = std::max(
                        shell_bound,
                        std::sqrt(std::max(std::abs(diagonal), 0.0))
                    );
                }
            }
            shell_pair_bounds[static_cast<std::size_t>(ish) * nshell + jsh] = shell_bound;
            shell_pair_bounds[static_cast<std::size_t>(jsh) * nshell + ish] = shell_bound;
        }
    }
}

struct CartesianToSphericalOutput {
    unsigned char indices[4]{};
};

struct CartesianToSphericalPlan {
    std::vector<CartesianToSphericalOutput> outputs;
    std::vector<std::size_t> offsets;
    std::vector<std::uint16_t> cart_indices;
    std::vector<double> coefficients;
};

bool build_cartesian_to_spherical_plan(
    const SparseShellTransform& transform_p,
    const SparseShellTransform& transform_q,
    const SparseShellTransform& transform_r,
    const SparseShellTransform& transform_s,
    const RysCartesianOutputPlan& cartesian_plan,
    npy_intp nq,
    npy_intp nr,
    npy_intp ns,
    bool pq_same_shell,
    bool rs_same_shell,
    bool same_pair,
    CartesianToSphericalPlan& plan
) {
    if (cartesian_plan.outputs.size() > std::numeric_limits<std::uint16_t>::max()) {
        return false;
    }
    plan.outputs.clear();
    plan.offsets.assign(1, 0);
    plan.cart_indices.clear();
    plan.coefficients.clear();
    const npy_intp na = static_cast<npy_intp>(transform_p.offsets.size()) - 1;
    const npy_intp nb = static_cast<npy_intp>(transform_q.offsets.size()) - 1;
    const npy_intp nc = static_cast<npy_intp>(transform_r.offsets.size()) - 1;
    const npy_intp nd = static_cast<npy_intp>(transform_s.offsets.size()) - 1;
    std::vector<std::pair<std::uint16_t, double>> terms;
    for (npy_intp a = 0; a < na; ++a) {
        for (npy_intp b = 0; b < nb; ++b) {
            if (pq_same_shell && a < b) continue;
            const npy_intp pair_ab = pq_same_shell
                ? pair_index(static_cast<int>(a), static_cast<int>(b))
                : a * nb + b;
            for (npy_intp c = 0; c < nc; ++c) {
                for (npy_intp d = 0; d < nd; ++d) {
                    if (rs_same_shell && c < d) continue;
                    const npy_intp pair_cd = rs_same_shell
                        ? pair_index(static_cast<int>(c), static_cast<int>(d))
                        : c * nd + d;
                    if (same_pair && pair_ab < pair_cd) continue;
                    plan.outputs.push_back({{
                        static_cast<unsigned char>(a),
                        static_cast<unsigned char>(b),
                        static_cast<unsigned char>(c),
                        static_cast<unsigned char>(d),
                    }});
                    terms.clear();
                    for (
                        std::size_t ip = transform_p.offsets[a];
                        ip < transform_p.offsets[a + 1];
                        ++ip
                    ) {
                        const npy_intp p = transform_p.cart_indices[ip];
                        const double cp = transform_p.coefficients[ip];
                        for (
                            std::size_t iq = transform_q.offsets[b];
                            iq < transform_q.offsets[b + 1];
                            ++iq
                        ) {
                            const npy_intp q = transform_q.cart_indices[iq];
                            const double cpq = cp * transform_q.coefficients[iq];
                            for (
                                std::size_t ir = transform_r.offsets[c];
                                ir < transform_r.offsets[c + 1];
                                ++ir
                            ) {
                                const npy_intp r = transform_r.cart_indices[ir];
                                const double cpqr = cpq * transform_r.coefficients[ir];
                                for (
                                    std::size_t is = transform_s.offsets[d];
                                    is < transform_s.offsets[d + 1];
                                    ++is
                                ) {
                                    const npy_intp s = transform_s.cart_indices[is];
                                    const std::size_t dense =
                                        (((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns + s);
                                    terms.emplace_back(
                                        static_cast<std::uint16_t>(cartesian_plan.dense_to_output[dense]),
                                        cpqr * transform_s.coefficients[is]
                                    );
                                }
                            }
                        }
                    }
                    std::sort(
                        terms.begin(),
                        terms.end(),
                        [](const auto& left, const auto& right) {
                            return left.first < right.first;
                        }
                    );
                    for (std::size_t index = 0; index < terms.size();) {
                        const std::uint16_t cart_index = terms[index].first;
                        double coefficient = 0.0;
                        do {
                            coefficient += terms[index].second;
                            ++index;
                        } while (index < terms.size() && terms[index].first == cart_index);
                        if (coefficient == 0.0) continue;
                        plan.cart_indices.push_back(cart_index);
                        plan.coefficients.push_back(coefficient);
                    }
                    plan.offsets.push_back(plan.cart_indices.size());
                }
            }
        }
    }
    return !plan.outputs.empty();
}

struct SphericalContractionOutput {
    npy_intp p;
    npy_intp q;
    npy_intp r;
    npy_intp s;
};

struct SphericalTargetContractionPlan {
    OSHRRExpansionPlan hrr;
    OSPrimitiveWeightPlan primitive_weights;
    std::vector<ShellQuartetTarget> targets;
    std::vector<SphericalContractionOutput> outputs;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> target_indices;
    std::vector<double> coefficients;
};

struct SphericalTargetPlanCache {
    std::uint64_t key = 0;
    std::vector<SphericalTargetContractionPlan> plans;
    std::vector<unsigned char> valid;
};

struct SphericalWorkerScratch {
    std::vector<double> vrr_table;
    std::vector<double> vrr_table_secondary;
    std::vector<double> values;
    std::vector<double> cart;
};

bool build_rys_spherical_output_plan(
    const ShellBlock& pblk,
    const ShellBlock& qblk,
    const ShellBlock& rblk,
    const ShellBlock& sblk,
    npy_intp pa0,
    npy_intp pb0,
    npy_intp pc0,
    npy_intp pd0,
    const SparseShellTransform& transform_p,
    const SparseShellTransform& transform_q,
    const SparseShellTransform& transform_r,
    const SparseShellTransform& transform_s,
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    npy_intp max_prim,
    bool same_pair,
    RysSphericalOutputPlan& plan
) {
    plan.outputs.clear();
    plan.offsets.assign(1, 0);
    plan.terms.clear();
    for (int axis = 0; axis < 3; ++axis) {
        plan.axis_codes[axis].clear();
        plan.axis_offsets[axis].clear();
        plan.axis_terms[axis].clear();
    }
    const npy_intp na = static_cast<npy_intp>(transform_p.offsets.size()) - 1;
    const npy_intp nb = static_cast<npy_intp>(transform_q.offsets.size()) - 1;
    const npy_intp nc = static_cast<npy_intp>(transform_r.offsets.size()) - 1;
    const npy_intp nd = static_cast<npy_intp>(transform_s.offsets.size()) - 1;
    for (npy_intp a = 0; a < na; ++a) {
        const npy_intp ga = pa0 + a;
        for (npy_intp b = 0; b < nb; ++b) {
            const npy_intp gb = pb0 + b;
            if (ga < gb) continue;
            const npy_intp pair_ab = pair_index(static_cast<int>(ga), static_cast<int>(gb));
            for (npy_intp c = 0; c < nc; ++c) {
                const npy_intp gc = pc0 + c;
                for (npy_intp d = 0; d < nd; ++d) {
                    const npy_intp gd = pd0 + d;
                    if (gc < gd) continue;
                    const npy_intp pair_cd = pair_index(static_cast<int>(gc), static_cast<int>(gd));
                    if (same_pair && pair_ab < pair_cd) continue;
                    plan.outputs.push_back({ga, gb, gc, gd});
                    plan.outputs.back().symmetry_class = static_cast<unsigned char>(
                        (ga == gb) |
                        ((gc == gd) << 1) |
                        ((ga == gc && gb == gd) << 2)
                    );
                    for (std::size_t ip = transform_p.offsets[a]; ip < transform_p.offsets[a + 1]; ++ip) {
                        const npy_intp p = pblk.start + transform_p.cart_indices[ip];
                        for (std::size_t iq = transform_q.offsets[b]; iq < transform_q.offsets[b + 1]; ++iq) {
                            const npy_intp q = qblk.start + transform_q.cart_indices[iq];
                            for (std::size_t ir = transform_r.offsets[c]; ir < transform_r.offsets[c + 1]; ++ir) {
                                const npy_intp r = rblk.start + transform_r.cart_indices[ir];
                                for (std::size_t is = transform_s.offsets[d]; is < transform_s.offsets[d + 1]; ++is) {
                                    const npy_intp s = sblk.start + transform_s.cart_indices[is];
                                    RysSphericalTerm term;
                                    term.coefficient =
                                        transform_p.coefficients[ip] * transform_q.coefficients[iq] *
                                        transform_r.coefficients[ir] * transform_s.coefficients[is];
                                    term.weight_scale =
                                        weights[p * max_prim] / weights[pblk.start * max_prim] *
                                        weights[q * max_prim] / weights[qblk.start * max_prim] *
                                        weights[r * max_prim] / weights[rblk.start * max_prim] *
                                        weights[s * max_prim] / weights[sblk.start * max_prim];
                                    const npy_intp aos[4] = {p, q, r, s};
                                    for (int axis = 0; axis < 3; ++axis) {
                                        const unsigned char code = static_cast<unsigned char>(
                                            shells[3 * aos[0] + axis] |
                                            (shells[3 * aos[1] + axis] << 2) |
                                            (shells[3 * aos[2] + axis] << 4) |
                                            (shells[3 * aos[3] + axis] << 6)
                                        );
                                        auto& codes = plan.axis_codes[axis];
                                        auto found = std::find(codes.begin(), codes.end(), code);
                                        if (found == codes.end()) {
                                            codes.push_back(code);
                                            term.axis_slot[axis] = static_cast<unsigned char>(codes.size() - 1);
                                        } else {
                                            term.axis_slot[axis] = static_cast<unsigned char>(found - codes.begin());
                                        }
                                    }
                                    plan.terms.push_back(term);
                                }
                            }
                        }
                    }
                    plan.offsets.push_back(plan.terms.size());
                }
            }
        }
    }
    const double AB[3] = {
        origins[3 * pblk.start] - origins[3 * qblk.start],
        origins[3 * pblk.start + 1] - origins[3 * qblk.start + 1],
        origins[3 * pblk.start + 2] - origins[3 * qblk.start + 2],
    };
    const double CD[3] = {
        origins[3 * rblk.start] - origins[3 * sblk.start],
        origins[3 * rblk.start + 1] - origins[3 * sblk.start + 1],
        origins[3 * rblk.start + 2] - origins[3 * sblk.start + 2],
    };
    for (int axis = 0; axis < 3; ++axis) {
        auto& expansion_offsets = plan.axis_offsets[axis];
        auto& expansion_terms = plan.axis_terms[axis];
        expansion_offsets.assign(1, 0);
        for (unsigned char code : plan.axis_codes[axis]) {
            const int a = code & 3;
            const int b = (code >> 2) & 3;
            const int c = (code >> 4) & 3;
            const int d = (code >> 6) & 3;
            for (int ib = 0; ib <= b; ++ib) {
                const double bra_coefficient =
                    os_binom_small(b, ib) * os_pow_small(AB[axis], b - ib);
                for (int id = 0; id <= d; ++id) {
                    const double coefficient = bra_coefficient *
                        os_binom_small(d, id) * os_pow_small(CD[axis], d - id);
                    if (coefficient == 0.0) continue;
                    expansion_terms.push_back({
                        static_cast<unsigned char>(a + ib),
                        static_cast<unsigned char>(c + id),
                        coefficient,
                    });
                }
            }
            expansion_offsets.push_back(expansion_terms.size());
        }
    }
    return !plan.outputs.empty();
}

struct SphericalDirectJKPlan {
    std::array<PyArrayObject*, 7> arrays{};
    const std::int64_t* shells = nullptr;
    const double* origins = nullptr;
    const double* exps = nullptr;
    const double* weights = nullptr;
    const std::int64_t* nprim = nullptr;
    const double* pair_bounds = nullptr;
    const double* transform = nullptr;
    npy_intp nao = 0;
    npy_intp nsph = 0;
    npy_intp max_prim = 0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int rys_max_rank = -1;
    int build_workers = 1;
    double screen_tol = 0.0;
    std::vector<ShellBlock> shell_blocks;
    std::vector<npy_intp> sph_starts;
    std::vector<npy_intp> sph_stops;
    std::vector<SparseShellTransform> sparse_shell_transforms;
    ShellPairGeomData pair_geom;
    std::vector<double> shell_pair_bounds;
    std::vector<double> spherical_pair_diagonal;
    std::vector<ShellQuartetTask> tasks;
    std::vector<std::size_t> coulomb_task_offsets;
    std::vector<std::uint32_t> coulomb_task_indices;
    bool needs_target_plans = false;
    std::size_t batched_execution_task_count = 0;
    int contraction_batch_max_l = -1;
    int contraction_batch_max_rank = -1;
    std::vector<std::int32_t> rys_spherical_plan_slots;
    std::vector<RysSphericalOutputPlan> rys_spherical_plans;
    std::array<RysCartesianOutputPlan, 2048> rys_cartesian_shape_plans;
    std::array<CartesianToSphericalPlan, 2048> cartesian_to_spherical_shape_plans;
    std::vector<std::int32_t> rys_recurrence_cache_slots;
    mutable std::vector<RysQuadratureCache> rys_recurrence_caches;
    std::vector<RysQuadraturePointData> rys_recurrence_points;
    std::size_t rys_recurrence_cache_bytes = 0;
    std::size_t rys_recurrence_cache_max_bytes = DEFAULT_RYS_RECURRENCE_CACHE_MAX_BYTES;
    std::size_t rys_spherical_plan_cache_bytes = 0;
    std::size_t rys_spherical_plan_cache_max_bytes =
        DEFAULT_RYS_SPHERICAL_CACHE_MAX_BYTES;
    mutable SphericalTargetPlanCache target_cache;
    std::size_t vrr_table_cap = 1;
    long long shell_screened = 0;
    double geometry_build_seconds = 0.0;
    double task_build_seconds = 0.0;
    double recurrence_select_seconds = 0.0;
    double recurrence_prepare_seconds = 0.0;
    double spherical_plan_seconds = 0.0;
    double total_build_seconds = 0.0;
};

void release_spherical_direct_jk_plan(SphericalDirectJKPlan* plan) {
    if (plan == nullptr) return;
    for (PyArrayObject* array : plan->arrays) {
        Py_XDECREF(array);
    }
    delete plan;
}

void destroy_spherical_direct_jk_plan(PyObject* capsule) {
    auto* plan = static_cast<SphericalDirectJKPlan*>(
        PyCapsule_GetPointer(capsule, "pyqed.SphericalDirectJKPlan")
    );
    if (plan == nullptr) {
        PyErr_Clear();
        return;
    }
    release_spherical_direct_jk_plan(plan);
}

void sort_spherical_tasks_by_cost(
    std::vector<ShellQuartetTask>& tasks,
    const std::vector<ShellBlock>& shell_blocks,
    const ShellPairGeomData& pair_geom,
    int nshell
) {
    auto task_cost = [&](const ShellQuartetTask& task) {
        const ShellBlock& pblk = shell_blocks[task.ish];
        const ShellBlock& qblk = shell_blocks[task.jsh];
        const ShellBlock& rblk = shell_blocks[task.ksh];
        const ShellBlock& sblk = shell_blocks[task.lsh];
        const std::size_t cart_count =
            static_cast<std::size_t>(pblk.stop - pblk.start) *
            static_cast<std::size_t>(qblk.stop - qblk.start) *
            static_cast<std::size_t>(rblk.stop - rblk.start) *
            static_cast<std::size_t>(sblk.stop - sblk.start);
        const std::size_t pq_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
        const std::size_t rs_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
        return cart_count *
            static_cast<std::size_t>(std::max(1, pair_geom.n[pq_idx])) *
            static_cast<std::size_t>(std::max(1, pair_geom.n[rs_idx]));
    };
    constexpr std::size_t NBUCKET = sizeof(std::size_t) * 8;
    std::array<std::size_t, NBUCKET> counts{};
    auto cost_bucket = [&](const ShellQuartetTask& task) {
        std::size_t cost = task_cost(task);
        std::size_t bucket = 0;
        while (cost >>= 1U) ++bucket;
        return bucket;
    };
    for (const ShellQuartetTask& task : tasks) {
        ++counts[cost_bucket(task)];
    }
    std::array<std::size_t, NBUCKET> offsets{};
    std::size_t cursor = 0;
    for (std::size_t bucket = NBUCKET; bucket-- > 0;) {
        offsets[bucket] = cursor;
        cursor += counts[bucket];
    }
    std::vector<ShellQuartetTask> ordered(tasks.size());
    for (const ShellQuartetTask& task : tasks) {
        ordered[offsets[cost_bucket(task)]++] = task;
    }
    tasks.swap(ordered);
}

std::vector<int> general_contraction_group_ids(
    const std::vector<ShellBlock>& shell_blocks,
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp max_prim
) {
    std::vector<int> group_ids(shell_blocks.size(), -1);
    int next_group = 0;
    for (std::size_t ish = 0; ish < shell_blocks.size(); ++ish) {
        for (std::size_t jsh = 0; jsh < ish; ++jsh) {
            if (same_shell_block_member(
                    shells, origins, exps, nprim, max_prim,
                    shell_blocks[jsh].start, shell_blocks[ish].start
                )) {
                group_ids[ish] = group_ids[jsh];
                break;
            }
        }
        if (group_ids[ish] < 0) group_ids[ish] = next_group++;
    }
    return group_ids;
}

bool shell_pair_geometry_matches(
    const ShellPairGeomData& pair_geom,
    std::size_t left,
    std::size_t right
) {
    if (pair_geom.n[left] != pair_geom.n[right]) return false;
    const std::size_t count = static_cast<std::size_t>(pair_geom.n[left]);
    const std::size_t left_offset = left * static_cast<std::size_t>(pair_geom.pair_cap);
    const std::size_t right_offset = right * static_cast<std::size_t>(pair_geom.pair_cap);
    auto matches = [&](const std::vector<double>& values) {
        return std::equal(
            values.begin() + static_cast<std::ptrdiff_t>(left_offset),
            values.begin() + static_cast<std::ptrdiff_t>(left_offset + count),
            values.begin() + static_cast<std::ptrdiff_t>(right_offset)
        );
    };
    return matches(pair_geom.p) && matches(pair_geom.px) &&
        matches(pair_geom.py) && matches(pair_geom.pz) && matches(pair_geom.k);
}

void build_spd_contraction_batches(
    std::vector<ShellQuartetTask>& tasks,
    const std::vector<ShellBlock>& shell_blocks,
    const ShellPairGeomData& pair_geom,
    const std::vector<int>& group_ids,
    int max_batch_l,
    int rys_max_rank
) {
    const std::size_t nshell = shell_blocks.size();
    std::unordered_map<std::uint64_t, std::size_t> leaders;
    leaders.reserve(tasks.size() / 8U + 1U);
    for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
        ShellQuartetTask& task = tasks[task_index];
        const int max_l = std::max({
                shell_blocks[task.ish].l,
                shell_blocks[task.jsh].l,
                shell_blocks[task.ksh].l,
                shell_blocks[task.lsh].l
            });
        const int rank =
            shell_blocks[task.ish].l + shell_blocks[task.jsh].l +
            shell_blocks[task.ksh].l + shell_blocks[task.lsh].l;
        if (max_l > max_batch_l || rank > rys_max_rank) continue;
        const std::uint64_t key =
            static_cast<std::uint64_t>(group_ids[task.ish]) |
            (static_cast<std::uint64_t>(group_ids[task.jsh]) << 16U) |
            (static_cast<std::uint64_t>(group_ids[task.ksh]) << 32U) |
            (static_cast<std::uint64_t>(group_ids[task.lsh]) << 48U);
        const auto inserted = leaders.emplace(key, task_index);
        if (inserted.second) continue;
        ShellQuartetTask& leader = tasks[inserted.first->second];
        if (leader.contraction_batch_size == std::numeric_limits<std::uint16_t>::max()) {
            continue;
        }
        const std::size_t leader_pq =
            static_cast<std::size_t>(leader.ish) * nshell + leader.jsh;
        const std::size_t leader_rs =
            static_cast<std::size_t>(leader.ksh) * nshell + leader.lsh;
        const std::size_t task_pq =
            static_cast<std::size_t>(task.ish) * nshell + task.jsh;
        const std::size_t task_rs =
            static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
        if (
            !shell_pair_geometry_matches(pair_geom, leader_pq, task_pq) ||
            !shell_pair_geometry_matches(pair_geom, leader_rs, task_rs)
        ) continue;
        task.contraction_batch_size = 0;
        task.contraction_batch_next = leader.contraction_batch_next;
        leader.contraction_batch_next = static_cast<std::int32_t>(task_index);
        ++leader.contraction_batch_size;
    }
}

void compact_batched_execution_prefix(
    std::vector<ShellQuartetTask>& tasks,
    const std::vector<ShellBlock>& shell_blocks,
    std::size_t& execution_task_count,
    int& batch_max_l,
    int& batch_max_rank
) {
    std::vector<std::uint32_t> old_to_new(tasks.size());
    std::vector<ShellQuartetTask> ordered;
    ordered.reserve(tasks.size());
    execution_task_count = 0;
    batch_max_l = -1;
    batch_max_rank = -1;
    for (std::size_t old_index = 0; old_index < tasks.size(); ++old_index) {
        const ShellQuartetTask& task = tasks[old_index];
        if (task.contraction_batch_size == 0) continue;
        old_to_new[old_index] = static_cast<std::uint32_t>(ordered.size());
        ordered.push_back(task);
        if (task.contraction_batch_size > 1) {
            const int max_l = std::max({
                shell_blocks[task.ish].l,
                shell_blocks[task.jsh].l,
                shell_blocks[task.ksh].l,
                shell_blocks[task.lsh].l
            });
            const int rank =
                shell_blocks[task.ish].l + shell_blocks[task.jsh].l +
                shell_blocks[task.ksh].l + shell_blocks[task.lsh].l;
            batch_max_l = std::max(batch_max_l, max_l);
            batch_max_rank = std::max(batch_max_rank, rank);
        }
    }
    execution_task_count = ordered.size();
    for (std::size_t old_index = 0; old_index < tasks.size(); ++old_index) {
        const ShellQuartetTask& task = tasks[old_index];
        if (task.contraction_batch_size != 0) continue;
        old_to_new[old_index] = static_cast<std::uint32_t>(ordered.size());
        ordered.push_back(task);
    }
    for (ShellQuartetTask& task : ordered) {
        if (task.contraction_batch_next >= 0) {
            task.contraction_batch_next = static_cast<std::int32_t>(
                old_to_new[static_cast<std::size_t>(task.contraction_batch_next)]
            );
        }
    }
    tasks.swap(ordered);
}

bool initialize_spherical_direct_jk_plan(SphericalDirectJKPlan& plan) {
    using BuildClock = std::chrono::steady_clock;
    const auto build_start = BuildClock::now();
    auto phase_seconds = [](const auto& start, const auto& stop) {
        return std::chrono::duration<double>(stop - start).count();
    };
    if (!try_build_shell_blocks(
            plan.shells,
            plan.origins,
            plan.exps,
            plan.nprim,
            plan.nao,
            plan.max_prim,
            plan.shell_blocks
        )) {
        return false;
    }
    const int nshell = static_cast<int>(plan.shell_blocks.size());
    plan.sph_starts.assign(static_cast<std::size_t>(nshell), 0);
    plan.sph_stops.assign(static_cast<std::size_t>(nshell), 0);
    npy_intp sph_cursor = 0;
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& block = plan.shell_blocks[ish];
        plan.sph_starts[static_cast<std::size_t>(ish)] = sph_cursor;
        sph_cursor += 2 * block.l + 1;
        plan.sph_stops[static_cast<std::size_t>(ish)] = sph_cursor;
    }
    if (sph_cursor != plan.nsph) return false;

    plan.sparse_shell_transforms.resize(static_cast<std::size_t>(nshell));
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& block = plan.shell_blocks[ish];
        const npy_intp sph_start = plan.sph_starts[static_cast<std::size_t>(ish)];
        const npy_intp nshell_sph =
            plan.sph_stops[static_cast<std::size_t>(ish)] - sph_start;
        SparseShellTransform& sparse = plan.sparse_shell_transforms[static_cast<std::size_t>(ish)];
        sparse.offsets.assign(static_cast<std::size_t>(nshell_sph) + 1, 0);
        for (npy_intp a = 0; a < nshell_sph; ++a) {
            for (npy_intp p = block.start; p < block.stop; ++p) {
                const double coefficient = plan.transform[p * plan.nsph + sph_start + a];
                if (coefficient == 0.0) continue;
                sparse.cart_indices.push_back(p - block.start);
                sparse.coefficients.push_back(coefficient);
            }
            sparse.offsets[static_cast<std::size_t>(a) + 1] = sparse.coefficients.size();
        }
    }

    plan.pair_geom = get_primary_shell_pair_geom(
        plan.shell_blocks,
        plan.shells,
        plan.origins,
        plan.exps,
        plan.weights,
        plan.nprim,
        plan.nao,
        plan.max_prim
    );
    build_spherical_shell_pair_bounds(
        plan.shell_blocks,
        plan.sparse_shell_transforms,
        plan.pair_bounds,
        plan.nao,
        plan.shell_pair_bounds
    );
    refine_spherical_shell_pair_bounds_exact(
        plan.shell_blocks,
        plan.sparse_shell_transforms,
        plan.sph_starts,
        plan.shells,
        plan.origins,
        plan.exps,
        plan.weights,
        plan.nprim,
        plan.pair_bounds,
        plan.nao,
        plan.nsph,
        plan.max_prim,
        plan.shell_pair_bounds,
        plan.spherical_pair_diagonal
    );
    const auto geometry_done = BuildClock::now();
    plan.geometry_build_seconds = phase_seconds(build_start, geometry_done);
    build_shell_quartet_tasks(
        plan.shell_blocks,
        plan.shell_pair_bounds,
        nshell,
        plan.screen_tol,
        plan.tasks,
        plan.shell_screened
    );
    sort_spherical_tasks_by_cost(
        plan.tasks,
        plan.shell_blocks,
        plan.pair_geom,
        nshell
    );
    build_spd_contraction_batches(
        plan.tasks,
        plan.shell_blocks,
        plan.pair_geom,
        general_contraction_group_ids(
            plan.shell_blocks,
            plan.shells,
            plan.origins,
            plan.exps,
            plan.nprim,
            plan.max_prim
        ),
        2,
        plan.rys_max_rank
    );
    compact_batched_execution_prefix(
        plan.tasks,
        plan.shell_blocks,
        plan.batched_execution_task_count,
        plan.contraction_batch_max_l,
        plan.contraction_batch_max_rank
    );
    const std::size_t shell_pair_task_count =
        static_cast<std::size_t>(nshell) * (static_cast<std::size_t>(nshell) + 1) / 2;
    plan.coulomb_task_offsets.assign(shell_pair_task_count + 1, 0);
    for (const ShellQuartetTask& task : plan.tasks) {
        const std::size_t pq = static_cast<std::size_t>(pair_index(task.ish, task.jsh));
        const std::size_t rs = static_cast<std::size_t>(pair_index(task.ksh, task.lsh));
        ++plan.coulomb_task_offsets[pq + 1];
        if (rs != pq) ++plan.coulomb_task_offsets[rs + 1];
        const ShellBlock& pblk = plan.shell_blocks[task.ish];
        const ShellBlock& qblk = plan.shell_blocks[task.jsh];
        const ShellBlock& rblk = plan.shell_blocks[task.ksh];
        const ShellBlock& sblk = plan.shell_blocks[task.lsh];
        const int rank = pblk.l + qblk.l + rblk.l + sblk.l;
        const bool rys_sp_eligible =
            plan.rys_max_rank >= 0 &&
            std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 1 &&
            rank <= plan.rys_max_rank;
        const bool rys_l3_eligible =
            plan.rys_max_rank >= 4 &&
            std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 3 &&
            rank <= plan.rys_max_rank;
        plan.needs_target_plans = plan.needs_target_plans ||
            (!rys_sp_eligible && !rys_l3_eligible);
    }
    for (std::size_t pair = 0; pair < shell_pair_task_count; ++pair) {
        plan.coulomb_task_offsets[pair + 1] += plan.coulomb_task_offsets[pair];
    }
    if (plan.tasks.size() > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    plan.coulomb_task_indices.resize(plan.coulomb_task_offsets.back());
    std::vector<std::size_t> task_cursors = plan.coulomb_task_offsets;
    for (std::size_t task_index = 0; task_index < plan.tasks.size(); ++task_index) {
        const ShellQuartetTask& task = plan.tasks[task_index];
        const std::size_t pq = static_cast<std::size_t>(pair_index(task.ish, task.jsh));
        const std::size_t rs = static_cast<std::size_t>(pair_index(task.ksh, task.lsh));
        plan.coulomb_task_indices[task_cursors[pq]++] =
            static_cast<std::uint32_t>(task_index);
        if (rs != pq) {
            plan.coulomb_task_indices[task_cursors[rs]++] =
                static_cast<std::uint32_t>(task_index);
        }
    }
    const auto tasks_done = BuildClock::now();
    plan.task_build_seconds = phase_seconds(geometry_done, tasks_done);
    plan.rys_spherical_plan_slots.assign(plan.tasks.size(), -1);
    plan.rys_recurrence_cache_slots.assign(plan.tasks.size(), -1);
    std::size_t recurrence_cache_bytes = 0;
    plan.rys_recurrence_caches.reserve(std::min<std::size_t>(
        plan.tasks.size(), plan.rys_recurrence_cache_max_bytes / 512U
    ));
    std::vector<const ShellQuartetTask*> recurrence_cache_build_tasks;
    recurrence_cache_build_tasks.reserve(plan.rys_recurrence_caches.capacity());
    std::size_t recurrence_point_count = 0;
    std::unordered_map<std::uint64_t, std::int32_t> selected_quadrature_slots;
    selected_quadrature_slots.reserve(std::min<std::size_t>(
        plan.tasks.size(),
        plan.rys_recurrence_cache_max_bytes /
            std::max<std::size_t>(1, sizeof(RysQuadratureCache))
    ));
    const std::size_t shell_pair_count =
        static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
    std::vector<std::uint64_t> pair_geometry_keys(shell_pair_count);
    for (std::size_t pair_idx = 0; pair_idx < shell_pair_count; ++pair_idx) {
        const std::size_t pair_off =
            pair_idx * static_cast<std::size_t>(plan.pair_geom.pair_cap);
        pair_geometry_keys[pair_idx] = rys_pair_geometry_key(
            plan.pair_geom.p.data() + pair_off,
            plan.pair_geom.px.data() + pair_off,
            plan.pair_geom.py.data() + pair_off,
            plan.pair_geom.pz.data() + pair_off,
            plan.pair_geom.k.data() + pair_off,
            plan.pair_geom.n[pair_idx]
        );
    }
    std::array<std::vector<const ShellQuartetTask*>, 8> tasks_by_nroots;
    for (const ShellQuartetTask& task : plan.tasks) {
        const int rank = plan.shell_blocks[task.ish].l +
            plan.shell_blocks[task.jsh].l + plan.shell_blocks[task.ksh].l +
            plan.shell_blocks[task.lsh].l;
        const int nroots = rank / 2 + 1;
        if (nroots >= 1 && nroots <= 7) {
            tasks_by_nroots[static_cast<std::size_t>(nroots)].push_back(&task);
        }
    }
    for (int preferred_nroots = 1; preferred_nroots <= 7; ++preferred_nroots) {
        for (const ShellQuartetTask* task_ptr : tasks_by_nroots[preferred_nroots]) {
            const ShellQuartetTask& task = *task_ptr;
            const ShellBlock& pblk = plan.shell_blocks[task.ish];
            const ShellBlock& qblk = plan.shell_blocks[task.jsh];
            const ShellBlock& rblk = plan.shell_blocks[task.ksh];
            const ShellBlock& sblk = plan.shell_blocks[task.lsh];
            const int rank = pblk.l + qblk.l + rblk.l + sblk.l;
            const int nroots = preferred_nroots;
            const int max_l = std::max({pblk.l, qblk.l, rblk.l, sblk.l});
            const bool rys_eligible =
                (max_l <= 1 && plan.rys_max_rank >= 0 &&
                 rank <= plan.rys_max_rank) ||
                (max_l <= 3 && plan.rys_max_rank >= 4 &&
                 rank <= plan.rys_max_rank);
            if (!rys_eligible) continue;
            const std::size_t pq_idx =
                static_cast<std::size_t>(task.ish) * nshell + task.jsh;
            const std::size_t rs_idx =
                static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
            const std::size_t npq = static_cast<std::size_t>(plan.pair_geom.n[pq_idx]);
            const std::size_t nrs = static_cast<std::size_t>(plan.pair_geom.n[rs_idx]);
            const std::uint64_t geometry_key = rys_quadrature_geometry_key(
                nroots,
                pair_geometry_keys[pq_idx],
                pair_geometry_keys[rs_idx]
            );
            const auto shared = selected_quadrature_slots.find(geometry_key);
            if (shared != selected_quadrature_slots.end()) {
                plan.rys_recurrence_cache_slots[task.cache_index] = shared->second;
                continue;
            }
            if (
                npq > std::numeric_limits<std::size_t>::max() /
                    std::max<std::size_t>(1, nrs) ||
                npq * nrs > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(nroots) ||
                npq * nrs * static_cast<std::size_t>(nroots) >
                    std::numeric_limits<std::size_t>::max() /
                        sizeof(RysQuadraturePointData)
            ) continue;
            const std::size_t point_count =
                npq * nrs * static_cast<std::size_t>(nroots);
            const std::size_t bytes = sizeof(RysQuadratureCache) +
                point_count * sizeof(RysQuadraturePointData);
            if (
                bytes > plan.rys_recurrence_cache_max_bytes -
                    recurrence_cache_bytes
            ) {
                continue;
            }
            const std::int32_t slot = static_cast<std::int32_t>(
                plan.rys_recurrence_caches.size()
            );
            plan.rys_recurrence_cache_slots[task.cache_index] = slot;
            plan.rys_recurrence_caches.emplace_back();
            RysQuadratureCache& cache = plan.rys_recurrence_caches.back();
            cache.offset = recurrence_point_count;
            cache.size = point_count;
            recurrence_point_count += point_count;
            recurrence_cache_build_tasks.push_back(&task);
            selected_quadrature_slots.emplace(geometry_key, slot);
            recurrence_cache_bytes += bytes;
        }
    }
    const auto recurrence_selected = BuildClock::now();
    plan.recurrence_select_seconds = phase_seconds(tasks_done, recurrence_selected);
    plan.rys_recurrence_points.resize(recurrence_point_count);
    std::atomic<std::size_t> next_cache_slot{0};
    std::atomic<bool> cache_build_failed{false};
    auto prepare_cache_worker = [&](int) {
        while (!cache_build_failed.load(std::memory_order_relaxed)) {
            const std::size_t slot = next_cache_slot.fetch_add(
                1, std::memory_order_relaxed
            );
            if (slot >= recurrence_cache_build_tasks.size()) break;
            const ShellQuartetTask& task = *recurrence_cache_build_tasks[slot];
            const ShellBlock& pblk = plan.shell_blocks[task.ish];
            const ShellBlock& qblk = plan.shell_blocks[task.jsh];
            const ShellBlock& rblk = plan.shell_blocks[task.ksh];
            const ShellBlock& sblk = plan.shell_blocks[task.lsh];
            const int nroots = (pblk.l + qblk.l + rblk.l + sblk.l) / 2 + 1;
            const std::size_t pq_idx =
                static_cast<std::size_t>(task.ish) * nshell + task.jsh;
            const std::size_t rs_idx =
                static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
            const std::size_t pair_cap = static_cast<std::size_t>(plan.pair_geom.pair_cap);
            const std::size_t pq_off = pq_idx * pair_cap;
            const std::size_t rs_off = rs_idx * pair_cap;
            if (!prepare_rys_quadrature_cache(
                    plan.rys_recurrence_caches[slot],
                    plan.rys_recurrence_points,
                    nroots,
                    plan.pair_geom.p.data() + pq_off,
                    plan.pair_geom.px.data() + pq_off,
                    plan.pair_geom.py.data() + pq_off,
                    plan.pair_geom.pz.data() + pq_off,
                    plan.pair_geom.k.data() + pq_off,
                    plan.pair_geom.n[pq_idx],
                    plan.pair_geom.p.data() + rs_off,
                    plan.pair_geom.px.data() + rs_off,
                    plan.pair_geom.py.data() + rs_off,
                    plan.pair_geom.pz.data() + rs_off,
                    plan.pair_geom.k.data() + rs_off,
                    plan.pair_geom.n[rs_idx]
                )) {
                cache_build_failed.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };
    const int cache_build_threads = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(std::max(1, plan.build_workers)),
        std::max<std::size_t>(1, recurrence_cache_build_tasks.size())
    ));
    if (cache_build_threads <= 1) {
        prepare_cache_worker(0);
    } else {
        try {
            native_integral_worker_pool().run(cache_build_threads, prepare_cache_worker);
        } catch (...) {
            cache_build_failed.store(true, std::memory_order_relaxed);
        }
    }
    if (cache_build_failed.load(std::memory_order_relaxed)) return false;
    for (RysQuadratureCache& cache : plan.rys_recurrence_caches) {
        cache.points = plan.rys_recurrence_points.data() + cache.offset;
    }
    plan.rys_recurrence_cache_bytes = recurrence_cache_bytes;
    const auto recurrence_prepared = BuildClock::now();
    plan.recurrence_prepare_seconds = phase_seconds(
        recurrence_selected, recurrence_prepared
    );
    std::size_t spherical_plan_cache_bytes = 0;
    bool spherical_plan_cache_full =
        plan.rys_spherical_plan_cache_max_bytes == 0;
    for (const ShellQuartetTask& task : plan.tasks) {
        const ShellBlock& pblk = plan.shell_blocks[task.ish];
        const ShellBlock& qblk = plan.shell_blocks[task.jsh];
        const ShellBlock& rblk = plan.shell_blocks[task.ksh];
        const ShellBlock& sblk = plan.shell_blocks[task.lsh];
        if (
            plan.rys_max_rank >= 4 &&
            std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 3 &&
            std::max({pblk.l, qblk.l, rblk.l, sblk.l}) >= 2
        ) {
            const int cartesian_plan_key = rys_cartesian_plan_key(
                pblk.l, qblk.l, rblk.l, sblk.l,
                task.ish == task.jsh,
                task.ksh == task.lsh,
                task.ish == task.ksh && task.jsh == task.lsh
            );
            RysCartesianOutputPlan& cartesian_plan =
                plan.rys_cartesian_shape_plans[cartesian_plan_key];
            if (cartesian_plan.outputs.empty() && !build_rys_cartesian_output_plan(
                    plan.shells, plan.weights, plan.max_prim,
                    pblk.start, pblk.stop,
                    qblk.start, qblk.stop,
                    rblk.start, rblk.stop,
                    sblk.start, sblk.stop,
                    true,
                    cartesian_plan
                )) return false;
            CartesianToSphericalPlan& transform_plan =
                plan.cartesian_to_spherical_shape_plans[cartesian_plan_key];
            if (
                transform_plan.outputs.empty() &&
                !build_cartesian_to_spherical_plan(
                    plan.sparse_shell_transforms[static_cast<std::size_t>(task.ish)],
                    plan.sparse_shell_transforms[static_cast<std::size_t>(task.jsh)],
                    plan.sparse_shell_transforms[static_cast<std::size_t>(task.ksh)],
                    plan.sparse_shell_transforms[static_cast<std::size_t>(task.lsh)],
                    cartesian_plan,
                    qblk.stop - qblk.start,
                    rblk.stop - rblk.start,
                    sblk.stop - sblk.start,
                    task.ish == task.jsh,
                    task.ksh == task.lsh,
                    task.ish == task.ksh && task.jsh == task.lsh,
                    transform_plan
                )
            ) return false;
            if (
                task.contraction_batch_size == 1 &&
                !spherical_plan_cache_full
            ) {
                RysSphericalOutputPlan candidate;
                if (!build_rys_spherical_output_plan(
                        pblk, qblk, rblk, sblk,
                        plan.sph_starts[static_cast<std::size_t>(task.ish)],
                        plan.sph_starts[static_cast<std::size_t>(task.jsh)],
                        plan.sph_starts[static_cast<std::size_t>(task.ksh)],
                        plan.sph_starts[static_cast<std::size_t>(task.lsh)],
                        plan.sparse_shell_transforms[static_cast<std::size_t>(task.ish)],
                        plan.sparse_shell_transforms[static_cast<std::size_t>(task.jsh)],
                        plan.sparse_shell_transforms[static_cast<std::size_t>(task.ksh)],
                        plan.sparse_shell_transforms[static_cast<std::size_t>(task.lsh)],
                        plan.shells,
                        plan.origins,
                        plan.weights,
                        plan.max_prim,
                        task.ish == task.ksh && task.jsh == task.lsh,
                        candidate
                    )) return false;
                const std::size_t cart_count =
                    static_cast<std::size_t>(pblk.stop - pblk.start) *
                    static_cast<std::size_t>(qblk.stop - qblk.start) *
                    static_cast<std::size_t>(rblk.stop - rblk.start) *
                    static_cast<std::size_t>(sblk.stop - sblk.start);
                if (candidate.terms.size() <= cart_count) {
                    const std::size_t bytes =
                        rys_spherical_output_plan_bytes(candidate);
                    if (
                        bytes <= plan.rys_spherical_plan_cache_max_bytes -
                            spherical_plan_cache_bytes
                    ) {
                        plan.rys_spherical_plan_slots[task.cache_index] =
                            static_cast<std::int32_t>(
                                plan.rys_spherical_plans.size()
                            );
                        plan.rys_spherical_plans.push_back(std::move(candidate));
                        spherical_plan_cache_bytes += bytes;
                    } else {
                        spherical_plan_cache_full = true;
                    }
                }
            }
        }
        const int lab = plan.shell_blocks[task.ish].l + plan.shell_blocks[task.jsh].l;
        const int lcd = plan.shell_blocks[task.ksh].l + plan.shell_blocks[task.lsh].l;
        if (lab <= OS_VRR_PAIR_MAX_L && lcd <= OS_VRR_PAIR_MAX_L) {
            plan.vrr_table_cap = std::max(
                plan.vrr_table_cap,
                os_vrr_table_size(lab, lcd, lab + lcd)
            );
        }
    }
    plan.rys_spherical_plan_cache_bytes = spherical_plan_cache_bytes;
    const auto spherical_plans_done = BuildClock::now();
    plan.spherical_plan_seconds = phase_seconds(
        recurrence_prepared, spherical_plans_done
    );
    plan.total_build_seconds = phase_seconds(build_start, spherical_plans_done);
    return true;
}

bool build_spherical_target_contraction_plan(
    const std::vector<ShellQuartetTarget>& targets,
    npy_intp p_cart_start,
    npy_intp q_cart_start,
    npy_intp r_cart_start,
    npy_intp s_cart_start,
    npy_intp pa0,
    npy_intp pb0,
    npy_intp pc0,
    npy_intp pd0,
    const SparseShellTransform& transform_p,
    const SparseShellTransform& transform_q,
    const SparseShellTransform& transform_r,
    const SparseShellTransform& transform_s,
    bool same_pair,
    SphericalTargetContractionPlan& plan
) {
    plan.outputs.clear();
    plan.offsets.assign(1, 0);
    plan.target_indices.clear();
    plan.coefficients.clear();
    std::unordered_map<npy_intp, std::size_t> target_lookup;
    target_lookup.reserve(targets.size());
    for (std::size_t it = 0; it < targets.size(); ++it) {
        target_lookup.emplace(targets[it].s8_index, it);
    }
    const npy_intp na = static_cast<npy_intp>(transform_p.offsets.size()) - 1;
    const npy_intp nb = static_cast<npy_intp>(transform_q.offsets.size()) - 1;
    const npy_intp nc = static_cast<npy_intp>(transform_r.offsets.size()) - 1;
    const npy_intp nd = static_cast<npy_intp>(transform_s.offsets.size()) - 1;
    for (npy_intp a = 0; a < na; ++a) {
        const npy_intp ga = pa0 + a;
        for (npy_intp b = 0; b < nb; ++b) {
            const npy_intp gb = pb0 + b;
            if (ga < gb) continue;
            const npy_intp pair_ab = pair_index(static_cast<int>(ga), static_cast<int>(gb));
            for (npy_intp c = 0; c < nc; ++c) {
                const npy_intp gc = pc0 + c;
                for (npy_intp d = 0; d < nd; ++d) {
                    const npy_intp gd = pd0 + d;
                    if (gc < gd) continue;
                    const npy_intp pair_cd = pair_index(static_cast<int>(gc), static_cast<int>(gd));
                    if (same_pair && pair_ab < pair_cd) continue;
                    plan.outputs.push_back({ga, gb, gc, gd});
                    for (std::size_t ip = transform_p.offsets[a]; ip < transform_p.offsets[a + 1]; ++ip) {
                        const npy_intp p = p_cart_start + transform_p.cart_indices[ip];
                        const double cp = transform_p.coefficients[ip];
                        for (std::size_t iq = transform_q.offsets[b]; iq < transform_q.offsets[b + 1]; ++iq) {
                            const npy_intp q = q_cart_start + transform_q.cart_indices[iq];
                            const double cpq = cp * transform_q.coefficients[iq];
                            for (std::size_t ir = transform_r.offsets[c]; ir < transform_r.offsets[c + 1]; ++ir) {
                                const npy_intp r = r_cart_start + transform_r.cart_indices[ir];
                                const double cpqr = cpq * transform_r.coefficients[ir];
                                for (std::size_t is = transform_s.offsets[d]; is < transform_s.offsets[d + 1]; ++is) {
                                    const npy_intp s = s_cart_start + transform_s.cart_indices[is];
                                    const npy_intp s8_index = pair_pair_index(
                                        pair_index(static_cast<int>(p), static_cast<int>(q)),
                                        pair_index(static_cast<int>(r), static_cast<int>(s))
                                    );
                                    const auto found = target_lookup.find(s8_index);
                                    if (found == target_lookup.end()) {
                                        return false;
                                    }
                                    plan.target_indices.push_back(found->second);
                                    plan.coefficients.push_back(cpqr * transform_s.coefficients[is]);
                                }
                            }
                        }
                    }
                    plan.offsets.push_back(plan.target_indices.size());
                }
            }
        }
    }
    return true;
}

void consume_spherical_target_contraction_plan(
    const SphericalTargetContractionPlan& plan,
    const std::vector<double>& values,
    double screen_tol,
    double* eri,
    const double* dm,
    double* vj,
    double* vk,
    npy_intp nsph,
    bool symmetric_density,
    long long& computed,
    long long& skipped
) {
    const bool build_direct = dm != nullptr && vj != nullptr;
    for (std::size_t output = 0; output < plan.outputs.size(); ++output) {
        double value = 0.0;
        for (std::size_t term = plan.offsets[output]; term < plan.offsets[output + 1]; ++term) {
            value += plan.coefficients[term] * values[plan.target_indices[term]];
        }
        if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
            ++skipped;
            continue;
        }
        ++computed;
        const SphericalContractionOutput& item = plan.outputs[output];
        if (build_direct) {
            if (symmetric_density) {
                direct_jk_add_unique_permutations_symmetric_density(
                    vj, vk, dm, nsph, value, item.p, item.q, item.r, item.s
                );
            } else {
                direct_jk_add_unique_permutations(
                    vj, vk, dm, nsph, value, item.p, item.q, item.r, item.s
                );
            }
        } else {
            fill_symmetric_eri(eri, nsph, item.p, item.q, item.r, item.s, value);
        }
    }
}

void build_indexed_shell_density_bounds(
    const std::vector<npy_intp>& starts,
    const std::vector<npy_intp>& stops,
    const double* dm,
    int nshell,
    npy_intp nao,
    std::vector<double>& bounds
) {
    bounds.assign(static_cast<std::size_t>(nshell) * nshell, 0.0);
    for (int ish = 0; ish < nshell; ++ish) {
        for (int jsh = 0; jsh < nshell; ++jsh) {
            double bound = 0.0;
            for (npy_intp i = starts[static_cast<std::size_t>(ish)]; i < stops[static_cast<std::size_t>(ish)]; ++i) {
                for (npy_intp j = starts[static_cast<std::size_t>(jsh)]; j < stops[static_cast<std::size_t>(jsh)]; ++j) {
                    bound = std::max(
                        bound,
                        std::abs(dm[static_cast<std::size_t>(i) * nao + j])
                    );
                }
            }
            bounds[static_cast<std::size_t>(ish) * nshell + jsh] = bound;
        }
    }
}

inline double direct_jk_density_bound_for_shell_task(
    const std::vector<double>& bounds,
    int nshell,
    int ish,
    int jsh,
    int ksh,
    int lsh
) {
    auto get = [&](int a, int b) {
        return bounds[static_cast<std::size_t>(a) * nshell + b];
    };
    return std::max({
        get(ish, jsh), get(jsh, ish),
        get(ksh, lsh), get(lsh, ksh),
        get(ish, ksh), get(ksh, ish),
        get(ish, lsh), get(lsh, ish),
        get(jsh, ksh), get(ksh, jsh),
        get(jsh, lsh), get(lsh, jsh),
    });
}

inline double direct_j_density_bound_for_shell_task(
    const std::vector<double>& bounds,
    int nshell,
    int ish,
    int jsh,
    int ksh,
    int lsh
) {
    auto get = [&](int a, int b) {
        return bounds[static_cast<std::size_t>(a) * nshell + b];
    };
    return std::max({
        get(ish, jsh), get(jsh, ish),
        get(ksh, lsh), get(lsh, ksh),
    });
}

struct SSSSContractionBatchEntry {
    const ShellQuartetTask* task = nullptr;
    const double* pq_weight = nullptr;
    const double* rs_weight = nullptr;
    double value = 0.0;
};

void consume_ssss_contraction_batch_direct_jk(
    std::size_t leader_index,
    const std::vector<ShellQuartetTask>& tasks,
    const std::vector<npy_intp>& sph_starts,
    const ShellPairGeomData& pair_geom,
    const std::vector<double>& shell_pair_bounds,
    const std::vector<double>& shell_dm_bounds,
    int nshell,
    double screen_tol,
    const double* dm,
    double* vj,
    double* vk,
    npy_intp nao,
    bool symmetric_density,
    const std::vector<std::int32_t>& recurrence_slots,
    const std::vector<RysQuadratureCache>& recurrence_caches,
    long long& computed,
    long long& skipped
) {
    static thread_local std::vector<SSSSContractionBatchEntry> entries;
    entries.clear();
    const ShellQuartetTask& leader = tasks[leader_index];
    entries.reserve(leader.contraction_batch_size);
    std::int32_t task_index = static_cast<std::int32_t>(leader_index);
    std::int32_t recurrence_slot = -2;
    for (std::uint16_t member = 0; member < leader.contraction_batch_size; ++member) {
        const ShellQuartetTask& task = tasks[static_cast<std::size_t>(task_index)];
        const std::size_t pq_idx =
            static_cast<std::size_t>(task.ish) * nshell + task.jsh;
        const std::size_t rs_idx =
            static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
        if (screen_tol > 0.0 || vk == nullptr) {
            const double density_bound = vk != nullptr
                ? direct_jk_density_bound_for_shell_task(
                    shell_dm_bounds,
                    nshell,
                    task.ish,
                    task.jsh,
                    task.ksh,
                    task.lsh
                )
                : direct_j_density_bound_for_shell_task(
                    shell_dm_bounds,
                    nshell,
                    task.ish,
                    task.jsh,
                    task.ksh,
                    task.lsh
                );
            if (
                density_bound == 0.0 ||
                (screen_tol > 0.0 &&
                 shell_pair_bounds[pq_idx] * shell_pair_bounds[rs_idx] *
                    density_bound < screen_tol)
            ) {
                skipped += task.shell_count;
                task_index = task.contraction_batch_next;
                continue;
            }
        }
        entries.push_back({
            &task,
            pair_geom.weight.data() + pq_idx * pair_geom.pair_cap,
            pair_geom.weight.data() + rs_idx * pair_geom.pair_cap,
            0.0,
        });
        const std::int32_t slot = recurrence_slots[task.cache_index];
        if (recurrence_slot == -2) {
            recurrence_slot = slot;
        } else if (recurrence_slot != slot) {
            recurrence_slot = -1;
        }
        task_index = task.contraction_batch_next;
    }
    if (entries.empty()) return;

    const ShellQuartetTask& reference = *entries.front().task;
    const std::size_t pq_idx =
        static_cast<std::size_t>(reference.ish) * nshell + reference.jsh;
    const std::size_t rs_idx =
        static_cast<std::size_t>(reference.ksh) * nshell + reference.lsh;
    const std::size_t pq_offset = pq_idx * pair_geom.pair_cap;
    const std::size_t rs_offset = rs_idx * pair_geom.pair_cap;
    const int npq = pair_geom.n[pq_idx];
    const int nrs = pair_geom.n[rs_idx];
    const RysQuadratureCache* quadrature_cache = recurrence_slot >= 0
        ? &recurrence_caches[static_cast<std::size_t>(recurrence_slot)]
        : nullptr;
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            double primitive_value = 0.0;
            if (quadrature_cache != nullptr) {
                primitive_value = quadrature_cache->points[
                    static_cast<std::size_t>(idx_pq) * nrs + idx_rs
                ].root_prefactor;
            } else {
                const double pq_exponent = pair_geom.p[pq_offset + idx_pq];
                const double rs_exponent = pair_geom.p[rs_offset + idx_rs];
                const double zeta = pq_exponent + rs_exponent;
                const double alpha = pq_exponent * rs_exponent / zeta;
                const double dx = pair_geom.px[pq_offset + idx_pq] -
                    pair_geom.px[rs_offset + idx_rs];
                const double dy = pair_geom.py[pq_offset + idx_pq] -
                    pair_geom.py[rs_offset + idx_rs];
                const double dz = pair_geom.pz[pq_offset + idx_pq] -
                    pair_geom.pz[rs_offset + idx_rs];
                primitive_value = ERI_PREFAC *
                    pair_geom.k[pq_offset + idx_pq] *
                    pair_geom.k[rs_offset + idx_rs] *
                    boys0(alpha * (dx * dx + dy * dy + dz * dz)) /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
            }
            for (SSSSContractionBatchEntry& entry : entries) {
                entry.value += entry.pq_weight[idx_pq] *
                    entry.rs_weight[idx_rs] * primitive_value;
            }
        }
    }
    for (const SSSSContractionBatchEntry& entry : entries) {
        const ShellQuartetTask& task = *entry.task;
        const npy_intp p = sph_starts[static_cast<std::size_t>(task.ish)];
        const npy_intp q = sph_starts[static_cast<std::size_t>(task.jsh)];
        const npy_intp r = sph_starts[static_cast<std::size_t>(task.ksh)];
        const npy_intp s = sph_starts[static_cast<std::size_t>(task.lsh)];
        if (symmetric_density) {
            direct_jk_add_unique_permutations_symmetric_density(
                vj, vk, dm, nao, entry.value, p, q, r, s
            );
        } else {
            direct_jk_add_unique_permutations(
                vj, vk, dm, nao, entry.value, p, q, r, s
            );
        }
        ++computed;
    }
}

struct SPDContractionBatchEntry {
    const ShellQuartetTask* task = nullptr;
    const double* pq_weight = nullptr;
    const double* rs_weight = nullptr;
    npy_intp cart_starts[4]{};
    CartesianDirectJKConsumer direct;
};

template<int LP, int LQ, int LR, int LS>
void consume_spd_contraction_batch_values(
    const double* values,
    const SPDContractionBatchEntry& entry,
    const double* weights,
    npy_intp max_prim
) {
    constexpr int NP = (LP + 1) * (LP + 2) / 2;
    constexpr int NQ = (LQ + 1) * (LQ + 2) / 2;
    constexpr int NR = (LR + 1) * (LR + 2) / 2;
    constexpr int NS = (LS + 1) * (LS + 2) / 2;
    double component_scale[4][6]{};
    const npy_intp starts[4] = {
        entry.cart_starts[0], entry.cart_starts[1],
        entry.cart_starts[2], entry.cart_starts[3],
    };
    constexpr int counts[4] = {NP, NQ, NR, NS};
    for (int center = 0; center < 4; ++center) {
        npy_intp primitive = 0;
        while (
            primitive < max_prim &&
            weights[starts[center] * max_prim + primitive] == 0.0
        ) {
            ++primitive;
        }
        if (primitive == max_prim) continue;
        const double reference_weight =
            weights[starts[center] * max_prim + primitive];
        for (int component = 0; component < counts[center]; ++component) {
            component_scale[center][component] =
                weights[(starts[center] + component) * max_prim + primitive] /
                reference_weight;
        }
    }
    const CartesianDirectJKConsumer& direct = entry.direct;
    const bool all_shells_distinct =
        direct.starts[0] != direct.starts[1] &&
        direct.starts[0] != direct.starts[2] &&
        direct.starts[0] != direct.starts[3] &&
        direct.starts[1] != direct.starts[2] &&
        direct.starts[1] != direct.starts[3] &&
        direct.starts[2] != direct.starts[3];
    if (direct.symmetric_density && all_shells_distinct) {
        direct_jk_accumulate_distinct_shell_tiles<NP, NQ, NR, NS>(
            direct,
            [&](std::size_t index, int ip, int iq, int ir, int is) {
                return values[index] *
                    component_scale[0][ip] * component_scale[1][iq] *
                    component_scale[2][ir] * component_scale[3][is];
            }
        );
        return;
    }
    std::size_t index = 0;
    for (int ip = 0; ip < NP; ++ip) {
        const npy_intp p = direct.starts[0] + ip;
        for (int iq = 0; iq < NQ; ++iq) {
            const npy_intp q = direct.starts[1] + iq;
            for (int ir = 0; ir < NR; ++ir) {
                const npy_intp r = direct.starts[2] + ir;
                for (int is = 0; is < NS; ++is, ++index) {
                    const npy_intp s = direct.starts[3] + is;
                    if (direct.pq_same_shell && ip < iq) continue;
                    if (direct.rs_same_shell && ir < is) continue;
                    if (
                        direct.same_pair &&
                        (ip < ir || (ip == ir && iq < is))
                    ) continue;
                    const double value = values[index] *
                        component_scale[0][ip] * component_scale[1][iq] *
                        component_scale[2][ir] * component_scale[3][is];
                    if (direct.symmetric_density) {
                        direct_jk_add_unique_permutations_symmetric_density(
                            direct.vj, direct.vk, direct.dm, direct.nao,
                            value, p, q, r, s
                        );
                    } else {
                        direct_jk_add_unique_permutations(
                            direct.vj, direct.vk, direct.dm, direct.nao,
                            value, p, q, r, s
                        );
                    }
                    ++*direct.computed;
                }
            }
        }
    }
}

template<int LP, int LQ, int LR, int LS>
bool compute_spd_contraction_batch_direct_jk_fixed(
    std::vector<SPDContractionBatchEntry>& entries,
    const double* origins,
    const double* weights,
    npy_intp max_prim,
    const ShellPairGeomData& pair_geom,
    std::size_t pq_idx,
    std::size_t rs_idx,
    const RysQuadratureCache* quadrature_cache,
    std::vector<double>& values
) {
    constexpr int NP = (LP + 1) * (LP + 2) / 2;
    constexpr int NQ = (LQ + 1) * (LQ + 2) / 2;
    constexpr int NR = (LR + 1) * (LR + 2) / 2;
    constexpr int NS = (LS + 1) * (LS + 2) / 2;
    constexpr int NROOTS = (LP + LQ + LR + LS) / 2 + 1;
    constexpr std::size_t NOUT =
        static_cast<std::size_t>(NP) * NQ * NR * NS;
    values.assign(entries.size() * NOUT, 0.0);
    static thread_local std::vector<double> contraction_prefactors;
    contraction_prefactors.resize(entries.size());
    const std::size_t pq_offset = pq_idx * pair_geom.pair_cap;
    const std::size_t rs_offset = rs_idx * pair_geom.pair_cap;
    const int npq = pair_geom.n[pq_idx];
    const int nrs = pair_geom.n[rs_idx];
    const npy_intp p0 = entries.front().cart_starts[0];
    const npy_intp q0 = entries.front().cart_starts[1];
    const npy_intp r0 = entries.front().cart_starts[2];
    const npy_intp s0 = entries.front().cart_starts[3];
    const double AB[3] = {
        origins[3 * p0] - origins[3 * q0],
        origins[3 * p0 + 1] - origins[3 * q0 + 1],
        origins[3 * p0 + 2] - origins[3 * q0 + 2],
    };
    const double CD[3] = {
        origins[3 * r0] - origins[3 * s0],
        origins[3 * r0 + 1] - origins[3 * s0 + 1],
        origins[3 * r0 + 2] - origins[3 * s0 + 2],
    };
    const auto& output_plan = rys_fixed_direct_plan<
        LP, LQ, LR, LS, false, false, false
    >();
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const double pq_exponent = pair_geom.p[pq_offset + idx_pq];
            const double rs_exponent = pair_geom.p[rs_offset + idx_rs];
            const double zeta = pq_exponent + rs_exponent;
            const double alpha = pq_exponent * rs_exponent / zeta;
            const double PQ[3] = {
                pair_geom.px[pq_offset + idx_pq] - pair_geom.px[rs_offset + idx_rs],
                pair_geom.py[pq_offset + idx_pq] - pair_geom.py[rs_offset + idx_rs],
                pair_geom.pz[pq_offset + idx_pq] - pair_geom.pz[rs_offset + idx_rs],
            };
            double local_roots[7]{};
            double local_rys_weights[7]{};
            double primitive_prefactor = 0.0;
            if (quadrature_cache == nullptr) {
                const double T = alpha * (
                    PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]
                );
                if (!rys_roots_weights_general(
                        NROOTS, T, local_roots, local_rys_weights
                    )) return false;
                primitive_prefactor = ERI_PREFAC *
                    pair_geom.k[pq_offset + idx_pq] *
                    pair_geom.k[rs_offset + idx_rs] /
                    (pq_exponent * rs_exponent * std::sqrt(zeta));
            }
            for (int root = 0; root < NROOTS; ++root) {
                double u2 = 0.0;
                double root_prefactor = 0.0;
                const std::size_t quadrature_offset =
                    (static_cast<std::size_t>(idx_pq) * nrs + idx_rs) * NROOTS + root;
                if (quadrature_cache != nullptr) {
                    const RysQuadraturePointData& item =
                        quadrature_cache->points[quadrature_offset];
                    u2 = item.u2;
                    root_prefactor = item.root_prefactor;
                } else {
                    u2 = alpha * local_roots[root];
                    root_prefactor = primitive_prefactor * local_rys_weights[root];
                }
                for (std::size_t member = 0; member < entries.size(); ++member) {
                    contraction_prefactors[member] = root_prefactor *
                        entries[member].pq_weight[idx_pq] *
                        entries[member].rs_weight[idx_rs];
                }
                const double tmp4 = 0.5 /
                    (u2 * zeta + pq_exponent * rs_exponent);
                const double tmp5 = u2 * tmp4;
                const double tmp2 = 2.0 * tmp5 * rs_exponent;
                const double tmp3 = 2.0 * tmp5 * pq_exponent;
                const double b00 = tmp5;
                const double b10 = tmp5 + tmp4 * rs_exponent;
                const double b01 = tmp5 + tmp4 * pq_exponent;
                double c00[3]{};
                double c0p[3]{};
                c00[0] = pair_geom.px[pq_offset + idx_pq] - origins[3 * p0] - tmp2 * PQ[0];
                c00[1] = pair_geom.py[pq_offset + idx_pq] - origins[3 * p0 + 1] - tmp2 * PQ[1];
                c00[2] = pair_geom.pz[pq_offset + idx_pq] - origins[3 * p0 + 2] - tmp2 * PQ[2];
                c0p[0] = pair_geom.px[rs_offset + idx_rs] - origins[3 * r0] + tmp3 * PQ[0];
                c0p[1] = pair_geom.py[rs_offset + idx_rs] - origins[3 * r0 + 1] + tmp3 * PQ[1];
                c0p[2] = pair_geom.pz[rs_offset + idx_rs] - origins[3 * r0 + 2] + tmp3 * PQ[2];
                constexpr std::size_t NDISTRIBUTED = static_cast<std::size_t>(
                    LP | (LQ << 2) | (LR << 4) | (LS << 6)
                ) + 1;
                double distributed[3][NDISTRIBUTED];
                rys_build_shell_shape_tables_fixed<LP, LQ, LR, LS>(
                    c00, c0p, b10, b01, b00, AB, CD, distributed
                );
                for (std::size_t output = 0; output < NOUT; ++output) {
                    const RysFixedDirectOutput& item = output_plan.outputs[output];
                    const double product =
                        distributed[0][item.axis_index[0]] *
                        distributed[1][item.axis_index[1]] *
                        distributed[2][item.axis_index[2]];
                    for (std::size_t member = 0; member < entries.size(); ++member) {
                        values[member * NOUT + output] +=
                            contraction_prefactors[member] * product;
                    }
                }
            }
        }
    }
    for (std::size_t member = 0; member < entries.size(); ++member) {
        consume_spd_contraction_batch_values<LP, LQ, LR, LS>(
            values.data() + member * NOUT,
            entries[member],
            weights,
            max_prim
        );
    }
    return true;
}

bool consume_spd_contraction_batch_direct_jk(
    std::size_t leader_index,
    const std::vector<ShellQuartetTask>& tasks,
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<npy_intp>& sph_starts,
    const ShellPairGeomData& pair_geom,
    const std::vector<double>& shell_pair_bounds,
    const std::vector<double>& shell_dm_bounds,
    int nshell,
    double screen_tol,
    const double* origins,
    const double* weights,
    npy_intp max_prim,
    const double* dm,
    double* vj,
    double* vk,
    npy_intp nao,
    bool cartesian_output,
    bool symmetric_density,
    const std::vector<std::int32_t>& recurrence_slots,
    const std::vector<RysQuadratureCache>& recurrence_caches,
    long long& computed,
    long long& skipped,
    std::vector<double>& values
) {
    static thread_local std::vector<SPDContractionBatchEntry> entries;
    entries.clear();
    const ShellQuartetTask& leader = tasks[leader_index];
    entries.reserve(leader.contraction_batch_size);
    std::int32_t task_index = static_cast<std::int32_t>(leader_index);
    std::int32_t recurrence_slot = -2;
    for (std::uint16_t member = 0; member < leader.contraction_batch_size; ++member) {
        const ShellQuartetTask& task = tasks[static_cast<std::size_t>(task_index)];
        const std::size_t pq_idx =
            static_cast<std::size_t>(task.ish) * nshell + task.jsh;
        const std::size_t rs_idx =
            static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
        if (screen_tol > 0.0 || vk == nullptr) {
            const double density_bound = vk != nullptr
                ? direct_jk_density_bound_for_shell_task(
                    shell_dm_bounds, nshell,
                    task.ish, task.jsh, task.ksh, task.lsh
                )
                : direct_j_density_bound_for_shell_task(
                    shell_dm_bounds, nshell,
                    task.ish, task.jsh, task.ksh, task.lsh
                );
            if (
                density_bound == 0.0 ||
                (screen_tol > 0.0 &&
                 shell_pair_bounds[pq_idx] * shell_pair_bounds[rs_idx] *
                    density_bound < screen_tol)
            ) {
                skipped += task.shell_count;
                task_index = task.contraction_batch_next;
                continue;
            }
        }
        SPDContractionBatchEntry entry;
        entry.task = &task;
        entry.pq_weight = pair_geom.weight.data() + pq_idx * pair_geom.pair_cap;
        entry.rs_weight = pair_geom.weight.data() + rs_idx * pair_geom.pair_cap;
        const int shell_ids[4] = {task.ish, task.jsh, task.ksh, task.lsh};
        entry.direct.dm = dm;
        entry.direct.vj = vj;
        entry.direct.vk = vk;
        entry.direct.nao = nao;
        entry.direct.symmetric_density = symmetric_density;
        entry.direct.pq_same_shell = task.ish == task.jsh;
        entry.direct.rs_same_shell = task.ksh == task.lsh;
        entry.direct.same_pair = task.ish == task.ksh && task.jsh == task.lsh;
        entry.direct.computed = &computed;
        for (int center = 0; center < 4; ++center) {
            entry.cart_starts[center] = shell_blocks[shell_ids[center]].start;
            entry.direct.starts[center] = cartesian_output
                ? shell_blocks[shell_ids[center]].start
                : sph_starts[shell_ids[center]];
        }
        entries.push_back(entry);
        const std::int32_t slot = recurrence_slots[task.cache_index];
        if (recurrence_slot == -2) recurrence_slot = slot;
        else if (recurrence_slot != slot) recurrence_slot = -1;
        task_index = task.contraction_batch_next;
    }
    if (entries.empty()) return true;
    const ShellQuartetTask& reference = *entries.front().task;
    const std::size_t pq_idx =
        static_cast<std::size_t>(reference.ish) * nshell + reference.jsh;
    const std::size_t rs_idx =
        static_cast<std::size_t>(reference.ksh) * nshell + reference.lsh;
    const RysQuadratureCache* quadrature_cache = recurrence_slot >= 0
        ? &recurrence_caches[static_cast<std::size_t>(recurrence_slot)]
        : nullptr;
    const int shape = shell_blocks[reference.ish].l |
        (shell_blocks[reference.jsh].l << 2) |
        (shell_blocks[reference.ksh].l << 4) |
        (shell_blocks[reference.lsh].l << 6);
#define PYQED_SPD_BATCH_CASE(lp, lq, lr, ls) \
    case ((lp) | ((lq) << 2) | ((lr) << 4) | ((ls) << 6)): \
        return compute_spd_contraction_batch_direct_jk_fixed<lp, lq, lr, ls>( \
            entries, origins, weights, max_prim, pair_geom, pq_idx, rs_idx, \
            quadrature_cache, values \
        )
#define PYQED_SPD_BATCH_LS(lp, lq, lr) \
    PYQED_SPD_BATCH_CASE(lp, lq, lr, 0); \
    PYQED_SPD_BATCH_CASE(lp, lq, lr, 1); \
    PYQED_SPD_BATCH_CASE(lp, lq, lr, 2)
#define PYQED_SPD_BATCH_LR(lp, lq) \
    PYQED_SPD_BATCH_LS(lp, lq, 0); \
    PYQED_SPD_BATCH_LS(lp, lq, 1); \
    PYQED_SPD_BATCH_LS(lp, lq, 2)
#define PYQED_SPD_BATCH_LQ(lp) \
    PYQED_SPD_BATCH_LR(lp, 0); \
    PYQED_SPD_BATCH_LR(lp, 1); \
    PYQED_SPD_BATCH_LR(lp, 2)
    switch (shape) {
        PYQED_SPD_BATCH_LQ(0);
        PYQED_SPD_BATCH_LQ(1);
        PYQED_SPD_BATCH_LQ(2);
        default:
            break;
    }
#undef PYQED_SPD_BATCH_LQ
#undef PYQED_SPD_BATCH_LR
#undef PYQED_SPD_BATCH_LS
#undef PYQED_SPD_BATCH_CASE
    return false;
}

void consume_cartesian_shell_quartet_spherical_sparse_fused(
    npy_intp nq,
    npy_intp nr,
    npy_intp ns,
    npy_intp na,
    npy_intp nb,
    npy_intp nc,
    npy_intp nd,
    npy_intp pa0,
    npy_intp pb0,
    npy_intp pc0,
    npy_intp pd0,
    const SparseShellTransform& transform_p,
    const SparseShellTransform& transform_q,
    const SparseShellTransform& transform_r,
    const SparseShellTransform& transform_s,
    const std::vector<double>& cart,
    const std::vector<std::size_t>* dense_to_cart,
    const CartesianToSphericalPlan* reusable_transform_plan,
    double screen_tol,
    double* eri,
    const double* dm,
    double* vj,
    double* vk,
    npy_intp nsph,
    bool symmetric_density,
    bool same_pair,
    long long& computed,
    long long& skipped
) {
    const bool build_direct = dm != nullptr && vj != nullptr;
    if (reusable_transform_plan != nullptr && !reusable_transform_plan->outputs.empty()) {
        const std::vector<CartesianToSphericalOutput>& outputs =
            reusable_transform_plan->outputs;
        const std::vector<std::size_t>& offsets = reusable_transform_plan->offsets;
        const std::uint16_t* cart_indices = reusable_transform_plan->cart_indices.data();
        const double* coefficients = reusable_transform_plan->coefficients.data();
        const double* cart_values = cart.data();
        for (std::size_t output = 0; output < outputs.size(); ++output) {
            double value = 0.0;
            for (std::size_t term = offsets[output]; term < offsets[output + 1]; ++term) {
                value += coefficients[term] * cart_values[cart_indices[term]];
            }
            if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                ++skipped;
                continue;
            }
            ++computed;
            const CartesianToSphericalOutput& item = outputs[output];
            const npy_intp ga = pa0 + item.indices[0];
            const npy_intp gb = pb0 + item.indices[1];
            const npy_intp gc = pc0 + item.indices[2];
            const npy_intp gd = pd0 + item.indices[3];
            if (build_direct) {
                if (symmetric_density) {
                    direct_jk_add_unique_permutations_symmetric_density(
                        vj, vk, dm, nsph, value, ga, gb, gc, gd
                    );
                } else {
                    direct_jk_add_unique_permutations(
                        vj, vk, dm, nsph, value, ga, gb, gc, gd
                    );
                }
            } else {
                fill_symmetric_eri(eri, nsph, ga, gb, gc, gd, value);
            }
        }
        return;
    }
    auto cart_index = [&](npy_intp p, npy_intp q, npy_intp r, npy_intp s) {
        const std::size_t dense =
            (((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns + s);
        return dense_to_cart != nullptr ? (*dense_to_cart)[dense] : dense;
    };
    for (npy_intp a = 0; a < na; ++a) {
        const npy_intp ga = pa0 + a;
        for (npy_intp b = 0; b < nb; ++b) {
            const npy_intp gb = pb0 + b;
            if (ga < gb) {
                continue;
            }
            const npy_intp pair_ab = pair_index(static_cast<int>(ga), static_cast<int>(gb));
            for (npy_intp c = 0; c < nc; ++c) {
                const npy_intp gc = pc0 + c;
                for (npy_intp d = 0; d < nd; ++d) {
                    const npy_intp gd = pd0 + d;
                    if (gc < gd) {
                        continue;
                    }
                    const npy_intp pair_cd = pair_index(static_cast<int>(gc), static_cast<int>(gd));
                    if (same_pair && pair_ab < pair_cd) {
                        continue;
                    }
                    double value = 0.0;
                    for (std::size_t ip = transform_p.offsets[a]; ip < transform_p.offsets[a + 1]; ++ip) {
                        const npy_intp p = transform_p.cart_indices[ip];
                        const double cp = transform_p.coefficients[ip];
                        for (std::size_t iq = transform_q.offsets[b]; iq < transform_q.offsets[b + 1]; ++iq) {
                            const npy_intp q = transform_q.cart_indices[iq];
                            const double cpq = cp * transform_q.coefficients[iq];
                            for (std::size_t ir = transform_r.offsets[c]; ir < transform_r.offsets[c + 1]; ++ir) {
                                const npy_intp r = transform_r.cart_indices[ir];
                                const double cpqr = cpq * transform_r.coefficients[ir];
                                for (std::size_t is = transform_s.offsets[d]; is < transform_s.offsets[d + 1]; ++is) {
                                    value += cpqr * transform_s.coefficients[is] * cart[cart_index(
                                        p,
                                        q,
                                        r,
                                        transform_s.cart_indices[is]
                                    )];
                                }
                            }
                        }
                    }
                    if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    ++computed;
                    if (build_direct) {
                        if (symmetric_density) {
                            direct_jk_add_unique_permutations_symmetric_density(
                                vj, vk, dm, nsph, value, ga, gb, gc, gd
                            );
                        } else {
                            direct_jk_add_unique_permutations(
                                vj, vk, dm, nsph, value, ga, gb, gc, gd
                            );
                        }
                    } else {
                        fill_symmetric_eri(eri, nsph, ga, gb, gc, gd, value);
                    }
                }
            }
        }
    }
}

bool sparse_shell_transform_is_identity(const SparseShellTransform& transform) {
    if (transform.offsets.empty()) return false;
    const std::size_t size = transform.offsets.size() - 1;
    if (transform.cart_indices.size() != size || transform.coefficients.size() != size) {
        return false;
    }
    for (std::size_t index = 0; index < size; ++index) {
        if (
            transform.offsets[index] != index ||
            transform.offsets[index + 1] != index + 1 ||
            transform.cart_indices[index] != static_cast<npy_intp>(index) ||
            transform.coefficients[index] != 1.0
        ) return false;
    }
    return true;
}

void transform_spherical_matrix_to_cartesian(
    const double* transform,
    const double* spherical,
    npy_intp ncart,
    npy_intp nsph,
    std::vector<double>& cartesian
) {
    std::vector<double> intermediate(
        static_cast<std::size_t>(ncart) * static_cast<std::size_t>(nsph),
        0.0
    );
    for (npy_intp p = 0; p < ncart; ++p) {
        const double* transform_row = transform + static_cast<std::size_t>(p) * nsph;
        double* intermediate_row = intermediate.data() + static_cast<std::size_t>(p) * nsph;
        for (npy_intp a = 0; a < nsph; ++a) {
            const double coefficient = transform_row[a];
            if (coefficient == 0.0) continue;
            const double* spherical_row = spherical + static_cast<std::size_t>(a) * nsph;
            for (npy_intp b = 0; b < nsph; ++b) {
                intermediate_row[b] += coefficient * spherical_row[b];
            }
        }
    }
    cartesian.assign(
        static_cast<std::size_t>(ncart) * static_cast<std::size_t>(ncart),
        0.0
    );
    for (npy_intp p = 0; p < ncart; ++p) {
        const double* intermediate_row = intermediate.data() + static_cast<std::size_t>(p) * nsph;
        double* cartesian_row = cartesian.data() + static_cast<std::size_t>(p) * ncart;
        for (npy_intp q = 0; q < ncart; ++q) {
            const double* transform_row = transform + static_cast<std::size_t>(q) * nsph;
            double value = 0.0;
            for (npy_intp b = 0; b < nsph; ++b) {
                value += intermediate_row[b] * transform_row[b];
            }
            cartesian_row[q] = value;
        }
    }
}

void add_cartesian_matrix_to_spherical(
    const double* transform,
    const double* cartesian,
    npy_intp ncart,
    npy_intp nsph,
    double* spherical
) {
    std::vector<double> intermediate(
        static_cast<std::size_t>(ncart) * static_cast<std::size_t>(nsph),
        0.0
    );
    for (npy_intp p = 0; p < ncart; ++p) {
        const double* cartesian_row = cartesian + static_cast<std::size_t>(p) * ncart;
        double* intermediate_row = intermediate.data() + static_cast<std::size_t>(p) * nsph;
        for (npy_intp q = 0; q < ncart; ++q) {
            const double value = cartesian_row[q];
            if (value == 0.0) continue;
            const double* transform_row = transform + static_cast<std::size_t>(q) * nsph;
            for (npy_intp b = 0; b < nsph; ++b) {
                intermediate_row[b] += value * transform_row[b];
            }
        }
    }
    for (npy_intp p = 0; p < ncart; ++p) {
        const double* transform_row = transform + static_cast<std::size_t>(p) * nsph;
        const double* intermediate_row = intermediate.data() + static_cast<std::size_t>(p) * nsph;
        for (npy_intp a = 0; a < nsph; ++a) {
            const double coefficient = transform_row[a];
            if (coefficient == 0.0) continue;
            double* spherical_row = spherical + static_cast<std::size_t>(a) * nsph;
            for (npy_intp b = 0; b < nsph; ++b) {
                spherical_row[b] += coefficient * intermediate_row[b];
            }
        }
    }
}

void consume_cartesian_output_plan_direct_jk(
    const RysCartesianOutputPlan& plan,
    const std::vector<double>& values,
    npy_intp p0,
    npy_intp q0,
    npy_intp r0,
    npy_intp s0,
    const double* dm,
    npy_intp nao,
    bool symmetric_density,
    double* vj,
    double* vk,
    long long& computed
) {
    // Screen before this point at the shell-quartet level: an individual
    // Cartesian component is not a safe bound for its spherical combination.
    for (std::size_t output = 0; output < plan.outputs.size(); ++output) {
        const RysCartesianOutputAngular& item = plan.outputs[output];
        const npy_intp p = p0 + item.local_index[0];
        const npy_intp q = q0 + item.local_index[1];
        const npy_intp r = r0 + item.local_index[2];
        const npy_intp s = s0 + item.local_index[3];
        if (symmetric_density) {
            direct_jk_add_symmetric_class_code(
                vj, vk, dm, nao, values[output], p, q, r, s,
                item.symmetry_class
            );
        } else {
            direct_jk_add_unique_permutations(
                vj, vk, dm, nao, values[output], p, q, r, s
            );
        }
        ++computed;
    }
}

bool compute_dense_eri_spherical_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* pair_bounds,
    const double* transform,
    npy_intp nao,
    npy_intp nsph,
    npy_intp max_prim,
    double screen_tol,
    double* eri,
    const double* dm,
    double* vj,
    double* vk,
    long long& computed,
    long long& skipped,
    int workers,
    int rys_max_rank = -1,
    const SphericalDirectJKPlan* reusable_plan = nullptr,
    int symmetric_density_hint = -1
) {
    const bool build_direct = dm != nullptr && vj != nullptr;
    const bool build_exchange = build_direct && vk != nullptr;
    if (!build_direct && eri == nullptr) {
        return false;
    }
    const bool symmetric_density = build_direct && (
        symmetric_density_hint >= 0
            ? symmetric_density_hint != 0
            : direct_density_is_symmetric(dm, nsph)
    );
    const double output_screen_tol = build_direct ? 0.0 : screen_tol;
    std::vector<ShellBlock> shell_blocks_storage;
    std::vector<npy_intp> sph_starts_storage;
    std::vector<npy_intp> sph_stops_storage;
    std::vector<std::vector<double>> shell_transforms_storage;
    std::vector<SparseShellTransform> sparse_shell_transforms_storage;
    const std::vector<ShellBlock>* shell_blocks_ptr = nullptr;
    const std::vector<npy_intp>* sph_starts_ptr = nullptr;
    const std::vector<npy_intp>* sph_stops_ptr = nullptr;
    const std::vector<SparseShellTransform>* sparse_shell_transforms_ptr = nullptr;
    if (reusable_plan != nullptr) {
        if (
            reusable_plan->nao != nao || reusable_plan->nsph != nsph ||
            reusable_plan->max_prim != max_prim
        ) return false;
        shell_blocks_ptr = &reusable_plan->shell_blocks;
        sph_starts_ptr = &reusable_plan->sph_starts;
        sph_stops_ptr = &reusable_plan->sph_stops;
        sparse_shell_transforms_ptr = &reusable_plan->sparse_shell_transforms;
    } else {
        if (!try_build_shell_blocks(
                shells, origins, exps, nprim, nao, max_prim, shell_blocks_storage
            )) return false;
        const int local_nshell = static_cast<int>(shell_blocks_storage.size());
        sph_starts_storage.assign(static_cast<std::size_t>(local_nshell), 0);
        sph_stops_storage.assign(static_cast<std::size_t>(local_nshell), 0);
        npy_intp sph_cursor = 0;
        for (int ish = 0; ish < local_nshell; ++ish) {
            const ShellBlock& block = shell_blocks_storage[ish];
            sph_starts_storage[static_cast<std::size_t>(ish)] = sph_cursor;
            sph_cursor += 2 * block.l + 1;
            sph_stops_storage[static_cast<std::size_t>(ish)] = sph_cursor;
        }
        if (sph_cursor != nsph) return false;
        shell_transforms_storage.resize(static_cast<std::size_t>(local_nshell));
        sparse_shell_transforms_storage.resize(static_cast<std::size_t>(local_nshell));
        for (int ish = 0; ish < local_nshell; ++ish) {
            const ShellBlock& block = shell_blocks_storage[ish];
            const npy_intp sph_start = sph_starts_storage[static_cast<std::size_t>(ish)];
            const npy_intp nshell_sph =
                sph_stops_storage[static_cast<std::size_t>(ish)] - sph_start;
            std::vector<double>& local = shell_transforms_storage[static_cast<std::size_t>(ish)];
            local.resize(static_cast<std::size_t>(block.stop - block.start) * nshell_sph);
            for (npy_intp p = block.start; p < block.stop; ++p) {
                for (npy_intp a = 0; a < nshell_sph; ++a) {
                    local[static_cast<std::size_t>(p - block.start) * nshell_sph + a] =
                        transform[p * nsph + sph_start + a];
                }
            }
            SparseShellTransform& sparse =
                sparse_shell_transforms_storage[static_cast<std::size_t>(ish)];
            sparse.offsets.resize(static_cast<std::size_t>(nshell_sph) + 1, 0);
            for (npy_intp a = 0; a < nshell_sph; ++a) {
                for (npy_intp p = block.start; p < block.stop; ++p) {
                    const double coefficient =
                        local[static_cast<std::size_t>(p - block.start) * nshell_sph + a];
                    if (coefficient == 0.0) continue;
                    sparse.cart_indices.push_back(p - block.start);
                    sparse.coefficients.push_back(coefficient);
                }
                sparse.offsets[static_cast<std::size_t>(a) + 1] = sparse.coefficients.size();
            }
        }
        shell_blocks_ptr = &shell_blocks_storage;
        sph_starts_ptr = &sph_starts_storage;
        sph_stops_ptr = &sph_stops_storage;
        sparse_shell_transforms_ptr = &sparse_shell_transforms_storage;
    }
    const std::vector<ShellBlock>& shell_blocks = *shell_blocks_ptr;
    const std::vector<npy_intp>& sph_starts = *sph_starts_ptr;
    const std::vector<npy_intp>& sph_stops = *sph_stops_ptr;
    const std::vector<SparseShellTransform>& sparse_shell_transforms =
        *sparse_shell_transforms_ptr;
    const int nshell = static_cast<int>(shell_blocks.size());
    try {
        const ShellPairGeomData& pair_geom = reusable_plan != nullptr
            ? reusable_plan->pair_geom
            : get_primary_shell_pair_geom(
                shell_blocks, shells, origins, exps, weights, nprim, nao, max_prim
            );
        const npy_intp pair_cap = pair_geom.pair_cap;
        std::vector<double> shell_pair_bounds_storage;
        if (reusable_plan == nullptr) {
            build_spherical_shell_pair_bounds(
                shell_blocks,
                sparse_shell_transforms,
                pair_bounds,
                nao,
                shell_pair_bounds_storage
            );
        }
        const std::vector<double>& shell_pair_bounds = reusable_plan != nullptr
            ? reusable_plan->shell_pair_bounds
            : shell_pair_bounds_storage;
        std::vector<ShellQuartetTask> tasks_storage;
        long long shell_screened = 0;
        const std::vector<ShellQuartetTask>* tasks_ptr = nullptr;
        const bool matching_reusable_tasks =
            reusable_plan != nullptr && screen_tol >= reusable_plan->screen_tol;
        if (matching_reusable_tasks) {
            tasks_ptr = &reusable_plan->tasks;
            shell_screened = reusable_plan->shell_screened;
        } else {
            build_shell_quartet_tasks(
                shell_blocks, shell_pair_bounds, nshell, screen_tol,
                tasks_storage, shell_screened
            );
            if (workers > 1 && tasks_storage.size() > 1) {
                sort_spherical_tasks_by_cost(tasks_storage, shell_blocks, pair_geom, nshell);
            }
            tasks_ptr = &tasks_storage;
        }
        const std::vector<ShellQuartetTask>& tasks = *tasks_ptr;
        std::vector<double> shell_dm_bounds;
        if (build_direct) {
            build_indexed_shell_density_bounds(
                sph_starts,
                sph_stops,
                dm,
                nshell,
                nsph,
                shell_dm_bounds
            );
        }
        const std::vector<std::uint32_t>* indexed_tasks = nullptr;
        std::size_t indexed_tasks_start = 0;
        std::size_t indexed_tasks_stop = 0;
        if (
            build_direct && !build_exchange && matching_reusable_tasks &&
            !reusable_plan->coulomb_task_offsets.empty()
        ) {
            int active_shell_pair = -1;
            bool multiple_active_shell_pairs = false;
            for (int ish = 0; ish < nshell && !multiple_active_shell_pairs; ++ish) {
                for (int jsh = 0; jsh <= ish; ++jsh) {
                    const double bound = std::max(
                        shell_dm_bounds[static_cast<std::size_t>(ish) * nshell + jsh],
                        shell_dm_bounds[static_cast<std::size_t>(jsh) * nshell + ish]
                    );
                    if (bound == 0.0) continue;
                    const int shell_pair = pair_index(ish, jsh);
                    if (active_shell_pair < 0) active_shell_pair = shell_pair;
                    else if (active_shell_pair != shell_pair) {
                        multiple_active_shell_pairs = true;
                        break;
                    }
                }
            }
            if (!multiple_active_shell_pairs && active_shell_pair >= 0) {
                indexed_tasks = &reusable_plan->coulomb_task_indices;
                indexed_tasks_start = reusable_plan->coulomb_task_offsets[
                    static_cast<std::size_t>(active_shell_pair)
                ];
                indexed_tasks_stop = reusable_plan->coulomb_task_offsets[
                    static_cast<std::size_t>(active_shell_pair) + 1
                ];
            }
        }
        static thread_local SphericalTargetPlanCache target_plan_cache;
        SphericalTargetPlanCache* active_target_plan_cache_ptr = nullptr;
        const bool matching_reusable_angular_rank =
            matching_reusable_tasks && rys_max_rank == reusable_plan->rys_max_rank;
        bool need_target_plans = matching_reusable_angular_rank
            ? reusable_plan->needs_target_plans
            : false;
        if (!matching_reusable_angular_rank) {
            for (const ShellQuartetTask& task : tasks) {
                const ShellBlock& pblk = shell_blocks[task.ish];
                const ShellBlock& qblk = shell_blocks[task.jsh];
                const ShellBlock& rblk = shell_blocks[task.ksh];
                const ShellBlock& sblk = shell_blocks[task.lsh];
                const int rank = pblk.l + qblk.l + rblk.l + sblk.l;
                const bool rys_sp_eligible =
                    rys_max_rank >= 0 &&
                    std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 1 &&
                    rank <= rys_max_rank;
                const bool rys_l3_eligible =
                    rys_max_rank >= 4 &&
                    std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 3 &&
                    rank <= rys_max_rank;
                if (!rys_sp_eligible && !rys_l3_eligible) {
                    need_target_plans = true;
                    break;
                }
            }
        }
        if (
            need_target_plans &&
            reusable_plan != nullptr &&
            screen_tol == reusable_plan->screen_tol
        ) {
            SphericalTargetPlanCache& cache = reusable_plan->target_cache;
            if (cache.plans.size() != tasks.size()) {
                cache.plans.clear();
                cache.plans.resize(tasks.size());
                cache.valid.assign(tasks.size(), 0);
            }
            active_target_plan_cache_ptr = &cache;
        } else if (need_target_plans) {
            std::uint64_t target_plan_key = primary_shell_pair_geom_key(
                shells, origins, exps, weights, nprim, nao, max_prim
            );
            target_plan_key = fnv1a_mix(target_plan_key, &nsph, sizeof(nsph));
            target_plan_key = fnv1a_mix(target_plan_key, &screen_tol, sizeof(screen_tol));
            target_plan_key = fnv1a_mix(
                target_plan_key,
                transform,
                static_cast<std::size_t>(nao) * static_cast<std::size_t>(nsph) * sizeof(double)
            );
            target_plan_key = fnv1a_mix(
                target_plan_key,
                weights,
                static_cast<std::size_t>(nao) * static_cast<std::size_t>(max_prim) * sizeof(double)
            );
            for (const ShellQuartetTask& task : tasks) {
                target_plan_key = fnv1a_mix(target_plan_key, &task.ish, sizeof(task.ish));
                target_plan_key = fnv1a_mix(target_plan_key, &task.jsh, sizeof(task.jsh));
                target_plan_key = fnv1a_mix(target_plan_key, &task.ksh, sizeof(task.ksh));
                target_plan_key = fnv1a_mix(target_plan_key, &task.lsh, sizeof(task.lsh));
            }
            if (
                target_plan_cache.key != target_plan_key ||
                target_plan_cache.plans.size() != tasks.size()
            ) {
                target_plan_cache.key = target_plan_key;
                target_plan_cache.plans.clear();
                target_plan_cache.plans.resize(tasks.size());
                target_plan_cache.valid.assign(tasks.size(), 0);
            }
            active_target_plan_cache_ptr = &target_plan_cache;
        }
        std::size_t vrr_table_cap = reusable_plan != nullptr &&
            screen_tol == reusable_plan->screen_tol
                ? reusable_plan->vrr_table_cap
                : 1;
        if (reusable_plan == nullptr || screen_tol != reusable_plan->screen_tol) {
            for (const ShellQuartetTask& task : tasks) {
                const int lab = shell_blocks[task.ish].l + shell_blocks[task.jsh].l;
                const int lcd = shell_blocks[task.ksh].l + shell_blocks[task.lsh].l;
                if (lab > OS_VRR_PAIR_MAX_L || lcd > OS_VRR_PAIR_MAX_L) {
                    continue;
                }
                const std::size_t task_cap = os_vrr_table_size(lab, lcd, lab + lcd);
                vrr_table_cap = std::max(vrr_table_cap, task_cap);
            }
        }
        const std::size_t selected_task_count = indexed_tasks != nullptr
            ? indexed_tasks_stop - indexed_tasks_start
            : tasks.size();
        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(std::min<std::size_t>(
                selected_task_count,
                static_cast<std::size_t>(std::numeric_limits<int>::max())
            )))
        );
        const std::size_t sph_n2 =
            static_cast<std::size_t>(nsph) * static_cast<std::size_t>(nsph);
        bool use_fused_cartesian_output = false;
        if (build_direct && matching_reusable_tasks) {
            const std::size_t inspect_count = indexed_tasks != nullptr
                ? indexed_tasks_stop - indexed_tasks_start
                : tasks.size();
            for (std::size_t position = 0; position < inspect_count; ++position) {
                const std::size_t task_index = indexed_tasks != nullptr
                    ? static_cast<std::size_t>(
                        (*indexed_tasks)[indexed_tasks_start + position]
                    )
                    : position;
                const ShellQuartetTask& task = tasks[task_index];
                const ShellBlock& pblk = shell_blocks[task.ish];
                const ShellBlock& qblk = shell_blocks[task.jsh];
                const ShellBlock& rblk = shell_blocks[task.ksh];
                const ShellBlock& sblk = shell_blocks[task.lsh];
                const int rank = pblk.l + qblk.l + rblk.l + sblk.l;
                if (
                    rys_max_rank >= 4 && rank <= rys_max_rank &&
                    std::max({pblk.l, qblk.l, rblk.l, sblk.l}) >= 2 &&
                    std::max({pblk.l, qblk.l, rblk.l, sblk.l}) <= 3 &&
                    reusable_plan->rys_spherical_plan_slots[task.cache_index] < 0
                ) {
                    use_fused_cartesian_output = true;
                    break;
                }
            }
        }
        const bool use_batched_execution_prefix =
            build_exchange && matching_reusable_tasks &&
            reusable_plan->batched_execution_task_count < tasks.size() &&
            rys_max_rank >= reusable_plan->contraction_batch_max_rank &&
            (
                reusable_plan->contraction_batch_max_l <= 1 ||
                (
                    reusable_plan->contraction_batch_max_l <= 2 &&
                    use_fused_cartesian_output
                )
            );
        const std::size_t execution_task_count = indexed_tasks != nullptr
            ? indexed_tasks_stop - indexed_tasks_start
            : use_batched_execution_prefix
                ? reusable_plan->batched_execution_task_count
                : tasks.size();
        const std::size_t cart_n2 =
            static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
        static thread_local std::vector<double> scratch_local_vj;
        static thread_local std::vector<double> scratch_local_vk;
        static thread_local std::vector<double> scratch_cart_dm;
        static thread_local std::vector<double> scratch_cart_vj;
        static thread_local std::vector<double> scratch_cart_vk;
        static thread_local std::vector<double> scratch_local_cart_vj;
        static thread_local std::vector<double> scratch_local_cart_vk;
        static thread_local std::vector<long long> scratch_thread_computed;
        static thread_local std::vector<long long> scratch_thread_skipped;
        std::vector<double>& local_vj = scratch_local_vj;
        std::vector<double>& local_vk = scratch_local_vk;
        std::vector<double>& cart_dm = scratch_cart_dm;
        std::vector<double>& cart_vj = scratch_cart_vj;
        std::vector<double>& cart_vk = scratch_cart_vk;
        std::vector<double>& local_cart_vj = scratch_local_cart_vj;
        std::vector<double>& local_cart_vk = scratch_local_cart_vk;
        std::vector<long long>& thread_computed = scratch_thread_computed;
        std::vector<long long>& thread_skipped = scratch_thread_skipped;
        const bool use_private_direct = build_direct && nthread > 1;
        if (build_direct) {
            std::fill(vj, vj + sph_n2, 0.0);
        }
        if (build_exchange) {
            std::fill(vk, vk + sph_n2, 0.0);
        }
        local_vj.assign(
            use_private_direct ? static_cast<std::size_t>(nthread) * sph_n2 : 0,
            0.0
        );
        local_vk.assign(
            build_exchange && nthread > 1
                ? static_cast<std::size_t>(nthread) * sph_n2
                : 0,
            0.0
        );
        if (use_fused_cartesian_output) {
            transform_spherical_matrix_to_cartesian(
                transform, dm, nao, nsph, cart_dm
            );
            cart_vj.assign(cart_n2, 0.0);
            if (build_exchange) cart_vk.assign(cart_n2, 0.0);
            else cart_vk.clear();
        } else {
            cart_dm.clear();
            cart_vj.clear();
            cart_vk.clear();
        }
        local_cart_vj.assign(
            use_fused_cartesian_output && nthread > 1
                ? static_cast<std::size_t>(nthread) * cart_n2
                : 0,
            0.0
        );
        local_cart_vk.assign(
            use_fused_cartesian_output && build_exchange && nthread > 1
                ? static_cast<std::size_t>(nthread) * cart_n2
                : 0,
            0.0
        );
        thread_computed.assign(static_cast<std::size_t>(nthread), 0);
        thread_skipped.assign(static_cast<std::size_t>(nthread), 0);
        constexpr std::size_t task_dispatch_chunk = 32;
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};

        auto run_worker = [&](int tid) {
            long long local_computed = 0;
            long long local_skipped = 0;
            std::size_t serial_task_index = 0;
            std::size_t task_cursor = 0;
            std::size_t task_end = 0;
            static thread_local SphericalWorkerScratch scratch;
            scratch.vrr_table.resize(vrr_table_cap);
            scratch.vrr_table_secondary.resize(vrr_table_cap);
            std::vector<double>& vrr_table = scratch.vrr_table;
            std::vector<double>& vrr_table_secondary = scratch.vrr_table_secondary;
            std::vector<double>& values = scratch.values;
            std::vector<double>& cart = scratch.cart;
            double* vj_local = use_private_direct
                ? local_vj.data() + static_cast<std::size_t>(tid) * sph_n2
                : vj;
            double* vk_local = !build_exchange
                ? nullptr
                : nthread > 1
                    ? local_vk.data() + static_cast<std::size_t>(tid) * sph_n2
                    : vk;
            double* cart_vj_local = use_fused_cartesian_output && nthread > 1
                ? local_cart_vj.data() + static_cast<std::size_t>(tid) * cart_n2
                : cart_vj.data();
            double* cart_vk_local = !build_exchange
                ? nullptr
                : use_fused_cartesian_output && nthread > 1
                    ? local_cart_vk.data() + static_cast<std::size_t>(tid) * cart_n2
                    : cart_vk.data();
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    if (task_cursor >= task_end) {
                        task_cursor = nthread == 1
                            ? serial_task_index
                            : next_task.fetch_add(
                                task_dispatch_chunk, std::memory_order_relaxed
                            );
                        if (task_cursor >= execution_task_count) {
                            break;
                        }
                        task_end = nthread == 1
                            ? execution_task_count
                            : std::min(
                                task_cursor + task_dispatch_chunk,
                                execution_task_count
                            );
                        serial_task_index = task_end;
                    }
                    const std::size_t task_position = task_cursor++;
                    const std::size_t task_index = indexed_tasks != nullptr
                        ? static_cast<std::size_t>(
                            (*indexed_tasks)[indexed_tasks_start + task_position]
                        )
                        : task_position;
                    const ShellQuartetTask& task = tasks[task_index];
                    int batch_rank = -1;
                    int batch_max_l = -1;
                    bool contraction_batch_enabled = false;
                    if (build_exchange && task.contraction_batch_size != 1) {
                        batch_rank =
                            shell_blocks[task.ish].l + shell_blocks[task.jsh].l +
                            shell_blocks[task.ksh].l + shell_blocks[task.lsh].l;
                        batch_max_l = std::max({
                            shell_blocks[task.ish].l,
                            shell_blocks[task.jsh].l,
                            shell_blocks[task.ksh].l,
                            shell_blocks[task.lsh].l
                        });
                        contraction_batch_enabled = batch_rank <= rys_max_rank &&
                            (batch_max_l <= 1 || (
                                batch_max_l <= 2 && use_fused_cartesian_output
                            ));
                    }
                    if (
                        task.contraction_batch_size == 0 &&
                        contraction_batch_enabled
                    ) {
                        continue;
                    }
                    if (
                        task.contraction_batch_size > 1 &&
                        contraction_batch_enabled
                    ) {
                        bool batch_ok = true;
                        if (batch_rank == 0) {
                            consume_ssss_contraction_batch_direct_jk(
                                task_index,
                                tasks,
                                sph_starts,
                                pair_geom,
                                shell_pair_bounds,
                                shell_dm_bounds,
                                nshell,
                                screen_tol,
                                dm,
                                vj_local,
                                vk_local,
                                nsph,
                                symmetric_density,
                                reusable_plan->rys_recurrence_cache_slots,
                                reusable_plan->rys_recurrence_caches,
                                local_computed,
                                local_skipped
                            );
                        } else {
                            const bool cartesian_batch = batch_max_l >= 2;
                            batch_ok = consume_spd_contraction_batch_direct_jk(
                                task_index,
                                tasks,
                                shell_blocks,
                                sph_starts,
                                pair_geom,
                                shell_pair_bounds,
                                shell_dm_bounds,
                                nshell,
                                screen_tol,
                                origins,
                                weights,
                                max_prim,
                                cartesian_batch ? cart_dm.data() : dm,
                                cartesian_batch ? cart_vj_local : vj_local,
                                cartesian_batch ? cart_vk_local : vk_local,
                                cartesian_batch ? nao : nsph,
                                cartesian_batch,
                                symmetric_density,
                                reusable_plan->rys_recurrence_cache_slots,
                                reusable_plan->rys_recurrence_caches,
                                local_computed,
                                local_skipped,
                                values
                            );
                        }
                        if (!batch_ok) {
                            throw std::runtime_error(
                                "General-contraction Rys batch failed"
                            );
                        }
                        continue;
                    }
                    const ShellBlock& pblk = shell_blocks[task.ish];
                    const ShellBlock& qblk = shell_blocks[task.jsh];
                    const ShellBlock& rblk = shell_blocks[task.ksh];
                    const ShellBlock& sblk = shell_blocks[task.lsh];
                    const std::size_t pq_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
                    const std::size_t rs_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
                    if (build_direct) {
                        const double integral_bound =
                            shell_pair_bounds[pq_idx] * shell_pair_bounds[rs_idx];
                        const double density_bound = build_exchange
                            ? direct_jk_density_bound_for_shell_task(
                                shell_dm_bounds,
                                nshell,
                                task.ish,
                                task.jsh,
                                task.ksh,
                                task.lsh
                            )
                            : direct_j_density_bound_for_shell_task(
                                shell_dm_bounds,
                                nshell,
                                task.ish,
                                task.jsh,
                                task.ksh,
                                task.lsh
                            );
                        if (
                            density_bound == 0.0 ||
                            (screen_tol > 0.0 &&
                             integral_bound * density_bound < screen_tol)
                        ) {
                            local_skipped += task.shell_count;
                            continue;
                        }
                    }
                    const std::size_t pq_off = pq_idx * pair_cap;
                    const std::size_t rs_off = rs_idx * pair_cap;
                    const bool rys_sp_eligible =
                        rys_max_rank >= 0 &&
                        pblk.l <= 1 && qblk.l <= 1 && rblk.l <= 1 && sblk.l <= 1 &&
                        pblk.l + qblk.l + rblk.l + sblk.l <= rys_max_rank;
                    const bool rys_l3_eligible =
                        rys_max_rank >= 4 &&
                        pblk.l <= 3 && qblk.l <= 3 && rblk.l <= 3 && sblk.l <= 3 &&
                        pblk.l + qblk.l + rblk.l + sblk.l <= rys_max_rank;
                    if (rys_sp_eligible || rys_l3_eligible) {
                        const bool same_pair_shells =
                            task.ish == task.ksh && task.jsh == task.lsh;
                        const RysQuadratureCache* quadrature_cache = nullptr;
                        const std::int32_t recurrence_cache_slot =
                            matching_reusable_tasks
                                ? reusable_plan->rys_recurrence_cache_slots[task.cache_index]
                                : -1;
                        if (recurrence_cache_slot >= 0) {
                            quadrature_cache = &reusable_plan->rys_recurrence_caches[
                                static_cast<std::size_t>(recurrence_cache_slot)
                            ];
                        }
                        const bool identity_sp = rys_sp_eligible && build_direct &&
                            sparse_shell_transform_is_identity(
                                sparse_shell_transforms[static_cast<std::size_t>(task.ish)]
                            ) &&
                            sparse_shell_transform_is_identity(
                                sparse_shell_transforms[static_cast<std::size_t>(task.jsh)]
                            ) &&
                            sparse_shell_transform_is_identity(
                                sparse_shell_transforms[static_cast<std::size_t>(task.ksh)]
                            ) &&
                            sparse_shell_transform_is_identity(
                                sparse_shell_transforms[static_cast<std::size_t>(task.lsh)]
                            );
                        const bool unique_sp_direct = identity_sp && same_pair_shells;
                        const int shell_mask = pblk.l | (qblk.l << 1) |
                            (rblk.l << 2) | (sblk.l << 3);
                        const std::int32_t spherical_plan_slot =
                            matching_reusable_tasks
                                ? reusable_plan->rys_spherical_plan_slots[task.cache_index]
                                : -1;
                        const RysSphericalOutputPlan* direct_rys_plan =
                            build_direct && !rys_sp_eligible && spherical_plan_slot >= 0
                                ? &reusable_plan->rys_spherical_plans[
                                    static_cast<std::size_t>(spherical_plan_slot)
                                ]
                                : nullptr;
                        const int cartesian_plan_key = rys_cartesian_plan_key(
                            pblk.l, qblk.l, rblk.l, sblk.l,
                            task.ish == task.jsh,
                            task.ksh == task.lsh,
                            same_pair_shells
                        );
                        const RysCartesianOutputPlan* cartesian_rys_plan =
                            reusable_plan != nullptr && !rys_sp_eligible &&
                            !reusable_plan->rys_cartesian_shape_plans[cartesian_plan_key].outputs.empty()
                                ? &reusable_plan->rys_cartesian_shape_plans[cartesian_plan_key]
                                : nullptr;
                        const CartesianToSphericalPlan* cartesian_to_spherical_plan =
                            reusable_plan != nullptr && !rys_sp_eligible &&
                            !reusable_plan->cartesian_to_spherical_shape_plans[
                                cartesian_plan_key
                            ].outputs.empty()
                                ? &reusable_plan->cartesian_to_spherical_shape_plans[
                                    cartesian_plan_key
                                ]
                                : nullptr;
                        CartesianDirectJKConsumer cartesian_direct_consumer;
                        const CartesianDirectJKConsumer* cartesian_direct_consumer_ptr = nullptr;
                        if (
                            use_fused_cartesian_output && !rys_sp_eligible &&
                            direct_rys_plan == nullptr && cartesian_rys_plan != nullptr
                        ) {
                            cartesian_direct_consumer.dm = cart_dm.data();
                            cartesian_direct_consumer.vj = cart_vj_local;
                            cartesian_direct_consumer.vk = cart_vk_local;
                            cartesian_direct_consumer.nao = nao;
                            cartesian_direct_consumer.starts[0] = pblk.start;
                            cartesian_direct_consumer.starts[1] = qblk.start;
                            cartesian_direct_consumer.starts[2] = rblk.start;
                            cartesian_direct_consumer.starts[3] = sblk.start;
                            cartesian_direct_consumer.symmetric_density = symmetric_density;
                            cartesian_direct_consumer.pq_same_shell = task.ish == task.jsh;
                            cartesian_direct_consumer.rs_same_shell = task.ksh == task.lsh;
                            cartesian_direct_consumer.same_pair = same_pair_shells;
                            cartesian_direct_consumer.computed = &local_computed;
                            cartesian_direct_consumer_ptr = &cartesian_direct_consumer;
                        }
                        bool used_rys = false;
                        if (rys_sp_eligible) {
                            SPDirectJKConsumer direct_consumer;
                            const SPDirectJKConsumer* direct_consumer_ptr = nullptr;
                            if (identity_sp) {
                                direct_consumer.dm = dm;
                                direct_consumer.vj = vj_local;
                                direct_consumer.vk = vk_local;
                                direct_consumer.nao = nsph;
                                direct_consumer.starts[0] = sph_starts[static_cast<std::size_t>(task.ish)];
                                direct_consumer.starts[1] = sph_starts[static_cast<std::size_t>(task.jsh)];
                                direct_consumer.starts[2] = sph_starts[static_cast<std::size_t>(task.ksh)];
                                direct_consumer.starts[3] = sph_starts[static_cast<std::size_t>(task.lsh)];
                                direct_consumer.integral_bound =
                                    shell_pair_bounds[pq_idx] * shell_pair_bounds[rs_idx];
                                direct_consumer.screen_tol = 0.0;
                                direct_consumer.symmetric_density = symmetric_density;
                                direct_consumer.same_pair = same_pair_shells;
                                direct_consumer.computed = &local_computed;
                                direct_consumer.skipped = &local_skipped;
                                direct_consumer_ptr = &direct_consumer;
                            }
                            used_rys = compute_sp_shell_quartet_rys_values_specialized(
                                shell_mask,
                                origins,
                                weights,
                                nprim,
                                max_prim,
                                pblk.start,
                                qblk.start,
                                rblk.start,
                                sblk.start,
                                pair_geom.p.data() + pq_off,
                                pair_geom.px.data() + pq_off,
                                pair_geom.py.data() + pq_off,
                                pair_geom.pz.data() + pq_off,
                                pair_geom.k.data() + pq_off,
                                pair_geom.weight.data() + pq_off,
                                pair_geom.n[pq_idx],
                                pair_geom.p.data() + rs_off,
                                pair_geom.px.data() + rs_off,
                                pair_geom.py.data() + rs_off,
                                pair_geom.pz.data() + rs_off,
                                pair_geom.k.data() + rs_off,
                                pair_geom.weight.data() + rs_off,
                                pair_geom.n[rs_idx],
                                quadrature_cache,
                                unique_sp_direct,
                                task.ish == task.jsh,
                                task.ksh == task.lsh,
                                same_pair_shells,
                                direct_consumer_ptr,
                                cart
                            );
                        } else if (direct_rys_plan != nullptr) {
                            SPDirectJKConsumer direct_consumer;
                            direct_consumer.dm = dm;
                            direct_consumer.vj = vj_local;
                            direct_consumer.vk = vk_local;
                            direct_consumer.nao = nsph;
                            direct_consumer.integral_bound =
                                shell_pair_bounds[pq_idx] * shell_pair_bounds[rs_idx];
                            direct_consumer.screen_tol = 0.0;
                            direct_consumer.symmetric_density = symmetric_density;
                            direct_consumer.same_pair = same_pair_shells;
                            direct_consumer.computed = &local_computed;
                            direct_consumer.skipped = &local_skipped;
                            used_rys = compute_l3_spherical_rys_values(
                                *direct_rys_plan,
                                origins,
                                weights,
                                nprim,
                                max_prim,
                                pblk.start,
                                qblk.start,
                                rblk.start,
                                sblk.start,
                                pblk.l,
                                qblk.l,
                                rblk.l,
                                sblk.l,
                                pair_geom.p.data() + pq_off,
                                pair_geom.px.data() + pq_off,
                                pair_geom.py.data() + pq_off,
                                pair_geom.pz.data() + pq_off,
                                pair_geom.k.data() + pq_off,
                                pair_geom.weight.data() + pq_off,
                                pair_geom.n[pq_idx],
                                pair_geom.p.data() + rs_off,
                                pair_geom.px.data() + rs_off,
                                pair_geom.py.data() + rs_off,
                                pair_geom.pz.data() + rs_off,
                                pair_geom.k.data() + rs_off,
                                pair_geom.weight.data() + rs_off,
                                pair_geom.n[rs_idx],
                                &direct_consumer,
                                values,
                                quadrature_cache
                            );
                        } else {
                            used_rys = compute_l3_shell_quartet_rys_values(
                                shells,
                                origins,
                                weights,
                                nprim,
                                max_prim,
                                pblk.start,
                                pblk.stop,
                                qblk.start,
                                qblk.stop,
                                rblk.start,
                                rblk.stop,
                                sblk.start,
                                sblk.stop,
                                pair_geom.p.data() + pq_off,
                                pair_geom.px.data() + pq_off,
                                pair_geom.py.data() + pq_off,
                                pair_geom.pz.data() + pq_off,
                                pair_geom.k.data() + pq_off,
                                pair_geom.weight.data() + pq_off,
                                pair_geom.n[pq_idx],
                                pair_geom.p.data() + rs_off,
                                pair_geom.px.data() + rs_off,
                                pair_geom.py.data() + rs_off,
                                pair_geom.pz.data() + rs_off,
                                pair_geom.k.data() + rs_off,
                                pair_geom.weight.data() + rs_off,
                                pair_geom.n[rs_idx],
                                cart,
                                cartesian_rys_plan,
                                quadrature_cache,
                                cartesian_direct_consumer_ptr
                            );
                        }
                        if (used_rys) {
                            if (direct_rys_plan != nullptr) {
                                continue;
                            }
                            if (cartesian_direct_consumer_ptr != nullptr) {
                                continue;
                            }
                            if (
                                use_fused_cartesian_output &&
                                cartesian_rys_plan != nullptr
                            ) {
                                consume_cartesian_output_plan_direct_jk(
                                    *cartesian_rys_plan,
                                    cart,
                                    pblk.start,
                                    qblk.start,
                                    rblk.start,
                                    sblk.start,
                                    cart_dm.data(),
                                    nao,
                                    symmetric_density,
                                    cart_vj_local,
                                    cart_vk_local,
                                    local_computed
                                );
                                continue;
                            }
                            const npy_intp pa0 = sph_starts[static_cast<std::size_t>(task.ish)];
                            const npy_intp pb0 = sph_starts[static_cast<std::size_t>(task.jsh)];
                            const npy_intp pc0 = sph_starts[static_cast<std::size_t>(task.ksh)];
                            const npy_intp pd0 = sph_starts[static_cast<std::size_t>(task.lsh)];
                            if (identity_sp) {
                                continue;
                            } else {
                                consume_cartesian_shell_quartet_spherical_sparse_fused(
                                    qblk.stop - qblk.start,
                                    rblk.stop - rblk.start,
                                    sblk.stop - sblk.start,
                                    sph_stops[static_cast<std::size_t>(task.ish)] - pa0,
                                    sph_stops[static_cast<std::size_t>(task.jsh)] - pb0,
                                    sph_stops[static_cast<std::size_t>(task.ksh)] - pc0,
                                    sph_stops[static_cast<std::size_t>(task.lsh)] - pd0,
                                    pa0,
                                    pb0,
                                    pc0,
                                    pd0,
                                    sparse_shell_transforms[static_cast<std::size_t>(task.ish)],
                                    sparse_shell_transforms[static_cast<std::size_t>(task.jsh)],
                                    sparse_shell_transforms[static_cast<std::size_t>(task.ksh)],
                                    sparse_shell_transforms[static_cast<std::size_t>(task.lsh)],
                                    cart,
                                    cartesian_rys_plan != nullptr
                                        ? &cartesian_rys_plan->dense_to_output
                                        : nullptr,
                                    cartesian_to_spherical_plan,
                                    output_screen_tol,
                                    eri,
                                    dm,
                                    vj_local,
                                    vk_local,
                                    nsph,
                                    symmetric_density,
                                    task.ish == task.ksh && task.jsh == task.lsh,
                                    local_computed,
                                    local_skipped
                                );
                            }
                            continue;
                        }
                    }
                    if (active_target_plan_cache_ptr == nullptr) {
                        throw std::runtime_error("Rys shell-quartet evaluation failed");
                    }
                    SphericalTargetPlanCache& active_target_plan_cache =
                        *active_target_plan_cache_ptr;
                    SphericalTargetContractionPlan& spherical_plan =
                        active_target_plan_cache.plans[task.cache_index];
                    std::vector<ShellQuartetTarget>& targets = spherical_plan.targets;
                    long long target_screened = 0;
                    const bool used_vrr = compute_shell_quartet_vrr_hrr_target_values(
                            shells,
                            origins,
                            weights,
                            nprim,
                            max_prim,
                            nao,
                            0.0,
                            nullptr,
                            pblk.start,
                            pblk.stop,
                            qblk.start,
                            qblk.stop,
                            rblk.start,
                            rblk.stop,
                            sblk.start,
                            sblk.stop,
                            pair_geom.a.data() + pq_off,
                            pair_geom.b.data() + pq_off,
                            pair_geom.p.data() + pq_off,
                            pair_geom.px.data() + pq_off,
                            pair_geom.py.data() + pq_off,
                            pair_geom.pz.data() + pq_off,
                            pair_geom.k.data() + pq_off,
                            pair_geom.n[pq_idx],
                            pair_geom.a.data() + rs_off,
                            pair_geom.b.data() + rs_off,
                            pair_geom.p.data() + rs_off,
                            pair_geom.px.data() + rs_off,
                            pair_geom.py.data() + rs_off,
                            pair_geom.pz.data() + rs_off,
                            pair_geom.k.data() + rs_off,
                            pair_geom.n[rs_idx],
                            vrr_table.data(),
                            vrr_table_cap,
                            targets,
                            values,
                            target_screened,
                            nullptr,
                            0.0,
                            &spherical_plan.hrr,
                            true,
                            &spherical_plan.primitive_weights,
                            vrr_table_secondary.data()
                        );

                    if (used_vrr) {
                        const npy_intp pa0 = sph_starts[static_cast<std::size_t>(task.ish)];
                        const npy_intp pb0 = sph_starts[static_cast<std::size_t>(task.jsh)];
                        const npy_intp pc0 = sph_starts[static_cast<std::size_t>(task.ksh)];
                        const npy_intp pd0 = sph_starts[static_cast<std::size_t>(task.lsh)];
                        bool plan_valid =
                            active_target_plan_cache.valid[task.cache_index] != 0;
                        if (!plan_valid) {
                            plan_valid = build_spherical_target_contraction_plan(
                                targets,
                                pblk.start,
                                qblk.start,
                                rblk.start,
                                sblk.start,
                                pa0,
                                pb0,
                                pc0,
                                pd0,
                                sparse_shell_transforms[static_cast<std::size_t>(task.ish)],
                                sparse_shell_transforms[static_cast<std::size_t>(task.jsh)],
                                sparse_shell_transforms[static_cast<std::size_t>(task.ksh)],
                                sparse_shell_transforms[static_cast<std::size_t>(task.lsh)],
                                task.ish == task.ksh && task.jsh == task.lsh,
                                spherical_plan
                            );
                            active_target_plan_cache.valid[task.cache_index] =
                                plan_valid ? 1 : 0;
                        }
                        if (plan_valid) {
                            consume_spherical_target_contraction_plan(
                                spherical_plan,
                                values,
                                output_screen_tol,
                                eri,
                                dm,
                                vj_local,
                                vk_local,
                                nsph,
                                symmetric_density,
                                local_computed,
                                local_skipped
                            );
                            continue;
                        }
                    }

                    const npy_intp np = pblk.stop - pblk.start;
                    const npy_intp nq = qblk.stop - qblk.start;
                    const npy_intp nr = rblk.stop - rblk.start;
                    const npy_intp ns = sblk.stop - sblk.start;
                    const std::size_t cart_size = static_cast<std::size_t>(np) * nq * nr * ns;
                    cart.assign(cart_size, 0.0);
                    auto cart_index = [&](npy_intp p, npy_intp q, npy_intp r, npy_intp s) {
                        return (((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns + s);
                    };
                    auto set_cart = [&](npy_intp p, npy_intp q, npy_intp r, npy_intp s, double value) {
                        if (
                            p >= pblk.start && p < pblk.stop &&
                            q >= qblk.start && q < qblk.stop &&
                            r >= rblk.start && r < rblk.stop &&
                            s >= sblk.start && s < sblk.stop
                        ) {
                            cart[cart_index(
                                p - pblk.start,
                                q - qblk.start,
                                r - rblk.start,
                                s - sblk.start
                            )] = value;
                        }
                    };
                    if (used_vrr) {
                        for (std::size_t it = 0; it < targets.size(); ++it) {
                            const ShellQuartetTarget& target = targets[it];
                            const double value = values[it];
                            set_cart(target.ao_p, target.ao_q, target.ao_r, target.ao_s, value);
                            set_cart(target.ao_q, target.ao_p, target.ao_r, target.ao_s, value);
                            set_cart(target.ao_p, target.ao_q, target.ao_s, target.ao_r, value);
                            set_cart(target.ao_q, target.ao_p, target.ao_s, target.ao_r, value);
                            set_cart(target.ao_r, target.ao_s, target.ao_p, target.ao_q, value);
                            set_cart(target.ao_s, target.ao_r, target.ao_p, target.ao_q, value);
                            set_cart(target.ao_r, target.ao_s, target.ao_q, target.ao_p, value);
                            set_cart(target.ao_s, target.ao_r, target.ao_q, target.ao_p, value);
                        }
                    } else {
                        for (npy_intp p = 0; p < np; ++p) {
                            const npy_intp gp = pblk.start + p;
                            for (npy_intp q = 0; q < nq; ++q) {
                                const npy_intp gq = qblk.start + q;
                                if (gp < gq) continue;
                                const npy_intp pair_pq = pair_index(
                                    static_cast<int>(gp), static_cast<int>(gq)
                                );
                                for (npy_intp r = 0; r < nr; ++r) {
                                    const npy_intp gr = rblk.start + r;
                                    for (npy_intp s = 0; s < ns; ++s) {
                                        const npy_intp gs = sblk.start + s;
                                        if (gr < gs) continue;
                                        const npy_intp pair_rs = pair_index(
                                            static_cast<int>(gr), static_cast<int>(gs)
                                        );
                                        if (
                                            pblk.start == rblk.start &&
                                            qblk.start == sblk.start &&
                                            pair_pq < pair_rs
                                        ) continue;
                                        const double value = contracted_eri_cartesian(
                                            shells,
                                            origins,
                                            exps,
                                            weights,
                                            nprim,
                                            max_prim,
                                            gp,
                                            gq,
                                            gr,
                                            gs
                                        );
                                        set_cart(gp, gq, gr, gs, value);
                                        set_cart(gq, gp, gr, gs, value);
                                        set_cart(gp, gq, gs, gr, value);
                                        set_cart(gq, gp, gs, gr, value);
                                        set_cart(gr, gs, gp, gq, value);
                                        set_cart(gs, gr, gp, gq, value);
                                        set_cart(gr, gs, gq, gp, value);
                                        set_cart(gs, gr, gq, gp, value);
                                    }
                                }
                            }
                        }
                    }

                    const npy_intp pa0 = sph_starts[static_cast<std::size_t>(task.ish)];
                    const npy_intp pb0 = sph_starts[static_cast<std::size_t>(task.jsh)];
                    const npy_intp pc0 = sph_starts[static_cast<std::size_t>(task.ksh)];
                    const npy_intp pd0 = sph_starts[static_cast<std::size_t>(task.lsh)];
                    const npy_intp na = sph_stops[static_cast<std::size_t>(task.ish)] - pa0;
                    const npy_intp nb = sph_stops[static_cast<std::size_t>(task.jsh)] - pb0;
                    const npy_intp nc = sph_stops[static_cast<std::size_t>(task.ksh)] - pc0;
                    const npy_intp nd = sph_stops[static_cast<std::size_t>(task.lsh)] - pd0;
                    consume_cartesian_shell_quartet_spherical_sparse_fused(
                        nq,
                        nr,
                        ns,
                        na,
                        nb,
                        nc,
                        nd,
                        pa0,
                        pb0,
                        pc0,
                        pd0,
                        sparse_shell_transforms[static_cast<std::size_t>(task.ish)],
                        sparse_shell_transforms[static_cast<std::size_t>(task.jsh)],
                        sparse_shell_transforms[static_cast<std::size_t>(task.ksh)],
                        sparse_shell_transforms[static_cast<std::size_t>(task.lsh)],
                        cart,
                        nullptr,
                        nullptr,
                        output_screen_tol,
                        eri,
                        dm,
                        vj_local,
                        vk_local,
                        nsph,
                        symmetric_density,
                        task.ish == task.ksh && task.jsh == task.lsh,
                        local_computed,
                        local_skipped
                    );
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            thread_computed[static_cast<std::size_t>(tid)] = local_computed;
            thread_skipped[static_cast<std::size_t>(tid)] = local_skipped;
        };

        if (nthread <= 1) {
            run_worker(0);
        } else {
            try {
                native_integral_worker_pool().run(nthread, run_worker);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        }
        if (failed.load(std::memory_order_relaxed)) {
            return false;
        }
        computed = 0;
        skipped = shell_screened;
        for (int tid = 0; tid < nthread; ++tid) {
            computed += thread_computed[static_cast<std::size_t>(tid)];
            skipped += thread_skipped[static_cast<std::size_t>(tid)];
            if (use_private_direct) {
                const double* vj_local =
                    local_vj.data() + static_cast<std::size_t>(tid) * sph_n2;
                for (std::size_t index = 0; index < sph_n2; ++index) {
                    vj[index] += vj_local[index];
                }
                if (build_exchange) {
                    const double* vk_local =
                        local_vk.data() + static_cast<std::size_t>(tid) * sph_n2;
                    for (std::size_t index = 0; index < sph_n2; ++index) {
                        vk[index] += vk_local[index];
                    }
                }
            }
        }
        if (use_fused_cartesian_output) {
            if (nthread > 1) {
                for (int tid = 0; tid < nthread; ++tid) {
                    const double* local_cart_j = local_cart_vj.data() +
                        static_cast<std::size_t>(tid) * cart_n2;
                    for (std::size_t index = 0; index < cart_n2; ++index) {
                        cart_vj[index] += local_cart_j[index];
                    }
                    if (build_exchange) {
                        const double* local_cart_k = local_cart_vk.data() +
                            static_cast<std::size_t>(tid) * cart_n2;
                        for (std::size_t index = 0; index < cart_n2; ++index) {
                            cart_vk[index] += local_cart_k[index];
                        }
                    }
                }
            }
            add_cartesian_matrix_to_spherical(
                transform, cart_vj.data(), nao, nsph, vj
            );
            if (build_exchange) {
                add_cartesian_matrix_to_spherical(
                    transform, cart_vk.data(), nao, nsph, vk
                );
            }
        }
    } catch (const std::bad_alloc&) {
        return false;
    } catch (...) {
        return false;
    }
    return true;
}

bool compute_eri_s8_cartesian_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* pair_bounds,
    npy_intp nao,
    npy_intp max_prim,
    double screen_tol,
    double* eri_s8,
    long long& computed,
    long long& skipped,
    int workers
) {
    std::vector<ShellBlock> shell_blocks;
    if (!try_build_shell_blocks(shells, origins, exps, nprim, nao, max_prim, shell_blocks)) {
        return false;
    }

    const int nshell = static_cast<int>(shell_blocks.size());
    for (int ish = 0; ish < nshell; ++ish) {
        for (int jsh = 0; jsh <= ish; ++jsh) {
            if (shell_blocks[ish].l + shell_blocks[jsh].l > OS_VRR_PAIR_MAX_L) {
                return false;
            }
            for (int ksh = 0; ksh <= ish; ++ksh) {
                const int lsh_max = (ksh == ish) ? jsh : ksh;
                for (int lsh = 0; lsh <= lsh_max; ++lsh) {
                    if (shell_blocks[ksh].l + shell_blocks[lsh].l > OS_VRR_PAIR_MAX_L) {
                        return false;
                    }
                }
            }
        }
    }

    try {
        const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
            shell_blocks,
            shells,
            origins,
            exps,
            weights,
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        const std::size_t vrr_table_cap = os_vrr_table_size(
            OS_VRR_PAIR_MAX_L,
            OS_VRR_PAIR_MAX_L,
            2 * OS_VRR_PAIR_MAX_L
        );
        for (int ish = 0; ish < nshell; ++ish) {
            const ShellBlock& pblk = shell_blocks[ish];
            for (int jsh = 0; jsh < nshell; ++jsh) {
                const ShellBlock& qblk = shell_blocks[jsh];
                double bound = 0.0;
                for (npy_intp ip = pblk.start; ip < pblk.stop; ++ip) {
                    for (npy_intp iq = qblk.start; iq < qblk.stop; ++iq) {
                        bound = std::max(bound, pair_bounds[ip * nao + iq]);
                    }
                }
                const std::size_t pair_idx = static_cast<std::size_t>(ish) * nshell + jsh;
                shell_pair_bounds[pair_idx] = bound;
            }
        }

        std::vector<ShellQuartetTask> tasks;
        long long shell_screened = 0;
        build_shell_quartet_tasks(
            shell_blocks,
            shell_pair_bounds,
            nshell,
            screen_tol,
            tasks,
            shell_screened
        );

        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(tasks.size()))
        );
        std::vector<long long> thread_computed(static_cast<std::size_t>(nthread), 0);
        std::vector<long long> thread_skipped(static_cast<std::size_t>(nthread), 0);
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};

        auto run_worker = [&](int tid) {
            long long local_computed = 0;
            long long local_skipped = 0;
            std::vector<double> vrr_table(vrr_table_cap, 0.0);
            std::vector<ShellQuartetTarget> targets;
            std::vector<double> target_values;
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size()) {
                        break;
                    }
                    const ShellQuartetTask& task = tasks[task_index];
                    const ShellBlock& pblk = shell_blocks[task.ish];
                    const ShellBlock& qblk = shell_blocks[task.jsh];
                    const ShellBlock& rblk = shell_blocks[task.ksh];
                    const ShellBlock& sblk = shell_blocks[task.lsh];
                    const std::size_t pq_pair_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
                    const std::size_t rs_pair_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
                    const std::size_t pq_offset = pq_pair_idx * pair_cap;
                    const std::size_t rs_offset = rs_pair_idx * pair_cap;
                    long long target_screened = 0;
                    if (!compute_shell_quartet_vrr_hrr_target_values(
                            shells,
                            origins,
                            weights,
                            nprim,
                            max_prim,
                            nao,
                            screen_tol,
                            pair_bounds,
                            pblk.start,
                            pblk.stop,
                            qblk.start,
                            qblk.stop,
                            rblk.start,
                            rblk.stop,
                            sblk.start,
                            sblk.stop,
                            pair_geom.a.data() + pq_offset,
                            pair_geom.b.data() + pq_offset,
                            pair_geom.p.data() + pq_offset,
                            pair_geom.px.data() + pq_offset,
                            pair_geom.py.data() + pq_offset,
                            pair_geom.pz.data() + pq_offset,
                            pair_geom.k.data() + pq_offset,
                            pair_geom.n[pq_pair_idx],
                            pair_geom.a.data() + rs_offset,
                            pair_geom.b.data() + rs_offset,
                            pair_geom.p.data() + rs_offset,
                            pair_geom.px.data() + rs_offset,
                            pair_geom.py.data() + rs_offset,
                            pair_geom.pz.data() + rs_offset,
                            pair_geom.k.data() + rs_offset,
                            pair_geom.n[rs_pair_idx],
                            vrr_table.data(),
                            vrr_table_cap,
                            targets,
                            target_values,
                            target_screened
                        )) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                    local_skipped += target_screened;

                    for (std::size_t it = 0; it < targets.size(); ++it) {
                        const double value = target_values[it];
                        if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                            ++local_skipped;
                            continue;
                        }
                        eri_s8[targets[it].s8_index] = value;
                        ++local_computed;
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            thread_computed[static_cast<std::size_t>(tid)] = local_computed;
            thread_skipped[static_cast<std::size_t>(tid)] = local_skipped;
        };

        if (nthread <= 1) {
            run_worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(nthread));
            try {
                for (int tid = 0; tid < nthread; ++tid) {
                    threads.emplace_back(run_worker, tid);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        if (failed.load(std::memory_order_relaxed)) {
            return false;
        }

        computed = 0;
        skipped = shell_screened;
        for (int tid = 0; tid < nthread; ++tid) {
            computed += thread_computed[static_cast<std::size_t>(tid)];
            skipped += thread_skipped[static_cast<std::size_t>(tid)];
        }
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

inline int target_angular_power(const ShellQuartetTarget& target, int index) {
    const int angular[12] = {
        target.ax, target.ay, target.az,
        target.bx, target.by, target.bz,
        target.cx, target.cy, target.cz,
        target.dx, target.dy, target.dz,
    };
    return angular[index];
}

struct HrrExpansionTerm {
    std::size_t index;
    double coefficient;
};

struct HrrExpansion {
    std::size_t offset;
    std::size_t size;
};

struct HrrExpansionCache {
    std::unordered_map<std::uint64_t, int> lookup;
    std::vector<HrrExpansion> expansions;
    std::vector<HrrExpansionTerm> terms;

    void clear() {
        lookup.clear();
        expansions.clear();
        terms.clear();
        if (lookup.bucket_count() < 512) {
            lookup.reserve(512);
        }
    }
};

struct HrrFirstDerivativeRecipe {
    int high = -1;
    int low = -1;
    int power = 0;
};

struct HrrSecondDerivativeRecipe {
    int pp = -1;
    int pm = -1;
    int mp = -1;
    int mm = -1;
    int power_a = 0;
    int power_b = 0;
    bool same = false;
};

inline std::uint64_t hrr_angular_key(const std::array<int, 12>& angular) {
    std::uint64_t key = 0;
    for (int value : angular) {
        key = key * 8U + static_cast<std::uint64_t>(value);
    }
    return key;
}

int get_hrr_expansion(
    HrrExpansionCache& cache,
    const std::array<int, 12>& angular,
    int max_a_l,
    int max_c_l,
    int max_m_l,
    const double* AB,
    const double* CD
) {
    for (int value : angular) {
        if (value < 0) {
            return -1;
        }
    }
    const std::uint64_t key = hrr_angular_key(angular);
    const auto found = cache.lookup.find(key);
    if (found != cache.lookup.end()) {
        return found->second;
    }

    const int adim = max_a_l + 1;
    const int cdim = max_c_l + 1;
    const int mdim = max_m_l + 1;
    const std::size_t offset = cache.terms.size();
    const int ax = angular[0];
    const int ay = angular[1];
    const int az = angular[2];
    const int bx = angular[3];
    const int by = angular[4];
    const int bz = angular[5];
    const int cx = angular[6];
    const int cy = angular[7];
    const int cz = angular[8];
    const int dx = angular[9];
    const int dy = angular[10];
    const int dz = angular[11];
    for (int ix = 0; ix <= bx; ++ix) {
        for (int iy = 0; iy <= by; ++iy) {
            for (int iz = 0; iz <= bz; ++iz) {
                const double coeff_b =
                    os_binom_small(bx, ix) * os_pow_small(AB[0], bx - ix)
                    * os_binom_small(by, iy) * os_pow_small(AB[1], by - iy)
                    * os_binom_small(bz, iz) * os_pow_small(AB[2], bz - iz);
                if (coeff_b == 0.0) {
                    continue;
                }
                for (int jx = 0; jx <= dx; ++jx) {
                    for (int jy = 0; jy <= dy; ++jy) {
                        for (int jz = 0; jz <= dz; ++jz) {
                            const double coefficient = coeff_b
                                * os_binom_small(dx, jx) * os_pow_small(CD[0], dx - jx)
                                * os_binom_small(dy, jy) * os_pow_small(CD[1], dy - jy)
                                * os_binom_small(dz, jz) * os_pow_small(CD[2], dz - jz);
                            if (coefficient == 0.0) {
                                continue;
                            }
                            cache.terms.push_back({
                                os_vrr_idx(
                                    ax + ix,
                                    ay + iy,
                                    az + iz,
                                    cx + jx,
                                    cy + jy,
                                    cz + jz,
                                    0,
                                    adim,
                                    cdim,
                                    mdim
                                ),
                                coefficient,
                            });
                        }
                    }
                }
            }
        }
    }
    const int expansion = static_cast<int>(cache.expansions.size());
    cache.expansions.push_back({offset, cache.terms.size() - offset});
    cache.lookup.emplace(key, expansion);
    return expansion;
}

inline double evaluate_hrr_expansion(
    const HrrExpansionCache& cache,
    int expansion,
    const double* table
) {
    if (expansion < 0) {
        return 0.0;
    }
    const HrrExpansion& entry = cache.expansions[static_cast<std::size_t>(expansion)];
    const HrrExpansionTerm* terms = cache.terms.data() + entry.offset;
    double value = 0.0;
    for (std::size_t i = 0; i < entry.size; ++i) {
        value += terms[i].coefficient * table[terms[i].index];
    }
    return value;
}

int get_shifted_hrr_expansion(
    HrrExpansionCache& cache,
    const ShellQuartetTarget& target,
    int slot_a,
    int axis_a,
    int shift_a,
    int slot_b,
    int axis_b,
    int shift_b,
    int max_a_l,
    int max_c_l,
    int max_m_l,
    const double* AB,
    const double* CD
) {
    std::array<int, 12> angular = {
        target.ax, target.ay, target.az,
        target.bx, target.by, target.bz,
        target.cx, target.cy, target.cz,
        target.dx, target.dy, target.dz,
    };
    angular[3 * slot_a + axis_a] += shift_a;
    if (slot_b >= 0) {
        angular[3 * slot_b + axis_b] += shift_b;
    }
    return get_hrr_expansion(
        cache, angular, max_a_l, max_c_l, max_m_l, AB, CD
    );
}

HrrFirstDerivativeRecipe build_hrr_first_recipe(
    HrrExpansionCache& cache,
    const ShellQuartetTarget& target,
    int slot,
    int axis,
    int max_a_l,
    int max_c_l,
    int max_m_l,
    const double* AB,
    const double* CD
) {
    HrrFirstDerivativeRecipe recipe;
    recipe.power = target_angular_power(target, 3 * slot + axis);
    recipe.high = get_shifted_hrr_expansion(
        cache, target, slot, axis, 1, -1, 0, 0,
        max_a_l, max_c_l, max_m_l, AB, CD
    );
    if (recipe.power > 0) {
        recipe.low = get_shifted_hrr_expansion(
            cache, target, slot, axis, -1, -1, 0, 0,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
    }
    return recipe;
}

HrrSecondDerivativeRecipe build_hrr_second_recipe(
    HrrExpansionCache& cache,
    const ShellQuartetTarget& target,
    int slot_a,
    int axis_a,
    int slot_b,
    int axis_b,
    int max_a_l,
    int max_c_l,
    int max_m_l,
    const double* AB,
    const double* CD
) {
    HrrSecondDerivativeRecipe recipe;
    const int idx_a = 3 * slot_a + axis_a;
    const int idx_b = 3 * slot_b + axis_b;
    recipe.power_a = target_angular_power(target, idx_a);
    recipe.power_b = target_angular_power(target, idx_b);
    recipe.same = idx_a == idx_b;
    if (recipe.same) {
        recipe.pp = get_shifted_hrr_expansion(
            cache, target, slot_a, axis_a, 2, -1, 0, 0,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
        recipe.pm = get_shifted_hrr_expansion(
            cache, target, slot_a, axis_a, 0, -1, 0, 0,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
        if (recipe.power_a > 1) {
            recipe.mm = get_shifted_hrr_expansion(
                cache, target, slot_a, axis_a, -2, -1, 0, 0,
                max_a_l, max_c_l, max_m_l, AB, CD
            );
        }
        return recipe;
    }

    recipe.pp = get_shifted_hrr_expansion(
        cache, target, slot_a, axis_a, 1, slot_b, axis_b, 1,
        max_a_l, max_c_l, max_m_l, AB, CD
    );
    if (recipe.power_b > 0) {
        recipe.pm = get_shifted_hrr_expansion(
            cache, target, slot_a, axis_a, 1, slot_b, axis_b, -1,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
    }
    if (recipe.power_a > 0) {
        recipe.mp = get_shifted_hrr_expansion(
            cache, target, slot_a, axis_a, -1, slot_b, axis_b, 1,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
    }
    if (recipe.power_a > 0 && recipe.power_b > 0) {
        recipe.mm = get_shifted_hrr_expansion(
            cache, target, slot_a, axis_a, -1, slot_b, axis_b, -1,
            max_a_l, max_c_l, max_m_l, AB, CD
        );
    }
    return recipe;
}

inline double evaluate_hrr_first_recipe(
    const HrrExpansionCache& cache,
    const HrrFirstDerivativeRecipe& recipe,
    double exponent,
    const double* table
) {
    return 2.0 * exponent * evaluate_hrr_expansion(cache, recipe.high, table)
        - recipe.power * evaluate_hrr_expansion(cache, recipe.low, table);
}

inline double evaluate_hrr_second_recipe(
    const HrrExpansionCache& cache,
    const HrrSecondDerivativeRecipe& recipe,
    double exponent_a,
    double exponent_b,
    const double* table
) {
    if (recipe.same) {
        return 4.0 * exponent_a * exponent_a
                * evaluate_hrr_expansion(cache, recipe.pp, table)
            - 2.0 * exponent_a * (2.0 * recipe.power_a + 1.0)
                * evaluate_hrr_expansion(cache, recipe.pm, table)
            + recipe.power_a * (recipe.power_a - 1.0)
                * evaluate_hrr_expansion(cache, recipe.mm, table);
    }
    return 4.0 * exponent_a * exponent_b
            * evaluate_hrr_expansion(cache, recipe.pp, table)
        - 2.0 * exponent_a * recipe.power_b
            * evaluate_hrr_expansion(cache, recipe.pm, table)
        - 2.0 * exponent_b * recipe.power_a
            * evaluate_hrr_expansion(cache, recipe.mp, table)
        + recipe.power_a * recipe.power_b
            * evaluate_hrr_expansion(cache, recipe.mm, table);
}

inline double primitive_eri_target_angular(
    const ShellQuartetTarget& target,
    const int* angular,
    const double* exponents,
    const double* origins
) {
    return primitive_eri_cartesian(
        exponents[0],
        angular[0], angular[1], angular[2],
        origins + 3 * target.ao_p,
        exponents[1],
        angular[3], angular[4], angular[5],
        origins + 3 * target.ao_q,
        exponents[2],
        angular[6], angular[7], angular[8],
        origins + 3 * target.ao_r,
        exponents[3],
        angular[9], angular[10], angular[11],
        origins + 3 * target.ao_s
    );
}

inline double primitive_center_first_derivative_target(
    const ShellQuartetTarget& target,
    int slot,
    int axis,
    const double* exponents,
    const double* origins
) {
    int angular[12] = {
        target.ax, target.ay, target.az,
        target.bx, target.by, target.bz,
        target.cx, target.cy, target.cz,
        target.dx, target.dy, target.dz,
    };
    const int idx = 3 * slot + axis;
    const int power = angular[idx];
    angular[idx] += 1;
    double value = 2.0 * exponents[slot] * primitive_eri_target_angular(
        target, angular, exponents, origins
    );
    if (power > 0) {
        angular[idx] -= 2;
        value -= power * primitive_eri_target_angular(
            target, angular, exponents, origins
        );
    }
    return value;
}

inline double primitive_center_second_derivative_target(
    const ShellQuartetTarget& target,
    int slot_a,
    int axis_a,
    int slot_b,
    int axis_b,
    const double* exponents,
    const double* origins
) {
    int angular[12] = {
        target.ax, target.ay, target.az,
        target.bx, target.by, target.bz,
        target.cx, target.cy, target.cz,
        target.dx, target.dy, target.dz,
    };
    const int idx_a = 3 * slot_a + axis_a;
    const int idx_b = 3 * slot_b + axis_b;
    const int power_a = angular[idx_a];
    const int power_b = angular[idx_b];
    if (idx_a == idx_b) {
        angular[idx_a] += 2;
        double value = 4.0 * exponents[slot_a] * exponents[slot_a] *
            primitive_eri_target_angular(target, angular, exponents, origins);
        angular[idx_a] -= 2;
        value -= 2.0 * exponents[slot_a] * (2.0 * power_a + 1.0) *
            primitive_eri_target_angular(target, angular, exponents, origins);
        if (power_a > 1) {
            angular[idx_a] -= 2;
            value += power_a * (power_a - 1.0) *
                primitive_eri_target_angular(target, angular, exponents, origins);
        }
        return value;
    }

    angular[idx_a] += 1;
    angular[idx_b] += 1;
    double value = 4.0 * exponents[slot_a] * exponents[slot_b] *
        primitive_eri_target_angular(target, angular, exponents, origins);
    angular[idx_a] -= 1;
    angular[idx_b] -= 1;
    if (power_b > 0) {
        angular[idx_a] += 1;
        angular[idx_b] -= 1;
        value -= 2.0 * exponents[slot_a] * power_b *
            primitive_eri_target_angular(target, angular, exponents, origins);
        angular[idx_a] -= 1;
        angular[idx_b] += 1;
    }
    if (power_a > 0) {
        angular[idx_a] -= 1;
        angular[idx_b] += 1;
        value -= 2.0 * exponents[slot_b] * power_a *
            primitive_eri_target_angular(target, angular, exponents, origins);
        angular[idx_a] += 1;
        angular[idx_b] -= 1;
    }
    if (power_a > 0 && power_b > 0) {
        angular[idx_a] -= 1;
        angular[idx_b] -= 1;
        value += power_a * power_b *
            primitive_eri_target_angular(target, angular, exponents, origins);
    }
    return value;
}

inline double eri_scalar_weight_unique(
    npy_intp nao,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s,
    const double* dm_left,
    const double* dm_right
) {
    const std::array<std::array<npy_intp, 4>, 8> permutations = {{
        {{p, q, r, s}},
        {{q, p, r, s}},
        {{p, q, s, r}},
        {{q, p, s, r}},
        {{r, s, p, q}},
        {{s, r, p, q}},
        {{r, s, q, p}},
        {{s, r, q, p}},
    }};
    std::array<npy_intp, 8> seen{};
    int nseen = 0;
    double coulomb = 0.0;
    double exchange = 0.0;
    for (const auto& index : permutations) {
        const npy_intp dense = dense_index(
            nao, index[0], index[1], index[2], index[3]
        );
        bool duplicate = false;
        for (int previous = 0; previous < nseen; ++previous) {
            if (seen[previous] == dense) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        seen[nseen++] = dense;
        coulomb += dm_left[index[0] * nao + index[1]]
            * dm_right[index[2] * nao + index[3]];
        exchange += dm_left[index[0] * nao + index[2]]
            * dm_right[index[1] * nao + index[3]];
    }
    return coulomb - 0.5 * exchange;
}

bool compute_directional_shell_quartet_derivatives(
    const std::int64_t* shells,
    const double* origins,
    const double* weights,
    const std::int64_t* nprim,
    const std::int64_t* atom_ids,
    const double* directions,
    npy_intp natm,
    npy_intp nmodes,
    npy_intp nao,
    npy_intp max_prim,
    const ShellBlock& pblk,
    const ShellBlock& qblk,
    const ShellBlock& rblk,
    const ShellBlock& sblk,
    bool same_shell_pair,
    const double* pq_a,
    const double* pq_b,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    const double* pq_k,
    int npq,
    const double* rs_a,
    const double* rs_b,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    const double* rs_k,
    int nrs,
    int order,
    double* out,
    std::vector<double>& vrr_table,
    std::vector<ShellQuartetTarget>& targets,
    std::vector<double>& derivative_block,
    std::vector<double>& mode_coeffs,
    HrrExpansionCache& hrr_cache,
    std::vector<HrrFirstDerivativeRecipe>& first_recipes,
    std::vector<HrrSecondDerivativeRecipe>& second_recipes,
    const double* dm_left,
    const double* dm_right
) {
    const int max_a_l = pblk.l + qblk.l + order;
    const int max_c_l = rblk.l + sblk.l + order;
    const int max_m_l = max_a_l + max_c_l;
    const bool use_vrr =
        max_a_l <= OS_VRR_PAIR_MAX_L && max_c_l <= OS_VRR_PAIR_MAX_L;

    targets.clear();
    const std::size_t target_cap =
        static_cast<std::size_t>(pblk.stop - pblk.start) *
        static_cast<std::size_t>(qblk.stop - qblk.start) *
        static_cast<std::size_t>(rblk.stop - rblk.start) *
        static_cast<std::size_t>(sblk.stop - sblk.start);
    targets.reserve(target_cap);
    for (npy_intp ao_p = pblk.start; ao_p < pblk.stop; ++ao_p) {
        for (npy_intp ao_q = qblk.start; ao_q < qblk.stop; ++ao_q) {
            if (ao_p < ao_q) {
                continue;
            }
            const npy_intp pair_pq = pair_index(static_cast<int>(ao_p), static_cast<int>(ao_q));
            for (npy_intp ao_r = rblk.start; ao_r < rblk.stop; ++ao_r) {
                for (npy_intp ao_s = sblk.start; ao_s < sblk.stop; ++ao_s) {
                    if (ao_r < ao_s) {
                        continue;
                    }
                    const npy_intp pair_rs = pair_index(static_cast<int>(ao_r), static_cast<int>(ao_s));
                    if (same_shell_pair && pair_pq < pair_rs) {
                        continue;
                    }
                    targets.push_back({
                        static_cast<int>(shells[3 * ao_p]),
                        static_cast<int>(shells[3 * ao_p + 1]),
                        static_cast<int>(shells[3 * ao_p + 2]),
                        static_cast<int>(shells[3 * ao_q]),
                        static_cast<int>(shells[3 * ao_q + 1]),
                        static_cast<int>(shells[3 * ao_q + 2]),
                        static_cast<int>(shells[3 * ao_r]),
                        static_cast<int>(shells[3 * ao_r + 1]),
                        static_cast<int>(shells[3 * ao_r + 2]),
                        static_cast<int>(shells[3 * ao_s]),
                        static_cast<int>(shells[3 * ao_s + 1]),
                        static_cast<int>(shells[3 * ao_s + 2]),
                        ao_p,
                        ao_q,
                        ao_r,
                        ao_s,
                        ao_p * max_prim,
                        ao_q * max_prim,
                        ao_r * max_prim,
                        ao_s * max_prim,
                        pair_pair_index(pair_pq, pair_rs),
                    });
                }
            }
        }
    }
    if (targets.empty()) {
        return true;
    }

    constexpr int ncenter_derivatives = 9;
    const int nderiv = order == 1 ? ncenter_derivatives : ncenter_derivatives * ncenter_derivatives;
    derivative_block.assign(targets.size() * static_cast<std::size_t>(nderiv), 0.0);
    mode_coeffs.assign(static_cast<std::size_t>(nmodes) * ncenter_derivatives, 0.0);
    bool derivative_active[ncenter_derivatives] = {};
    const npy_intp atom_slots[4] = {
        atom_ids[pblk.start],
        atom_ids[qblk.start],
        atom_ids[rblk.start],
        atom_ids[sblk.start],
    };
    int reference_slot = 3;
    int reference_count = 0;
    int reference_hrr_cost = std::numeric_limits<int>::max();
    for (int candidate = 0; candidate < 4; ++candidate) {
        int count = 0;
        int hrr_cost = 0;
        for (int slot = 0; slot < 4; ++slot) {
            count += atom_slots[slot] == atom_slots[candidate];
            if (atom_slots[slot] != atom_slots[candidate]) {
                hrr_cost += (slot == 1 || slot == 3) ? 4 : 1;
            }
        }
        if (
            count > reference_count
            || (count == reference_count && hrr_cost < reference_hrr_cost)
        ) {
            reference_slot = candidate;
            reference_count = count;
            reference_hrr_cost = hrr_cost;
        }
    }
    int derivative_slots[3];
    int derivative_slot_count = 0;
    for (int slot = 0; slot < 4; ++slot) {
        if (slot != reference_slot) {
            derivative_slots[derivative_slot_count++] = slot;
        }
    }
    for (int derivative = 0; derivative < ncenter_derivatives; ++derivative) {
        const int slot = derivative_slots[derivative / 3];
        const int axis = derivative % 3;
        for (npy_intp mode = 0; mode < nmodes; ++mode) {
            const double coeff =
                directions[(mode * natm + atom_slots[slot]) * 3 + axis] -
                directions[(mode * natm + atom_slots[reference_slot]) * 3 + axis];
            mode_coeffs[static_cast<std::size_t>(mode) * ncenter_derivatives + derivative] = coeff;
            derivative_active[derivative] = derivative_active[derivative] || coeff != 0.0;
        }
    }

    const double AB[3] = {
        origins[3 * pblk.start] - origins[3 * qblk.start],
        origins[3 * pblk.start + 1] - origins[3 * qblk.start + 1],
        origins[3 * pblk.start + 2] - origins[3 * qblk.start + 2],
    };
    const double CD[3] = {
        origins[3 * rblk.start] - origins[3 * sblk.start],
        origins[3 * rblk.start + 1] - origins[3 * sblk.start + 1],
        origins[3 * rblk.start + 2] - origins[3 * sblk.start + 2],
    };
    hrr_cache.clear();
    first_recipes.clear();
    second_recipes.clear();
    if (use_vrr && order == 1) {
        first_recipes.resize(targets.size() * ncenter_derivatives);
        for (std::size_t it = 0; it < targets.size(); ++it) {
            for (int derivative = 0; derivative < ncenter_derivatives; ++derivative) {
                if (!derivative_active[derivative]) {
                    continue;
                }
                first_recipes[it * ncenter_derivatives + derivative] =
                    build_hrr_first_recipe(
                        hrr_cache,
                        targets[it],
                        derivative_slots[derivative / 3],
                        derivative % 3,
                        max_a_l,
                        max_c_l,
                        max_m_l,
                        AB,
                        CD
                    );
            }
        }
    } else if (use_vrr) {
        second_recipes.resize(
            targets.size() * ncenter_derivatives * ncenter_derivatives
        );
        for (std::size_t it = 0; it < targets.size(); ++it) {
            for (int derivative_a = 0; derivative_a < ncenter_derivatives; ++derivative_a) {
                if (!derivative_active[derivative_a]) {
                    continue;
                }
                for (int derivative_b = 0; derivative_b <= derivative_a; ++derivative_b) {
                    if (!derivative_active[derivative_b]) {
                        continue;
                    }
                    second_recipes[
                        (it * ncenter_derivatives + derivative_a)
                            * ncenter_derivatives + derivative_b
                    ] = build_hrr_second_recipe(
                        hrr_cache,
                        targets[it],
                        derivative_slots[derivative_a / 3],
                        derivative_a % 3,
                        derivative_slots[derivative_b / 3],
                        derivative_b % 3,
                        max_a_l,
                        max_c_l,
                        max_m_l,
                        AB,
                        CD
                    );
                }
            }
        }
    }
    const int nprim_q = static_cast<int>(nprim[qblk.start]);
    const int nprim_s = static_cast<int>(nprim[sblk.start]);
    for (int idx_pq = 0; idx_pq < npq; ++idx_pq) {
        const int ip = idx_pq / nprim_q;
        const int iq = idx_pq - ip * nprim_q;
        for (int idx_rs = 0; idx_rs < nrs; ++idx_rs) {
            const int ir = idx_rs / nprim_s;
            const int is = idx_rs - ir * nprim_s;
            const double zeta = pq_p[idx_pq] + rs_p[idx_rs];
            const double alpha = pq_p[idx_pq] * rs_p[idx_rs] / zeta;
            const double pqx = pq_px[idx_pq] - rs_px[idx_rs];
            const double pqy = pq_py[idx_pq] - rs_py[idx_rs];
            const double pqz = pq_pz[idx_pq] - rs_pz[idx_rs];
            const double base_pref =
                ERI_PREFAC * pq_k[idx_pq] * rs_k[idx_rs] /
                (pq_p[idx_pq] * rs_p[idx_rs] * std::sqrt(zeta));
            const double PA[3] = {
                pq_px[idx_pq] - origins[3 * pblk.start],
                pq_py[idx_pq] - origins[3 * pblk.start + 1],
                pq_pz[idx_pq] - origins[3 * pblk.start + 2],
            };
            const double QC[3] = {
                rs_px[idx_rs] - origins[3 * rblk.start],
                rs_py[idx_rs] - origins[3 * rblk.start + 1],
                rs_pz[idx_rs] - origins[3 * rblk.start + 2],
            };
            const double PQ[3] = {pqx, pqy, pqz};
            if (use_vrr) {
                os_fill_vrr_table(
                    vrr_table.data(),
                    max_a_l,
                    max_c_l,
                    max_m_l,
                    pq_p[idx_pq],
                    rs_p[idx_rs],
                    zeta,
                    alpha * (pqx * pqx + pqy * pqy + pqz * pqz),
                    base_pref,
                    PA,
                    QC,
                    PQ
                );
            }
            const double exponents[4] = {
                pq_a[idx_pq], pq_b[idx_pq], rs_a[idx_rs], rs_b[idx_rs]
            };

            for (std::size_t it = 0; it < targets.size(); ++it) {
                const ShellQuartetTarget& target = targets[it];
                const double prefac =
                    weights[target.weight_p + ip] *
                    weights[target.weight_q + iq] *
                    weights[target.weight_r + ir] *
                    weights[target.weight_s + is];
                if (order == 1) {
                    for (int derivative = 0; derivative < ncenter_derivatives; ++derivative) {
                        if (!derivative_active[derivative]) {
                            continue;
                        }
                        const int slot = derivative_slots[derivative / 3];
                        const int axis = derivative % 3;
                        const double derivative_value = use_vrr
                            ? evaluate_hrr_first_recipe(
                                hrr_cache,
                                first_recipes[it * ncenter_derivatives + derivative],
                                exponents[slot],
                                vrr_table.data()
                            )
                            : primitive_center_first_derivative_target(
                                target,
                                slot,
                                axis,
                                exponents,
                                origins
                            );
                        derivative_block[it * ncenter_derivatives + derivative] +=
                            prefac * derivative_value;
                    }
                } else {
                    for (int derivative_a = 0; derivative_a < ncenter_derivatives; ++derivative_a) {
                        if (!derivative_active[derivative_a]) {
                            continue;
                        }
                        const int slot_a = derivative_slots[derivative_a / 3];
                        const int axis_a = derivative_a % 3;
                        for (int derivative_b = 0; derivative_b <= derivative_a; ++derivative_b) {
                            if (!derivative_active[derivative_b]) {
                                continue;
                            }
                            const int slot_b = derivative_slots[derivative_b / 3];
                            const int axis_b = derivative_b % 3;
                            const double derivative_value = use_vrr
                                ? evaluate_hrr_second_recipe(
                                    hrr_cache,
                                    second_recipes[
                                        (it * ncenter_derivatives + derivative_a)
                                            * ncenter_derivatives + derivative_b
                                    ],
                                    exponents[slot_a],
                                    exponents[slot_b],
                                    vrr_table.data()
                                )
                                : primitive_center_second_derivative_target(
                                    target,
                                    slot_a,
                                    axis_a,
                                    slot_b,
                                    axis_b,
                                    exponents,
                                    origins
                                );
                            const double value = prefac * derivative_value;
                            derivative_block[
                                (it * ncenter_derivatives + derivative_a) * ncenter_derivatives + derivative_b
                            ] += value;
                            if (derivative_a != derivative_b) {
                                derivative_block[
                                    (it * ncenter_derivatives + derivative_b) * ncenter_derivatives + derivative_a
                                ] += value;
                            }
                        }
                    }
                }
            }
        }
    }

    const std::size_t eri_size =
        static_cast<std::size_t>(nao) * nao * nao * nao;
    for (std::size_t it = 0; it < targets.size(); ++it) {
        const ShellQuartetTarget& target = targets[it];
        const bool contract_scalar = dm_left != nullptr && dm_right != nullptr;
        const double scalar_weight = contract_scalar
            ? eri_scalar_weight_unique(
                nao,
                target.ao_p,
                target.ao_q,
                target.ao_r,
                target.ao_s,
                dm_left,
                dm_right
            )
            : 0.0;
        if (order == 1) {
            for (npy_intp mode = 0; mode < nmodes; ++mode) {
                double value = 0.0;
                for (int derivative = 0; derivative < ncenter_derivatives; ++derivative) {
                    value +=
                        mode_coeffs[static_cast<std::size_t>(mode) * ncenter_derivatives + derivative] *
                        derivative_block[it * ncenter_derivatives + derivative];
                }
                if (contract_scalar) {
                    out[mode] += scalar_weight * value;
                } else {
                    add_eri_symmetries_unique(
                        out + static_cast<std::size_t>(mode) * eri_size,
                        nao,
                        target.ao_p,
                        target.ao_q,
                        target.ao_r,
                        target.ao_s,
                        value
                    );
                }
            }
        } else {
            for (npy_intp mode_a = 0; mode_a < nmodes; ++mode_a) {
                for (npy_intp mode_b = 0; mode_b < nmodes; ++mode_b) {
                    double value = 0.0;
                    for (int derivative_a = 0; derivative_a < ncenter_derivatives; ++derivative_a) {
                        const double coeff_a = mode_coeffs[
                            static_cast<std::size_t>(mode_a) * ncenter_derivatives + derivative_a
                        ];
                        if (coeff_a == 0.0) {
                            continue;
                        }
                        for (int derivative_b = 0; derivative_b < ncenter_derivatives; ++derivative_b) {
                            value += coeff_a * mode_coeffs[
                                static_cast<std::size_t>(mode_b) * ncenter_derivatives + derivative_b
                            ] * derivative_block[
                                (it * ncenter_derivatives + derivative_a) * ncenter_derivatives + derivative_b
                            ];
                        }
                    }
                    if (contract_scalar) {
                        out[static_cast<std::size_t>(mode_a) * nmodes + mode_b] +=
                            scalar_weight * value;
                    } else {
                        add_eri_symmetries_unique(
                            out + (static_cast<std::size_t>(mode_a) * nmodes + mode_b) * eri_size,
                            nao,
                            target.ao_p,
                            target.ao_q,
                            target.ao_r,
                            target.ao_s,
                            value
                        );
                    }
                }
            }
        }
    }
    return true;
}

bool compute_directional_eri_derivatives_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const std::int64_t* atom_ids,
    const double* directions,
    npy_intp natm,
    npy_intp nmodes,
    npy_intp nao,
    npy_intp max_prim,
    int order,
    int workers,
    double* out,
    const std::vector<ShellBlock>& shell_blocks,
    const double* dm_left,
    const double* dm_right
) {
    try {
        const int nshell = static_cast<int>(shell_blocks.size());
        const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
            shell_blocks, shells, origins, exps, weights, nprim, nao, max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        std::vector<double> shell_pair_bounds(
            static_cast<std::size_t>(nshell) * nshell,
            1.0
        );
        std::vector<ShellQuartetTask> tasks;
        long long shell_screened = 0;
        build_shell_quartet_tasks(
            shell_blocks,
            shell_pair_bounds,
            nshell,
            0.0,
            tasks,
            shell_screened
        );

        const std::size_t vrr_table_cap = os_vrr_table_size(
            OS_VRR_PAIR_MAX_L,
            OS_VRR_PAIR_MAX_L,
            2 * OS_VRR_PAIR_MAX_L
        );
        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(tasks.size()))
        );
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};
        std::mutex output_mutex;
        const bool contract_scalar = dm_left != nullptr && dm_right != nullptr;
        const std::size_t scalar_size = order == 1
            ? static_cast<std::size_t>(nmodes)
            : static_cast<std::size_t>(nmodes) * nmodes;

        auto run_worker = [&]() {
            std::vector<double> vrr_table(vrr_table_cap, 0.0);
            std::vector<ShellQuartetTarget> targets;
            std::vector<double> derivative_block;
            std::vector<double> mode_coeffs;
            HrrExpansionCache hrr_cache;
            std::vector<HrrFirstDerivativeRecipe> first_recipes;
            std::vector<HrrSecondDerivativeRecipe> second_recipes;
            std::vector<double> scalar_output(
                contract_scalar ? scalar_size : 0,
                0.0
            );
            double* worker_output = contract_scalar ? scalar_output.data() : out;
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size()) {
                        break;
                    }
                    const ShellQuartetTask& task = tasks[task_index];
                    const std::size_t pq_pair_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
                    const std::size_t rs_pair_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
                    const std::size_t pq_offset = pq_pair_idx * pair_cap;
                    const std::size_t rs_offset = rs_pair_idx * pair_cap;
                    if (!compute_directional_shell_quartet_derivatives(
                            shells,
                            origins,
                            weights,
                            nprim,
                            atom_ids,
                            directions,
                            natm,
                            nmodes,
                            nao,
                            max_prim,
                            shell_blocks[task.ish],
                            shell_blocks[task.jsh],
                            shell_blocks[task.ksh],
                            shell_blocks[task.lsh],
                            task.ish == task.ksh && task.jsh == task.lsh,
                            pair_geom.a.data() + pq_offset,
                            pair_geom.b.data() + pq_offset,
                            pair_geom.p.data() + pq_offset,
                            pair_geom.px.data() + pq_offset,
                            pair_geom.py.data() + pq_offset,
                            pair_geom.pz.data() + pq_offset,
                            pair_geom.k.data() + pq_offset,
                            pair_geom.n[pq_pair_idx],
                            pair_geom.a.data() + rs_offset,
                            pair_geom.b.data() + rs_offset,
                            pair_geom.p.data() + rs_offset,
                            pair_geom.px.data() + rs_offset,
                            pair_geom.py.data() + rs_offset,
                            pair_geom.pz.data() + rs_offset,
                            pair_geom.k.data() + rs_offset,
                            pair_geom.n[rs_pair_idx],
                            order,
                            worker_output,
                            vrr_table,
                            targets,
                            derivative_block,
                            mode_coeffs,
                            hrr_cache,
                            first_recipes,
                            second_recipes,
                            dm_left,
                            dm_right
                        )) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
                if (contract_scalar && !failed.load(std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    for (std::size_t index = 0; index < scalar_size; ++index) {
                        out[index] += scalar_output[index];
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        };

        if (nthread == 1) {
            run_worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(nthread));
            try {
                for (int tid = 0; tid < nthread; ++tid) {
                    threads.emplace_back(run_worker);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        return !failed.load(std::memory_order_relaxed);
    } catch (const std::bad_alloc&) {
        return false;
    }
}

void fill_symmetric_eri(
    double* eri,
    npy_intp nao,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s,
    double value
) {
    eri[dense_index(nao, p, q, r, s)] = value;
    eri[dense_index(nao, q, p, r, s)] = value;
    eri[dense_index(nao, p, q, s, r)] = value;
    eri[dense_index(nao, q, p, s, r)] = value;
    eri[dense_index(nao, r, s, p, q)] = value;
    eri[dense_index(nao, s, r, p, q)] = value;
    eri[dense_index(nao, r, s, q, p)] = value;
    eri[dense_index(nao, s, r, q, p)] = value;
}

bool validate_dense_inputs(
    PyArrayObject* shells,
    PyArrayObject* origins,
    PyArrayObject* exps,
    PyArrayObject* weights,
    PyArrayObject* nprim,
    PyArrayObject* pair_bounds
) {
    if (
        PyArray_NDIM(shells) != 2 ||
        PyArray_NDIM(origins) != 2 ||
        PyArray_NDIM(exps) != 2 ||
        PyArray_NDIM(weights) != 2 ||
        PyArray_NDIM(nprim) != 1 ||
        PyArray_NDIM(pair_bounds) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "compute_dense_eri_ssss expects shells/origins/exps/weights/pair_bounds as 2D arrays and nprim as a 1D array.");
        return false;
    }

    const npy_intp nao = PyArray_DIM(shells, 0);
    const npy_intp max_prim = PyArray_DIM(exps, 1);
    if (
        PyArray_DIM(shells, 1) != 3 ||
        PyArray_DIM(origins, 0) != nao ||
        PyArray_DIM(origins, 1) != 3 ||
        PyArray_DIM(exps, 0) != nao ||
        PyArray_DIM(weights, 0) != nao ||
        PyArray_DIM(weights, 1) != max_prim ||
        PyArray_DIM(nprim, 0) != nao ||
        PyArray_DIM(pair_bounds, 0) != nao ||
        PyArray_DIM(pair_bounds, 1) != nao
    ) {
        PyErr_SetString(PyExc_ValueError, "compute_dense_eri_ssss received inconsistent array shapes.");
        return false;
    }

    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim));
    for (npy_intp i = 0; i < nao; ++i) {
        if (nprim_data[i] < 0 || nprim_data[i] > max_prim) {
            PyErr_SetString(PyExc_ValueError, "nprim entries must be between 0 and max_prim.");
            return false;
        }
    }
    return true;
}

bool validate_cartesian_shell_lmax(PyArrayObject* shells, int max_l) {
    const npy_intp nao = PyArray_DIM(shells, 0);
    const auto* shell_data = static_cast<const std::int64_t*>(PyArray_DATA(shells));
    for (npy_intp i = 0; i < nao; ++i) {
        const std::int64_t lx = shell_data[3 * i + 0];
        const std::int64_t ly = shell_data[3 * i + 1];
        const std::int64_t lz = shell_data[3 * i + 2];
        if (lx < 0 || ly < 0 || lz < 0) {
            PyErr_SetString(PyExc_ValueError, "Cartesian shell angular momentum entries must be nonnegative.");
            return false;
        }
        if (lx + ly + lz > max_l) {
            PyErr_SetString(PyExc_NotImplementedError, "C++ ERI backend currently supports Cartesian shells only through the requested max_l.");
            return false;
        }
    }
    return true;
}

bool validate_nprim_bounds(PyArrayObject* nprim, npy_intp ncenter, npy_intp max_prim, const char* name);
bool validate_nonnegative_shells(PyArrayObject* shells, const char* name);

bool validate_directional_eri_inputs(
    PyArrayObject* shells,
    PyArrayObject* origins,
    PyArrayObject* exps,
    PyArrayObject* weights,
    PyArrayObject* nprim,
    PyArrayObject* atom_ids,
    PyArrayObject* directions
) {
    if (
        PyArray_NDIM(shells) != 2 ||
        PyArray_NDIM(origins) != 2 ||
        PyArray_NDIM(exps) != 2 ||
        PyArray_NDIM(weights) != 2 ||
        PyArray_NDIM(nprim) != 1 ||
        PyArray_NDIM(atom_ids) != 1 ||
        PyArray_NDIM(directions) != 3
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "directional ERI derivatives received arrays with invalid dimensions."
        );
        return false;
    }

    const npy_intp nao = PyArray_DIM(shells, 0);
    const npy_intp max_prim = PyArray_DIM(exps, 1);
    const npy_intp natm = PyArray_DIM(directions, 1);
    if (
        PyArray_DIM(shells, 1) != 3 ||
        PyArray_DIM(origins, 0) != nao ||
        PyArray_DIM(origins, 1) != 3 ||
        PyArray_DIM(exps, 0) != nao ||
        PyArray_DIM(weights, 0) != nao ||
        PyArray_DIM(weights, 1) != max_prim ||
        PyArray_DIM(nprim, 0) != nao ||
        PyArray_DIM(atom_ids, 0) != nao ||
        PyArray_DIM(directions, 2) != 3
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "directional ERI derivatives received inconsistent array shapes."
        );
        return false;
    }
    if (!validate_nprim_bounds(nprim, nao, max_prim, "nprim")) {
        return false;
    }
    if (!validate_nonnegative_shells(shells, "primary")) {
        return false;
    }

    const auto* atom_data = static_cast<const std::int64_t*>(PyArray_DATA(atom_ids));
    for (npy_intp ao = 0; ao < nao; ++ao) {
        if (atom_data[ao] < 0 || atom_data[ao] >= natm) {
            PyErr_SetString(PyExc_ValueError, "atom_ids entries must index the directions atom axis.");
            return false;
        }
    }
    return true;
}

bool validate_nprim_bounds(PyArrayObject* nprim, npy_intp ncenter, npy_intp max_prim, const char* name) {
    if (PyArray_DIM(nprim, 0) != ncenter) {
        PyErr_Format(PyExc_ValueError, "%s has inconsistent length.", name);
        return false;
    }
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim));
    for (npy_intp i = 0; i < ncenter; ++i) {
        if (nprim_data[i] < 0 || nprim_data[i] > max_prim) {
            PyErr_Format(PyExc_ValueError, "%s entries must be between 0 and max_prim.", name);
            return false;
        }
    }
    return true;
}

bool validate_nonnegative_shells(PyArrayObject* shells, const char* name) {
    const npy_intp ncenter = PyArray_DIM(shells, 0);
    const auto* shell_data = static_cast<const std::int64_t*>(PyArray_DATA(shells));
    for (npy_intp i = 0; i < ncenter; ++i) {
        const std::int64_t lx = shell_data[3 * i + 0];
        const std::int64_t ly = shell_data[3 * i + 1];
        const std::int64_t lz = shell_data[3 * i + 2];
        if (lx < 0 || ly < 0 || lz < 0) {
            PyErr_Format(PyExc_ValueError, "%s Cartesian shell entries must be nonnegative.", name);
            return false;
        }
    }
    return true;
}

bool validate_ri_inputs(
    PyArrayObject* shells,
    PyArrayObject* origins,
    PyArrayObject* exps,
    PyArrayObject* weights,
    PyArrayObject* nprim,
    PyArrayObject* aux_shells,
    PyArrayObject* aux_origins,
    PyArrayObject* aux_exps,
    PyArrayObject* aux_weights,
    PyArrayObject* aux_nprim,
    PyArrayObject* pair_bounds
) {
    if (
        PyArray_NDIM(shells) != 2 ||
        PyArray_NDIM(origins) != 2 ||
        PyArray_NDIM(exps) != 2 ||
        PyArray_NDIM(weights) != 2 ||
        PyArray_NDIM(nprim) != 1 ||
        PyArray_NDIM(aux_shells) != 2 ||
        PyArray_NDIM(aux_origins) != 2 ||
        PyArray_NDIM(aux_exps) != 2 ||
        PyArray_NDIM(aux_weights) != 2 ||
        PyArray_NDIM(aux_nprim) != 1 ||
        PyArray_NDIM(pair_bounds) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "compute_ri_tensors_packed received arrays with invalid dimensions.");
        return false;
    }

    const npy_intp nao = PyArray_DIM(shells, 0);
    const npy_intp max_prim = PyArray_DIM(exps, 1);
    const npy_intp naux = PyArray_DIM(aux_shells, 0);
    const npy_intp aux_max_prim = PyArray_DIM(aux_exps, 1);
    if (
        PyArray_DIM(shells, 1) != 3 ||
        PyArray_DIM(origins, 0) != nao ||
        PyArray_DIM(origins, 1) != 3 ||
        PyArray_DIM(exps, 0) != nao ||
        PyArray_DIM(weights, 0) != nao ||
        PyArray_DIM(weights, 1) != max_prim ||
        PyArray_DIM(pair_bounds, 0) != nao ||
        PyArray_DIM(pair_bounds, 1) != nao ||
        PyArray_DIM(aux_shells, 1) != 3 ||
        PyArray_DIM(aux_origins, 0) != naux ||
        PyArray_DIM(aux_origins, 1) != 3 ||
        PyArray_DIM(aux_exps, 0) != naux ||
        PyArray_DIM(aux_weights, 0) != naux ||
        PyArray_DIM(aux_weights, 1) != aux_max_prim
    ) {
        PyErr_SetString(PyExc_ValueError, "compute_ri_tensors_packed received inconsistent array shapes.");
        return false;
    }
    return (
        validate_nprim_bounds(nprim, nao, max_prim, "nprim") &&
        validate_nprim_bounds(aux_nprim, naux, aux_max_prim, "aux_nprim") &&
        validate_nonnegative_shells(shells, "primary") &&
        validate_nonnegative_shells(aux_shells, "auxiliary")
    );
}

PyObject* compute_dense_eri_ssss(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    double screen_tol = 0.0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOd",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &screen_tol
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, 0)) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    npy_intp dims[4] = {nao, nao, nao, nao};
    PyObject* eri_obj = PyArray_ZEROS(4, dims, NPY_DOUBLE, 0);
    if (eri_obj == nullptr) {
        return nullptr;
    }

    auto* eri = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(eri_obj)));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));

    long long computed = 0;
    long long skipped = 0;

    for (npy_intp p = 0; p < nao; ++p) {
        for (npy_intp q = 0; q <= p; ++q) {
            const double bound_pq = bounds_data[p * nao + q];
            for (npy_intp r = 0; r <= p; ++r) {
                const npy_intp s_max = (r == p) ? q : r;
                for (npy_intp s = 0; s <= s_max; ++s) {
                    if (screen_tol > 0.0 && bound_pq * bounds_data[r * nao + s] < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    const double value = contracted_eri_ssss(
                        origins_data,
                        exps_data,
                        weights_data,
                        nprim_data,
                        max_prim,
                        p,
                        q,
                        r,
                        s
                    );
                    if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    fill_symmetric_eri(eri, nao, p, q, r, s, value);
                    ++computed;
                }
            }
        }
    }

    PyObject* result = Py_BuildValue("NLL", eri_obj, computed, skipped);
    return result;
}

PyObject* compute_dense_eri_cartesian(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOdi|i",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &screen_tol,
            &max_l,
            &workers
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    npy_intp dims[4] = {nao, nao, nao, nao};
    PyObject* eri_obj = PyArray_ZEROS(4, dims, NPY_DOUBLE, 0);
    if (eri_obj == nullptr) {
        return nullptr;
    }

    auto* eri = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(eri_obj)));
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));

    long long computed = 0;
    long long skipped = 0;
    if (compute_dense_eri_cartesian_blocked(
            shells_data,
            origins_data,
            exps_data,
            weights_data,
            nprim_data,
            bounds_data,
            nao,
            max_prim,
            screen_tol,
            eri,
            computed,
            skipped,
            workers
        )) {
        return Py_BuildValue("NLL", eri_obj, computed, skipped);
    }

    std::fill(eri, eri + nao * nao * nao * nao, 0.0);
    computed = 0;
    skipped = 0;
    for (npy_intp p = 0; p < nao; ++p) {
        for (npy_intp q = 0; q <= p; ++q) {
            const double bound_pq = bounds_data[p * nao + q];
            for (npy_intp r = 0; r <= p; ++r) {
                const npy_intp s_max = (r == p) ? q : r;
                for (npy_intp s = 0; s <= s_max; ++s) {
                    if (screen_tol > 0.0 && bound_pq * bounds_data[r * nao + s] < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    const double value = contracted_eri_cartesian(
                        shells_data,
                        origins_data,
                        exps_data,
                        weights_data,
                        nprim_data,
                        max_prim,
                        p,
                        q,
                        r,
                        s
                    );
                    if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    fill_symmetric_eri(eri, nao, p, q, r, s, value);
                    ++computed;
                }
            }
        }
    }

    return Py_BuildValue("NLL", eri_obj, computed, skipped);
}

PyObject* compute_dense_eri_spherical(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    PyObject* transform_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;
    int rys_max_rank = -1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOdi|ii",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &transform_obj,
            &screen_tol,
            &max_l,
            &workers,
            &rys_max_rank
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef transform(transform_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds || !transform) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }
    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (
        PyArray_NDIM(transform.obj) != 2 ||
        PyArray_DIM(transform.obj, 0) != nao ||
        PyArray_DIM(transform.obj, 1) <= 0
    ) {
        PyErr_SetString(PyExc_ValueError, "transform must have shape (nao_cart, nao_spherical).");
        return nullptr;
    }
    const npy_intp nsph = PyArray_DIM(transform.obj, 1);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    npy_intp dims[4] = {nsph, nsph, nsph, nsph};
    PyObject* eri_obj = PyArray_ZEROS(4, dims, NPY_DOUBLE, 0);
    if (eri_obj == nullptr) {
        return nullptr;
    }

    long long computed = 0;
    long long skipped = 0;
    const bool ok = compute_dense_eri_spherical_blocked(
        static_cast<const std::int64_t*>(PyArray_DATA(shells.obj)),
        static_cast<const double*>(PyArray_DATA(origins.obj)),
        static_cast<const double*>(PyArray_DATA(exps.obj)),
        static_cast<const double*>(PyArray_DATA(weights.obj)),
        static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj)),
        static_cast<const double*>(PyArray_DATA(pair_bounds.obj)),
        static_cast<const double*>(PyArray_DATA(transform.obj)),
        nao,
        nsph,
        max_prim,
        screen_tol,
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(eri_obj))),
        nullptr,
        nullptr,
        nullptr,
        computed,
        skipped,
        std::max(1, workers),
        rys_max_rank
    );
    if (!ok) {
        Py_DECREF(eri_obj);
        PyErr_SetString(PyExc_RuntimeError, "Native spherical shell-blocked ERI construction failed.");
        return nullptr;
    }
    return Py_BuildValue("NLL", eri_obj, computed, skipped);
}

PyObject* compute_shell_quartet_rys_l3(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    int p0, p1, q0, q1, r0, r1, s0, s1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOiiiiiiii",
            &shells_obj, &origins_obj, &exps_obj, &weights_obj, &nprim_obj,
            &p0, &p1, &q0, &q1, &r0, &r1, &s0, &s1
        )) return nullptr;
    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim) return nullptr;
    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    if (
        p0 < 0 || p0 >= p1 || p1 > nao ||
        q0 < 0 || q0 >= q1 || q1 > nao ||
        r0 < 0 || r0 >= r1 || r1 > nao ||
        s0 < 0 || s0 >= s1 || s1 > nao
    ) {
        PyErr_SetString(PyExc_ValueError, "Invalid shell ranges for Rys block validation.");
        return nullptr;
    }
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const int npq = static_cast<int>(nprim_data[p0] * nprim_data[q0]);
    const int nrs = static_cast<int>(nprim_data[r0] * nprim_data[s0]);
    std::vector<double> pq_a(npq), pq_b(npq), pq_p(npq), pq_px(npq), pq_py(npq),
        pq_pz(npq), pq_k(npq), pq_weight(npq);
    std::vector<double> rs_a(nrs), rs_b(nrs), rs_p(nrs), rs_px(nrs), rs_py(nrs),
        rs_pz(nrs), rs_k(nrs), rs_weight(nrs);
    precompute_primitive_pair_geom(
        p0, q0, origins_data, exps_data, nprim_data, max_prim,
        pq_a.data(), pq_b.data(), pq_p.data(), pq_px.data(), pq_py.data(),
        pq_pz.data(), pq_k.data(), weights_data, pq_weight.data()
    );
    precompute_primitive_pair_geom(
        r0, s0, origins_data, exps_data, nprim_data, max_prim,
        rs_a.data(), rs_b.data(), rs_p.data(), rs_px.data(), rs_py.data(),
        rs_pz.data(), rs_k.data(), weights_data, rs_weight.data()
    );
    std::vector<double> values;
    if (!compute_l3_shell_quartet_rys_values(
            shells_data, origins_data, weights_data, nprim_data, max_prim,
            p0, p1, q0, q1, r0, r1, s0, s1,
            pq_p.data(), pq_px.data(), pq_py.data(), pq_pz.data(), pq_k.data(), pq_weight.data(), npq,
            rs_p.data(), rs_px.data(), rs_py.data(), rs_pz.data(), rs_k.data(), rs_weight.data(), nrs,
            values
        )) {
        PyErr_SetString(PyExc_RuntimeError, "Native l<=3 Rys shell block failed.");
        return nullptr;
    }
    npy_intp dims[4] = {p1 - p0, q1 - q0, r1 - r0, s1 - s0};
    PyObject* block_obj = PyArray_SimpleNew(4, dims, NPY_DOUBLE);
    if (block_obj == nullptr) return nullptr;
    std::copy(
        values.begin(), values.end(),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(block_obj)))
    );
    return block_obj;
}

PyObject* build_spherical_direct_jk_plan(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    PyObject* transform_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int rys_max_rank = -1;
    unsigned long long rys_recurrence_cache_max_bytes =
        DEFAULT_RYS_RECURRENCE_CACHE_MAX_BYTES;
    int build_workers = 1;
    unsigned long long rys_spherical_plan_cache_max_bytes =
        DEFAULT_RYS_SPHERICAL_CACHE_MAX_BYTES;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOdiiK|iK",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &transform_obj,
            &screen_tol,
            &max_l,
            &rys_max_rank,
            &rys_recurrence_cache_max_bytes,
            &build_workers,
            &rys_spherical_plan_cache_max_bytes
        )) return nullptr;
    if (rys_recurrence_cache_max_bytes == 0) {
        PyErr_SetString(PyExc_ValueError, "Rys recurrence cache budget must be positive.");
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef transform(transform_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds || !transform) {
        return nullptr;
    }
    if (!validate_dense_inputs(
            shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj
        )) return nullptr;
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) return nullptr;
    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (
        PyArray_NDIM(transform.obj) != 2 ||
        PyArray_DIM(transform.obj, 0) != nao ||
        PyArray_DIM(transform.obj, 1) <= 0
    ) {
        PyErr_SetString(PyExc_ValueError, "transform must have shape (nao_cart, nao_spherical).");
        return nullptr;
    }

    auto* plan = new (std::nothrow) SphericalDirectJKPlan();
    if (plan == nullptr) return PyErr_NoMemory();
    plan->arrays = {
        shells.obj, origins.obj, exps.obj, weights.obj,
        nprim.obj, pair_bounds.obj, transform.obj
    };
    for (PyArrayObject* array : plan->arrays) Py_INCREF(array);
    plan->shells = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    plan->origins = static_cast<const double*>(PyArray_DATA(origins.obj));
    plan->exps = static_cast<const double*>(PyArray_DATA(exps.obj));
    plan->weights = static_cast<const double*>(PyArray_DATA(weights.obj));
    plan->nprim = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    plan->pair_bounds = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));
    plan->transform = static_cast<const double*>(PyArray_DATA(transform.obj));
    plan->nao = nao;
    plan->nsph = PyArray_DIM(transform.obj, 1);
    plan->max_prim = PyArray_DIM(exps.obj, 1);
    plan->max_l = max_l;
    plan->rys_max_rank = rys_max_rank;
    plan->build_workers = std::max(1, build_workers);
    plan->rys_recurrence_cache_max_bytes = static_cast<std::size_t>(
        std::min<unsigned long long>(
            rys_recurrence_cache_max_bytes,
            std::numeric_limits<std::size_t>::max()
        )
    );
    plan->rys_spherical_plan_cache_max_bytes = static_cast<std::size_t>(
        std::min<unsigned long long>(
            rys_spherical_plan_cache_max_bytes,
            std::numeric_limits<std::size_t>::max()
        )
    );
    plan->screen_tol = screen_tol;
    try {
        if (!initialize_spherical_direct_jk_plan(*plan)) {
            release_spherical_direct_jk_plan(plan);
            PyErr_SetString(PyExc_RuntimeError, "Failed to construct native spherical direct-J/K plan.");
            return nullptr;
        }
    } catch (const std::bad_alloc&) {
        release_spherical_direct_jk_plan(plan);
        return PyErr_NoMemory();
    } catch (...) {
        release_spherical_direct_jk_plan(plan);
        PyErr_SetString(PyExc_RuntimeError, "Failed to construct native spherical direct-J/K plan.");
        return nullptr;
    }
    return PyCapsule_New(
        plan,
        "pyqed.SphericalDirectJKPlan",
        destroy_spherical_direct_jk_plan
    );
}

PyObject* spherical_direct_jk_plan_stats(PyObject*, PyObject* args) {
    PyObject* plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) return nullptr;
    auto* plan = static_cast<SphericalDirectJKPlan*>(PyCapsule_GetPointer(
        plan_obj, "pyqed.SphericalDirectJKPlan"
    ));
    if (plan == nullptr) return nullptr;
    std::array<unsigned long long, 13> rank_tasks{};
    std::array<unsigned long long, 13> rank_recurrence_cached{};
    std::array<unsigned long long, 13> rank_spherical_planned{};
    unsigned long long primitive_pair_terms = 0;
    unsigned long long primitive_pair_terms_unpruned = 0;
    unsigned long long contraction_batches = 0;
    unsigned long long batched_tasks = 0;
    unsigned long long ssss_contraction_batches = 0;
    unsigned long long ssss_batched_tasks = 0;
    const std::size_t nshell = plan->shell_blocks.size();
    for (std::size_t ish = 0; ish < nshell; ++ish) {
        const npy_intp p = plan->shell_blocks[ish].start;
        for (std::size_t jsh = 0; jsh < nshell; ++jsh) {
            const npy_intp q = plan->shell_blocks[jsh].start;
            primitive_pair_terms += static_cast<unsigned long long>(
                plan->pair_geom.n[ish * nshell + jsh]
            );
            primitive_pair_terms_unpruned +=
                static_cast<unsigned long long>(plan->nprim[p]) *
                static_cast<unsigned long long>(plan->nprim[q]);
        }
    }
    for (const ShellQuartetTask& task : plan->tasks) {
        const int rank = plan->shell_blocks[task.ish].l +
            plan->shell_blocks[task.jsh].l + plan->shell_blocks[task.ksh].l +
            plan->shell_blocks[task.lsh].l;
        if (task.contraction_batch_size > 1) {
            ++contraction_batches;
            batched_tasks += task.contraction_batch_size;
            if (rank == 0) {
                ++ssss_contraction_batches;
                ssss_batched_tasks += task.contraction_batch_size;
            }
        }
        if (rank < 0 || rank >= static_cast<int>(rank_tasks.size())) continue;
        ++rank_tasks[static_cast<std::size_t>(rank)];
        if (plan->rys_recurrence_cache_slots[task.cache_index] >= 0) {
            ++rank_recurrence_cached[static_cast<std::size_t>(rank)];
        }
        if (plan->rys_spherical_plan_slots[task.cache_index] >= 0) {
            ++rank_spherical_planned[static_cast<std::size_t>(rank)];
        }
    }
    auto make_rank_list = [](const auto& counts) {
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(counts.size()));
        if (list == nullptr) return static_cast<PyObject*>(nullptr);
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            PyList_SET_ITEM(
                list, static_cast<Py_ssize_t>(rank),
                PyLong_FromUnsignedLongLong(counts[rank])
            );
        }
        return list;
    };
    PyObject* task_list = make_rank_list(rank_tasks);
    PyObject* recurrence_list = make_rank_list(rank_recurrence_cached);
    PyObject* spherical_list = make_rank_list(rank_spherical_planned);
    if (task_list == nullptr || recurrence_list == nullptr || spherical_list == nullptr) {
        Py_XDECREF(task_list);
        Py_XDECREF(recurrence_list);
        Py_XDECREF(spherical_list);
        return nullptr;
    }
    PyObject* result = Py_BuildValue(
        "{sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sK,sN,sN,sN}",
        "tasks", static_cast<unsigned long long>(plan->tasks.size()),
        "task_entry_bytes", static_cast<unsigned long long>(sizeof(ShellQuartetTask)),
        "task_bytes", static_cast<unsigned long long>(
            plan->tasks.size() * sizeof(ShellQuartetTask)
        ),
        "recurrence_cache_max_bytes", static_cast<unsigned long long>(plan->rys_recurrence_cache_max_bytes),
        "recurrence_cache_bytes", static_cast<unsigned long long>(plan->rys_recurrence_cache_bytes),
        "spherical_plan_cache_bytes", static_cast<unsigned long long>(plan->rys_spherical_plan_cache_bytes),
        "recurrence_caches", static_cast<unsigned long long>(plan->rys_recurrence_caches.size()),
        "spherical_plans", static_cast<unsigned long long>(plan->rys_spherical_plans.size()),
        "primitive_pair_terms", primitive_pair_terms,
        "primitive_pair_terms_unpruned", primitive_pair_terms_unpruned,
        "contraction_batches", contraction_batches,
        "batched_tasks", batched_tasks,
        "ssss_contraction_batches", ssss_contraction_batches,
        "ssss_batched_tasks", ssss_batched_tasks,
        "rank_tasks", task_list,
        "rank_recurrence_cached", recurrence_list,
        "rank_spherical_planned", spherical_list
    );
    if (result == nullptr) return nullptr;
    PyObject* execution_tasks = PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(plan->batched_execution_task_count)
    );
    if (execution_tasks == nullptr) {
        Py_DECREF(result);
        return nullptr;
    }
    const int execution_status = PyDict_SetItemString(
        result, "execution_tasks", execution_tasks
    );
    Py_DECREF(execution_tasks);
    if (execution_status < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    PyObject* coulomb_index_bytes = PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(
            plan->coulomb_task_offsets.size() * sizeof(std::size_t) +
            plan->coulomb_task_indices.size() * sizeof(std::uint32_t)
        )
    );
    if (coulomb_index_bytes == nullptr) {
        Py_DECREF(result);
        return nullptr;
    }
    const int coulomb_index_status = PyDict_SetItemString(
        result, "coulomb_task_index_bytes", coulomb_index_bytes
    );
    Py_DECREF(coulomb_index_bytes);
    if (coulomb_index_status < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    PyObject* spherical_cache_max = PyLong_FromUnsignedLongLong(
        static_cast<unsigned long long>(plan->rys_spherical_plan_cache_max_bytes)
    );
    if (spherical_cache_max == nullptr) {
        Py_DECREF(result);
        return nullptr;
    }
    const int spherical_cache_status = PyDict_SetItemString(
        result, "spherical_plan_cache_max_bytes", spherical_cache_max
    );
    Py_DECREF(spherical_cache_max);
    if (spherical_cache_status < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    auto add_timing = [&](const char* name, double value) {
        PyObject* timing = PyFloat_FromDouble(value);
        if (timing == nullptr) return false;
        const int status = PyDict_SetItemString(result, name, timing);
        Py_DECREF(timing);
        return status == 0;
    };
    if (
        !add_timing("geometry_build_seconds", plan->geometry_build_seconds) ||
        !add_timing("task_build_seconds", plan->task_build_seconds) ||
        !add_timing("recurrence_select_seconds", plan->recurrence_select_seconds) ||
        !add_timing("recurrence_prepare_seconds", plan->recurrence_prepare_seconds) ||
        !add_timing("spherical_plan_seconds", plan->spherical_plan_seconds) ||
        !add_timing("total_build_seconds", plan->total_build_seconds)
    ) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}

PyObject* spherical_direct_jk_plan_diagonal(PyObject*, PyObject* args) {
    PyObject* plan_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &plan_obj)) return nullptr;
    auto* plan = static_cast<SphericalDirectJKPlan*>(PyCapsule_GetPointer(
        plan_obj, "pyqed.SphericalDirectJKPlan"
    ));
    if (plan == nullptr) return nullptr;
    npy_intp dims[1] = {
        static_cast<npy_intp>(plan->spherical_pair_diagonal.size())
    };
    PyObject* diagonal_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (diagonal_obj == nullptr) return nullptr;
    double* diagonal = static_cast<double*>(PyArray_DATA(
        reinterpret_cast<PyArrayObject*>(diagonal_obj)
    ));
    std::copy(
        plan->spherical_pair_diagonal.begin(),
        plan->spherical_pair_diagonal.end(),
        diagonal
    );
    return diagonal_obj;
}

PyObject* direct_jk_spherical_plan(PyObject*, PyObject* args) {
    PyObject* plan_obj = nullptr;
    PyObject* dm_obj = nullptr;
    double screen_tol = 0.0;
    int workers = 1;
    int rys_max_rank = -1;
    int symmetric_density_hint = -1;
    if (!PyArg_ParseTuple(
            args,
            "OOdii|i",
            &plan_obj,
            &dm_obj,
            &screen_tol,
            &workers,
            &rys_max_rank,
            &symmetric_density_hint
        )) return nullptr;
    auto* plan = static_cast<SphericalDirectJKPlan*>(
        PyCapsule_GetPointer(plan_obj, "pyqed.SphericalDirectJKPlan")
    );
    if (plan == nullptr) return nullptr;
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!dm) return nullptr;
    if (
        PyArray_NDIM(dm.obj) != 2 ||
        PyArray_DIM(dm.obj, 0) != plan->nsph ||
        PyArray_DIM(dm.obj, 1) != plan->nsph
    ) {
        PyErr_SetString(PyExc_ValueError, "direct_jk_spherical_plan received an inconsistent density shape.");
        return nullptr;
    }
    npy_intp dims[2] = {plan->nsph, plan->nsph};
    PyObject* vj_obj = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) return nullptr;
    PyObject* vk_obj = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (vk_obj == nullptr) {
        Py_DECREF(vj_obj);
        return nullptr;
    }
    long long computed = 0;
    long long skipped = 0;
    const bool ok = compute_dense_eri_spherical_blocked(
        plan->shells,
        plan->origins,
        plan->exps,
        plan->weights,
        plan->nprim,
        plan->pair_bounds,
        plan->transform,
        plan->nao,
        plan->nsph,
        plan->max_prim,
        screen_tol,
        nullptr,
        static_cast<const double*>(PyArray_DATA(dm.obj)),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj))),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vk_obj))),
        computed,
        skipped,
        std::max(1, workers),
        rys_max_rank,
        plan,
        symmetric_density_hint
    );
    if (!ok) {
        Py_DECREF(vj_obj);
        Py_DECREF(vk_obj);
        PyErr_SetString(PyExc_RuntimeError, "Native planned spherical J/K construction failed.");
        return nullptr;
    }
    return Py_BuildValue("NNLL", vj_obj, vk_obj, computed, skipped);
}

PyObject* direct_j_spherical_plan(PyObject*, PyObject* args) {
    PyObject* plan_obj = nullptr;
    PyObject* dm_obj = nullptr;
    double screen_tol = 0.0;
    int workers = 1;
    int rys_max_rank = -1;
    int symmetric_density_hint = -1;
    if (!PyArg_ParseTuple(
            args,
            "OOdii|i",
            &plan_obj,
            &dm_obj,
            &screen_tol,
            &workers,
            &rys_max_rank,
            &symmetric_density_hint
        )) return nullptr;
    auto* plan = static_cast<SphericalDirectJKPlan*>(
        PyCapsule_GetPointer(plan_obj, "pyqed.SphericalDirectJKPlan")
    );
    if (plan == nullptr) return nullptr;
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!dm) return nullptr;
    if (
        PyArray_NDIM(dm.obj) != 2 ||
        PyArray_DIM(dm.obj, 0) != plan->nsph ||
        PyArray_DIM(dm.obj, 1) != plan->nsph
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "direct_j_spherical_plan received an inconsistent density shape."
        );
        return nullptr;
    }
    npy_intp dims[2] = {plan->nsph, plan->nsph};
    PyObject* vj_obj = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) return nullptr;
    long long computed = 0;
    long long skipped = 0;
    const bool ok = compute_dense_eri_spherical_blocked(
        plan->shells,
        plan->origins,
        plan->exps,
        plan->weights,
        plan->nprim,
        plan->pair_bounds,
        plan->transform,
        plan->nao,
        plan->nsph,
        plan->max_prim,
        screen_tol,
        nullptr,
        static_cast<const double*>(PyArray_DATA(dm.obj)),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj))),
        nullptr,
        computed,
        skipped,
        std::max(1, workers),
        rys_max_rank,
        plan,
        symmetric_density_hint
    );
    if (!ok) {
        Py_DECREF(vj_obj);
        PyErr_SetString(
            PyExc_RuntimeError,
            "Native planned spherical J construction failed."
        );
        return nullptr;
    }
    return Py_BuildValue("NLL", vj_obj, computed, skipped);
}

PyObject* direct_jk_spherical(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    PyObject* transform_obj = nullptr;
    PyObject* dm_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;
    int rys_max_rank = -1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOdi|ii",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &transform_obj,
            &dm_obj,
            &screen_tol,
            &max_l,
            &workers,
            &rys_max_rank
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef transform(transform_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds || !transform || !dm) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }
    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (
        PyArray_NDIM(transform.obj) != 2 ||
        PyArray_DIM(transform.obj, 0) != nao ||
        PyArray_DIM(transform.obj, 1) <= 0
    ) {
        PyErr_SetString(PyExc_ValueError, "transform must have shape (nao_cart, nao_spherical).");
        return nullptr;
    }
    const npy_intp nsph = PyArray_DIM(transform.obj, 1);
    if (
        PyArray_NDIM(dm.obj) != 2 ||
        PyArray_DIM(dm.obj, 0) != nsph ||
        PyArray_DIM(dm.obj, 1) != nsph
    ) {
        PyErr_SetString(PyExc_ValueError, "direct_jk_spherical received an inconsistent density shape.");
        return nullptr;
    }

    npy_intp dims[2] = {nsph, nsph};
    PyObject* vj_obj = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) return nullptr;
    PyObject* vk_obj = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    if (vk_obj == nullptr) {
        Py_DECREF(vj_obj);
        return nullptr;
    }
    long long computed = 0;
    long long skipped = 0;
    const bool ok = compute_dense_eri_spherical_blocked(
        static_cast<const std::int64_t*>(PyArray_DATA(shells.obj)),
        static_cast<const double*>(PyArray_DATA(origins.obj)),
        static_cast<const double*>(PyArray_DATA(exps.obj)),
        static_cast<const double*>(PyArray_DATA(weights.obj)),
        static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj)),
        static_cast<const double*>(PyArray_DATA(pair_bounds.obj)),
        static_cast<const double*>(PyArray_DATA(transform.obj)),
        nao,
        nsph,
        PyArray_DIM(exps.obj, 1),
        screen_tol,
        nullptr,
        static_cast<const double*>(PyArray_DATA(dm.obj)),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj))),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vk_obj))),
        computed,
        skipped,
        std::max(1, workers),
        rys_max_rank
    );
    if (!ok) {
        Py_DECREF(vj_obj);
        Py_DECREF(vk_obj);
        PyErr_SetString(PyExc_RuntimeError, "Native direct spherical J/K construction failed.");
        return nullptr;
    }
    return Py_BuildValue("NNLL", vj_obj, vk_obj, computed, skipped);
}

PyObject* compute_eri_s8_cartesian(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOdi|i",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &screen_tol,
            &max_l,
            &workers
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp ns8 = npair * (npair + 1) / 2;
    npy_intp dims[1] = {ns8};
    PyObject* eri_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (eri_obj == nullptr) {
        return nullptr;
    }

    auto* eri_s8 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(eri_obj)));
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));

    long long computed = 0;
    long long skipped = 0;
    if (compute_eri_s8_cartesian_blocked(
            shells_data,
            origins_data,
            exps_data,
            weights_data,
            nprim_data,
            bounds_data,
            nao,
            max_prim,
            screen_tol,
            eri_s8,
            computed,
            skipped,
            workers
        )) {
        return Py_BuildValue("NLL", eri_obj, computed, skipped);
    }

    std::fill(eri_s8, eri_s8 + ns8, 0.0);
    computed = 0;
    skipped = 0;
    for (npy_intp p = 0; p < nao; ++p) {
        for (npy_intp q = 0; q <= p; ++q) {
            const double bound_pq = bounds_data[p * nao + q];
            const npy_intp ij = pair_index(static_cast<int>(p), static_cast<int>(q));
            for (npy_intp r = 0; r <= p; ++r) {
                const npy_intp s_max = (r == p) ? q : r;
                for (npy_intp s = 0; s <= s_max; ++s) {
                    if (screen_tol > 0.0 && bound_pq * bounds_data[r * nao + s] < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    const double value = contracted_eri_cartesian(
                        shells_data,
                        origins_data,
                        exps_data,
                        weights_data,
                        nprim_data,
                        max_prim,
                        p,
                        q,
                        r,
                        s
                    );
                    if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    const npy_intp kl = pair_index(static_cast<int>(r), static_cast<int>(s));
                    eri_s8[pair_pair_index(ij, kl)] = value;
                    ++computed;
                }
            }
        }
    }

    return Py_BuildValue("NLL", eri_obj, computed, skipped);
}

PyObject* compute_directional_eri_derivatives(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* atom_ids_obj = nullptr;
    PyObject* directions_obj = nullptr;
    int order = 1;
    int workers = 1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOi|i",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &atom_ids_obj,
            &directions_obj,
            &order,
            &workers
        )) {
        return nullptr;
    }
    if (order != 1 && order != 2) {
        PyErr_SetString(PyExc_ValueError, "order must be 1 or 2.");
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef atom_ids(atom_ids_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef directions(directions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim || !atom_ids || !directions) {
        return nullptr;
    }
    if (!validate_directional_eri_inputs(
            shells.obj,
            origins.obj,
            exps.obj,
            weights.obj,
            nprim.obj,
            atom_ids.obj,
            directions.obj
        )) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const npy_intp nmodes = PyArray_DIM(directions.obj, 0);
    const npy_intp natm = PyArray_DIM(directions.obj, 1);
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* atom_ids_data = static_cast<const std::int64_t*>(PyArray_DATA(atom_ids.obj));
    const auto* directions_data = static_cast<const double*>(PyArray_DATA(directions.obj));

    std::vector<ShellBlock> shell_blocks;
    if (!try_build_shell_blocks(
            shells_data,
            origins_data,
            exps_data,
            nprim_data,
            nao,
            max_prim,
            shell_blocks
        )) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "C++ directional ERI derivatives require contiguous complete Cartesian shell blocks."
        );
        return nullptr;
    }
    npy_intp dims1[5] = {nmodes, nao, nao, nao, nao};
    npy_intp dims2[6] = {nmodes, nmodes, nao, nao, nao, nao};
    PyObject* out_obj = PyArray_ZEROS(
        order == 1 ? 5 : 6,
        order == 1 ? dims1 : dims2,
        NPY_DOUBLE,
        0
    );
    if (out_obj == nullptr) {
        return nullptr;
    }
    auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = compute_directional_eri_derivatives_blocked(
        shells_data,
        origins_data,
        exps_data,
        weights_data,
        nprim_data,
        atom_ids_data,
        directions_data,
        natm,
        nmodes,
        nao,
        max_prim,
        order,
        std::max(1, workers),
        out,
        shell_blocks,
        nullptr,
        nullptr
    );
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out_obj);
        PyErr_SetString(PyExc_RuntimeError, "C++ directional ERI derivative evaluation failed.");
        return nullptr;
    }
    return out_obj;
}

PyObject* compute_directional_eri_derivative_scalar(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* atom_ids_obj = nullptr;
    PyObject* directions_obj = nullptr;
    PyObject* dm_left_obj = nullptr;
    PyObject* dm_right_obj = nullptr;
    int order = 2;
    int workers = 1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOii",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &atom_ids_obj,
            &directions_obj,
            &dm_left_obj,
            &dm_right_obj,
            &order,
            &workers
        )) {
        return nullptr;
    }
    if (order != 1 && order != 2) {
        PyErr_SetString(PyExc_ValueError, "order must be 1 or 2.");
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef atom_ids(atom_ids_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef directions(directions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm_left(dm_left_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm_right(dm_right_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (
        !shells || !origins || !exps || !weights || !nprim || !atom_ids
        || !directions || !dm_left || !dm_right
    ) {
        return nullptr;
    }
    if (!validate_directional_eri_inputs(
            shells.obj,
            origins.obj,
            exps.obj,
            weights.obj,
            nprim.obj,
            atom_ids.obj,
            directions.obj
        )) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (
        PyArray_NDIM(dm_left.obj) != 2
        || PyArray_NDIM(dm_right.obj) != 2
        || PyArray_DIM(dm_left.obj, 0) != nao
        || PyArray_DIM(dm_left.obj, 1) != nao
        || PyArray_DIM(dm_right.obj, 0) != nao
        || PyArray_DIM(dm_right.obj, 1) != nao
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "density matrices must both have shape (nao_cart, nao_cart)."
        );
        return nullptr;
    }

    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const npy_intp nmodes = PyArray_DIM(directions.obj, 0);
    const npy_intp natm = PyArray_DIM(directions.obj, 1);
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* atom_ids_data = static_cast<const std::int64_t*>(PyArray_DATA(atom_ids.obj));
    const auto* directions_data = static_cast<const double*>(PyArray_DATA(directions.obj));
    const auto* dm_left_data = static_cast<const double*>(PyArray_DATA(dm_left.obj));
    const auto* dm_right_data = static_cast<const double*>(PyArray_DATA(dm_right.obj));

    std::vector<ShellBlock> shell_blocks;
    if (!try_build_shell_blocks(
            shells_data,
            origins_data,
            exps_data,
            nprim_data,
            nao,
            max_prim,
            shell_blocks
        )) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "C++ derivative contractions require contiguous complete Cartesian shell blocks."
        );
        return nullptr;
    }

    npy_intp dims1[1] = {nmodes};
    npy_intp dims2[2] = {nmodes, nmodes};
    PyObject* out_obj = PyArray_ZEROS(
        order == 1 ? 1 : 2,
        order == 1 ? dims1 : dims2,
        NPY_DOUBLE,
        0
    );
    if (out_obj == nullptr) {
        return nullptr;
    }
    auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = compute_directional_eri_derivatives_blocked(
        shells_data,
        origins_data,
        exps_data,
        weights_data,
        nprim_data,
        atom_ids_data,
        directions_data,
        natm,
        nmodes,
        nao,
        max_prim,
        order,
        std::max(1, workers),
        out,
        shell_blocks,
        dm_left_data,
        dm_right_data
    );
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out_obj);
        PyErr_SetString(PyExc_RuntimeError, "C++ derivative contraction failed.");
        return nullptr;
    }
    return out_obj;
}

PyObject* compute_directional_one_electron_derivatives(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* atom_ids_obj = nullptr;
    PyObject* atom_coords_obj = nullptr;
    PyObject* charges_obj = nullptr;
    PyObject* directions_obj = nullptr;
    int kernel = ONE_HCORE;
    int order = 1;
    int workers = 1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOiii",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &atom_ids_obj,
            &atom_coords_obj,
            &charges_obj,
            &directions_obj,
            &kernel,
            &order,
            &workers
        )) {
        return nullptr;
    }
    if (kernel < ONE_OVERLAP || kernel > ONE_HCORE) {
        PyErr_SetString(PyExc_ValueError, "unknown one-electron integral kernel.");
        return nullptr;
    }
    if (order != 1 && order != 2) {
        PyErr_SetString(PyExc_ValueError, "order must be 1 or 2.");
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef atom_ids(atom_ids_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef atom_coords(atom_coords_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef charges(charges_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef directions(directions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (
        !shells || !origins || !exps || !weights || !nprim || !atom_ids
        || !atom_coords || !charges || !directions
    ) {
        return nullptr;
    }
    if (!validate_directional_eri_inputs(
            shells.obj,
            origins.obj,
            exps.obj,
            weights.obj,
            nprim.obj,
            atom_ids.obj,
            directions.obj
        )) {
        return nullptr;
    }
    const npy_intp natm = PyArray_DIM(directions.obj, 1);
    if (
        PyArray_NDIM(atom_coords.obj) != 2
        || PyArray_DIM(atom_coords.obj, 0) != natm
        || PyArray_DIM(atom_coords.obj, 1) != 3
        || PyArray_NDIM(charges.obj) != 1
        || PyArray_DIM(charges.obj, 0) != natm
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "atom_coords and charges are inconsistent with the directions atom axis."
        );
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const npy_intp nmodes = PyArray_DIM(directions.obj, 0);
    npy_intp dims1[3] = {nmodes, nao, nao};
    npy_intp dims2[4] = {nmodes, nmodes, nao, nao};
    PyObject* out_obj = PyArray_ZEROS(
        order == 1 ? 3 : 4,
        order == 1 ? dims1 : dims2,
        NPY_DOUBLE,
        0
    );
    if (out_obj == nullptr) {
        return nullptr;
    }

    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = compute_directional_one_electron_derivatives_native(
        static_cast<const std::int64_t*>(PyArray_DATA(shells.obj)),
        static_cast<const double*>(PyArray_DATA(origins.obj)),
        static_cast<const double*>(PyArray_DATA(exps.obj)),
        static_cast<const double*>(PyArray_DATA(weights.obj)),
        static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj)),
        static_cast<const std::int64_t*>(PyArray_DATA(atom_ids.obj)),
        static_cast<const double*>(PyArray_DATA(atom_coords.obj)),
        static_cast<const double*>(PyArray_DATA(charges.obj)),
        static_cast<const double*>(PyArray_DATA(directions.obj)),
        natm,
        nmodes,
        nao,
        max_prim,
        kernel,
        order,
        std::max(1, workers),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)))
    );
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out_obj);
        PyErr_SetString(
            PyExc_RuntimeError,
            "C++ directional one-electron derivative evaluation failed."
        );
        return nullptr;
    }
    return out_obj;
}

PyObject* compute_one_index_one_electron_derivatives(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* atom_ids_obj = nullptr;
    int natm = 0;
    int kernel = ONE_OVERLAP;
    int index_slot = 1;
    int order = 1;
    int workers = 1;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOiiiii",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &atom_ids_obj,
            &natm,
            &kernel,
            &index_slot,
            &order,
            &workers
        )) {
        return nullptr;
    }
    if (natm <= 0) {
        PyErr_SetString(PyExc_ValueError, "natm must be positive.");
        return nullptr;
    }
    if (kernel != ONE_OVERLAP && kernel != ONE_KINETIC) {
        PyErr_SetString(PyExc_ValueError, "one-index derivatives support overlap or kinetic.");
        return nullptr;
    }
    if (index_slot != 0 && index_slot != 1) {
        PyErr_SetString(PyExc_ValueError, "index_slot must be 0 (bra) or 1 (ket).");
        return nullptr;
    }
    if (order != 1 && order != 2) {
        PyErr_SetString(PyExc_ValueError, "order must be 1 or 2.");
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef atom_ids(atom_ids_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!shells || !origins || !exps || !weights || !nprim || !atom_ids) {
        return nullptr;
    }
    if (
        PyArray_NDIM(shells.obj) != 2
        || PyArray_NDIM(origins.obj) != 2
        || PyArray_NDIM(exps.obj) != 2
        || PyArray_NDIM(weights.obj) != 2
        || PyArray_NDIM(nprim.obj) != 1
        || PyArray_NDIM(atom_ids.obj) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "one-index derivative arrays have invalid dimensions.");
        return nullptr;
    }
    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    if (
        PyArray_DIM(shells.obj, 1) != 3
        || PyArray_DIM(origins.obj, 0) != nao
        || PyArray_DIM(origins.obj, 1) != 3
        || PyArray_DIM(exps.obj, 0) != nao
        || PyArray_DIM(weights.obj, 0) != nao
        || PyArray_DIM(weights.obj, 1) != max_prim
        || PyArray_DIM(nprim.obj, 0) != nao
        || PyArray_DIM(atom_ids.obj, 0) != nao
    ) {
        PyErr_SetString(PyExc_ValueError, "one-index derivative arrays have inconsistent shapes.");
        return nullptr;
    }
    if (
        !validate_nprim_bounds(nprim.obj, nao, max_prim, "nprim")
        || !validate_nonnegative_shells(shells.obj, "primary")
    ) {
        return nullptr;
    }
    const auto* atom_data = static_cast<const std::int64_t*>(PyArray_DATA(atom_ids.obj));
    for (npy_intp ao = 0; ao < nao; ++ao) {
        if (atom_data[ao] < 0 || atom_data[ao] >= natm) {
            PyErr_SetString(PyExc_ValueError, "atom_ids entries must be valid atom indices.");
            return nullptr;
        }
    }

    npy_intp dims1[4] = {natm, 3, nao, nao};
    npy_intp dims2[6] = {natm, 3, natm, 3, nao, nao};
    PyObject* out_obj = PyArray_ZEROS(
        order == 1 ? 4 : 6,
        order == 1 ? dims1 : dims2,
        NPY_DOUBLE,
        0
    );
    if (out_obj == nullptr) {
        return nullptr;
    }
    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = compute_one_index_one_electron_derivatives_native(
        static_cast<const std::int64_t*>(PyArray_DATA(shells.obj)),
        static_cast<const double*>(PyArray_DATA(origins.obj)),
        static_cast<const double*>(PyArray_DATA(exps.obj)),
        static_cast<const double*>(PyArray_DATA(weights.obj)),
        static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj)),
        atom_data,
        natm,
        nao,
        max_prim,
        kernel,
        index_slot,
        order,
        std::max(1, workers),
        static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)))
    );
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out_obj);
        PyErr_SetString(PyExc_RuntimeError, "C++ one-index derivative evaluation failed.");
        return nullptr;
    }
    return out_obj;
}

inline bool direct_perm_seen(
    const npy_intp* aa,
    const npy_intp* bb,
    const npy_intp* cc,
    const npy_intp* dd,
    int n,
    npy_intp a,
    npy_intp b,
    npy_intp c,
    npy_intp d
) {
    for (int idx = 0; idx < n; ++idx) {
        if (aa[idx] == a && bb[idx] == b && cc[idx] == c && dd[idx] == d) {
            return true;
        }
    }
    return false;
}

inline void direct_jk_add_perm(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp a,
    npy_intp b,
    npy_intp c,
    npy_intp d
) {
    vj[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(b)] +=
        dm[static_cast<std::size_t>(d) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(c)] * value;
    if (vk != nullptr) {
        vk[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(d)] +=
            dm[static_cast<std::size_t>(b) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(c)] * value;
    }
}

inline void direct_jk_add_unique_permutations(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    npy_intp aa[8], bb[8], cc[8], dd[8];
    int nperm = 0;
    auto add = [&](npy_intp a, npy_intp b, npy_intp c, npy_intp d) {
        if (direct_perm_seen(aa, bb, cc, dd, nperm, a, b, c, d)) {
            return;
        }
        aa[nperm] = a;
        bb[nperm] = b;
        cc[nperm] = c;
        dd[nperm] = d;
        ++nperm;
        direct_jk_add_perm(vj, vk, dm, nao, value, a, b, c, d);
    };

    add(p, q, r, s);
    add(q, p, r, s);
    add(p, q, s, r);
    add(q, p, s, r);
    add(r, s, p, q);
    add(s, r, p, q);
    add(r, s, q, p);
    add(s, r, q, p);
}

inline bool direct_density_is_symmetric(const double* dm, npy_intp nao) {
    double max_abs = 0.0;
    double max_diff = 0.0;
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j < i; ++j) {
            const double a = dm[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
            const double b = dm[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)];
            max_abs = std::max(max_abs, std::max(std::abs(a), std::abs(b)));
            max_diff = std::max(max_diff, std::abs(a - b));
        }
    }
    return max_diff <= 1.0e-12 * std::max(1.0, max_abs);
}

inline void direct_add_k_exchange_pair_symmetric_density(
    double* vk,
    npy_intp nao,
    npy_intp i,
    npy_intp j,
    double value,
    bool duplicate_tuple
) {
    const double scale = (i == j && !duplicate_tuple) ? 2.0 : 1.0;
    vk[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] += scale * value;
    if (i != j) {
        vk[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)] += value;
    }
}

template<bool PQ_SAME, bool RS_SAME, bool SAME_PAIR>
inline void direct_jk_add_symmetric_class(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    auto dm_at = [&](npy_intp a, npy_intp b) {
        return dm[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(b)];
    };

    const double jpq = (RS_SAME ? 1.0 : 2.0) * dm_at(r, s) * value;
    vj[static_cast<std::size_t>(p) * nao + q] += jpq;
    if constexpr (!PQ_SAME) {
        vj[static_cast<std::size_t>(q) * nao + p] += jpq;
    }
    if constexpr (!SAME_PAIR) {
        const double jrs = (PQ_SAME ? 1.0 : 2.0) * dm_at(p, q) * value;
        vj[static_cast<std::size_t>(r) * nao + s] += jrs;
        if constexpr (!RS_SAME) {
            vj[static_cast<std::size_t>(s) * nao + r] += jrs;
        }
    }

    if (vk == nullptr) return;

    direct_add_k_exchange_pair_symmetric_density(
        vk,
        nao,
        p,
        s,
        dm_at(q, r) * value,
        p == s && q == r
    );
    if constexpr (!PQ_SAME) {
        direct_add_k_exchange_pair_symmetric_density(
            vk,
            nao,
            q,
            s,
            dm_at(p, r) * value,
            q == s && p == r
        );
    }
    if constexpr (!RS_SAME) {
        direct_add_k_exchange_pair_symmetric_density(
            vk,
            nao,
            p,
            r,
            dm_at(q, s) * value,
            p == r && q == s
        );
    }
    if constexpr (!PQ_SAME && !RS_SAME && !SAME_PAIR) {
        direct_add_k_exchange_pair_symmetric_density(
            vk,
            nao,
            q,
            r,
            dm_at(p, s) * value,
            q == r && p == s
        );
    }
}

inline void direct_jk_add_symmetric_class_code(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s,
    unsigned char symmetry_class
) {
    switch (symmetry_class) {
        case 7:
            direct_jk_add_symmetric_class<true, true, true>(
                vj, vk, dm, nao, value, p, q, r, s
            );
            return;
        case 4:
            direct_jk_add_symmetric_class<false, false, true>(
                vj, vk, dm, nao, value, p, q, r, s
            );
            return;
        case 3:
            direct_jk_add_symmetric_class<true, true, false>(
                vj, vk, dm, nao, value, p, q, r, s
            );
            return;
        case 1:
            direct_jk_add_symmetric_class<true, false, false>(
                vj, vk, dm, nao, value, p, q, r, s
            );
            return;
        case 2:
            direct_jk_add_symmetric_class<false, true, false>(
                vj, vk, dm, nao, value, p, q, r, s
            );
            return;
        default:
            direct_jk_add_symmetric_class<false, false, false>(
                vj, vk, dm, nao, value, p, q, r, s
            );
    }
}

inline void direct_jk_add_unique_permutations_symmetric_density(
    double* vj,
    double* vk,
    const double* dm,
    npy_intp nao,
    double value,
    npy_intp p,
    npy_intp q,
    npy_intp r,
    npy_intp s
) {
    const unsigned char symmetry_class = static_cast<unsigned char>(
        (p == q) | ((r == s) << 1) | ((p == r && q == s) << 2)
    );
    direct_jk_add_symmetric_class_code(
        vj, vk, dm, nao, value, p, q, r, s, symmetry_class
    );
}

void build_shell_density_bounds(
    const std::vector<ShellBlock>& shell_blocks,
    const double* dm,
    int nshell,
    npy_intp nao,
    std::vector<double>& shell_dm_bounds
) {
    shell_dm_bounds.assign(
        static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell),
        0.0
    );
    for (int ish = 0; ish < nshell; ++ish) {
        const ShellBlock& pblk = shell_blocks[ish];
        for (int jsh = 0; jsh < nshell; ++jsh) {
            const ShellBlock& qblk = shell_blocks[jsh];
            double bound = 0.0;
            for (npy_intp p = pblk.start; p < pblk.stop; ++p) {
                for (npy_intp q = qblk.start; q < qblk.stop; ++q) {
                    bound = std::max(
                        bound,
                        std::abs(dm[static_cast<std::size_t>(p) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(q)])
                    );
                }
            }
            shell_dm_bounds[static_cast<std::size_t>(ish) * static_cast<std::size_t>(nshell) + static_cast<std::size_t>(jsh)] = bound;
        }
    }
}

inline double direct_jk_density_bound_for_task(
    const std::vector<double>& shell_dm_bounds,
    int nshell,
    int ish,
    int jsh,
    int ksh,
    int lsh
) {
    auto bound = [&](int a, int b) {
        return shell_dm_bounds[static_cast<std::size_t>(a) * static_cast<std::size_t>(nshell) + static_cast<std::size_t>(b)];
    };
    double dm_bound = 0.0;
    dm_bound = std::max(dm_bound, bound(ish, jsh));
    dm_bound = std::max(dm_bound, bound(jsh, ish));
    dm_bound = std::max(dm_bound, bound(ksh, lsh));
    dm_bound = std::max(dm_bound, bound(lsh, ksh));
    dm_bound = std::max(dm_bound, bound(ish, ksh));
    dm_bound = std::max(dm_bound, bound(ksh, ish));
    dm_bound = std::max(dm_bound, bound(ish, lsh));
    dm_bound = std::max(dm_bound, bound(lsh, ish));
    dm_bound = std::max(dm_bound, bound(jsh, ksh));
    dm_bound = std::max(dm_bound, bound(ksh, jsh));
    dm_bound = std::max(dm_bound, bound(jsh, lsh));
    dm_bound = std::max(dm_bound, bound(lsh, jsh));
    return dm_bound;
}

void contract_direct_targets_to_jk(
    const std::vector<ShellQuartetTarget>& targets,
    const std::vector<double>& values,
    const double* dm,
    npy_intp nao,
    double screen_tol,
    bool symmetric_density,
    double* vj,
    double* vk,
    long long& computed,
    long long& skipped
) {
    for (std::size_t it = 0; it < targets.size(); ++it) {
        const double value = values[it];
        if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
            ++skipped;
            continue;
        }
        const ShellQuartetTarget& target = targets[it];
        if (symmetric_density) {
            direct_jk_add_unique_permutations_symmetric_density(
                vj,
                vk,
                dm,
                nao,
                value,
                target.ao_p,
                target.ao_q,
                target.ao_r,
                target.ao_s
            );
        } else {
            direct_jk_add_unique_permutations(
                vj,
                vk,
                dm,
                nao,
                value,
                target.ao_p,
                target.ao_q,
                target.ao_r,
                target.ao_s
            );
        }
        ++computed;
    }
}

void contract_direct_shell_quartet_scalar_to_jk(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* pair_bounds,
    npy_intp nao,
    npy_intp max_prim,
    double screen_tol,
    npy_intp p0,
    npy_intp p1,
    npy_intp q0,
    npy_intp q1,
    npy_intp r0,
    npy_intp r1,
    npy_intp s0,
    npy_intp s1,
    const double* dm,
    bool symmetric_density,
    double* vj,
    double* vk,
    long long& computed,
    long long& skipped
) {
    for (npy_intp p = p0; p < p1; ++p) {
        for (npy_intp q = q0; q < q1; ++q) {
            if (p < q) {
                continue;
            }
            const npy_intp ij = pair_index(static_cast<int>(p), static_cast<int>(q));
            const double bound_pq = pair_bounds[p * nao + q];
            for (npy_intp r = r0; r < r1; ++r) {
                for (npy_intp s = s0; s < s1; ++s) {
                    if (r < s) {
                        continue;
                    }
                    const npy_intp kl = pair_index(static_cast<int>(r), static_cast<int>(s));
                    if (ij < kl) {
                        continue;
                    }
                    if (screen_tol > 0.0 && bound_pq * pair_bounds[r * nao + s] < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    const double value = contracted_eri_cartesian(
                        shells,
                        origins,
                        exps,
                        weights,
                        nprim,
                        max_prim,
                        p,
                        q,
                        r,
                        s
                    );
                    if (screen_tol > 0.0 && std::abs(value) < screen_tol) {
                        ++skipped;
                        continue;
                    }
                    if (symmetric_density) {
                        direct_jk_add_unique_permutations_symmetric_density(vj, vk, dm, nao, value, p, q, r, s);
                    } else {
                        direct_jk_add_unique_permutations(vj, vk, dm, nao, value, p, q, r, s);
                    }
                    ++computed;
                }
            }
        }
    }
}

bool compute_direct_jk_cartesian_blocked(
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const double* weights,
    const std::int64_t* nprim,
    const double* pair_bounds,
    const double* dm,
    npy_intp nao,
    npy_intp max_prim,
    double screen_tol,
    double* vj,
    double* vk,
    long long& computed,
    long long& skipped,
    int workers
) {
    std::vector<ShellBlock> shell_blocks;
    if (!try_build_shell_blocks(shells, origins, exps, nprim, nao, max_prim, shell_blocks)) {
        return false;
    }

    const int nshell = static_cast<int>(shell_blocks.size());
    try {
        const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
            shell_blocks,
            shells,
            origins,
            exps,
            weights,
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        const std::size_t vrr_table_cap = os_vrr_table_size(
            OS_VRR_PAIR_MAX_L,
            OS_VRR_PAIR_MAX_L,
            2 * OS_VRR_PAIR_MAX_L
        );
        for (int ish = 0; ish < nshell; ++ish) {
            const ShellBlock& pblk = shell_blocks[ish];
            for (int jsh = 0; jsh < nshell; ++jsh) {
                const ShellBlock& qblk = shell_blocks[jsh];
                double bound = 0.0;
                for (npy_intp ip = pblk.start; ip < pblk.stop; ++ip) {
                    for (npy_intp iq = qblk.start; iq < qblk.stop; ++iq) {
                        bound = std::max(bound, pair_bounds[ip * nao + iq]);
                    }
                }
                shell_pair_bounds[static_cast<std::size_t>(ish) * nshell + jsh] = bound;
            }
        }
        std::vector<double> shell_dm_bounds;
        build_shell_density_bounds(shell_blocks, dm, nshell, nao, shell_dm_bounds);
        const double global_dm_bound = shell_dm_bounds.empty()
            ? 0.0
            : *std::max_element(shell_dm_bounds.begin(), shell_dm_bounds.end());
        const bool use_target_density_screen = screen_tol > 0.0 && global_dm_bound < 1.0e-6;
        const bool symmetric_density = direct_density_is_symmetric(dm, nao);

        std::vector<ShellQuartetTask> tasks;
        long long shell_screened = 0;
        build_shell_quartet_tasks(
            shell_blocks,
            shell_pair_bounds,
            nshell,
            screen_tol,
            tasks,
            shell_screened
        );

        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(tasks.size()))
        );
        const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
        std::vector<double> local_vj(static_cast<std::size_t>(nthread) * n2, 0.0);
        std::vector<double> local_vk(static_cast<std::size_t>(nthread) * n2, 0.0);
        std::vector<long long> thread_computed(static_cast<std::size_t>(nthread), 0);
        std::vector<long long> thread_skipped(static_cast<std::size_t>(nthread), 0);
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};

        auto run_worker = [&](int tid) {
            long long local_computed = 0;
            long long local_skipped = 0;
            double* vj_local = local_vj.data() + static_cast<std::size_t>(tid) * n2;
            double* vk_local = local_vk.data() + static_cast<std::size_t>(tid) * n2;
            std::vector<double> vrr_table(vrr_table_cap, 0.0);
            std::vector<ShellQuartetTarget> targets;
            std::vector<double> target_values;
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size()) {
                        break;
                    }
                    const ShellQuartetTask& task = tasks[task_index];
                    const ShellBlock& pblk = shell_blocks[task.ish];
                    const ShellBlock& qblk = shell_blocks[task.jsh];
                    const ShellBlock& rblk = shell_blocks[task.ksh];
                    const ShellBlock& sblk = shell_blocks[task.lsh];
                    const std::size_t pq_pair_idx = static_cast<std::size_t>(task.ish) * nshell + task.jsh;
                    const std::size_t rs_pair_idx = static_cast<std::size_t>(task.ksh) * nshell + task.lsh;
                    if (screen_tol > 0.0) {
                        const double integral_bound =
                            shell_pair_bounds[pq_pair_idx] * shell_pair_bounds[rs_pair_idx];
                        const double dm_bound = direct_jk_density_bound_for_task(
                            shell_dm_bounds,
                            nshell,
                            task.ish,
                            task.jsh,
                            task.ksh,
                            task.lsh
                        );
                        if (integral_bound * dm_bound < screen_tol) {
                            local_skipped += task.shell_count;
                            continue;
                        }
                    }
                    const std::size_t pq_offset = pq_pair_idx * pair_cap;
                    const std::size_t rs_offset = rs_pair_idx * pair_cap;
                    long long target_screened = 0;
                    const bool used_shell_block = compute_shell_quartet_vrr_hrr_target_values(
                        shells,
                        origins,
                        weights,
                        nprim,
                        max_prim,
                        nao,
                        screen_tol,
                        pair_bounds,
                        pblk.start,
                        pblk.stop,
                        qblk.start,
                        qblk.stop,
                        rblk.start,
                        rblk.stop,
                        sblk.start,
                        sblk.stop,
                        pair_geom.a.data() + pq_offset,
                        pair_geom.b.data() + pq_offset,
                        pair_geom.p.data() + pq_offset,
                        pair_geom.px.data() + pq_offset,
                        pair_geom.py.data() + pq_offset,
                        pair_geom.pz.data() + pq_offset,
                        pair_geom.k.data() + pq_offset,
                        pair_geom.n[pq_pair_idx],
                        pair_geom.a.data() + rs_offset,
                        pair_geom.b.data() + rs_offset,
                        pair_geom.p.data() + rs_offset,
                        pair_geom.px.data() + rs_offset,
                        pair_geom.py.data() + rs_offset,
                        pair_geom.pz.data() + rs_offset,
                        pair_geom.k.data() + rs_offset,
                        pair_geom.n[rs_pair_idx],
                        vrr_table.data(),
                        vrr_table_cap,
                        targets,
                        target_values,
                        target_screened,
                        use_target_density_screen ? dm : nullptr,
                        screen_tol
                    );
                    if (used_shell_block) {
                        local_skipped += target_screened;
                        contract_direct_targets_to_jk(
                            targets,
                            target_values,
                            dm,
                            nao,
                            screen_tol,
                            symmetric_density,
                            vj_local,
                            vk_local,
                            local_computed,
                            local_skipped
                        );
                    } else {
                        contract_direct_shell_quartet_scalar_to_jk(
                            shells,
                            origins,
                            exps,
                            weights,
                            nprim,
                            pair_bounds,
                            nao,
                            max_prim,
                            screen_tol,
                            pblk.start,
                            pblk.stop,
                            qblk.start,
                            qblk.stop,
                            rblk.start,
                            rblk.stop,
                            sblk.start,
                            sblk.stop,
                            dm,
                            symmetric_density,
                            vj_local,
                            vk_local,
                            local_computed,
                            local_skipped
                        );
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            thread_computed[static_cast<std::size_t>(tid)] = local_computed;
            thread_skipped[static_cast<std::size_t>(tid)] = local_skipped;
        };

        if (nthread <= 1) {
            run_worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(nthread));
            try {
                for (int tid = 0; tid < nthread; ++tid) {
                    threads.emplace_back(run_worker, tid);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (std::thread& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }
        if (failed.load(std::memory_order_relaxed)) {
            return false;
        }

        computed = 0;
        skipped = shell_screened;
        std::fill(vj, vj + n2, 0.0);
        std::fill(vk, vk + n2, 0.0);
        for (int tid = 0; tid < nthread; ++tid) {
            computed += thread_computed[static_cast<std::size_t>(tid)];
            skipped += thread_skipped[static_cast<std::size_t>(tid)];
            const double* vj_local = local_vj.data() + static_cast<std::size_t>(tid) * n2;
            const double* vk_local = local_vk.data() + static_cast<std::size_t>(tid) * n2;
            for (std::size_t idx = 0; idx < n2; ++idx) {
                vj[idx] += vj_local[idx];
                vk[idx] += vk_local[idx];
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

PyObject* direct_jk_cartesian(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    PyObject* dm_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOdi|i",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &dm_obj,
            &screen_tol,
            &max_l,
            &workers
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds || !dm) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }
    if (PyArray_NDIM(dm.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "direct_jk_cartesian expects dm as a 2D array.");
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (PyArray_DIM(dm.obj, 0) != nao || PyArray_DIM(dm.obj, 1) != nao) {
        PyErr_SetString(PyExc_ValueError, "direct_jk_cartesian received a density shape inconsistent with the AO basis.");
        return nullptr;
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* vj_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) {
        return nullptr;
    }
    PyObject* vk_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vk_obj == nullptr) {
        Py_DECREF(vj_obj);
        return nullptr;
    }

    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));
    const auto* dm_data = static_cast<const double*>(PyArray_DATA(dm.obj));
    auto* vj = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj)));
    auto* vk = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vk_obj)));

    long long computed = 0;
    long long skipped = 0;
    if (!compute_direct_jk_cartesian_blocked(
            shells_data,
            origins_data,
            exps_data,
            weights_data,
            nprim_data,
            bounds_data,
            dm_data,
            nao,
            max_prim,
            screen_tol,
            vj,
            vk,
            computed,
            skipped,
            std::max(1, workers)
        )) {
        Py_DECREF(vj_obj);
        Py_DECREF(vk_obj);
        PyErr_SetString(PyExc_RuntimeError, "direct_jk_cartesian failed to build shell blocks.");
        return nullptr;
    }

    return Py_BuildValue("NNLL", vj_obj, vk_obj, computed, skipped);
}

PyObject* direct_veff_cartesian(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    PyObject* dm_obj = nullptr;
    double screen_tol = 0.0;
    int max_l = CARTESIAN_SCALAR_MAX_L;
    int workers = 1;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOdi|i",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &pair_bounds_obj,
            &dm_obj,
            &screen_tol,
            &max_l,
            &workers
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!shells || !origins || !exps || !weights || !nprim || !pair_bounds || !dm) {
        return nullptr;
    }
    if (!validate_dense_inputs(shells.obj, origins.obj, exps.obj, weights.obj, nprim.obj, pair_bounds.obj)) {
        return nullptr;
    }
    if (!validate_cartesian_shell_lmax(shells.obj, max_l)) {
        return nullptr;
    }
    if (PyArray_NDIM(dm.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "direct_veff_cartesian expects dm as a 2D array.");
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    if (PyArray_DIM(dm.obj, 0) != nao || PyArray_DIM(dm.obj, 1) != nao) {
        PyErr_SetString(PyExc_ValueError, "direct_veff_cartesian received a density shape inconsistent with the AO basis.");
        return nullptr;
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* veff_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (veff_obj == nullptr) {
        return nullptr;
    }

    const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    std::vector<double> vj(n2, 0.0);
    std::vector<double> vk(n2, 0.0);
    auto* veff = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(veff_obj)));

    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));
    const auto* dm_data = static_cast<const double*>(PyArray_DATA(dm.obj));

    long long computed = 0;
    long long skipped = 0;
    if (!compute_direct_jk_cartesian_blocked(
            shells_data,
            origins_data,
            exps_data,
            weights_data,
            nprim_data,
            bounds_data,
            dm_data,
            nao,
            max_prim,
            screen_tol,
            vj.data(),
            vk.data(),
            computed,
            skipped,
            std::max(1, workers)
        )) {
        Py_DECREF(veff_obj);
        PyErr_SetString(PyExc_RuntimeError, "direct_veff_cartesian failed to build shell blocks.");
        return nullptr;
    }

    for (std::size_t idx = 0; idx < n2; ++idx) {
        veff[idx] = vj[idx] - 0.5 * vk[idx];
    }

    return Py_BuildValue("NLL", veff_obj, computed, skipped);
}

bool density_is_symmetric(const double* dmat, npy_intp nao) {
    double max_abs = 0.0;
    double max_diff = 0.0;
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j < i; ++j) {
            const double a = dmat[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
            const double b = dmat[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)];
            max_abs = std::max(max_abs, std::max(std::abs(a), std::abs(b)));
            max_diff = std::max(max_diff, std::abs(a - b));
        }
    }
    return max_diff <= 1.0e-12 * std::max(1.0, max_abs);
}

void build_pair_density(const double* dmat, npy_intp nao, double* pair_dm) {
    npy_intp pair = 0;
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j <= i; ++j) {
            pair_dm[static_cast<std::size_t>(pair)] = dmat[j * nao + i];
            if (i != j) {
                pair_dm[static_cast<std::size_t>(pair)] += dmat[i * nao + j];
            }
            ++pair;
        }
    }
}

void contract_j_s8_pairs(
    const double* eri,
    const std::vector<double>& pair_dm,
    npy_intp npair,
    int workers,
    std::vector<double>& j_pairs
) {
    const std::size_t npair_size = static_cast<std::size_t>(npair);
    j_pairs.assign(npair_size, 0.0);
    const int nthread = std::min(
        std::max(1, workers),
        std::max(1, static_cast<int>(npair))
    );
    auto run_serial = [&]() {
        for (npy_intp ij = 0; ij < npair; ++ij) {
            const double dij = pair_dm[static_cast<std::size_t>(ij)];
            const npy_intp row_base = ij * (ij + 1) / 2;
            for (npy_intp kl = 0; kl <= ij; ++kl) {
                const double value = eri[row_base + kl];
                j_pairs[static_cast<std::size_t>(ij)] += value * pair_dm[static_cast<std::size_t>(kl)];
                if (ij != kl) {
                    j_pairs[static_cast<std::size_t>(kl)] += value * dij;
                }
            }
        }
    };
    if (nthread <= 1 || npair < 128) {
        run_serial();
        return;
    }

    const std::size_t chunk = 16;
    std::vector<double> local_j(static_cast<std::size_t>(nthread) * npair_size, 0.0);
    std::atomic<std::size_t> next_pair{0};
    auto run_worker = [&](int tid) {
        double* local = local_j.data() + static_cast<std::size_t>(tid) * npair_size;
        while (true) {
            const std::size_t begin = next_pair.fetch_add(chunk, std::memory_order_relaxed);
            if (begin >= npair_size) {
                break;
            }
            const std::size_t end = std::min(npair_size, begin + chunk);
            for (std::size_t ij_size = begin; ij_size < end; ++ij_size) {
                const double dij = pair_dm[ij_size];
                const std::size_t row_base = ij_size * (ij_size + 1) / 2;
                for (std::size_t kl_size = 0; kl_size <= ij_size; ++kl_size) {
                    const double value = eri[row_base + kl_size];
                    local[ij_size] += value * pair_dm[kl_size];
                    if (ij_size != kl_size) {
                        local[kl_size] += value * dij;
                    }
                }
            }
        }
    };

    bool failed = false;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nthread));
    try {
        for (int tid = 0; tid < nthread; ++tid) {
            threads.emplace_back(run_worker, tid);
        }
    } catch (...) {
        failed = true;
    }
    for (std::thread& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (failed) {
        run_serial();
        return;
    }
    for (int tid = 0; tid < nthread; ++tid) {
        const double* local = local_j.data() + static_cast<std::size_t>(tid) * npair_size;
        for (std::size_t ij = 0; ij < npair_size; ++ij) {
            j_pairs[ij] += local[ij];
        }
    }
}

void contract_k_s8_full_density(
    const double* eri,
    const double* dmat,
    npy_intp nao,
    const S8IndexCache& cache,
    npy_intp npair,
    int workers,
    double* vk
) {
    if (nao <= 0) {
        return;
    }
    const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    std::fill(vk, vk + n2, 0.0);
    const int nthread = std::min(
        std::max(1, workers),
        std::max(1, static_cast<int>(nao))
    );
    auto compute_row = [&](npy_intp i) {
        for (npy_intp j = 0; j < nao; ++j) {
            double total = 0.0;
            for (npy_intp k = 0; k < nao; ++k) {
                const npy_intp kj = cache.ao_pair_index[static_cast<std::size_t>(k) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
                for (npy_intp l = 0; l < nao; ++l) {
                    const npy_intp il = cache.ao_pair_index[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(l)];
                    const npy_intp idx = s8_lookup_index(cache, npair, il, kj);
                    total += dmat[static_cast<std::size_t>(l) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(k)] * eri[idx];
                }
            }
            vk[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] = total;
        }
    };
    auto run_serial = [&]() {
        for (npy_intp i = 0; i < nao; ++i) {
            compute_row(i);
        }
    };
    if (nthread <= 1) {
        run_serial();
        return;
    }

    std::atomic<std::size_t> next_row{0};
    auto run_worker = [&]() {
        while (true) {
            const std::size_t i = next_row.fetch_add(1, std::memory_order_relaxed);
            if (i >= static_cast<std::size_t>(nao)) {
                break;
            }
            compute_row(static_cast<npy_intp>(i));
        }
    };
    bool failed = false;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nthread));
    try {
        for (int tid = 0; tid < nthread; ++tid) {
            threads.emplace_back(run_worker);
        }
    } catch (...) {
        failed = true;
    }
    for (std::thread& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (failed) {
        run_serial();
    }
}

void contract_k_s8_symmetric_density(
    const double* eri,
    const double* dmat,
    npy_intp nao,
    const S8IndexCache& cache,
    npy_intp npair,
    int workers,
    double* vk
) {
    if (nao <= 0) {
        return;
    }
    const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    std::fill(vk, vk + n2, 0.0);
    const int nthread = std::min(
        std::max(1, workers),
        std::max(1, static_cast<int>(nao))
    );
    auto compute_row = [&](npy_intp i) {
        for (npy_intp j = 0; j <= i; ++j) {
            double total = 0.0;
            for (npy_intp k = 0; k < nao; ++k) {
                for (npy_intp l = 0; l <= k; ++l) {
                    if (k == l) {
                        const npy_intp ik = cache.ao_pair_index[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(k)];
                        const npy_intp kj = cache.ao_pair_index[static_cast<std::size_t>(k) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
                        const npy_intp idx = s8_lookup_index(cache, npair, ik, kj);
                        total += dmat[static_cast<std::size_t>(k) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(k)] * eri[idx];
                    } else {
                        const double dkl = 0.5 * (
                            dmat[static_cast<std::size_t>(l) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(k)] +
                            dmat[static_cast<std::size_t>(k) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(l)]
                        );
                        const npy_intp il = cache.ao_pair_index[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(l)];
                        const npy_intp kj = cache.ao_pair_index[static_cast<std::size_t>(k) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
                        const npy_intp ik = cache.ao_pair_index[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(k)];
                        const npy_intp lj = cache.ao_pair_index[static_cast<std::size_t>(l) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)];
                        const npy_intp idx1 = s8_lookup_index(cache, npair, il, kj);
                        const npy_intp idx2 = s8_lookup_index(cache, npair, ik, lj);
                        total += dkl * (eri[idx1] + eri[idx2]);
                    }
                }
            }
            vk[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] = total;
            vk[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)] = total;
        }
    };
    auto run_serial = [&]() {
        for (npy_intp i = 0; i < nao; ++i) {
            compute_row(i);
        }
    };
    if (nthread <= 1) {
        run_serial();
        return;
    }

    std::atomic<std::size_t> next_row{0};
    auto run_worker = [&]() {
        while (true) {
            const std::size_t i = next_row.fetch_add(1, std::memory_order_relaxed);
            if (i >= static_cast<std::size_t>(nao)) {
                break;
            }
            compute_row(static_cast<npy_intp>(i));
        }
    };
    bool failed = false;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(nthread));
    try {
        for (int tid = 0; tid < nthread; ++tid) {
            threads.emplace_back(run_worker);
        }
    } catch (...) {
        failed = true;
    }
    for (std::thread& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (failed) {
        run_serial();
    }
}

void contract_jk_s8_from_density(
    const double* eri,
    const double* dmat,
    npy_intp nao,
    double* vj,
    double* vk,
    int workers
) {
    const npy_intp npair = nao * (nao + 1) / 2;
    const S8IndexCache& cache = get_s8_index_cache(nao, npair);
    std::vector<double> pair_dm(static_cast<std::size_t>(npair), 0.0);
    std::vector<double> j_pairs;
    build_pair_density(dmat, nao, pair_dm.data());
    contract_j_s8_pairs(eri, pair_dm, npair, std::max(1, workers), j_pairs);
    for (npy_intp ij = 0; ij < npair; ++ij) {
        const npy_intp i = cache.pair_i[static_cast<std::size_t>(ij)];
        const npy_intp j = cache.pair_j[static_cast<std::size_t>(ij)];
        const double value = j_pairs[static_cast<std::size_t>(ij)];
        vj[i * nao + j] = value;
        vj[j * nao + i] = value;
    }

    if (density_is_symmetric(dmat, nao)) {
        contract_k_s8_symmetric_density(eri, dmat, nao, cache, npair, std::max(1, workers), vk);
    } else {
        contract_k_s8_full_density(eri, dmat, nao, cache, npair, std::max(1, workers), vk);
    }
}

PyObject* contract_jk_s8(PyObject*, PyObject* args) {
    PyObject* eri_obj = nullptr;
    PyObject* dm_obj = nullptr;
    int nao_arg = 0;
    int workers_arg = 1;
    if (!PyArg_ParseTuple(args, "OOi|i", &eri_obj, &dm_obj, &nao_arg, &workers_arg)) {
        return nullptr;
    }

    ArrayRef eri_s8(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!eri_s8 || !dm) {
        return nullptr;
    }
    if (nao_arg < 0 || PyArray_NDIM(eri_s8.obj) != 1 || PyArray_NDIM(dm.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "contract_jk_s8 expects eri_s8 as a 1D array and dm as a 2D array.");
        return nullptr;
    }

    const npy_intp nao = static_cast<npy_intp>(nao_arg);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp ns8 = npair * (npair + 1) / 2;
    if (
        PyArray_DIM(eri_s8.obj, 0) != ns8 ||
        PyArray_DIM(dm.obj, 0) != nao ||
        PyArray_DIM(dm.obj, 1) != nao
    ) {
        PyErr_SetString(PyExc_ValueError, "contract_jk_s8 received array shapes inconsistent with nao.");
        return nullptr;
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* vj_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) {
        return nullptr;
    }
    PyObject* vk_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vk_obj == nullptr) {
        Py_DECREF(vj_obj);
        return nullptr;
    }

    const auto* eri = static_cast<const double*>(PyArray_DATA(eri_s8.obj));
    const auto* dmat = static_cast<const double*>(PyArray_DATA(dm.obj));
    auto* vj = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj)));
    auto* vk = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vk_obj)));

    contract_jk_s8_from_density(eri, dmat, nao, vj, vk, std::max(1, workers_arg));
    return Py_BuildValue("NN", vj_obj, vk_obj);
}

PyObject* contract_veff_s8(PyObject*, PyObject* args) {
    PyObject* eri_obj = nullptr;
    PyObject* dm_obj = nullptr;
    int nao_arg = 0;
    int workers_arg = 1;
    if (!PyArg_ParseTuple(args, "OOi|i", &eri_obj, &dm_obj, &nao_arg, &workers_arg)) {
        return nullptr;
    }

    ArrayRef eri_s8(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef dm(dm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!eri_s8 || !dm) {
        return nullptr;
    }
    if (nao_arg < 0 || PyArray_NDIM(eri_s8.obj) != 1 || PyArray_NDIM(dm.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "contract_veff_s8 expects eri_s8 as a 1D array and dm as a 2D array.");
        return nullptr;
    }

    const npy_intp nao = static_cast<npy_intp>(nao_arg);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp ns8 = npair * (npair + 1) / 2;
    if (
        PyArray_DIM(eri_s8.obj, 0) != ns8 ||
        PyArray_DIM(dm.obj, 0) != nao ||
        PyArray_DIM(dm.obj, 1) != nao
    ) {
        PyErr_SetString(PyExc_ValueError, "contract_veff_s8 received array shapes inconsistent with nao.");
        return nullptr;
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* veff_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (veff_obj == nullptr) {
        return nullptr;
    }

    const auto* eri = static_cast<const double*>(PyArray_DATA(eri_s8.obj));
    const auto* dmat = static_cast<const double*>(PyArray_DATA(dm.obj));
    auto* veff = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(veff_obj)));
    std::vector<double> vj(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0.0);
    std::vector<double> vk(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0.0);
    contract_jk_s8_from_density(eri, dmat, nao, vj.data(), vk.data(), std::max(1, workers_arg));
    const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    for (std::size_t idx = 0; idx < n2; ++idx) {
        veff[idx] = vj[idx] - 0.5 * vk[idx];
    }
    return veff_obj;
}

PyObject* contract_veff_s8_occ(PyObject*, PyObject* args) {
    PyObject* eri_obj = nullptr;
    PyObject* mo_coeff_obj = nullptr;
    PyObject* mo_occ_obj = nullptr;
    int workers_arg = 1;
    if (!PyArg_ParseTuple(args, "OOO|i", &eri_obj, &mo_coeff_obj, &mo_occ_obj, &workers_arg)) {
        return nullptr;
    }

    ArrayRef eri_s8(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_coeff(mo_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_occ(mo_occ_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!eri_s8 || !mo_coeff || !mo_occ) {
        return nullptr;
    }
    if (PyArray_NDIM(eri_s8.obj) != 1 || PyArray_NDIM(mo_coeff.obj) != 2 || PyArray_NDIM(mo_occ.obj) != 1) {
        PyErr_SetString(PyExc_ValueError, "contract_veff_s8_occ expects eri_s8 as 1D, mo_coeff as 2D, and mo_occ as 1D.");
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(mo_coeff.obj, 0);
    const npy_intp nmo = PyArray_DIM(mo_coeff.obj, 1);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp ns8 = npair * (npair + 1) / 2;
    if (PyArray_DIM(eri_s8.obj, 0) != ns8 || PyArray_DIM(mo_occ.obj, 0) != nmo) {
        PyErr_SetString(PyExc_ValueError, "contract_veff_s8_occ received inconsistent array shapes.");
        return nullptr;
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* veff_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (veff_obj == nullptr) {
        return nullptr;
    }

    const auto* eri = static_cast<const double*>(PyArray_DATA(eri_s8.obj));
    const auto* coeff = static_cast<const double*>(PyArray_DATA(mo_coeff.obj));
    const auto* occ = static_cast<const double*>(PyArray_DATA(mo_occ.obj));
    auto* veff = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(veff_obj)));

    std::vector<double> dm(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0.0);
    for (npy_intp a = 0; a < nmo; ++a) {
        const double oa = occ[a];
        if (std::abs(oa) <= 1.0e-14) {
            continue;
        }
        for (npy_intp i = 0; i < nao; ++i) {
            const double cio = coeff[i * nmo + a] * oa;
            for (npy_intp j = 0; j < nao; ++j) {
                dm[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] +=
                    cio * coeff[j * nmo + a];
            }
        }
    }

    std::vector<double> vj(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0.0);
    std::vector<double> vk(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao), 0.0);
    contract_jk_s8_from_density(eri, dm.data(), nao, vj.data(), vk.data(), std::max(1, workers_arg));
    const std::size_t n2 = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    for (std::size_t idx = 0; idx < n2; ++idx) {
        veff[idx] = vj[idx] - 0.5 * vk[idx];
    }
    return veff_obj;
}

PyObject* contract_jk_ri_occ_packed(PyObject*, PyObject* args) {
    PyObject* pair_factors_obj = nullptr;
    PyObject* mo_coeff_obj = nullptr;
    PyObject* mo_occ_obj = nullptr;
    int nao_arg = 0;
    int workers_arg = 1;
    if (!PyArg_ParseTuple(args, "OOOi|i", &pair_factors_obj, &mo_coeff_obj, &mo_occ_obj, &nao_arg, &workers_arg)) {
        return nullptr;
    }

    ArrayRef pair_factors_arr(pair_factors_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_coeff_arr(mo_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_occ_arr(mo_occ_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!pair_factors_arr || !mo_coeff_arr || !mo_occ_arr) {
        return nullptr;
    }
    if (
        nao_arg < 0 ||
        PyArray_NDIM(pair_factors_arr.obj) != 2 ||
        PyArray_NDIM(mo_coeff_arr.obj) != 2 ||
        PyArray_NDIM(mo_occ_arr.obj) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "contract_jk_ri_occ_packed expects pair_factors/mo_coeff/mo_occ arrays.");
        return nullptr;
    }

    const npy_intp nao = static_cast<npy_intp>(nao_arg);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp naux = PyArray_DIM(pair_factors_arr.obj, 0);
    const npy_intp nmo = PyArray_DIM(mo_coeff_arr.obj, 1);
    if (
        PyArray_DIM(pair_factors_arr.obj, 1) != npair ||
        PyArray_DIM(mo_coeff_arr.obj, 0) != nao ||
        PyArray_DIM(mo_occ_arr.obj, 0) != nmo
    ) {
        PyErr_SetString(PyExc_ValueError, "contract_jk_ri_occ_packed received inconsistent array shapes.");
        return nullptr;
    }

    const auto* pair_factors = static_cast<const double*>(PyArray_DATA(pair_factors_arr.obj));
    const auto* mo_coeff = static_cast<const double*>(PyArray_DATA(mo_coeff_arr.obj));
    const auto* mo_occ = static_cast<const double*>(PyArray_DATA(mo_occ_arr.obj));

    std::vector<npy_intp> occ_index;
    std::vector<double> occ_value;
    occ_index.reserve(static_cast<std::size_t>(nmo));
    occ_value.reserve(static_cast<std::size_t>(nmo));
    for (npy_intp a = 0; a < nmo; ++a) {
        if (std::abs(mo_occ[a]) > 1.0e-14) {
            occ_index.push_back(a);
            occ_value.push_back(mo_occ[a]);
        }
    }

    npy_intp dims[2] = {nao, nao};
    PyObject* vj_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vj_obj == nullptr) {
        return nullptr;
    }
    PyObject* vk_obj = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (vk_obj == nullptr) {
        Py_DECREF(vj_obj);
        return nullptr;
    }

    auto* vj = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vj_obj)));
    auto* vk = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(vk_obj)));
    const npy_intp nocc = static_cast<npy_intp>(occ_index.size());
    if (nocc == 0) {
        return Py_BuildValue("NN", vj_obj, vk_obj);
    }

    std::vector<npy_intp> pair_i(static_cast<std::size_t>(npair), 0);
    std::vector<npy_intp> pair_j(static_cast<std::size_t>(npair), 0);
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j <= i; ++j) {
            const npy_intp pair = pair_index(static_cast<int>(i), static_cast<int>(j));
            pair_i[static_cast<std::size_t>(pair)] = i;
            pair_j[static_cast<std::size_t>(pair)] = j;
        }
    }

    std::vector<double> c_occ(static_cast<std::size_t>(nocc) * static_cast<std::size_t>(nao), 0.0);
    std::vector<double> occ_abs(static_cast<std::size_t>(nocc), 0.0);
    for (npy_intp a = 0; a < nocc; ++a) {
        const npy_intp mo = occ_index[static_cast<std::size_t>(a)];
        occ_abs[static_cast<std::size_t>(a)] = std::abs(occ_value[static_cast<std::size_t>(a)]);
        double* c = c_occ.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(nao);
        for (npy_intp i = 0; i < nao; ++i) {
            c[static_cast<std::size_t>(i)] = mo_coeff[i * nmo + mo];
        }
    }

    const std::size_t target_bc_bytes = 2u * 1024u * 1024u;
    const std::size_t bytes_per_aux_bc =
        std::max<std::size_t>(1, static_cast<std::size_t>(nao) * static_cast<std::size_t>(nocc) * sizeof(double));
    npy_intp block_size = static_cast<npy_intp>(std::max<std::size_t>(1, target_bc_bytes / bytes_per_aux_bc));
    block_size = std::max<npy_intp>(1, std::min<npy_intp>(64, block_size));
    block_size = std::min<npy_intp>(block_size, std::max<npy_intp>(1, naux));
    const npy_intp nblock = (naux + block_size - 1) / block_size;
    const int nthread = std::min(
        std::max(1, workers_arg),
        std::max(1, static_cast<int>(nblock))
    );

    if (nthread <= 1) {
        std::vector<double> j_pairs(static_cast<std::size_t>(npair), 0.0);
        std::vector<double> bc(static_cast<std::size_t>(nao) * static_cast<std::size_t>(nocc), 0.0);
        for (npy_intp aux = 0; aux < naux; ++aux) {
            std::fill(bc.begin(), bc.end(), 0.0);
            const double* packed = pair_factors + aux * npair;
            for (npy_intp i = 0; i < nao; ++i) {
                for (npy_intp j = 0; j <= i; ++j) {
                    const npy_intp pair = pair_index(static_cast<int>(i), static_cast<int>(j));
                    const double factor = packed[pair];
                    for (npy_intp a = 0; a < nocc; ++a) {
                        const npy_intp mo = occ_index[static_cast<std::size_t>(a)];
                        bc[i * nocc + a] += factor * mo_coeff[j * nmo + mo];
                        if (i != j) {
                            bc[j * nocc + a] += factor * mo_coeff[i * nmo + mo];
                        }
                    }
                }
            }

            double coeff = 0.0;
            for (npy_intp i = 0; i < nao; ++i) {
                for (npy_intp a = 0; a < nocc; ++a) {
                    const npy_intp mo = occ_index[static_cast<std::size_t>(a)];
                    coeff += mo_coeff[i * nmo + mo] * occ_value[static_cast<std::size_t>(a)] * bc[i * nocc + a];
                }
            }
            for (npy_intp pair = 0; pair < npair; ++pair) {
                j_pairs[static_cast<std::size_t>(pair)] += coeff * packed[pair];
            }

            for (npy_intp i = 0; i < nao; ++i) {
                for (npy_intp j = 0; j <= i; ++j) {
                    double value = 0.0;
                    for (npy_intp a = 0; a < nocc; ++a) {
                        value += std::abs(occ_value[static_cast<std::size_t>(a)]) * bc[i * nocc + a] * bc[j * nocc + a];
                    }
                    vk[i * nao + j] += value;
                    if (i != j) {
                        vk[j * nao + i] += value;
                    }
                }
            }
        }

        for (npy_intp i = 0; i < nao; ++i) {
            for (npy_intp j = 0; j <= i; ++j) {
                const double value = j_pairs[static_cast<std::size_t>(pair_index(static_cast<int>(i), static_cast<int>(j)))];
                vj[i * nao + j] = value;
                vj[j * nao + i] = value;
            }
        }

        return Py_BuildValue("NN", vj_obj, vk_obj);
    }

    const std::size_t npair_size = static_cast<std::size_t>(npair);
    const std::size_t n2_size = static_cast<std::size_t>(nao) * static_cast<std::size_t>(nao);
    std::vector<double> local_j(static_cast<std::size_t>(nthread) * npair_size, 0.0);
    std::vector<double> local_vk(static_cast<std::size_t>(nthread) * n2_size, 0.0);
    std::atomic<npy_intp> next_block{0};

    auto run_worker = [&](int tid) {
        double* local_j_pairs = local_j.data() + static_cast<std::size_t>(tid) * npair_size;
        double* local_vk_mat = local_vk.data() + static_cast<std::size_t>(tid) * n2_size;
        std::vector<double> bc(
            static_cast<std::size_t>(nocc) *
            static_cast<std::size_t>(nao) *
            static_cast<std::size_t>(block_size),
            0.0
        );
        std::vector<double> coeff(static_cast<std::size_t>(block_size), 0.0);

        while (true) {
            const npy_intp block = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block >= nblock) {
                break;
            }
            const npy_intp aux_begin = block * block_size;
            const npy_intp aux_end = std::min<npy_intp>(naux, aux_begin + block_size);
            const npy_intp bsize = aux_end - aux_begin;
            std::fill(bc.begin(), bc.end(), 0.0);
            std::fill(coeff.begin(), coeff.end(), 0.0);

            for (npy_intp b = 0; b < bsize; ++b) {
                const double* packed = pair_factors + (aux_begin + b) * npair;
                for (npy_intp a = 0; a < nocc; ++a) {
                    const double* c = c_occ.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(nao);
                    double* bc_a = bc.data() +
                        (static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) * static_cast<std::size_t>(block_size));
                    for (npy_intp pair = 0; pair < npair; ++pair) {
                        const npy_intp i = pair_i[static_cast<std::size_t>(pair)];
                        const npy_intp j = pair_j[static_cast<std::size_t>(pair)];
                        const double factor = packed[pair];
                        bc_a[static_cast<std::size_t>(i) * static_cast<std::size_t>(block_size) + static_cast<std::size_t>(b)] +=
                            factor * c[static_cast<std::size_t>(j)];
                        if (i != j) {
                            bc_a[static_cast<std::size_t>(j) * static_cast<std::size_t>(block_size) + static_cast<std::size_t>(b)] +=
                                factor * c[static_cast<std::size_t>(i)];
                        }
                    }
                }
            }

            for (npy_intp a = 0; a < nocc; ++a) {
                const double occ = occ_value[static_cast<std::size_t>(a)];
                const double* c = c_occ.data() + static_cast<std::size_t>(a) * static_cast<std::size_t>(nao);
                const double* bc_a = bc.data() +
                    (static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) * static_cast<std::size_t>(block_size));
                for (npy_intp i = 0; i < nao; ++i) {
                    const double ci_occ = c[static_cast<std::size_t>(i)] * occ;
                    const double* bc_ai = bc_a + static_cast<std::size_t>(i) * static_cast<std::size_t>(block_size);
                    for (npy_intp b = 0; b < bsize; ++b) {
                        coeff[static_cast<std::size_t>(b)] += ci_occ * bc_ai[static_cast<std::size_t>(b)];
                    }
                }
            }

            for (npy_intp b = 0; b < bsize; ++b) {
                const double* packed = pair_factors + (aux_begin + b) * npair;
                const double scale = coeff[static_cast<std::size_t>(b)];
                for (npy_intp pair = 0; pair < npair; ++pair) {
                    local_j_pairs[static_cast<std::size_t>(pair)] += scale * packed[pair];
                }
            }

            for (npy_intp a = 0; a < nocc; ++a) {
                const double occ = occ_abs[static_cast<std::size_t>(a)];
                const double* bc_a = bc.data() +
                    (static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) * static_cast<std::size_t>(block_size));
                for (npy_intp i = 0; i < nao; ++i) {
                    const double* bc_i = bc_a + static_cast<std::size_t>(i) * static_cast<std::size_t>(block_size);
                    for (npy_intp j = 0; j <= i; ++j) {
                        const double* bc_j = bc_a + static_cast<std::size_t>(j) * static_cast<std::size_t>(block_size);
                        double sum = 0.0;
                        for (npy_intp b = 0; b < bsize; ++b) {
                            sum += bc_i[static_cast<std::size_t>(b)] * bc_j[static_cast<std::size_t>(b)];
                        }
                        const double value = occ * sum;
                        local_vk_mat[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] += value;
                        if (i != j) {
                            local_vk_mat[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)] += value;
                        }
                    }
                }
            }
        }
    };

    bool thread_failed = false;
    if (nthread <= 1) {
        run_worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(nthread));
        try {
            for (int tid = 0; tid < nthread; ++tid) {
                threads.emplace_back(run_worker, tid);
            }
        } catch (...) {
            thread_failed = true;
        }
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        if (thread_failed) {
            std::fill(local_j.begin(), local_j.end(), 0.0);
            std::fill(local_vk.begin(), local_vk.end(), 0.0);
            next_block.store(0, std::memory_order_relaxed);
            run_worker(0);
        }
    }

    std::vector<double> j_pairs(npair_size, 0.0);
    for (int tid = 0; tid < nthread; ++tid) {
        const double* local_j_pairs = local_j.data() + static_cast<std::size_t>(tid) * npair_size;
        const double* local_vk_mat = local_vk.data() + static_cast<std::size_t>(tid) * n2_size;
        for (std::size_t pair = 0; pair < npair_size; ++pair) {
            j_pairs[pair] += local_j_pairs[pair];
        }
        for (std::size_t idx = 0; idx < n2_size; ++idx) {
            vk[idx] += local_vk_mat[idx];
        }
    }

    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j <= i; ++j) {
            const double value = j_pairs[static_cast<std::size_t>(pair_index(static_cast<int>(i), static_cast<int>(j)))];
            vj[i * nao + j] = value;
            vj[j * nao + i] = value;
        }
    }

    return Py_BuildValue("NN", vj_obj, vk_obj);
}

PyObject* transform_ri_factors_to_mo_pair(PyObject*, PyObject* args) {
    PyObject* pair_factors_obj = nullptr;
    PyObject* mo_left_obj = nullptr;
    PyObject* mo_right_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &pair_factors_obj, &mo_left_obj, &mo_right_obj)) {
        return nullptr;
    }

    ArrayRef pair_factors_arr(pair_factors_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_left_arr(mo_left_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_right_arr(mo_right_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!pair_factors_arr || !mo_left_arr || !mo_right_arr) {
        return nullptr;
    }
    if (
        PyArray_NDIM(pair_factors_arr.obj) != 2 ||
        PyArray_NDIM(mo_left_arr.obj) != 2 ||
        PyArray_NDIM(mo_right_arr.obj) != 2
    ) {
        PyErr_SetString(PyExc_ValueError, "transform_ri_factors_to_mo_pair expects 2D arrays.");
        return nullptr;
    }

    const npy_intp naux = PyArray_DIM(pair_factors_arr.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_factors_arr.obj, 1);
    const npy_intp nao = PyArray_DIM(mo_left_arr.obj, 0);
    const npy_intp nmo_left = PyArray_DIM(mo_left_arr.obj, 1);
    const npy_intp nmo_right = PyArray_DIM(mo_right_arr.obj, 1);
    if (
        PyArray_DIM(mo_right_arr.obj, 0) != nao ||
        npair != nao * (nao + 1) / 2
    ) {
        PyErr_SetString(PyExc_ValueError, "transform_ri_factors_to_mo_pair received inconsistent array shapes.");
        return nullptr;
    }

    npy_intp dims[3] = {naux, nmo_left, nmo_right};
    PyObject* out_obj = PyArray_ZEROS(3, dims, NPY_DOUBLE, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }

    const auto* pair_factors = static_cast<const double*>(PyArray_DATA(pair_factors_arr.obj));
    const auto* mo_left = static_cast<const double*>(PyArray_DATA(mo_left_arr.obj));
    const auto* mo_right = static_cast<const double*>(PyArray_DATA(mo_right_arr.obj));
    auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));

    for (npy_intp aux = 0; aux < naux; ++aux) {
        const double* packed = pair_factors + aux * npair;
        double* out_aux = out + aux * nmo_left * nmo_right;
        for (npy_intp i = 0; i < nao; ++i) {
            for (npy_intp j = 0; j <= i; ++j) {
                const double factor = packed[pair_index(static_cast<int>(i), static_cast<int>(j))];
                if (i == j) {
                    for (npy_intp p = 0; p < nmo_left; ++p) {
                        const double left_ip = mo_left[i * nmo_left + p];
                        for (npy_intp q = 0; q < nmo_right; ++q) {
                            out_aux[p * nmo_right + q] += factor * left_ip * mo_right[i * nmo_right + q];
                        }
                    }
                } else {
                    for (npy_intp p = 0; p < nmo_left; ++p) {
                        const double left_ip = mo_left[i * nmo_left + p];
                        const double left_jp = mo_left[j * nmo_left + p];
                        for (npy_intp q = 0; q < nmo_right; ++q) {
                            out_aux[p * nmo_right + q] += factor * (
                                left_ip * mo_right[j * nmo_right + q] +
                                left_jp * mo_right[i * nmo_right + q]
                            );
                        }
                    }
                }
            }
        }
    }

    return out_obj;
}

PyObject* ao2mo_s8(PyObject*, PyObject* args) {
    PyObject* eri_obj = nullptr;
    PyObject* mo_coeff_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &eri_obj, &mo_coeff_obj)) {
        return nullptr;
    }

    ArrayRef eri_s8_arr(eri_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef mo_coeff_arr(mo_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!eri_s8_arr || !mo_coeff_arr) {
        return nullptr;
    }
    if (PyArray_NDIM(eri_s8_arr.obj) != 1 || PyArray_NDIM(mo_coeff_arr.obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "ao2mo_s8 expects eri_s8 as a 1D array and mo_coeff as a 2D array.");
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(mo_coeff_arr.obj, 0);
    const npy_intp nmo = PyArray_DIM(mo_coeff_arr.obj, 1);
    const npy_intp npair = nao * (nao + 1) / 2;
    const npy_intp ns8 = npair * (npair + 1) / 2;
    if (PyArray_DIM(eri_s8_arr.obj, 0) != ns8) {
        PyErr_SetString(PyExc_ValueError, "ao2mo_s8 received eri_s8 shape inconsistent with mo_coeff.");
        return nullptr;
    }

    npy_intp dims[4] = {nmo, nmo, nmo, nmo};
    PyObject* out_obj = PyArray_ZEROS(4, dims, NPY_DOUBLE, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }

    const auto* eri = static_cast<const double*>(PyArray_DATA(eri_s8_arr.obj));
    const auto* coeff = static_cast<const double*>(PyArray_DATA(mo_coeff_arr.obj));
    auto* out = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out_obj)));
    const npy_intp nmo2 = nmo * nmo;

    std::vector<double> pair_mo(static_cast<std::size_t>(npair) * static_cast<std::size_t>(nmo2), 0.0);
    for (npy_intp i = 0; i < nao; ++i) {
        for (npy_intp j = 0; j <= i; ++j) {
            const npy_intp pair = pair_index(static_cast<int>(i), static_cast<int>(j));
            double* pair_block = pair_mo.data() + pair * nmo2;
            for (npy_intp p = 0; p < nmo; ++p) {
                const double cip = coeff[i * nmo + p];
                const double cjp = coeff[j * nmo + p];
                for (npy_intp q = 0; q < nmo; ++q) {
                    double value = cip * coeff[j * nmo + q];
                    if (i != j) {
                        value += cjp * coeff[i * nmo + q];
                    }
                    pair_block[p * nmo + q] = value;
                }
            }
        }
    }

    std::vector<double> tmp(static_cast<std::size_t>(npair), 0.0);
    for (npy_intp pq = 0; pq < nmo2; ++pq) {
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (npy_intp ao_pair_left = 0; ao_pair_left < npair; ++ao_pair_left) {
            const double left = pair_mo[ao_pair_left * nmo2 + pq];
            if (left == 0.0) {
                continue;
            }
            for (npy_intp ao_pair_right = 0; ao_pair_right < npair; ++ao_pair_right) {
                tmp[static_cast<std::size_t>(ao_pair_right)] += (
                    left * eri[pair_pair_index(ao_pair_left, ao_pair_right)]
                );
            }
        }
        for (npy_intp rs = 0; rs < nmo2; ++rs) {
            double value = 0.0;
            for (npy_intp ao_pair = 0; ao_pair < npair; ++ao_pair) {
                value += tmp[static_cast<std::size_t>(ao_pair)] * pair_mo[ao_pair * nmo2 + rs];
            }
            out[pq * nmo2 + rs] = value;
        }
    }

    return out_obj;
}

PyObject* compute_ri_tensors_packed(PyObject*, PyObject* args) {
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    PyObject* aux_shells_obj = nullptr;
    PyObject* aux_origins_obj = nullptr;
    PyObject* aux_exps_obj = nullptr;
    PyObject* aux_weights_obj = nullptr;
    PyObject* aux_nprim_obj = nullptr;
    PyObject* pair_bounds_obj = nullptr;
    double screen_tol = 0.0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOd",
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &aux_shells_obj,
            &aux_origins_obj,
            &aux_exps_obj,
            &aux_weights_obj,
            &aux_nprim_obj,
            &pair_bounds_obj,
            &screen_tol
        )) {
        return nullptr;
    }

    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_shells(aux_shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_origins(aux_origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_exps(aux_exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_weights(aux_weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_nprim(aux_nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_bounds(pair_bounds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (
        !shells || !origins || !exps || !weights || !nprim ||
        !aux_shells || !aux_origins || !aux_exps || !aux_weights || !aux_nprim ||
        !pair_bounds
    ) {
        return nullptr;
    }
    if (!validate_ri_inputs(
            shells.obj,
            origins.obj,
            exps.obj,
            weights.obj,
            nprim.obj,
            aux_shells.obj,
            aux_origins.obj,
            aux_exps.obj,
            aux_weights.obj,
            aux_nprim.obj,
            pair_bounds.obj
        )) {
        return nullptr;
    }

    const npy_intp nao = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    const npy_intp naux = PyArray_DIM(aux_shells.obj, 0);
    const npy_intp aux_max_prim = PyArray_DIM(aux_exps.obj, 1);
    const npy_intp npair = nao * (nao + 1) / 2;
    npy_intp metric_dims[2] = {naux, naux};
    npy_intp j3_dims[2] = {naux, npair};
    PyObject* metric_obj = PyArray_ZEROS(2, metric_dims, NPY_DOUBLE, 0);
    if (metric_obj == nullptr) {
        return nullptr;
    }
    PyObject* j3_obj = PyArray_ZEROS(2, j3_dims, NPY_DOUBLE, 0);
    if (j3_obj == nullptr) {
        Py_DECREF(metric_obj);
        return nullptr;
    }

    const auto* shells_data = static_cast<const std::int64_t*>(PyArray_DATA(shells.obj));
    const auto* origins_data = static_cast<const double*>(PyArray_DATA(origins.obj));
    const auto* exps_data = static_cast<const double*>(PyArray_DATA(exps.obj));
    const auto* weights_data = static_cast<const double*>(PyArray_DATA(weights.obj));
    const auto* nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(nprim.obj));
    const auto* aux_shells_data = static_cast<const std::int64_t*>(PyArray_DATA(aux_shells.obj));
    const auto* aux_origins_data = static_cast<const double*>(PyArray_DATA(aux_origins.obj));
    const auto* aux_exps_data = static_cast<const double*>(PyArray_DATA(aux_exps.obj));
    const auto* aux_weights_data = static_cast<const double*>(PyArray_DATA(aux_weights.obj));
    const auto* aux_nprim_data = static_cast<const std::int64_t*>(PyArray_DATA(aux_nprim.obj));
    const auto* bounds_data = static_cast<const double*>(PyArray_DATA(pair_bounds.obj));
    auto* metric = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(metric_obj)));
    auto* j3 = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(j3_obj)));

    for (npy_intp p = 0; p < naux; ++p) {
        for (npy_intp q = 0; q <= p; ++q) {
            const double value = contracted_two_center_coulomb(
                aux_shells_data,
                aux_origins_data,
                aux_exps_data,
                aux_weights_data,
                aux_nprim_data,
                aux_max_prim,
                p,
                q
            );
            metric[p * naux + q] = value;
            metric[q * naux + p] = value;
        }
    }

    std::vector<double> aux_diag(static_cast<std::size_t>(naux), 0.0);
    for (npy_intp a = 0; a < naux; ++a) {
        aux_diag[a] = std::sqrt(std::max(std::abs(metric[a * naux + a]), 0.0));
    }

    long long computed = 0;
    long long skipped = 0;
    if (compute_ri_j3_shell_blocked(
            shells_data,
            origins_data,
            exps_data,
            weights_data,
            nprim_data,
            max_prim,
            aux_shells_data,
            aux_origins_data,
            aux_exps_data,
            aux_weights_data,
            aux_nprim_data,
            aux_max_prim,
            bounds_data,
            aux_diag.data(),
            j3,
            nao,
            naux,
            screen_tol
        )) {
        for (npy_intp a = 0; a < naux; ++a) {
            for (npy_intp p = 0; p < nao; ++p) {
                for (npy_intp q = 0; q <= p; ++q) {
                    if (screen_tol > 0.0 && bounds_data[p * nao + q] * aux_diag[a] < screen_tol) {
                        ++skipped;
                    } else {
                        ++computed;
                    }
                }
            }
        }
        return Py_BuildValue("NNLLi", metric_obj, j3_obj, computed, skipped, 1);
    }

    std::fill(j3, j3 + naux * npair, 0.0);
    std::vector<PrimitivePair> pq_primitives;
    npy_intp pair = 0;
    for (npy_intp p = 0; p < nao; ++p) {
        const double* A = origins_data + 3 * p;
        const std::int64_t* shell_p = shells_data + 3 * p;
        const int l1 = static_cast<int>(shell_p[0]);
        const int m1 = static_cast<int>(shell_p[1]);
        const int n1 = static_cast<int>(shell_p[2]);
        for (npy_intp q = 0; q <= p; ++q) {
            const double* B = origins_data + 3 * q;
            const std::int64_t* shell_q = shells_data + 3 * q;
            const int l2 = static_cast<int>(shell_q[0]);
            const int m2 = static_cast<int>(shell_q[1]);
            const int n2 = static_cast<int>(shell_q[2]);
            const double abx = A[0] - B[0];
            const double aby = A[1] - B[1];
            const double abz = A[2] - B[2];
            const double pair_bound = bounds_data[p * nao + q];
            precompute_primitive_pairs(
                origins_data,
                exps_data,
                weights_data,
                nprim_data,
                max_prim,
                p,
                q,
                pq_primitives
            );

            for (npy_intp a = 0; a < naux; ++a) {
                const double aux_bound = aux_diag[a];
                if (screen_tol > 0.0 && pair_bound * aux_bound < screen_tol) {
                    ++skipped;
                    continue;
                }
                const double* C = aux_origins_data + 3 * a;
                const std::int64_t* shell_a = aux_shells_data + 3 * a;
                const int l3 = static_cast<int>(shell_a[0]);
                const int m3 = static_cast<int>(shell_a[1]);
                const int n3 = static_cast<int>(shell_a[2]);
                double value = 0.0;
                for (const PrimitivePair& pq_primitive : pq_primitives) {
                    for (std::int64_t ia = 0; ia < aux_nprim_data[a]; ++ia) {
                        const double aa = aux_exps_data[a * aux_max_prim + ia];
                        const double wa = aux_weights_data[a * aux_max_prim + ia];
                        value += pq_primitive.weight * wa * primitive_three_center_precomputed(
                            pq_primitive,
                            abx,
                            aby,
                            abz,
                            l1,
                            m1,
                            n1,
                            l2,
                            m2,
                            n2,
                            aa,
                            l3,
                            m3,
                            n3,
                            C
                        );
                    }
                }
                j3[a * npair + pair] = value;
                ++computed;
            }
            ++pair;
        }
    }

    return Py_BuildValue("NNLLi", metric_obj, j3_obj, computed, skipped, 0);
}

PyObject* available(PyObject*, PyObject*) {
    Py_RETURN_TRUE;
}

PyObject* compute_rys_roots_weights(PyObject*, PyObject* args) {
    int nroots = 0;
    double T = 0.0;
    if (!PyArg_ParseTuple(args, "id", &nroots, &T)) return nullptr;
    double roots[7]{};
    double weights[7]{};
    if (!rys_roots_weights_general(nroots, T, roots, weights)) {
        PyErr_SetString(PyExc_ValueError, "Unable to construct the requested Rys rule.");
        return nullptr;
    }
    PyObject* roots_obj = PyTuple_New(nroots);
    PyObject* weights_obj = PyTuple_New(nroots);
    if (roots_obj == nullptr || weights_obj == nullptr) {
        Py_XDECREF(roots_obj);
        Py_XDECREF(weights_obj);
        return nullptr;
    }
    for (int root = 0; root < nroots; ++root) {
        PyTuple_SET_ITEM(roots_obj, root, PyFloat_FromDouble(roots[root]));
        PyTuple_SET_ITEM(weights_obj, root, PyFloat_FromDouble(weights[root]));
    }
    return Py_BuildValue("NN", roots_obj, weights_obj);
}

PyMethodDef methods[] = {
    {"available", available, METH_NOARGS, "Return True when the C++ qchem integral extension is loaded."},
    {"compute_rys_roots_weights", compute_rys_roots_weights, METH_VARARGS, "Compute a native Rys quadrature rule."},
    {"compute_dense_eri_ssss", compute_dense_eri_ssss, METH_VARARGS, "Compute dense Cartesian s-shell ERIs."},
    {"compute_dense_eri_cartesian", compute_dense_eri_cartesian, METH_VARARGS, "Compute dense Cartesian ERIs through max_l with scalar fallback for unsupported shell-block quartets."},
    {"compute_dense_eri_spherical", compute_dense_eri_spherical, METH_VARARGS, "Compute dense real-spherical ERIs by transforming native Cartesian shell quartets in place."},
    {"compute_shell_quartet_rys_l3", compute_shell_quartet_rys_l3, METH_VARARGS, "Compute one Cartesian shell-quartet block with the native l<=3 Rys kernel."},
    {"build_spherical_direct_jk_plan", build_spherical_direct_jk_plan, METH_VARARGS, "Build a persistent native spherical direct-J/K basis plan."},
    {"spherical_direct_jk_plan_stats", spherical_direct_jk_plan_stats, METH_VARARGS, "Return cache and angular-rank statistics for a spherical direct-J/K plan."},
    {"spherical_direct_jk_plan_diagonal", spherical_direct_jk_plan_diagonal, METH_VARARGS, "Return the packed exact spherical Coulomb diagonal for a native plan."},
    {"direct_jk_spherical_plan", direct_jk_spherical_plan, METH_VARARGS, "Compute direct spherical J/K from a persistent native basis plan."},
    {"direct_j_spherical_plan", direct_j_spherical_plan, METH_VARARGS, "Compute direct spherical J only from a persistent native basis plan."},
    {"direct_jk_spherical", direct_jk_spherical, METH_VARARGS, "Compute direct real-spherical J/K from fused shell-quartet blocks."},
    {"compute_eri_s8_cartesian", compute_eri_s8_cartesian, METH_VARARGS, "Compute eight-fold packed Cartesian ERIs through max_l."},
    {"compute_directional_eri_derivatives", compute_directional_eri_derivatives, METH_VARARGS, "Compute first or second directional Cartesian ERI derivatives."},
    {"compute_directional_eri_derivative_scalar", compute_directional_eri_derivative_scalar, METH_VARARGS, "Contract directional ERI derivatives directly with two density matrices."},
    {"compute_directional_one_electron_derivatives", compute_directional_one_electron_derivatives, METH_VARARGS, "Compute first or second directional Cartesian one-electron derivatives."},
    {"compute_one_index_one_electron_derivatives", compute_one_index_one_electron_derivatives, METH_VARARGS, "Compute first or second one-index Cartesian overlap or kinetic derivatives."},
    {"direct_jk_cartesian", direct_jk_cartesian, METH_VARARGS, "Compute direct Cartesian J/K from AO shell data and a density matrix."},
    {"direct_veff_cartesian", direct_veff_cartesian, METH_VARARGS, "Compute direct Cartesian RHF J - 0.5 K from AO shell data and a density matrix."},
    {"compute_ri_tensors_packed", compute_ri_tensors_packed, METH_VARARGS, "Compute packed RI three-center tensors and the auxiliary Coulomb metric."},
    {"contract_jk_s8", contract_jk_s8, METH_VARARGS, "Contract eight-fold packed ERIs with a density matrix."},
    {"contract_veff_s8", contract_veff_s8, METH_VARARGS, "Contract eight-fold packed ERIs with a density matrix and return J - 0.5 K."},
    {"contract_veff_s8_occ", contract_veff_s8_occ, METH_VARARGS, "Contract eight-fold packed ERIs for an occupied RHF MO density and return J - 0.5 K."},
    {"contract_jk_ri_occ_packed", contract_jk_ri_occ_packed, METH_VARARGS, "Contract packed RI factors for occupied MO density."},
    {"mo_pair_factors", transform_ri_factors_to_mo_pair, METH_VARARGS, "Transform packed RI factors to an MO-pair tensor."},
    {"transform_ri_factors_to_mo_pair", transform_ri_factors_to_mo_pair, METH_VARARGS, "Transform packed RI factors to an MO-pair tensor."},
    {"ao2mo_s8", ao2mo_s8, METH_VARARGS, "Transform eight-fold packed AO ERIs to a dense chemist-notation MO tensor."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_integrals_cpp",
    "C++ qchem integral kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__integrals_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
