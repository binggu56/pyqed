#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
const double ERI_PREFAC = 2.0 * PI * PI * std::sqrt(PI);
constexpr int OS_VRR_PAIR_MAX_L = 4;
constexpr int OS_VRR_MAX_CART = 15;
constexpr int CARTESIAN_SCALAR_MAX_L = 6;

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
    const double root = std::sqrt(T);
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
    double* k_out
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
    hash = fnv1a_mix(hash, nprim, static_cast<std::size_t>(nao) * sizeof(std::int64_t));
    return hash;
}

const ShellPairGeomData& get_primary_shell_pair_geom(
    const std::vector<ShellBlock>& shell_blocks,
    const std::int64_t* shells,
    const double* origins,
    const double* exps,
    const std::int64_t* nprim,
    npy_intp nao,
    npy_intp max_prim
) {
    const std::uint64_t key = primary_shell_pair_geom_key(shells, origins, exps, nprim, nao, max_prim);
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
                data.k.data() + off
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

inline std::size_t shell_block_index(int nq, int nr, int ns, int ia, int ib, int ic, int id) {
    return (((static_cast<std::size_t>(ia) * nq + ib) * nr + ic) * ns + id);
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

void fill_boys_values(int max_n, double T, double* values) {
    if (T < 1.0e-14) {
        for (int n = 0; n <= max_n; ++n) {
            values[n] = 1.0 / (2.0 * n + 1.0);
        }
        return;
    }

    const double exp_T = std::exp(-T);
    if (T < 4.0) {
        long double value = 1.0L / (2.0L * max_n + 1.0L);
        long double term = 1.0L;
        const long double t_ld = static_cast<long double>(T);
        for (int k = 1; k < 300; ++k) {
            term *= -t_ld / k;
            const long double add = term / (2.0L * max_n + 2.0L * k + 1.0L);
            value += add;
            if (std::abs(static_cast<double>(add)) < 1.0e-18) {
                break;
            }
        }
        values[max_n] = static_cast<double>(value);
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
    return (((((static_cast<std::size_t>(ax) * adim + ay) * adim + az) * cdim + cx) * cdim + cy) * cdim + cz) * mdim + m;
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
    const int adim = max_a + 1;
    const int cdim = max_c + 1;
    const int mdim = max_m + 1;

    double boys_values[2 * OS_VRR_PAIR_MAX_L + 1];
    fill_boys_values(max_m, T, boys_values);
    for (int m = 0; m <= max_m; ++m) {
        os_vrr_set(
            table,
            0, 0, 0,
            0, 0, 0,
            m,
            adim,
            cdim,
            mdim,
            base_pref * boys_values[m]
        );
    }

    for (int total = 1; total <= max_m; ++total) {
        const int asum_min = std::max(0, total - max_c);
        const int asum_max = std::min(max_a, total);
        for (int asum = asum_min; asum <= asum_max; ++asum) {
            const int csum = total - asum;
            for (int ax = 0; ax <= asum; ++ax) {
                for (int ay = 0; ay <= asum - ax; ++ay) {
                    const int az = asum - ax - ay;
                    for (int cx = 0; cx <= csum; ++cx) {
                        for (int cy = 0; cy <= csum - cx; ++cy) {
                            const int cz = csum - cx - cy;
                                for (int m = 0; m <= max_m - total; ++m) {
                                    double value = 0.0;
                                    if (asum > 0) {
                                        const int axis = os_axis_of_max3(ax, ay, az);
                                        if (axis == 0) {
                                            value = PA[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[0] * os_vrr_get_raw(table, ax - 1, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim);
                                            if (ax - 1 > 0) {
                                                value += (ax - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax - 2, ay, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (cx > 0) {
                                                value += cx / (2.0 * z) * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim);
                                            }
                                        } else if (axis == 1) {
                                            value = PA[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[1] * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy, cz, m + 1, adim, cdim, mdim);
                                            if (ay - 1 > 0) {
                                                value += (ay - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax, ay - 2, az, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (cy > 0) {
                                                value += cy / (2.0 * z) * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim);
                                            }
                                        } else {
                                            value = PA[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m, adim, cdim, mdim)
                                                - q / z * PQ[2] * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz, m + 1, adim, cdim, mdim);
                                            if (az - 1 > 0) {
                                                value += (az - 1) / (2.0 * p) * (
                                                    os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m, adim, cdim, mdim)
                                                    - q / z * os_vrr_get_raw(table, ax, ay, az - 2, cx, cy, cz, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (cz > 0) {
                                                value += cz / (2.0 * z) * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim);
                                            }
                                        }
                                    } else {
                                        const int axis = os_axis_of_max3(cx, cy, cz);
                                        if (axis == 0) {
                                            value = QC[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m, adim, cdim, mdim)
                                                + p / z * PQ[0] * os_vrr_get_raw(table, ax, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim);
                                            if (cx - 1 > 0) {
                                                value += (cx - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx - 2, cy, cz, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (ax > 0) {
                                                value += ax / (2.0 * z) * os_vrr_get_raw(table, ax - 1, ay, az, cx - 1, cy, cz, m + 1, adim, cdim, mdim);
                                            }
                                        } else if (axis == 1) {
                                            value = QC[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m, adim, cdim, mdim)
                                                + p / z * PQ[1] * os_vrr_get_raw(table, ax, ay, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim);
                                            if (cy - 1 > 0) {
                                                value += (cy - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx, cy - 2, cz, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (ay > 0) {
                                                value += ay / (2.0 * z) * os_vrr_get_raw(table, ax, ay - 1, az, cx, cy - 1, cz, m + 1, adim, cdim, mdim);
                                            }
                                        } else {
                                            value = QC[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m, adim, cdim, mdim)
                                                + p / z * PQ[2] * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 1, m + 1, adim, cdim, mdim);
                                            if (cz - 1 > 0) {
                                                value += (cz - 1) / (2.0 * q) * (
                                                    os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m, adim, cdim, mdim)
                                                    - p / z * os_vrr_get_raw(table, ax, ay, az, cx, cy, cz - 2, m + 1, adim, cdim, mdim)
                                                );
                                            }
                                            if (az > 0) {
                                                value += az / (2.0 * z) * os_vrr_get_raw(table, ax, ay, az - 1, cx, cy, cz - 1, m + 1, adim, cdim, mdim);
                                            }
                                        }
                                    }
                                    os_vrr_set(table, ax, ay, az, cx, cy, cz, m, adim, cdim, mdim, value);
                                }
                        }
                    }
                }
            }
        }
    }
}

inline double os_pow_small(double x, int n) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return x;
    }
    if (n == 2) {
        return x * x;
    }
    if (n == 3) {
        return x * x * x;
    }
    return x * x * x * x;
}

inline double os_binom_small(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    if (k == 0 || k == n) {
        return 1.0;
    }
    if (n == 2) {
        return 2.0;
    }
    if (n == 3) {
        return (k == 1 || k == 2) ? 3.0 : 1.0;
    }
    if (n == 4) {
        if (k == 1 || k == 3) {
            return 4.0;
        }
        if (k == 2) {
            return 6.0;
        }
    }
    return 1.0;
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
    double* k_out
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
            a_out[idx] = a;
            b_out[idx] = b;
            p_out[idx] = pexp;
            px_out[idx] = (a * A[0] + b * B[0]) / pexp;
            py_out[idx] = (a * A[1] + b * B[1]) / pexp;
            pz_out[idx] = (a * A[2] + b * B[2]) / pexp;
            k_out[idx] = std::exp(-(a * b / pexp) * ab2);
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
    long long shell_count;
};

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
                    tasks.push_back({ish, jsh, ksh, lsh, shell_count});
                }
            }
        }
    }
}

struct ShellQuartetTaskCache {
    bool valid = false;
    std::uint64_t key = 0;
    int nshell = 0;
    double screen_tol = 0.0;
    std::size_t shell_pair_count = 0;
    std::vector<ShellQuartetTask> tasks;
    long long shell_screened = 0;
};

ShellQuartetTaskCache shell_quartet_task_cache;

std::uint64_t shell_quartet_task_key(
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<double>& shell_pair_bounds,
    int nshell,
    double screen_tol
) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_mix(hash, &nshell, sizeof(nshell));
    hash = fnv1a_mix(hash, &screen_tol, sizeof(screen_tol));
    const std::size_t nblocks = shell_blocks.size();
    const std::size_t nbounds = shell_pair_bounds.size();
    hash = fnv1a_mix(hash, &nblocks, sizeof(nblocks));
    hash = fnv1a_mix(hash, &nbounds, sizeof(nbounds));
    for (const ShellBlock& block : shell_blocks) {
        hash = fnv1a_mix(hash, &block.start, sizeof(block.start));
        hash = fnv1a_mix(hash, &block.stop, sizeof(block.stop));
        hash = fnv1a_mix(hash, &block.l, sizeof(block.l));
    }
    if (!shell_pair_bounds.empty()) {
        hash = fnv1a_mix(
            hash,
            shell_pair_bounds.data(),
            shell_pair_bounds.size() * sizeof(double)
        );
    }
    return hash;
}

void get_shell_quartet_tasks_cached(
    const std::vector<ShellBlock>& shell_blocks,
    const std::vector<double>& shell_pair_bounds,
    int nshell,
    double screen_tol,
    std::vector<ShellQuartetTask>& tasks,
    long long& shell_screened
) {
    const std::uint64_t key = shell_quartet_task_key(
        shell_blocks,
        shell_pair_bounds,
        nshell,
        screen_tol
    );
    if (
        shell_quartet_task_cache.valid &&
        shell_quartet_task_cache.key == key &&
        shell_quartet_task_cache.nshell == nshell &&
        shell_quartet_task_cache.screen_tol == screen_tol &&
        shell_quartet_task_cache.shell_pair_count == shell_pair_bounds.size()
    ) {
        tasks = shell_quartet_task_cache.tasks;
        shell_screened = shell_quartet_task_cache.shell_screened;
        return;
    }

    build_shell_quartet_tasks(
        shell_blocks,
        shell_pair_bounds,
        nshell,
        screen_tol,
        tasks,
        shell_screened
    );
    shell_quartet_task_cache.valid = true;
    shell_quartet_task_cache.key = key;
    shell_quartet_task_cache.nshell = nshell;
    shell_quartet_task_cache.screen_tol = screen_tol;
    shell_quartet_task_cache.shell_pair_count = shell_pair_bounds.size();
    shell_quartet_task_cache.tasks = tasks;
    shell_quartet_task_cache.shell_screened = shell_screened;
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
    double direct_screen_tol = 0.0
) {
    targets.clear();
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

    const std::size_t vrr_table_size =
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_m_l + 1);
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
    if (targets.empty()) {
        return true;
    }
    values.assign(targets.size(), 0.0);

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
                    values[it] += (
                        weights[target.weight_p + ip] *
                        weights[target.weight_q + iq] *
                        weights[target.weight_r + ir] *
                        weights[target.weight_s + is] *
                        primitive
                    );
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
            const double base_pref =
                ERI_PREFAC * pq_k[idx_pq] * rs_k[idx_rs] /
                (pq_p[idx_pq] * rs_p[idx_rs] * std::sqrt(zeta));
            const double PA[3] = {
                pq_px[idx_pq] - origins[3 * p0],
                pq_py[idx_pq] - origins[3 * p0 + 1],
                pq_pz[idx_pq] - origins[3 * p0 + 2],
            };
            const double QC[3] = {
                rs_px[idx_rs] - origins[3 * r0],
                rs_py[idx_rs] - origins[3 * r0 + 1],
                rs_pz[idx_rs] - origins[3 * r0 + 2],
            };
            const double PQ[3] = {pqx, pqy, pqz};

            os_fill_vrr_table(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                rs_p[idx_rs],
                zeta,
                alpha * pq2,
                base_pref,
                PA,
                QC,
                PQ
            );

            for (std::size_t it = 0; it < targets.size(); ++it) {
                const ShellQuartetTarget& target = targets[it];
                const double prefac =
                    weights[target.weight_p + ip] *
                    weights[target.weight_q + iq] *
                    weights[target.weight_r + ir] *
                    weights[target.weight_s + is];
                values[it] += prefac * os_vrr_hrr_eval_expanded(
                    vrr_table,
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
                    0,
                    max_a_l,
                    max_c_l,
                    max_m_l,
                    AB,
                    CD
                );
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

double primitive_three_center_coulomb(
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
    const double* C
) {
    const double p = a + b;
    const double q = c;
    const double alpha = p * q / (p + q);
    const double px = (a * A[0] + b * B[0]) / p;
    const double py = (a * A[1] + b * B[1]) / p;
    const double pz = (a * A[2] + b * B[2]) / p;
    const double dx = px - C[0];
    const double dy = py - C[1];
    const double dz = pz - C[2];

    HermiteEMemo ex_ab(l1, l2, A[0] - B[0], a, b);
    HermiteEMemo ey_ab(m1, m2, A[1] - B[1], a, b);
    HermiteEMemo ez_ab(n1, n2, A[2] - B[2], a, b);
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

[[maybe_unused]] double contracted_three_center_coulomb(
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
    npy_intp p,
    npy_intp q,
    npy_intp a
) {
    const double* A = origins + 3 * p;
    const double* B = origins + 3 * q;
    const double* C = aux_origins + 3 * a;
    const std::int64_t* shell_p = shells + 3 * p;
    const std::int64_t* shell_q = shells + 3 * q;
    const std::int64_t* shell_a = aux_shells + 3 * a;

    double value = 0.0;
    for (std::int64_t ip = 0; ip < nprim[p]; ++ip) {
        const double ap = exps[p * max_prim + ip];
        const double wp = weights[p * max_prim + ip];
        for (std::int64_t iq = 0; iq < nprim[q]; ++iq) {
            const double aq = exps[q * max_prim + iq];
            const double wq = weights[q * max_prim + iq];
            for (std::int64_t ia = 0; ia < aux_nprim[a]; ++ia) {
                const double aa = aux_exps[a * aux_max_prim + ia];
                const double wa = aux_weights[a * aux_max_prim + ia];
                value += wp * wq * wa * primitive_three_center_coulomb(
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
                    aa,
                    static_cast<int>(shell_a[0]),
                    static_cast<int>(shell_a[1]),
                    static_cast<int>(shell_a[2]),
                    C
                );
            }
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

[[maybe_unused]] bool compute_shell_quartet_vrr_hrr_block(
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
    const double* pq_a,
    const double* pq_b,
    const double* pq_p,
    const double* pq_px,
    const double* pq_py,
    const double* pq_pz,
    int npq,
    const double* rs_a,
    const double* rs_b,
    const double* rs_p,
    const double* rs_px,
    const double* rs_py,
    const double* rs_pz,
    int nrs,
    double* vrr_table,
    std::size_t vrr_table_cap,
    std::vector<double>& block
) {
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

    const std::size_t vrr_table_size =
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_m_l + 1);
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

    const double abx = origins[3 * p0] - origins[3 * q0];
    const double aby = origins[3 * p0 + 1] - origins[3 * q0 + 1];
    const double abz = origins[3 * p0 + 2] - origins[3 * q0 + 2];
    const double cdx = origins[3 * r0] - origins[3 * s0];
    const double cdy = origins[3 * r0 + 1] - origins[3 * s0 + 1];
    const double cdz = origins[3 * r0 + 2] - origins[3 * s0 + 2];
    const double ab2 = abx * abx + aby * aby + abz * abz;
    const double cd2 = cdx * cdx + cdy * cdy + cdz * cdz;
    const double AB[3] = {abx, aby, abz};
    const double CD[3] = {cdx, cdy, cdz};

    block.assign(static_cast<std::size_t>(np) * nq * nr * ns, 0.0);
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
            const double base_pref =
                ERI_PREFAC *
                std::exp(-(pq_a[idx_pq] * pq_b[idx_pq] / pq_p[idx_pq]) * ab2) *
                std::exp(-(rs_a[idx_rs] * rs_b[idx_rs] / rs_p[idx_rs]) * cd2) /
                (pq_p[idx_pq] * rs_p[idx_rs] * std::sqrt(zeta));
            const double PA[3] = {
                pq_px[idx_pq] - origins[3 * p0],
                pq_py[idx_pq] - origins[3 * p0 + 1],
                pq_pz[idx_pq] - origins[3 * p0 + 2],
            };
            const double QC[3] = {
                rs_px[idx_rs] - origins[3 * r0],
                rs_py[idx_rs] - origins[3 * r0 + 1],
                rs_pz[idx_rs] - origins[3 * r0 + 2],
            };
            const double PQ[3] = {pqx, pqy, pqz};

            os_fill_vrr_table(
                vrr_table,
                max_a_l,
                max_c_l,
                max_m_l,
                pq_p[idx_pq],
                rs_p[idx_rs],
                zeta,
                alpha * pq2,
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
                    const double wpq = weights[ao_p * max_prim + ip] * weights[ao_q * max_prim + iq];
                    for (int ic = 0; ic < nr; ++ic) {
                        const npy_intp ao_r = r0 + ic;
                        for (int id = 0; id < ns; ++id) {
                            const npy_intp ao_s = s0 + id;
                            if (ao_r < ao_s) {
                                continue;
                            }
                            if (p0 == r0 && q0 == s0) {
                                const long long pair_pq_ao = (static_cast<long long>(ao_p) * (ao_p + 1)) / 2 + ao_q;
                                const long long pair_rs_ao = (static_cast<long long>(ao_r) * (ao_r + 1)) / 2 + ao_s;
                                if (pair_pq_ao < pair_rs_ao) {
                                    continue;
                                }
                            }
                            const double prefac = wpq * weights[ao_r * max_prim + ir] * weights[ao_s * max_prim + is];
                            block[shell_block_index(nq, nr, ns, ia, ib, ic, id)] += prefac * os_vrr_hrr_eval_expanded(
                                vrr_table,
                                ax[ia],
                                ay[ia],
                                az[ia],
                                bx[ib],
                                by[ib],
                                bz[ib],
                                cx[ic],
                                cy[ic],
                                cz[ic],
                                dx_l[id],
                                dy_l[id],
                                dz_l[id],
                                0,
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
        }
    }
    return true;
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

    const std::size_t vrr_table_size =
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_a_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_c_l + 1) *
        static_cast<std::size_t>(max_m_l + 1);
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
        nprim,
        nao,
        max_prim
    );
    const npy_intp pair_cap = pair_geom.pair_cap;

    const std::size_t vrr_table_cap =
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
        static_cast<std::size_t>(2 * OS_VRR_PAIR_MAX_L + 1);
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
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        const std::size_t vrr_table_cap =
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(2 * OS_VRR_PAIR_MAX_L + 1);
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
        get_shell_quartet_tasks_cached(
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
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        const std::size_t vrr_table_cap =
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(2 * OS_VRR_PAIR_MAX_L + 1);
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
        get_shell_quartet_tasks_cached(
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
    std::vector<HrrSecondDerivativeRecipe>& second_recipes
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
        if (order == 1) {
            for (npy_intp mode = 0; mode < nmodes; ++mode) {
                double value = 0.0;
                for (int derivative = 0; derivative < ncenter_derivatives; ++derivative) {
                    value +=
                        mode_coeffs[static_cast<std::size_t>(mode) * ncenter_derivatives + derivative] *
                        derivative_block[it * ncenter_derivatives + derivative];
                }
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
    const std::vector<ShellBlock>& shell_blocks
) {
    try {
        const int nshell = static_cast<int>(shell_blocks.size());
        const ShellPairGeomData& pair_geom = get_primary_shell_pair_geom(
            shell_blocks, shells, origins, exps, nprim, nao, max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        std::vector<double> shell_pair_bounds(
            static_cast<std::size_t>(nshell) * nshell,
            1.0
        );
        std::vector<ShellQuartetTask> tasks;
        long long shell_screened = 0;
        get_shell_quartet_tasks_cached(
            shell_blocks,
            shell_pair_bounds,
            nshell,
            0.0,
            tasks,
            shell_screened
        );

        const std::size_t vrr_table_cap =
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(2 * OS_VRR_PAIR_MAX_L + 1);
        const int nthread = std::min(
            std::max(1, workers),
            std::max(1, static_cast<int>(tasks.size()))
        );
        std::atomic<std::size_t> next_task{0};
        std::atomic<bool> failed{false};

        auto run_worker = [&]() {
            std::vector<double> vrr_table(vrr_table_cap, 0.0);
            std::vector<ShellQuartetTarget> targets;
            std::vector<double> derivative_block;
            std::vector<double> mode_coeffs;
            HrrExpansionCache hrr_cache;
            std::vector<HrrFirstDerivativeRecipe> first_recipes;
            std::vector<HrrSecondDerivativeRecipe> second_recipes;
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
                            out,
                            vrr_table,
                            targets,
                            derivative_block,
                            mode_coeffs,
                            hrr_cache,
                            first_recipes,
                            second_recipes
                        )) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
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
        shell_blocks
    );
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out_obj);
        PyErr_SetString(PyExc_RuntimeError, "C++ directional ERI derivative evaluation failed.");
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
    vk[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(d)] +=
        dm[static_cast<std::size_t>(b) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(c)] * value;
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

inline void direct_add_symmetric_matrix_entry(
    double* mat,
    npy_intp nao,
    npy_intp i,
    npy_intp j,
    double value
) {
    mat[static_cast<std::size_t>(i) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(j)] += value;
    if (i != j) {
        mat[static_cast<std::size_t>(j) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(i)] += value;
    }
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
    const bool pq_same = (p == q);
    const bool rs_same = (r == s);
    const bool same_pair = (p == r && q == s);
    auto dm_at = [&](npy_intp a, npy_intp b) {
        return dm[static_cast<std::size_t>(a) * static_cast<std::size_t>(nao) + static_cast<std::size_t>(b)];
    };

    direct_add_symmetric_matrix_entry(
        vj,
        nao,
        p,
        q,
        (rs_same ? 1.0 : 2.0) * dm_at(r, s) * value
    );
    if (!same_pair) {
        direct_add_symmetric_matrix_entry(
            vj,
            nao,
            r,
            s,
            (pq_same ? 1.0 : 2.0) * dm_at(p, q) * value
        );
    }

    direct_add_k_exchange_pair_symmetric_density(
        vk,
        nao,
        p,
        s,
        dm_at(q, r) * value,
        p == s && q == r
    );
    if (!pq_same) {
        direct_add_k_exchange_pair_symmetric_density(
            vk,
            nao,
            q,
            s,
            dm_at(p, r) * value,
            q == s && p == r
        );
    }
    if (!rs_same) {
        direct_add_k_exchange_pair_symmetric_density(
            vk,
            nao,
            p,
            r,
            dm_at(q, s) * value,
            p == r && q == s
        );
    }
    if (!pq_same && !rs_same && !same_pair) {
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
            nprim,
            nao,
            max_prim
        );
        const npy_intp pair_cap = pair_geom.pair_cap;
        const std::size_t shell_pair_count = static_cast<std::size_t>(nshell) * static_cast<std::size_t>(nshell);
        std::vector<double> shell_pair_bounds(shell_pair_count, 0.0);
        const std::size_t vrr_table_cap =
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(OS_VRR_PAIR_MAX_L + 1) *
            static_cast<std::size_t>(2 * OS_VRR_PAIR_MAX_L + 1);
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
        get_shell_quartet_tasks_cached(
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

PyMethodDef methods[] = {
    {"available", available, METH_NOARGS, "Return True when the C++ qchem integral extension is loaded."},
    {"compute_dense_eri_ssss", compute_dense_eri_ssss, METH_VARARGS, "Compute dense Cartesian s-shell ERIs."},
    {"compute_dense_eri_cartesian", compute_dense_eri_cartesian, METH_VARARGS, "Compute dense Cartesian ERIs through max_l with scalar fallback for unsupported shell-block quartets."},
    {"compute_eri_s8_cartesian", compute_eri_s8_cartesian, METH_VARARGS, "Compute eight-fold packed Cartesian ERIs through max_l."},
    {"compute_directional_eri_derivatives", compute_directional_eri_derivatives, METH_VARARGS, "Compute first or second directional Cartesian ERI derivatives."},
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
