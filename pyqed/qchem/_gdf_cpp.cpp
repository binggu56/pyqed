#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

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

inline void complex_mul(
    double ar,
    double ai,
    double br,
    double bi,
    double& cr,
    double& ci) {
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

inline void sincos_values(double value, double& sin_value, double& cos_value) {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_sincos(value, &sin_value, &cos_value);
#else
    sin_value = std::sin(value);
    cos_value = std::cos(value);
#endif
}

inline void multiply_power_factor(
    double& prod_real,
    double& prod_imag,
    const std::vector<double>& power_real,
    const std::vector<double>& power_imag,
    npy_int64 power) {
    if (power == 0) {
        return;
    }
    double next_real = 0.0;
    double next_imag = 0.0;
    complex_mul(
        prod_real,
        prod_imag,
        power_real[static_cast<size_t>(power)],
        power_imag[static_cast<size_t>(power)],
        next_real,
        next_imag);
    prod_real = next_real;
    prod_imag = next_imag;
}

inline void fill_neg_i_power_table(
    double g,
    npy_int64 max_power,
    std::vector<double>& real,
    std::vector<double>& imag) {
    real.resize(static_cast<size_t>(max_power + 1));
    imag.resize(static_cast<size_t>(max_power + 1));
    double mag = 1.0;
    for (npy_int64 n = 0; n <= max_power; ++n) {
        switch (n & 3) {
        case 0:
            real[static_cast<size_t>(n)] = mag;
            imag[static_cast<size_t>(n)] = 0.0;
            break;
        case 1:
            real[static_cast<size_t>(n)] = 0.0;
            imag[static_cast<size_t>(n)] = -mag;
            break;
        case 2:
            real[static_cast<size_t>(n)] = -mag;
            imag[static_cast<size_t>(n)] = 0.0;
            break;
        default:
            real[static_cast<size_t>(n)] = 0.0;
            imag[static_cast<size_t>(n)] = mag;
            break;
        }
        mag *= g;
    }
}

inline void fill_power_product_table(
    npy_int64 max_power_x,
    npy_int64 max_power_y,
    npy_int64 max_power_z,
    const std::vector<double>& gx_power_real,
    const std::vector<double>& gx_power_imag,
    const std::vector<double>& gy_power_real,
    const std::vector<double>& gy_power_imag,
    const std::vector<double>& gz_power_real,
    const std::vector<double>& gz_power_imag,
    std::vector<double>& product_real,
    std::vector<double>& product_imag) {
    max_power_x = std::max<npy_int64>(max_power_x, 0);
    max_power_y = std::max<npy_int64>(max_power_y, 0);
    max_power_z = std::max<npy_int64>(max_power_z, 0);
    const size_t ny = static_cast<size_t>(max_power_y + 1);
    const size_t nz = static_cast<size_t>(max_power_z + 1);
    const size_t size = static_cast<size_t>(max_power_x + 1) * ny * nz;
    product_real.resize(size);
    product_imag.resize(size);
    for (npy_int64 tx = 0; tx <= max_power_x; ++tx) {
        for (npy_int64 ty = 0; ty <= max_power_y; ++ty) {
            double xy_real = 0.0;
            double xy_imag = 0.0;
            complex_mul(
                gx_power_real[static_cast<size_t>(tx)],
                gx_power_imag[static_cast<size_t>(tx)],
                gy_power_real[static_cast<size_t>(ty)],
                gy_power_imag[static_cast<size_t>(ty)],
                xy_real,
                xy_imag);
            for (npy_int64 tz = 0; tz <= max_power_z; ++tz) {
                const size_t index =
                    (static_cast<size_t>(tx) * ny + static_cast<size_t>(ty)) * nz
                    + static_cast<size_t>(tz);
                complex_mul(
                    xy_real,
                    xy_imag,
                    gz_power_real[static_cast<size_t>(tz)],
                    gz_power_imag[static_cast<size_t>(tz)],
                    product_real[index],
                    product_imag[index]);
            }
        }
    }
}

inline void fill_gaussian_moment_table(
    npy_int64 order,
    double g,
    double exponent,
    std::vector<double>& real,
    std::vector<double>& imag) {
    order = std::max<npy_int64>(order, 0);
    real.resize(static_cast<size_t>(order + 1));
    imag.resize(static_cast<size_t>(order + 1));
    const double inv_2a = 0.5 / exponent;
    const double m0 = std::sqrt(kPi / exponent) * std::exp(-g * g * inv_2a * 0.5);
    real[0] = m0;
    imag[0] = 0.0;
    if (order == 0) {
        return;
    }
    real[1] = 0.0;
    imag[1] = -g * m0 * inv_2a;
    for (npy_int64 n = 1; n < order; ++n) {
        const double next_real = (n * real[static_cast<size_t>(n - 1)]
                                  + g * imag[static_cast<size_t>(n)])
            * inv_2a;
        const double next_imag = (n * imag[static_cast<size_t>(n - 1)]
                                  - g * real[static_cast<size_t>(n)])
            * inv_2a;
        real[static_cast<size_t>(n + 1)] = next_real;
        imag[static_cast<size_t>(n + 1)] = next_imag;
    }
}

inline long env_thread_count() {
    const char* value = std::getenv("PYQED_GDF_CPP_THREADS");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("OMP_NUM_THREADS");
    }
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 1) {
        return 1;
    }
    return parsed;
}

inline long selected_thread_count(int explicit_threads) {
    if (explicit_threads > 0) {
        return explicit_threads;
    }
    return env_thread_count();
}

enum class ContractThreadStrategy {
    Auto,
    Pair,
    PairG,
};

inline ContractThreadStrategy contract_thread_strategy() {
    const char* value = std::getenv("PYQED_GDF_CONTRACT_THREADING");
    if (value == nullptr || value[0] == '\0') {
        return ContractThreadStrategy::Auto;
    }
    if (std::strcmp(value, "pair_g") == 0 || std::strcmp(value, "pair-g") == 0
        || std::strcmp(value, "flat") == 0) {
        return ContractThreadStrategy::PairG;
    }
    if (std::strcmp(value, "pair") == 0 || std::strcmp(value, "pairs") == 0) {
        return ContractThreadStrategy::Pair;
    }
    return ContractThreadStrategy::Auto;
}

inline bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0
        || std::strcmp(value, "False") == 0 || std::strcmp(value, "off") == 0
        || std::strcmp(value, "no") == 0) {
        return false;
    }
    return true;
}

inline bool env_flag_is_false(const char* value) {
    return value != nullptr
        && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0
            || std::strcmp(value, "False") == 0 || std::strcmp(value, "off") == 0
            || std::strcmp(value, "no") == 0);
}

inline bool env_flag_is_true(const char* value) {
    return value != nullptr
        && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0
            || std::strcmp(value, "True") == 0 || std::strcmp(value, "on") == 0
            || std::strcmp(value, "yes") == 0);
}

inline size_t power_product_table_size(
    npy_int64 max_power_x,
    npy_int64 max_power_y,
    npy_int64 max_power_z) {
    max_power_x = std::max<npy_int64>(max_power_x, 0);
    max_power_y = std::max<npy_int64>(max_power_y, 0);
    max_power_z = std::max<npy_int64>(max_power_z, 0);
    return static_cast<size_t>(max_power_x + 1)
        * static_cast<size_t>(max_power_y + 1)
        * static_cast<size_t>(max_power_z + 1);
}

inline size_t power_product_cache_min_size() {
    const char* value = std::getenv("PYQED_GDF_POWER_PRODUCT_CACHE_MIN_SIZE");
    if (value == nullptr || value[0] == '\0') {
        return 64;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 1) {
        return 64;
    }
    return static_cast<size_t>(parsed);
}

inline bool power_product_cache_enabled(
    bool all_zero_powers,
    npy_int64 max_power_x,
    npy_int64 max_power_y,
    npy_int64 max_power_z) {
    if (all_zero_powers) {
        return false;
    }
    const char* value = std::getenv("PYQED_GDF_POWER_PRODUCT_CACHE");
    if (env_flag_is_false(value)) {
        return false;
    }
    if (env_flag_is_true(value)) {
        return true;
    }
    if (value != nullptr && std::strcmp(value, "auto") == 0) {
        return power_product_table_size(max_power_x, max_power_y, max_power_z)
            >= power_product_cache_min_size();
    }
    return true;
}

inline const npy_int64* int64_data(PyArrayObject* arr) {
    return reinterpret_cast<const npy_int64*>(PyArray_DATA(arr));
}

inline const double* double_data(PyArrayObject* arr) {
    return reinterpret_cast<const double*>(PyArray_DATA(arr));
}

inline const npy_complex128* complex_data(PyArrayObject* arr) {
    return reinterpret_cast<const npy_complex128*>(PyArray_DATA(arr));
}

inline void store_complex(npy_complex128& target, double real, double imag) {
    double* parts = reinterpret_cast<double*>(&target);
    parts[0] = real;
    parts[1] = imag;
}

inline void add_complex(npy_complex128& target, double real, double imag) {
    double* parts = reinterpret_cast<double*>(&target);
    parts[0] += real;
    parts[1] += imag;
}

inline bool same_primitive_product(
    npy_int64 term,
    npy_int64 image,
    double cx,
    double cy,
    double cz,
    double inv_4p,
    const npy_int64* image_ptr,
    const double* center_ptr,
    const double* inv_4p_ptr) {
    return image_ptr[term] == image && center_ptr[3 * term + 0] == cx
        && center_ptr[3 * term + 1] == cy && center_ptr[3 * term + 2] == cz
        && inv_4p_ptr[term] == inv_4p;
}

inline double cached_exp_scale(
    double g2,
    double inv_4p,
    std::vector<double>& cache_inv_4p,
    std::vector<double>& cache_values) {
    for (size_t index = 0; index < cache_inv_4p.size(); ++index) {
        if (cache_inv_4p[index] == inv_4p) {
            return cache_values[index];
        }
    }
    const double value = std::exp(-g2 * inv_4p);
    cache_inv_4p.push_back(inv_4p);
    cache_values.push_back(value);
    return value;
}

inline void gaussian_product_plane_factor(
    npy_int64 term,
    const double* center_ptr,
    const double* inv_4p_ptr,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    double& factor_real,
    double& factor_imag) {
    const double cx = center_ptr[3 * term + 0];
    const double cy = center_ptr[3 * term + 1];
    const double cz = center_ptr[3 * term + 2];
    const double phase_arg = gxv * cx + gyv * cy + gzv * cz;
    const double scale = std::exp(-g2 * inv_4p_ptr[term]);
    double sin_phase = 0.0;
    double cos_phase = 0.0;
    sincos_values(phase_arg, sin_phase, cos_phase);
    factor_real = scale * cos_phase;
    factor_imag = -scale * sin_phase;
}

inline void gaussian_product_plane_factor_values(
    double cx,
    double cy,
    double cz,
    double inv_4p,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    double& factor_real,
    double& factor_imag) {
    const double phase_arg = gxv * cx + gyv * cy + gzv * cz;
    const double scale = std::exp(-g2 * inv_4p);
    double sin_phase = 0.0;
    double cos_phase = 0.0;
    sincos_values(phase_arg, sin_phase, cos_phase);
    factor_real = scale * cos_phase;
    factor_imag = -scale * sin_phase;
}

inline void accumulate_primitive_product(
    npy_int64& term,
    npy_int64 term_stop,
    npy_intp nterm,
    const npy_int64* image_ptr,
    const double* center_ptr,
    const double* inv_4p_ptr,
    const double* coeff_ptr,
    const npy_int64* power_ptr,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    bool plane_z,
    bool all_zero_powers,
    const std::vector<double>& gx_power_real,
    const std::vector<double>& gx_power_imag,
    const std::vector<double>& gy_power_real,
    const std::vector<double>& gy_power_imag,
    const std::vector<double>& gz_power_real,
    const std::vector<double>& gz_power_imag,
    double& image_sum_real,
    double& image_sum_imag,
    std::vector<double>* scale_cache_inv_4p = nullptr,
    std::vector<double>* scale_cache_values = nullptr) {
    const npy_int64 image = image_ptr[term];
    const double cx = center_ptr[3 * term + 0];
    const double cy = center_ptr[3 * term + 1];
    const double cz = center_ptr[3 * term + 2];
    const double inv_4p = inv_4p_ptr[term];
    double poly_real = 0.0;
    double poly_imag = 0.0;

    do {
        if (all_zero_powers) {
            poly_real += coeff_ptr[term];
        } else {
            const npy_int64 tx = power_ptr[3 * term + 0];
            const npy_int64 ty = power_ptr[3 * term + 1];
            const npy_int64 tv = power_ptr[3 * term + 2];
            if ((!plane_z || tv == 0) && tx >= 0 && ty >= 0 && tv >= 0) {
                double prod_real = 1.0;
                double prod_imag = 0.0;
                multiply_power_factor(
                    prod_real, prod_imag, gx_power_real, gx_power_imag, tx);
                multiply_power_factor(
                    prod_real, prod_imag, gy_power_real, gy_power_imag, ty);
                multiply_power_factor(
                    prod_real, prod_imag, gz_power_real, gz_power_imag, tv);
                const double coeff = coeff_ptr[term];
                poly_real += coeff * prod_real;
                poly_imag += coeff * prod_imag;
            }
        }
        ++term;
    } while (term < term_stop && term >= 0 && term < nterm
             && same_primitive_product(
                 term,
                 image,
                 cx,
                 cy,
                 cz,
                 inv_4p,
                 image_ptr,
                 center_ptr,
                 inv_4p_ptr));

    if (poly_real == 0.0 && poly_imag == 0.0) {
        return;
    }
    const double phase_arg = gxv * cx + gyv * cy + gzv * cz;
    const double scale =
        (scale_cache_inv_4p != nullptr && scale_cache_values != nullptr)
        ? cached_exp_scale(g2, inv_4p, *scale_cache_inv_4p, *scale_cache_values)
        : std::exp(-g2 * inv_4p);
    double sin_phase = 0.0;
    double cos_phase = 0.0;
    sincos_values(phase_arg, sin_phase, cos_phase);
    image_sum_real += scale * (cos_phase * poly_real + sin_phase * poly_imag);
    image_sum_imag += scale * (cos_phase * poly_imag - sin_phase * poly_real);
}

inline void accumulate_product_group(
    npy_int64 term,
    npy_int64 term_stop,
    npy_intp nterm,
    const double* center_ptr,
    const double* inv_4p_ptr,
    const double* coeff_ptr,
    const npy_int64* power_ptr,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    bool plane_z,
    bool all_zero_powers,
    const std::vector<double>& gx_power_real,
    const std::vector<double>& gx_power_imag,
    const std::vector<double>& gy_power_real,
    const std::vector<double>& gy_power_imag,
    const std::vector<double>& gz_power_real,
    const std::vector<double>& gz_power_imag,
    double& image_sum_real,
    double& image_sum_imag,
    npy_int64 product_factor_id = -1,
    std::vector<npy_intp>* product_factor_seen = nullptr,
    std::vector<double>* product_factor_real = nullptr,
    std::vector<double>* product_factor_imag = nullptr,
    std::vector<double>* scale_cache_inv_4p = nullptr,
    std::vector<double>* scale_cache_values = nullptr,
    const std::vector<double>* power_product_real = nullptr,
    const std::vector<double>* power_product_imag = nullptr,
    npy_int64 power_product_max_x = 0,
    npy_int64 power_product_max_y = 0,
    npy_int64 power_product_max_z = 0,
    npy_intp product_factor_current_stamp = 0) {
    if (term < 0 || term >= nterm || term_stop <= term) {
        return;
    }
    const npy_int64 first_term = term;
    const double cx = center_ptr[3 * term + 0];
    const double cy = center_ptr[3 * term + 1];
    const double cz = center_ptr[3 * term + 2];
    const double inv_4p = inv_4p_ptr[term];
    double poly_real = 0.0;
    double poly_imag = 0.0;

    for (; term < term_stop; ++term) {
        if (term < 0 || term >= nterm) {
            continue;
        }
        if (all_zero_powers) {
            poly_real += coeff_ptr[term];
        } else {
            const npy_int64 tx = power_ptr[3 * term + 0];
            const npy_int64 ty = power_ptr[3 * term + 1];
            const npy_int64 tv = power_ptr[3 * term + 2];
            if ((!plane_z || tv == 0) && tx >= 0 && ty >= 0 && tv >= 0) {
                double prod_real = 1.0;
                double prod_imag = 0.0;
                const bool use_power_product =
                    power_product_real != nullptr && power_product_imag != nullptr
                    && tx <= power_product_max_x
                    && ty <= power_product_max_y
                    && tv <= power_product_max_z;
                if (use_power_product) {
                    const size_t ny = static_cast<size_t>(power_product_max_y + 1);
                    const size_t nz = static_cast<size_t>(power_product_max_z + 1);
                    const size_t index =
                        (static_cast<size_t>(tx) * ny + static_cast<size_t>(ty)) * nz
                        + static_cast<size_t>(tv);
                    prod_real = (*power_product_real)[index];
                    prod_imag = (*power_product_imag)[index];
                } else {
                    multiply_power_factor(
                        prod_real, prod_imag, gx_power_real, gx_power_imag, tx);
                    multiply_power_factor(
                        prod_real, prod_imag, gy_power_real, gy_power_imag, ty);
                    multiply_power_factor(
                        prod_real, prod_imag, gz_power_real, gz_power_imag, tv);
                }
                const double coeff = coeff_ptr[term];
                poly_real += coeff * prod_real;
                poly_imag += coeff * prod_imag;
            }
        }
    }

    if (poly_real == 0.0 && poly_imag == 0.0) {
        return;
    }
    double factor_real = 0.0;
    double factor_imag = 0.0;
    const bool use_product_factor_cache =
        product_factor_real != nullptr
        && product_factor_imag != nullptr
        && product_factor_id >= 0
        && static_cast<size_t>(product_factor_id) < product_factor_real->size()
        && static_cast<size_t>(product_factor_id) < product_factor_imag->size()
        && (
            product_factor_seen == nullptr
            || static_cast<size_t>(product_factor_id) < product_factor_seen->size());
    if (use_product_factor_cache) {
        const size_t factor_index = static_cast<size_t>(product_factor_id);
        if (product_factor_seen != nullptr
            && (*product_factor_seen)[factor_index] != product_factor_current_stamp) {
            gaussian_product_plane_factor(
                first_term,
                center_ptr,
                inv_4p_ptr,
                gxv,
                gyv,
                gzv,
                g2,
                (*product_factor_real)[factor_index],
                (*product_factor_imag)[factor_index]);
            (*product_factor_seen)[factor_index] = product_factor_current_stamp;
        }
        factor_real = (*product_factor_real)[factor_index];
        factor_imag = (*product_factor_imag)[factor_index];
    } else {
        const double phase_arg = gxv * cx + gyv * cy + gzv * cz;
        const double scale =
            (scale_cache_inv_4p != nullptr && scale_cache_values != nullptr)
            ? cached_exp_scale(g2, inv_4p, *scale_cache_inv_4p, *scale_cache_values)
            : std::exp(-g2 * inv_4p);
        double sin_phase = 0.0;
        double cos_phase = 0.0;
        sincos_values(phase_arg, sin_phase, cos_phase);
        factor_real = scale * cos_phase;
        factor_imag = -scale * sin_phase;
    }
    image_sum_real += factor_real * poly_real - factor_imag * poly_imag;
    image_sum_imag += factor_real * poly_imag + factor_imag * poly_real;
}

bool validate_image_group_metadata(
    const npy_int64* pair_group_starts_ptr,
    const npy_int64* group_image_ptr,
    const npy_int64* group_term_start_ptr,
    const npy_int64* group_term_stop_ptr,
    const npy_int64* group_product_start_ptr,
    const npy_int64* group_product_stop_ptr,
    const npy_int64* product_term_start_ptr,
    const npy_int64* product_term_stop_ptr,
    npy_intp npair,
    npy_intp ngroup,
    npy_intp nproduct,
    npy_intp nterm,
    npy_intp nimage) {
    if (pair_group_starts_ptr[0] != 0 || pair_group_starts_ptr[npair] != ngroup) {
        PyErr_SetString(
            PyExc_ValueError,
            "pair_image_group_starts must span image group arrays.");
        return false;
    }
    for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
        const npy_int64 begin = pair_group_starts_ptr[pair_idx];
        const npy_int64 end = pair_group_starts_ptr[pair_idx + 1];
        if (begin < 0 || end < begin || end > ngroup) {
            PyErr_SetString(PyExc_ValueError, "invalid pair image-group span.");
            return false;
        }
    }
    for (npy_intp group = 0; group < ngroup; ++group) {
        const npy_int64 image = group_image_ptr[group];
        const npy_int64 term_begin = group_term_start_ptr[group];
        const npy_int64 term_end = group_term_stop_ptr[group];
        const npy_int64 product_begin = group_product_start_ptr[group];
        const npy_int64 product_end = group_product_stop_ptr[group];
        if (image < 0 || image >= nimage) {
            PyErr_SetString(PyExc_ValueError, "image_group_image is out of range.");
            return false;
        }
        if (term_begin < 0 || term_end < term_begin || term_end > nterm) {
            PyErr_SetString(PyExc_ValueError, "invalid image-group term span.");
            return false;
        }
        if (product_begin < 0 || product_end < product_begin || product_end > nproduct) {
            PyErr_SetString(PyExc_ValueError, "invalid image-group product span.");
            return false;
        }
        if (product_begin < product_end) {
            if (product_term_start_ptr[product_begin] != term_begin
                || product_term_stop_ptr[product_end - 1] != term_end) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "image product groups must span image-group terms.");
                return false;
            }
        } else if (term_begin != term_end) {
            PyErr_SetString(
                PyExc_ValueError,
                "non-empty image group must have product groups.");
            return false;
        }
        for (npy_int64 product = product_begin; product < product_end; ++product) {
            const npy_int64 product_term_begin = product_term_start_ptr[product];
            const npy_int64 product_term_end = product_term_stop_ptr[product];
            if (product_term_begin < term_begin || product_term_end < product_term_begin
                || product_term_end > term_end) {
                PyErr_SetString(PyExc_ValueError, "invalid primitive-product term span.");
                return false;
            }
            if (product + 1 < product_end
                && product_term_end != product_term_start_ptr[product + 1]) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "primitive-product term spans must be contiguous.");
                return false;
            }
        }
    }
    return true;
}

bool validate_product_factor_ids(
    const npy_int64* product_factor_id_ptr,
    npy_intp nproduct,
    npy_intp& nfactor) {
    nfactor = 0;
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 factor_id = product_factor_id_ptr[product];
        if (factor_id < 0) {
            PyErr_SetString(PyExc_ValueError, "product_group_factor_id contains a negative id.");
            return false;
        }
        nfactor = std::max(nfactor, static_cast<npy_intp>(factor_id + 1));
    }
    return true;
}

template <typename AccumulateImage>
inline void for_each_image_group_sum(
    npy_intp pair_idx,
    const npy_int64* pair_group_starts_ptr,
    const npy_int64* group_image_ptr,
    const npy_int64* group_term_start_ptr,
    const npy_int64* group_term_stop_ptr,
    const npy_int64* group_product_start_ptr,
    const npy_int64* group_product_stop_ptr,
    const npy_int64* product_term_start_ptr,
    const npy_int64* product_term_stop_ptr,
    const npy_int64* product_factor_id_ptr,
    npy_intp ngroup,
    npy_intp nproduct,
    npy_intp nterm,
    const npy_int64* image_ptr,
    const double* center_ptr,
    const double* inv_4p_ptr,
    const double* coeff_ptr,
    const npy_int64* power_ptr,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    bool plane_z,
    bool all_zero_powers,
    const std::vector<double>& gx_power_real,
    const std::vector<double>& gx_power_imag,
    const std::vector<double>& gy_power_real,
    const std::vector<double>& gy_power_imag,
    const std::vector<double>& gz_power_real,
    const std::vector<double>& gz_power_imag,
    AccumulateImage accumulate_image,
    std::vector<npy_intp>* product_factor_seen = nullptr,
    std::vector<double>* product_factor_real = nullptr,
    std::vector<double>* product_factor_imag = nullptr,
    std::vector<double>* scale_cache_inv_4p = nullptr,
    std::vector<double>* scale_cache_values = nullptr,
    const std::vector<double>* power_product_real = nullptr,
    const std::vector<double>* power_product_imag = nullptr,
    npy_int64 power_product_max_x = 0,
    npy_int64 power_product_max_y = 0,
    npy_int64 power_product_max_z = 0,
    npy_intp product_factor_current_stamp = 0) {
    const npy_int64 group_begin = pair_group_starts_ptr[pair_idx];
    const npy_int64 group_end = pair_group_starts_ptr[pair_idx + 1];
    for (npy_int64 group = group_begin; group < group_end; ++group) {
        if (group < 0 || group >= ngroup) {
            continue;
        }
        const npy_int64 image = group_image_ptr[group];
        const npy_int64 product_begin = group_product_start_ptr[group];
        const npy_int64 product_end = group_product_stop_ptr[group];
        double image_sum_real = 0.0;
        double image_sum_imag = 0.0;
        for (npy_int64 product = product_begin; product < product_end; ++product) {
            if (product < 0 || product >= nproduct) {
                continue;
            }
            accumulate_product_group(
                product_term_start_ptr[product],
                product_term_stop_ptr[product],
                nterm,
                center_ptr,
                inv_4p_ptr,
                coeff_ptr,
                power_ptr,
                gxv,
                gyv,
                gzv,
                g2,
                plane_z,
                all_zero_powers,
                gx_power_real,
                gx_power_imag,
                gy_power_real,
                gy_power_imag,
                gz_power_real,
                gz_power_imag,
                image_sum_real,
                image_sum_imag,
                product_factor_id_ptr == nullptr ? -1 : product_factor_id_ptr[product],
                product_factor_seen,
                product_factor_real,
                product_factor_imag,
                scale_cache_inv_4p,
                scale_cache_values,
                power_product_real,
                power_product_imag,
                power_product_max_x,
                power_product_max_y,
                power_product_max_z,
                product_factor_current_stamp);
        }
        accumulate_image(image, image_sum_real, image_sum_imag);
    }
}

template <typename AccumulateImage>
inline void for_each_image_group_sum_zero_power(
    npy_intp pair_idx,
    const npy_int64* pair_group_starts_ptr,
    const npy_int64* group_image_ptr,
    const npy_int64* group_product_start_ptr,
    const npy_int64* group_product_stop_ptr,
    const npy_int64* product_term_start_ptr,
    const npy_int64* product_factor_id_ptr,
    npy_intp ngroup,
    npy_intp nproduct,
    npy_intp nterm,
    const double* center_ptr,
    const double* inv_4p_ptr,
    const double* product_coeff_sum_ptr,
    double gxv,
    double gyv,
    double gzv,
    double g2,
    AccumulateImage accumulate_image,
    std::vector<npy_intp>* product_factor_seen = nullptr,
    std::vector<double>* product_factor_real = nullptr,
    std::vector<double>* product_factor_imag = nullptr,
    npy_intp product_factor_current_stamp = 0) {
    const npy_int64 group_begin = pair_group_starts_ptr[pair_idx];
    const npy_int64 group_end = pair_group_starts_ptr[pair_idx + 1];
    for (npy_int64 group = group_begin; group < group_end; ++group) {
        if (group < 0 || group >= ngroup) {
            continue;
        }
        const npy_int64 image = group_image_ptr[group];
        const npy_int64 product_begin = group_product_start_ptr[group];
        const npy_int64 product_end = group_product_stop_ptr[group];
        double image_sum_real = 0.0;
        double image_sum_imag = 0.0;
        for (npy_int64 product = product_begin; product < product_end; ++product) {
            if (product < 0 || product >= nproduct) {
                continue;
            }
            const npy_int64 first_term = product_term_start_ptr[product];
            if (first_term < 0 || first_term >= nterm) {
                continue;
            }
            const double coeff = product_coeff_sum_ptr[product];
            if (coeff == 0.0) {
                continue;
            }

            double factor_real = 0.0;
            double factor_imag = 0.0;
            const npy_int64 product_factor_id =
                product_factor_id_ptr == nullptr ? -1 : product_factor_id_ptr[product];
            const bool use_product_factor_cache =
                product_factor_real != nullptr
                && product_factor_imag != nullptr
                && product_factor_id >= 0
                && static_cast<size_t>(product_factor_id) < product_factor_real->size()
                && static_cast<size_t>(product_factor_id) < product_factor_imag->size()
                && (
                    product_factor_seen == nullptr
                    || static_cast<size_t>(product_factor_id) < product_factor_seen->size());
            if (use_product_factor_cache) {
                const size_t factor_index = static_cast<size_t>(product_factor_id);
                if (product_factor_seen != nullptr
                    && (*product_factor_seen)[factor_index] != product_factor_current_stamp) {
                    gaussian_product_plane_factor(
                        first_term,
                        center_ptr,
                        inv_4p_ptr,
                        gxv,
                        gyv,
                        gzv,
                        g2,
                        (*product_factor_real)[factor_index],
                        (*product_factor_imag)[factor_index]);
                    (*product_factor_seen)[factor_index] = product_factor_current_stamp;
                }
                factor_real = (*product_factor_real)[factor_index];
                factor_imag = (*product_factor_imag)[factor_index];
            } else {
                gaussian_product_plane_factor(
                    first_term,
                    center_ptr,
                    inv_4p_ptr,
                    gxv,
                gyv,
                gzv,
                g2,
                factor_real,
                factor_imag);
            }
            image_sum_real += factor_real * coeff;
            image_sum_imag += factor_imag * coeff;
        }
        accumulate_image(image, image_sum_real, image_sum_imag);
    }
}

bool expect_ndim(PyArrayObject* arr, int ndim, const char* name) {
    if (PyArray_NDIM(arr) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions.", name, ndim);
        return false;
    }
    return true;
}

bool expect_last_dim3(PyArrayObject* arr, const char* name) {
    const int ndim = PyArray_NDIM(arr);
    if (ndim < 1 || PyArray_DIM(arr, ndim - 1) != 3) {
        PyErr_Format(PyExc_ValueError, "%s must have trailing dimension 3.", name);
        return false;
    }
    return true;
}

PyObject* gaussian_ft_batch(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* shells_obj = nullptr;
    PyObject* origins_obj = nullptr;
    PyObject* exps_obj = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* nprim_obj = nullptr;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOO|i",
            &gvecs_obj,
            &shells_obj,
            &origins_obj,
            &exps_obj,
            &weights_obj,
            &nprim_obj,
            &explicit_threads)) {
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef shells(shells_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef origins(origins_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef exps(exps_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef weights(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef nprim(nprim_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !shells || !origins || !exps || !weights || !nprim) {
        return nullptr;
    }
    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(shells.obj, 2, "shells") || !expect_last_dim3(shells.obj, "shells")
        || !expect_ndim(origins.obj, 2, "origins") || !expect_last_dim3(origins.obj, "origins")
        || !expect_ndim(exps.obj, 2, "exps")
        || !expect_ndim(weights.obj, 2, "weights")
        || !expect_ndim(nprim.obj, 1, "nprim")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp nfn = PyArray_DIM(shells.obj, 0);
    const npy_intp max_prim = PyArray_DIM(exps.obj, 1);
    if (PyArray_DIM(origins.obj, 0) != nfn || PyArray_DIM(exps.obj, 0) != nfn
        || PyArray_DIM(weights.obj, 0) != nfn || PyArray_DIM(nprim.obj, 0) != nfn
        || PyArray_DIM(weights.obj, 1) != max_prim) {
        PyErr_SetString(
            PyExc_ValueError,
            "shells, origins, exps, weights, and nprim have incompatible shapes.");
        return nullptr;
    }

    const npy_int64* nprim_ptr = int64_data(nprim.obj);
    for (npy_intp fn = 0; fn < nfn; ++fn) {
        if (nprim_ptr[fn] < 0 || nprim_ptr[fn] > max_prim) {
            PyErr_SetString(PyExc_ValueError, "nprim contains an out-of-range count.");
            return nullptr;
        }
    }

    npy_intp dims[2] = {ng, nfn};
    PyObject* out_obj = PyArray_ZEROS(2, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));

    const double* gvecs_ptr = double_data(gvecs.obj);
    const npy_int64* shells_ptr = int64_data(shells.obj);
    const double* origins_ptr = double_data(origins.obj);
    const double* exps_ptr = double_data(exps.obj);
    const double* weights_ptr = double_data(weights.obj);

    const npy_intp total_g = ng;
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, total_g))));

    auto worker = [&](npy_intp begin_g, npy_intp end_g) {
        std::vector<double> mx_real;
        std::vector<double> mx_imag;
        std::vector<double> my_real;
        std::vector<double> my_imag;
        std::vector<double> mz_real;
        std::vector<double> mz_imag;

        for (npy_intp gidx = begin_g; gidx < end_g; ++gidx) {
            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];

            for (npy_intp fn = 0; fn < nfn; ++fn) {
                const npy_int64 lx = shells_ptr[3 * fn + 0];
                const npy_int64 ly = shells_ptr[3 * fn + 1];
                const npy_int64 lz = shells_ptr[3 * fn + 2];
                if (lx < 0 || ly < 0 || lz < 0) {
                    continue;
                }
                double value_real = 0.0;
                double value_imag = 0.0;
                for (npy_int64 prim = 0; prim < nprim_ptr[fn]; ++prim) {
                    const npy_intp prim_index = fn * max_prim + prim;
                    const double exponent = exps_ptr[prim_index];
                    const double weight = weights_ptr[prim_index];
                    if (exponent <= 0.0 || weight == 0.0) {
                        continue;
                    }
                    fill_gaussian_moment_table(lx, gxv, exponent, mx_real, mx_imag);
                    fill_gaussian_moment_table(ly, gyv, exponent, my_real, my_imag);
                    fill_gaussian_moment_table(lz, gzv, exponent, mz_real, mz_imag);

                    double xy_real = 0.0;
                    double xy_imag = 0.0;
                    complex_mul(
                        mx_real[static_cast<size_t>(lx)],
                        mx_imag[static_cast<size_t>(lx)],
                        my_real[static_cast<size_t>(ly)],
                        my_imag[static_cast<size_t>(ly)],
                        xy_real,
                        xy_imag);
                    double xyz_real = 0.0;
                    double xyz_imag = 0.0;
                    complex_mul(
                        xy_real,
                        xy_imag,
                        mz_real[static_cast<size_t>(lz)],
                        mz_imag[static_cast<size_t>(lz)],
                        xyz_real,
                        xyz_imag);
                    value_real += weight * xyz_real;
                    value_imag += weight * xyz_imag;
                }

                const double phase_arg =
                    gxv * origins_ptr[3 * fn + 0]
                    + gyv * origins_ptr[3 * fn + 1]
                    + gzv * origins_ptr[3 * fn + 2];
                double sin_phase = 0.0;
                double cos_phase = 0.0;
                sincos_values(phase_arg, sin_phase, cos_phase);
                const double out_real = cos_phase * value_real + sin_phase * value_imag;
                const double out_imag = cos_phase * value_imag - sin_phase * value_real;
                store_complex(out_ptr[gidx * nfn + fn], out_real, out_imag);
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, total_g);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (total_g * tid) / nthreads;
            const npy_intp end = (total_g * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* cartesian_shell_transform(PyObject*, PyObject* args) {
    PyObject* cart_ft_obj = nullptr;
    PyObject* transform_obj = nullptr;
    PyObject* cart_start_obj = nullptr;
    PyObject* cart_stop_obj = nullptr;
    PyObject* aux_start_obj = nullptr;
    PyObject* aux_stop_obj = nullptr;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOO|i",
            &cart_ft_obj,
            &transform_obj,
            &cart_start_obj,
            &cart_stop_obj,
            &aux_start_obj,
            &aux_stop_obj,
            &explicit_threads)) {
        return nullptr;
    }

    ArrayRef cart_ft(cart_ft_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef transform(transform_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef cart_start(cart_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef cart_stop(cart_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_start(aux_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef aux_stop(aux_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!cart_ft || !transform || !cart_start || !cart_stop || !aux_start || !aux_stop) {
        return nullptr;
    }
    if (!expect_ndim(cart_ft.obj, 2, "cart_ft")
        || !expect_ndim(transform.obj, 2, "transform")
        || !expect_ndim(cart_start.obj, 1, "cart_start")
        || !expect_ndim(cart_stop.obj, 1, "cart_stop")
        || !expect_ndim(aux_start.obj, 1, "aux_start")
        || !expect_ndim(aux_stop.obj, 1, "aux_stop")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(cart_ft.obj, 0);
    const npy_intp ncart = PyArray_DIM(cart_ft.obj, 1);
    if (PyArray_DIM(transform.obj, 0) != ncart) {
        PyErr_SetString(PyExc_ValueError, "transform row count must match cart_ft columns.");
        return nullptr;
    }
    const npy_intp naux = PyArray_DIM(transform.obj, 1);
    const npy_intp nblock = PyArray_DIM(cart_start.obj, 0);
    if (PyArray_DIM(cart_stop.obj, 0) != nblock
        || PyArray_DIM(aux_start.obj, 0) != nblock
        || PyArray_DIM(aux_stop.obj, 0) != nblock) {
        PyErr_SetString(PyExc_ValueError, "shell slice arrays must have the same length.");
        return nullptr;
    }

    const npy_int64* cart_start_ptr = int64_data(cart_start.obj);
    const npy_int64* cart_stop_ptr = int64_data(cart_stop.obj);
    const npy_int64* aux_start_ptr = int64_data(aux_start.obj);
    const npy_int64* aux_stop_ptr = int64_data(aux_stop.obj);
    std::vector<unsigned char> aux_seen(static_cast<size_t>(naux), 0);
    for (npy_intp block = 0; block < nblock; ++block) {
        const npy_int64 c0 = cart_start_ptr[block];
        const npy_int64 c1 = cart_stop_ptr[block];
        const npy_int64 a0 = aux_start_ptr[block];
        const npy_int64 a1 = aux_stop_ptr[block];
        if (c0 < 0 || c1 < c0 || c1 > ncart || a0 < 0 || a1 < a0 || a1 > naux) {
            PyErr_SetString(PyExc_ValueError, "shell transform slice is out of bounds.");
            return nullptr;
        }
        for (npy_int64 aux = a0; aux < a1; ++aux) {
            const size_t index = static_cast<size_t>(aux);
            if (aux_seen[index]) {
                PyErr_SetString(PyExc_ValueError, "auxiliary transform shell slices overlap.");
                return nullptr;
            }
            aux_seen[index] = 1;
        }
    }
    for (npy_intp aux = 0; aux < naux; ++aux) {
        if (!aux_seen[static_cast<size_t>(aux)]) {
            PyErr_SetString(PyExc_ValueError, "auxiliary transform shell slices must cover all columns.");
            return nullptr;
        }
    }

    npy_intp dims[2] = {ng, naux};
    PyObject* out_obj = PyArray_EMPTY(2, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);

    const npy_complex128* cart_ptr =
        reinterpret_cast<const npy_complex128*>(PyArray_DATA(cart_ft.obj));
    const double* transform_ptr = double_data(transform.obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, ng))));

    auto worker = [&](npy_intp begin_g, npy_intp end_g) {
        for (npy_intp gidx = begin_g; gidx < end_g; ++gidx) {
            const npy_complex128* cart_row = cart_ptr + gidx * ncart;
            npy_complex128* out_row = out_ptr + gidx * naux;
            for (npy_intp block = 0; block < nblock; ++block) {
                const npy_int64 c0 = cart_start_ptr[block];
                const npy_int64 c1 = cart_stop_ptr[block];
                const npy_int64 a0 = aux_start_ptr[block];
                const npy_int64 a1 = aux_stop_ptr[block];
                for (npy_int64 aux = a0; aux < a1; ++aux) {
                    double sum_real = 0.0;
                    double sum_imag = 0.0;
                    for (npy_int64 cart = c0; cart < c1; ++cart) {
                        const double coeff = transform_ptr[cart * naux + aux];
                        if (coeff == 0.0) {
                            continue;
                        }
                        const double* value = reinterpret_cast<const double*>(&cart_row[cart]);
                        sum_real += coeff * value[0];
                        sum_imag += coeff * value[1];
                    }
                    store_complex(out_row[aux], sum_real, sum_imag);
                }
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, ng);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (ng * tid) / nthreads;
            const npy_intp end = (ng * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* periodic_pair_ft_primitive_sum(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* pair_p_obj = nullptr;
    PyObject* pair_q_obj = nullptr;
    PyObject* starts_obj = nullptr;
    PyObject* term_image_obj = nullptr;
    PyObject* term_center_obj = nullptr;
    PyObject* term_inv_4p_obj = nullptr;
    PyObject* term_coeff_obj = nullptr;
    PyObject* term_power_obj = nullptr;
    PyObject* pair_group_starts_obj = nullptr;
    PyObject* group_image_obj = nullptr;
    PyObject* group_term_start_obj = nullptr;
    PyObject* group_term_stop_obj = nullptr;
    PyObject* group_product_start_obj = nullptr;
    PyObject* group_product_stop_obj = nullptr;
    PyObject* product_term_start_obj = nullptr;
    PyObject* product_term_stop_obj = nullptr;
    PyObject* product_factor_id_obj = nullptr;
    PyObject* phases_obj = nullptr;
    int nleft = 0;
    int nright = 0;
    int plane_z_int = 0;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOiiOOOOOOOOOOOOOOOOp|i",
            &gvecs_obj,
            &pair_p_obj,
            &pair_q_obj,
            &nleft,
            &nright,
            &starts_obj,
            &term_image_obj,
            &term_center_obj,
            &term_inv_4p_obj,
            &term_coeff_obj,
            &term_power_obj,
            &pair_group_starts_obj,
            &group_image_obj,
            &group_term_start_obj,
            &group_term_stop_obj,
            &group_product_start_obj,
            &group_product_stop_obj,
            &product_term_start_obj,
            &product_term_stop_obj,
            &product_factor_id_obj,
            &phases_obj,
            &plane_z_int,
            &explicit_threads)) {
        return nullptr;
    }
    if (nleft < 0 || nright < 0) {
        PyErr_SetString(PyExc_ValueError, "nleft and nright must be non-negative.");
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_p(pair_p_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_q(pair_q_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef starts(starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_image(term_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_center(term_center_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_inv_4p(term_inv_4p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_coeff(term_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_power(term_power_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_group_starts(pair_group_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_image(group_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_start(group_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_stop(group_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_start(group_product_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_stop(group_product_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_start(product_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_stop(product_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_factor_id(product_factor_id_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef phases(phases_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !pair_p || !pair_q || !starts || !term_image || !term_center
        || !term_inv_4p || !term_coeff || !term_power || !pair_group_starts
        || !group_image || !group_term_start || !group_term_stop
        || !group_product_start || !group_product_stop || !product_term_start
        || !product_term_stop || !product_factor_id || !phases) {
        return nullptr;
    }

    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(pair_p.obj, 1, "pair_p")
        || !expect_ndim(pair_q.obj, 1, "pair_q")
        || !expect_ndim(starts.obj, 1, "pair_term_starts")
        || !expect_ndim(term_image.obj, 1, "term_image")
        || !expect_ndim(term_center.obj, 2, "term_center")
        || !expect_last_dim3(term_center.obj, "term_center")
        || !expect_ndim(term_inv_4p.obj, 1, "term_inv_4p")
        || !expect_ndim(term_coeff.obj, 1, "term_coeff")
        || !expect_ndim(term_power.obj, 2, "term_power")
        || !expect_last_dim3(term_power.obj, "term_power")
        || !expect_ndim(pair_group_starts.obj, 1, "pair_image_group_starts")
        || !expect_ndim(group_image.obj, 1, "image_group_image")
        || !expect_ndim(group_term_start.obj, 1, "image_group_term_start")
        || !expect_ndim(group_term_stop.obj, 1, "image_group_term_stop")
        || !expect_ndim(group_product_start.obj, 1, "image_group_product_start")
        || !expect_ndim(group_product_stop.obj, 1, "image_group_product_stop")
        || !expect_ndim(product_term_start.obj, 1, "product_group_term_start")
        || !expect_ndim(product_term_stop.obj, 1, "product_group_term_stop")
        || !expect_ndim(product_factor_id.obj, 1, "product_group_factor_id")
        || !expect_ndim(phases.obj, 1, "phases")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_p.obj, 0);
    const npy_intp nterm = PyArray_DIM(term_image.obj, 0);
    const npy_intp ngroup = PyArray_DIM(group_image.obj, 0);
    const npy_intp nproduct = PyArray_DIM(product_term_start.obj, 0);
    const npy_intp nimage = PyArray_DIM(phases.obj, 0);
    if (PyArray_DIM(pair_q.obj, 0) != npair) {
        PyErr_SetString(PyExc_ValueError, "pair_p and pair_q must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(starts.obj, 0) != npair + 1) {
        PyErr_SetString(PyExc_ValueError, "pair_term_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(term_center.obj, 0) != nterm
        || PyArray_DIM(term_inv_4p.obj, 0) != nterm
        || PyArray_DIM(term_coeff.obj, 0) != nterm
        || PyArray_DIM(term_power.obj, 0) != nterm) {
        PyErr_SetString(PyExc_ValueError, "primitive term arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(pair_group_starts.obj, 0) != npair + 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "pair_image_group_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(group_term_start.obj, 0) != ngroup
        || PyArray_DIM(group_term_stop.obj, 0) != ngroup
        || PyArray_DIM(group_product_start.obj, 0) != ngroup
        || PyArray_DIM(group_product_stop.obj, 0) != ngroup) {
        PyErr_SetString(PyExc_ValueError, "image group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_term_stop.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "primitive-product group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_factor_id.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "product_group_factor_id must have length nproduct.");
        return nullptr;
    }
    const npy_int64* group_starts_ptr = int64_data(pair_group_starts.obj);
    const npy_int64* group_image_ptr = int64_data(group_image.obj);
    const npy_int64* group_term_start_ptr = int64_data(group_term_start.obj);
    const npy_int64* group_term_stop_ptr = int64_data(group_term_stop.obj);
    const npy_int64* group_product_start_ptr = int64_data(group_product_start.obj);
    const npy_int64* group_product_stop_ptr = int64_data(group_product_stop.obj);
    const npy_int64* product_term_start_ptr = int64_data(product_term_start.obj);
    const npy_int64* product_term_stop_ptr = int64_data(product_term_stop.obj);
    const npy_int64* product_factor_id_ptr = int64_data(product_factor_id.obj);
    if (!validate_image_group_metadata(
            group_starts_ptr,
            group_image_ptr,
            group_term_start_ptr,
            group_term_stop_ptr,
            group_product_start_ptr,
            group_product_stop_ptr,
            product_term_start_ptr,
            product_term_stop_ptr,
            npair,
            ngroup,
            nproduct,
            nterm,
            nimage)) {
        return nullptr;
    }
    npy_intp nfactor = 0;
    if (!validate_product_factor_ids(product_factor_id_ptr, nproduct, nfactor)) {
        return nullptr;
    }

    npy_intp dims[3] = {ng, static_cast<npy_intp>(nleft), static_cast<npy_intp>(nright)};
    PyObject* out_obj = PyArray_ZEROS(3, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));

    const double* gvecs_ptr = double_data(gvecs.obj);
    const npy_int64* pair_p_ptr = int64_data(pair_p.obj);
    const npy_int64* pair_q_ptr = int64_data(pair_q.obj);
    const npy_int64* starts_ptr = int64_data(starts.obj);
    const npy_int64* image_ptr = int64_data(term_image.obj);
    const double* center_ptr = double_data(term_center.obj);
    const double* inv_4p_ptr = double_data(term_inv_4p.obj);
    const double* coeff_ptr = double_data(term_coeff.obj);
    const npy_int64* power_ptr = int64_data(term_power.obj);
    const npy_complex128* phase_ptr = complex_data(phases.obj);
    const bool plane_z = plane_z_int != 0;
    const npy_intp total = npair * ng;
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, total))));
    npy_int64 max_power_x = 0;
    npy_int64 max_power_y = 0;
    npy_int64 max_power_z = 0;
    bool all_zero_powers = true;
    for (npy_intp term = 0; term < nterm; ++term) {
        if (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
            || power_ptr[3 * term + 2] != 0) {
            all_zero_powers = false;
        }
        max_power_x = std::max(max_power_x, std::max<npy_int64>(0, power_ptr[3 * term + 0]));
        max_power_y = std::max(max_power_y, std::max<npy_int64>(0, power_ptr[3 * term + 1]));
        max_power_z = std::max(max_power_z, std::max<npy_int64>(0, power_ptr[3 * term + 2]));
    }
    const bool use_dynamic_pair_schedule =
        env_flag_enabled("PYQED_GDF_PAIR_DYNAMIC_SCHEDULE", false);
    const char* heavy_first_env = std::getenv("PYQED_GDF_PAIR_HEAVY_FIRST");
    bool use_heavy_first = false;
    if (heavy_first_env != nullptr && heavy_first_env[0] != '\0') {
        use_heavy_first = env_flag_enabled("PYQED_GDF_PAIR_HEAVY_FIRST", use_heavy_first);
    }
    use_heavy_first = use_dynamic_pair_schedule && use_heavy_first && nthreads > 1 && npair > 1;
    std::vector<npy_intp> pair_work_order;
    if (use_heavy_first) {
        pair_work_order.resize(static_cast<size_t>(npair));
        for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
            pair_work_order[static_cast<size_t>(pair_idx)] = pair_idx;
        }
        std::sort(
            pair_work_order.begin(),
            pair_work_order.end(),
            [&](npy_intp left, npy_intp right) {
                const npy_int64 left_terms = starts_ptr[left + 1] - starts_ptr[left];
                const npy_int64 right_terms = starts_ptr[right + 1] - starts_ptr[right];
                if (left_terms != right_terms) {
                    return left_terms > right_terms;
                }
                return left < right;
            });
    }
    auto worker = [&](npy_intp begin, npy_intp end) {
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;

        for (npy_intp flat = begin; flat < end; ++flat) {
            const npy_intp pair_idx = flat / ng;
            const npy_intp gidx = flat - pair_idx * ng;
            const npy_int64 pidx = pair_p_ptr[pair_idx];
            const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
            if (pidx < 0 || pidx >= nleft || qlocal < 0 || qlocal >= nright) {
                continue;
            }

            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];
            const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
            if (max_power_x > 0) {
                fill_neg_i_power_table(gxv, max_power_x, gx_power_real, gx_power_imag);
            }
            if (max_power_y > 0) {
                fill_neg_i_power_table(gyv, max_power_y, gy_power_real, gy_power_imag);
            }
            if (max_power_z > 0) {
                fill_neg_i_power_table(gzv, max_power_z, gz_power_real, gz_power_imag);
            }
            double value_real = 0.0;
            double value_imag = 0.0;

            for_each_image_group_sum(
                pair_idx,
                group_starts_ptr,
                group_image_ptr,
                group_term_start_ptr,
                group_term_stop_ptr,
                group_product_start_ptr,
                group_product_stop_ptr,
                product_term_start_ptr,
                product_term_stop_ptr,
                product_factor_id_ptr,
                ngroup,
                nproduct,
                nterm,
                image_ptr,
                center_ptr,
                inv_4p_ptr,
                coeff_ptr,
                power_ptr,
                gxv,
                gyv,
                gzv,
                g2,
                plane_z,
                all_zero_powers,
                gx_power_real,
                gx_power_imag,
                gy_power_real,
                gy_power_imag,
                gz_power_real,
                gz_power_imag,
                [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                    const double* image_phase = reinterpret_cast<const double*>(&phase_ptr[image]);
                    value_real +=
                        image_phase[0] * image_sum_real - image_phase[1] * image_sum_imag;
                    value_imag +=
                        image_phase[0] * image_sum_imag + image_phase[1] * image_sum_real;
                });

            const npy_intp out_index =
                (gidx * static_cast<npy_intp>(nleft) + pidx) * static_cast<npy_intp>(nright)
                + qlocal;
            store_complex(out_ptr[out_index], value_real, value_imag);
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, total);
    } else if (use_dynamic_pair_schedule) {
        const long pair_threads = std::max<long>(
            1,
            std::min<long>(nthreads, static_cast<long>(std::max<npy_intp>(1, npair))));
        std::atomic<npy_intp> next_pair(0);
        auto dynamic_pair_worker = [&]() {
            while (true) {
                const npy_intp work_idx = next_pair.fetch_add(1, std::memory_order_relaxed);
                if (work_idx >= npair) {
                    break;
                }
                const npy_intp pair_idx = use_heavy_first
                    ? pair_work_order[static_cast<size_t>(work_idx)]
                    : work_idx;
                worker(pair_idx * ng, (pair_idx + 1) * ng);
            }
        };
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(pair_threads));
        for (long tid = 0; tid < pair_threads; ++tid) {
            threads.emplace_back(dynamic_pair_worker);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (total * tid) / nthreads;
            const npy_intp end = (total * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* periodic_pair_ft_primitive_sum_many(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* pair_p_obj = nullptr;
    PyObject* pair_q_obj = nullptr;
    PyObject* starts_obj = nullptr;
    PyObject* term_image_obj = nullptr;
    PyObject* term_center_obj = nullptr;
    PyObject* term_inv_4p_obj = nullptr;
    PyObject* term_coeff_obj = nullptr;
    PyObject* term_power_obj = nullptr;
    PyObject* pair_group_starts_obj = nullptr;
    PyObject* group_image_obj = nullptr;
    PyObject* group_term_start_obj = nullptr;
    PyObject* group_term_stop_obj = nullptr;
    PyObject* group_product_start_obj = nullptr;
    PyObject* group_product_stop_obj = nullptr;
    PyObject* product_term_start_obj = nullptr;
    PyObject* product_term_stop_obj = nullptr;
    PyObject* product_factor_id_obj = nullptr;
    PyObject* phases_obj = nullptr;
    int nleft = 0;
    int nright = 0;
    int plane_z_int = 0;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOiiOOOOOOOOOOOOOOOOp|i",
            &gvecs_obj,
            &pair_p_obj,
            &pair_q_obj,
            &nleft,
            &nright,
            &starts_obj,
            &term_image_obj,
            &term_center_obj,
            &term_inv_4p_obj,
            &term_coeff_obj,
            &term_power_obj,
            &pair_group_starts_obj,
            &group_image_obj,
            &group_term_start_obj,
            &group_term_stop_obj,
            &group_product_start_obj,
            &group_product_stop_obj,
            &product_term_start_obj,
            &product_term_stop_obj,
            &product_factor_id_obj,
            &phases_obj,
            &plane_z_int,
            &explicit_threads)) {
        return nullptr;
    }
    if (nleft < 0 || nright < 0) {
        PyErr_SetString(PyExc_ValueError, "nleft and nright must be non-negative.");
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_p(pair_p_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_q(pair_q_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef starts(starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_image(term_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_center(term_center_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_inv_4p(term_inv_4p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_coeff(term_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_power(term_power_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_group_starts(pair_group_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_image(group_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_start(group_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_stop(group_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_start(group_product_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_stop(group_product_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_start(product_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_stop(product_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_factor_id(product_factor_id_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef phases(phases_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !pair_p || !pair_q || !starts || !term_image || !term_center
        || !term_inv_4p || !term_coeff || !term_power || !pair_group_starts
        || !group_image || !group_term_start || !group_term_stop
        || !group_product_start || !group_product_stop || !product_term_start
        || !product_term_stop || !product_factor_id || !phases) {
        return nullptr;
    }

    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(pair_p.obj, 1, "pair_p")
        || !expect_ndim(pair_q.obj, 1, "pair_q")
        || !expect_ndim(starts.obj, 1, "pair_term_starts")
        || !expect_ndim(term_image.obj, 1, "term_image")
        || !expect_ndim(term_center.obj, 2, "term_center")
        || !expect_last_dim3(term_center.obj, "term_center")
        || !expect_ndim(term_inv_4p.obj, 1, "term_inv_4p")
        || !expect_ndim(term_coeff.obj, 1, "term_coeff")
        || !expect_ndim(term_power.obj, 2, "term_power")
        || !expect_last_dim3(term_power.obj, "term_power")
        || !expect_ndim(pair_group_starts.obj, 1, "pair_image_group_starts")
        || !expect_ndim(group_image.obj, 1, "image_group_image")
        || !expect_ndim(group_term_start.obj, 1, "image_group_term_start")
        || !expect_ndim(group_term_stop.obj, 1, "image_group_term_stop")
        || !expect_ndim(group_product_start.obj, 1, "image_group_product_start")
        || !expect_ndim(group_product_stop.obj, 1, "image_group_product_stop")
        || !expect_ndim(product_term_start.obj, 1, "product_group_term_start")
        || !expect_ndim(product_term_stop.obj, 1, "product_group_term_stop")
        || !expect_ndim(product_factor_id.obj, 1, "product_group_factor_id")
        || !expect_ndim(phases.obj, 2, "phases")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_p.obj, 0);
    const npy_intp nterm = PyArray_DIM(term_image.obj, 0);
    const npy_intp nphase = PyArray_DIM(phases.obj, 0);
    const npy_intp nimage = PyArray_DIM(phases.obj, 1);
    const npy_intp ngroup = PyArray_DIM(group_image.obj, 0);
    const npy_intp nproduct = PyArray_DIM(product_term_start.obj, 0);
    if (PyArray_DIM(pair_q.obj, 0) != npair) {
        PyErr_SetString(PyExc_ValueError, "pair_p and pair_q must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(starts.obj, 0) != npair + 1) {
        PyErr_SetString(PyExc_ValueError, "pair_term_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(term_center.obj, 0) != nterm
        || PyArray_DIM(term_inv_4p.obj, 0) != nterm
        || PyArray_DIM(term_coeff.obj, 0) != nterm
        || PyArray_DIM(term_power.obj, 0) != nterm) {
        PyErr_SetString(PyExc_ValueError, "primitive term arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(pair_group_starts.obj, 0) != npair + 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "pair_image_group_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(group_term_start.obj, 0) != ngroup
        || PyArray_DIM(group_term_stop.obj, 0) != ngroup
        || PyArray_DIM(group_product_start.obj, 0) != ngroup
        || PyArray_DIM(group_product_stop.obj, 0) != ngroup) {
        PyErr_SetString(PyExc_ValueError, "image group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_term_stop.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "primitive-product group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_factor_id.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "product_group_factor_id must have length nproduct.");
        return nullptr;
    }
    const npy_int64* group_starts_ptr = int64_data(pair_group_starts.obj);
    const npy_int64* group_image_ptr = int64_data(group_image.obj);
    const npy_int64* group_term_start_ptr = int64_data(group_term_start.obj);
    const npy_int64* group_term_stop_ptr = int64_data(group_term_stop.obj);
    const npy_int64* group_product_start_ptr = int64_data(group_product_start.obj);
    const npy_int64* group_product_stop_ptr = int64_data(group_product_stop.obj);
    const npy_int64* product_term_start_ptr = int64_data(product_term_start.obj);
    const npy_int64* product_term_stop_ptr = int64_data(product_term_stop.obj);
    const npy_int64* product_factor_id_ptr = int64_data(product_factor_id.obj);
    if (!validate_image_group_metadata(
            group_starts_ptr,
            group_image_ptr,
            group_term_start_ptr,
            group_term_stop_ptr,
            group_product_start_ptr,
            group_product_stop_ptr,
            product_term_start_ptr,
            product_term_stop_ptr,
            npair,
            ngroup,
            nproduct,
            nterm,
            nimage)) {
        return nullptr;
    }
    npy_intp nfactor = 0;
    if (!validate_product_factor_ids(product_factor_id_ptr, nproduct, nfactor)) {
        return nullptr;
    }
    const bool use_precomputed_product_factors =
        nfactor > 0
        && nproduct >= 2 * nfactor
        && env_flag_enabled("PYQED_GDF_PRECOMPUTE_PRODUCT_FACTORS", true);
    const double* gvecs_ptr = double_data(gvecs.obj);
    const npy_int64* pair_p_ptr = int64_data(pair_p.obj);
    const npy_int64* pair_q_ptr = int64_data(pair_q.obj);
    const npy_int64* starts_ptr = int64_data(starts.obj);
    const npy_int64* image_ptr = int64_data(term_image.obj);
    const double* center_ptr = double_data(term_center.obj);
    const double* inv_4p_ptr = double_data(term_inv_4p.obj);
    const double* coeff_ptr = double_data(term_coeff.obj);
    const npy_int64* power_ptr = int64_data(term_power.obj);
    const npy_complex128* phase_ptr = complex_data(phases.obj);
    const bool plane_z = plane_z_int != 0;
    std::vector<npy_int64> product_factor_first_term(static_cast<size_t>(nfactor), -1);
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 factor_id = product_factor_id_ptr[product];
        if (factor_id < 0 || factor_id >= nfactor) {
            continue;
        }
        const size_t factor_index = static_cast<size_t>(factor_id);
        if (product_factor_first_term[factor_index] < 0) {
            product_factor_first_term[factor_index] = product_term_start_ptr[product];
        }
    }
    std::vector<double> product_factor_center_x(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_center_y(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_center_z(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_inv_4p(static_cast<size_t>(nfactor), 0.0);
    for (npy_intp factor = 0; factor < nfactor; ++factor) {
        const npy_int64 first_term =
            product_factor_first_term[static_cast<size_t>(factor)];
        if (first_term < 0 || first_term >= nterm) {
            continue;
        }
        const size_t factor_index = static_cast<size_t>(factor);
        product_factor_center_x[factor_index] = center_ptr[3 * first_term + 0];
        product_factor_center_y[factor_index] = center_ptr[3 * first_term + 1];
        product_factor_center_z[factor_index] = center_ptr[3 * first_term + 2];
        product_factor_inv_4p[factor_index] = inv_4p_ptr[first_term];
    }
    const npy_intp total_g = ng;
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, total_g))));
    npy_int64 max_power_x = 0;
    npy_int64 max_power_y = 0;
    npy_int64 max_power_z = 0;
    bool all_zero_powers = true;
    for (npy_intp term = 0; term < nterm; ++term) {
        if (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
            || power_ptr[3 * term + 2] != 0) {
            all_zero_powers = false;
        }
        max_power_x = std::max(max_power_x, std::max<npy_int64>(0, power_ptr[3 * term + 0]));
        max_power_y = std::max(max_power_y, std::max<npy_int64>(0, power_ptr[3 * term + 1]));
        max_power_z = std::max(max_power_z, std::max<npy_int64>(0, power_ptr[3 * term + 2]));
    }
    std::vector<double> product_coeff_sum(static_cast<size_t>(nproduct), 0.0);
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 term_begin = product_term_start_ptr[product];
        const npy_int64 term_end = product_term_stop_ptr[product];
        double coeff_sum = 0.0;
        for (npy_int64 term = term_begin; term < term_end; ++term) {
            if (term >= 0 && term < nterm) {
                coeff_sum += coeff_ptr[term];
            }
        }
        product_coeff_sum[static_cast<size_t>(product)] = coeff_sum;
    }
    std::vector<unsigned char> pair_has_power(static_cast<size_t>(npair), 0);
    if (!all_zero_powers) {
        for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
            npy_int64 term = starts_ptr[pair_idx];
            const npy_int64 term_stop = starts_ptr[pair_idx + 1];
            while (term < term_stop) {
                if (term >= 0 && term < nterm
                    && (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
                        || power_ptr[3 * term + 2] != 0)) {
                    pair_has_power[static_cast<size_t>(pair_idx)] = 1;
                    break;
                }
                ++term;
            }
        }
    }
    std::vector<npy_intp> active_pairs;
    active_pairs.reserve(static_cast<size_t>(npair));
    for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
        const npy_int64 pidx = pair_p_ptr[pair_idx];
        const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
        if (pidx < 0 || pidx >= nleft || qlocal < 0 || qlocal >= nright) {
            continue;
        }
        if (group_starts_ptr[pair_idx] == group_starts_ptr[pair_idx + 1]) {
            continue;
        }
        active_pairs.push_back(pair_idx);
    }
    npy_intp dims[4] = {
        nphase,
        ng,
        static_cast<npy_intp>(nleft),
        static_cast<npy_intp>(nright),
    };
    PyObject* out_obj = PyArray_ZEROS(4, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));

    const bool use_power_product_cache = power_product_cache_enabled(
        all_zero_powers,
        max_power_x,
        max_power_y,
        max_power_z);
    std::vector<double> phase_image_real(static_cast<size_t>(nimage * nphase));
    std::vector<double> phase_image_imag(static_cast<size_t>(nimage * nphase));
    for (npy_intp image = 0; image < nimage; ++image) {
        const size_t image_offset = static_cast<size_t>(image * nphase);
        for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
            const double* phase_value =
                reinterpret_cast<const double*>(&phase_ptr[iphase * nimage + image]);
            const size_t phase_index = image_offset + static_cast<size_t>(iphase);
            phase_image_real[phase_index] = phase_value[0];
            phase_image_imag[phase_index] = phase_value[1];
        }
    }
    auto worker = [&](npy_intp begin_g, npy_intp end_g) {
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;
        std::vector<double> value_real(static_cast<size_t>(nphase));
        std::vector<double> value_imag(static_cast<size_t>(nphase));
        std::vector<npy_intp> product_factor_seen(
            static_cast<size_t>(nfactor),
            static_cast<npy_intp>(-1));
        std::vector<double> product_factor_real(static_cast<size_t>(nfactor));
        std::vector<double> product_factor_imag(static_cast<size_t>(nfactor));
        std::vector<double> power_product_real;
        std::vector<double> power_product_imag;

        for (npy_intp gidx = begin_g; gidx < end_g; ++gidx) {
            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];
            const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
            const npy_intp product_factor_current_stamp = gidx;
            if (use_precomputed_product_factors) {
                for (npy_intp factor = 0; factor < nfactor; ++factor) {
                    const npy_int64 first_term =
                        product_factor_first_term[static_cast<size_t>(factor)];
                    if (first_term < 0 || first_term >= nterm) {
                        product_factor_real[static_cast<size_t>(factor)] = 0.0;
                        product_factor_imag[static_cast<size_t>(factor)] = 0.0;
                        continue;
                    }
                    const size_t factor_index = static_cast<size_t>(factor);
                    gaussian_product_plane_factor_values(
                        product_factor_center_x[factor_index],
                        product_factor_center_y[factor_index],
                        product_factor_center_z[factor_index],
                        product_factor_inv_4p[factor_index],
                        gxv,
                        gyv,
                        gzv,
                        g2,
                        product_factor_real[factor_index],
                        product_factor_imag[factor_index]);
                }
            }
            if (use_power_product_cache) {
                fill_neg_i_power_table(gxv, max_power_x, gx_power_real, gx_power_imag);
                fill_neg_i_power_table(gyv, max_power_y, gy_power_real, gy_power_imag);
                fill_neg_i_power_table(gzv, max_power_z, gz_power_real, gz_power_imag);
                fill_power_product_table(
                    max_power_x,
                    max_power_y,
                    max_power_z,
                    gx_power_real,
                    gx_power_imag,
                    gy_power_real,
                    gy_power_imag,
                    gz_power_real,
                    gz_power_imag,
                    power_product_real,
                    power_product_imag);
            } else if (max_power_x > 0) {
                fill_neg_i_power_table(gxv, max_power_x, gx_power_real, gx_power_imag);
            }
            if (!use_power_product_cache && max_power_y > 0) {
                fill_neg_i_power_table(gyv, max_power_y, gy_power_real, gy_power_imag);
            }
            if (!use_power_product_cache && max_power_z > 0) {
                fill_neg_i_power_table(gzv, max_power_z, gz_power_real, gz_power_imag);
            }

            for (npy_intp active_idx : active_pairs) {
                const npy_intp pair_idx = active_idx;
                const npy_int64 pidx = pair_p_ptr[pair_idx];
                const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
                const bool pair_needs_power =
                    pair_has_power[static_cast<size_t>(pair_idx)] != 0;
                auto accumulate_images = [&](auto&& accumulate_image) {
                    if (!pair_needs_power) {
                        for_each_image_group_sum_zero_power(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            center_ptr,
                            inv_4p_ptr,
                            product_coeff_sum.data(),
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            accumulate_image,
                            use_precomputed_product_factors ? nullptr : &product_factor_seen,
                            &product_factor_real,
                            &product_factor_imag,
                            product_factor_current_stamp);
                    } else {
                        for_each_image_group_sum(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_term_start_ptr,
                            group_term_stop_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_term_stop_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            image_ptr,
                            center_ptr,
                            inv_4p_ptr,
                            coeff_ptr,
                            power_ptr,
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            plane_z,
                            !pair_needs_power,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            accumulate_image,
                            use_precomputed_product_factors ? nullptr : &product_factor_seen,
                            &product_factor_real,
                            &product_factor_imag,
                            nullptr,
                            nullptr,
                            use_power_product_cache ? &power_product_real : nullptr,
                            use_power_product_cache ? &power_product_imag : nullptr,
                            max_power_x,
                            max_power_y,
                            max_power_z,
                            product_factor_current_stamp);
                    }
                };

                if (nphase == 1) {
                    double value0_real = 0.0;
                    double value0_imag = 0.0;

                    accumulate_images(
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                            const size_t phase_offset = static_cast<size_t>(image);
                            const double p0_real = phase_image_real[phase_offset];
                            const double p0_imag = phase_image_imag[phase_offset];
                            value0_real += p0_real * image_sum_real - p0_imag * image_sum_imag;
                            value0_imag += p0_real * image_sum_imag + p0_imag * image_sum_real;
                        });

                    const npy_intp out_index0 =
                        (gidx * static_cast<npy_intp>(nleft) + pidx)
                        * static_cast<npy_intp>(nright)
                        + qlocal;
                    store_complex(out_ptr[out_index0], value0_real, value0_imag);
                    continue;
                }

                if (nphase == 2) {
                    double value0_real = 0.0;
                    double value0_imag = 0.0;
                    double value1_real = 0.0;
                    double value1_imag = 0.0;

                    accumulate_images(
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                            const size_t phase_offset = static_cast<size_t>(image * nphase);
                            const double p0_real = phase_image_real[phase_offset];
                            const double p0_imag = phase_image_imag[phase_offset];
                            const double p1_real = phase_image_real[phase_offset + 1];
                            const double p1_imag = phase_image_imag[phase_offset + 1];
                            value0_real += p0_real * image_sum_real - p0_imag * image_sum_imag;
                            value0_imag += p0_real * image_sum_imag + p0_imag * image_sum_real;
                            value1_real += p1_real * image_sum_real - p1_imag * image_sum_imag;
                            value1_imag += p1_real * image_sum_imag + p1_imag * image_sum_real;
                        });

                    const npy_intp out_index0 =
                        (gidx * static_cast<npy_intp>(nleft) + pidx)
                        * static_cast<npy_intp>(nright)
                        + qlocal;
                    const npy_intp out_index1 =
                        ((ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                        * static_cast<npy_intp>(nright)
                        + qlocal;
                    store_complex(out_ptr[out_index0], value0_real, value0_imag);
                    store_complex(out_ptr[out_index1], value1_real, value1_imag);
                    continue;
                }

                if (nphase == 4) {
                    double value0_real = 0.0;
                    double value0_imag = 0.0;
                    double value1_real = 0.0;
                    double value1_imag = 0.0;
                    double value2_real = 0.0;
                    double value2_imag = 0.0;
                    double value3_real = 0.0;
                    double value3_imag = 0.0;

                    accumulate_images(
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                            const size_t phase_offset = static_cast<size_t>(image * nphase);
                            const double p0_real = phase_image_real[phase_offset];
                            const double p0_imag = phase_image_imag[phase_offset];
                            const double p1_real = phase_image_real[phase_offset + 1];
                            const double p1_imag = phase_image_imag[phase_offset + 1];
                            const double p2_real = phase_image_real[phase_offset + 2];
                            const double p2_imag = phase_image_imag[phase_offset + 2];
                            const double p3_real = phase_image_real[phase_offset + 3];
                            const double p3_imag = phase_image_imag[phase_offset + 3];
                            value0_real += p0_real * image_sum_real - p0_imag * image_sum_imag;
                            value0_imag += p0_real * image_sum_imag + p0_imag * image_sum_real;
                            value1_real += p1_real * image_sum_real - p1_imag * image_sum_imag;
                            value1_imag += p1_real * image_sum_imag + p1_imag * image_sum_real;
                            value2_real += p2_real * image_sum_real - p2_imag * image_sum_imag;
                            value2_imag += p2_real * image_sum_imag + p2_imag * image_sum_real;
                            value3_real += p3_real * image_sum_real - p3_imag * image_sum_imag;
                            value3_imag += p3_real * image_sum_imag + p3_imag * image_sum_real;
                        });

                    const npy_intp out_index0 =
                        (gidx * static_cast<npy_intp>(nleft) + pidx)
                        * static_cast<npy_intp>(nright)
                        + qlocal;
                    const npy_intp out_index1 =
                        ((ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                        * static_cast<npy_intp>(nright)
                        + qlocal;
                    const npy_intp out_index2 =
                        (((2 * ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    const npy_intp out_index3 =
                        (((3 * ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    store_complex(out_ptr[out_index0], value0_real, value0_imag);
                    store_complex(out_ptr[out_index1], value1_real, value1_imag);
                    store_complex(out_ptr[out_index2], value2_real, value2_imag);
                    store_complex(out_ptr[out_index3], value3_real, value3_imag);
                    continue;
                }

                if ((nphase > 0 && nphase <= 4) || (nphase >= 8 && nphase <= 32)) {
                    double value_real_small[32] = {};
                    double value_imag_small[32] = {};

                    accumulate_images(
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                            const size_t phase_offset = static_cast<size_t>(image * nphase);
                            for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                                const size_t phase_index =
                                    phase_offset + static_cast<size_t>(iphase);
                                const double phase_real = phase_image_real[phase_index];
                                const double phase_imag = phase_image_imag[phase_index];
                                value_real_small[iphase] +=
                                    phase_real * image_sum_real - phase_imag * image_sum_imag;
                                value_imag_small[iphase] +=
                                    phase_real * image_sum_imag + phase_imag * image_sum_real;
                            }
                        });

                    for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                        const npy_intp out_index =
                            (((iphase * ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                             * static_cast<npy_intp>(nright))
                            + qlocal;
                        store_complex(
                            out_ptr[out_index],
                            value_real_small[iphase],
                            value_imag_small[iphase]);
                    }
                    continue;
                }

                std::fill(value_real.begin(), value_real.end(), 0.0);
                std::fill(value_imag.begin(), value_imag.end(), 0.0);

                accumulate_images(
                    [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                        const size_t phase_offset = static_cast<size_t>(image * nphase);
                        for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                            const size_t phase_index =
                                phase_offset + static_cast<size_t>(iphase);
                            const double phase_real = phase_image_real[phase_index];
                            const double phase_imag = phase_image_imag[phase_index];
                            value_real[static_cast<size_t>(iphase)] +=
                                phase_real * image_sum_real - phase_imag * image_sum_imag;
                            value_imag[static_cast<size_t>(iphase)] +=
                                phase_real * image_sum_imag + phase_imag * image_sum_real;
                        }
                    });

                for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                    const npy_intp out_index =
                        (((iphase * ng + gidx) * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    store_complex(
                        out_ptr[out_index],
                        value_real[static_cast<size_t>(iphase)],
                        value_imag[static_cast<size_t>(iphase)]);
                }
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, total_g);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (total_g * tid) / nthreads;
            const npy_intp end = (total_g * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* periodic_pair_ft_primitive_image_group_sum(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* pair_p_obj = nullptr;
    PyObject* pair_q_obj = nullptr;
    PyObject* starts_obj = nullptr;
    PyObject* term_image_obj = nullptr;
    PyObject* term_center_obj = nullptr;
    PyObject* term_inv_4p_obj = nullptr;
    PyObject* term_coeff_obj = nullptr;
    PyObject* term_power_obj = nullptr;
    PyObject* pair_group_starts_obj = nullptr;
    PyObject* group_image_obj = nullptr;
    PyObject* group_term_start_obj = nullptr;
    PyObject* group_term_stop_obj = nullptr;
    PyObject* group_product_start_obj = nullptr;
    PyObject* group_product_stop_obj = nullptr;
    PyObject* product_term_start_obj = nullptr;
    PyObject* product_term_stop_obj = nullptr;
    PyObject* product_factor_id_obj = nullptr;
    int nleft = 0;
    int nright = 0;
    int plane_z_int = 0;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOiiOOOOOOOOOOOOOOOp|i",
            &gvecs_obj,
            &pair_p_obj,
            &pair_q_obj,
            &nleft,
            &nright,
            &starts_obj,
            &term_image_obj,
            &term_center_obj,
            &term_inv_4p_obj,
            &term_coeff_obj,
            &term_power_obj,
            &pair_group_starts_obj,
            &group_image_obj,
            &group_term_start_obj,
            &group_term_stop_obj,
            &group_product_start_obj,
            &group_product_stop_obj,
            &product_term_start_obj,
            &product_term_stop_obj,
            &product_factor_id_obj,
            &plane_z_int,
            &explicit_threads)) {
        return nullptr;
    }
    if (nleft < 0 || nright < 0) {
        PyErr_SetString(PyExc_ValueError, "nleft and nright must be non-negative.");
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_p(pair_p_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_q(pair_q_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef starts(starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_image(term_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_center(term_center_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_inv_4p(term_inv_4p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_coeff(term_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_power(term_power_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_group_starts(pair_group_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_image(group_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_start(group_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_stop(group_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_start(group_product_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_stop(group_product_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_start(product_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_stop(product_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_factor_id(product_factor_id_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !pair_p || !pair_q || !starts || !term_image || !term_center
        || !term_inv_4p || !term_coeff || !term_power || !pair_group_starts
        || !group_image || !group_term_start || !group_term_stop
        || !group_product_start || !group_product_stop || !product_term_start
        || !product_term_stop || !product_factor_id) {
        return nullptr;
    }

    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(pair_p.obj, 1, "pair_p")
        || !expect_ndim(pair_q.obj, 1, "pair_q")
        || !expect_ndim(starts.obj, 1, "pair_term_starts")
        || !expect_ndim(term_image.obj, 1, "term_image")
        || !expect_ndim(term_center.obj, 2, "term_center")
        || !expect_last_dim3(term_center.obj, "term_center")
        || !expect_ndim(term_inv_4p.obj, 1, "term_inv_4p")
        || !expect_ndim(term_coeff.obj, 1, "term_coeff")
        || !expect_ndim(term_power.obj, 2, "term_power")
        || !expect_last_dim3(term_power.obj, "term_power")
        || !expect_ndim(pair_group_starts.obj, 1, "pair_image_group_starts")
        || !expect_ndim(group_image.obj, 1, "image_group_image")
        || !expect_ndim(group_term_start.obj, 1, "image_group_term_start")
        || !expect_ndim(group_term_stop.obj, 1, "image_group_term_stop")
        || !expect_ndim(group_product_start.obj, 1, "image_group_product_start")
        || !expect_ndim(group_product_stop.obj, 1, "image_group_product_stop")
        || !expect_ndim(product_term_start.obj, 1, "product_group_term_start")
        || !expect_ndim(product_term_stop.obj, 1, "product_group_term_stop")
        || !expect_ndim(product_factor_id.obj, 1, "product_group_factor_id")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_p.obj, 0);
    const npy_intp nterm = PyArray_DIM(term_image.obj, 0);
    const npy_intp ngroup = PyArray_DIM(group_image.obj, 0);
    const npy_intp nproduct = PyArray_DIM(product_term_start.obj, 0);
    if (PyArray_DIM(pair_q.obj, 0) != npair) {
        PyErr_SetString(PyExc_ValueError, "pair_p and pair_q must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(starts.obj, 0) != npair + 1) {
        PyErr_SetString(PyExc_ValueError, "pair_term_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(term_center.obj, 0) != nterm
        || PyArray_DIM(term_inv_4p.obj, 0) != nterm
        || PyArray_DIM(term_coeff.obj, 0) != nterm
        || PyArray_DIM(term_power.obj, 0) != nterm) {
        PyErr_SetString(PyExc_ValueError, "primitive term arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(pair_group_starts.obj, 0) != npair + 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "pair_image_group_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(group_term_start.obj, 0) != ngroup
        || PyArray_DIM(group_term_stop.obj, 0) != ngroup
        || PyArray_DIM(group_product_start.obj, 0) != ngroup
        || PyArray_DIM(group_product_stop.obj, 0) != ngroup) {
        PyErr_SetString(PyExc_ValueError, "image group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_term_stop.obj, 0) != nproduct
        || PyArray_DIM(product_factor_id.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "primitive-product group arrays must have the same length.");
        return nullptr;
    }

    const npy_int64* group_starts_ptr = int64_data(pair_group_starts.obj);
    const npy_int64* group_image_ptr = int64_data(group_image.obj);
    const npy_int64* group_term_start_ptr = int64_data(group_term_start.obj);
    const npy_int64* group_term_stop_ptr = int64_data(group_term_stop.obj);
    const npy_int64* group_product_start_ptr = int64_data(group_product_start.obj);
    const npy_int64* group_product_stop_ptr = int64_data(group_product_stop.obj);
    const npy_int64* product_term_start_ptr = int64_data(product_term_start.obj);
    const npy_int64* product_term_stop_ptr = int64_data(product_term_stop.obj);
    const npy_int64* product_factor_id_ptr = int64_data(product_factor_id.obj);
    npy_int64 nimage = 0;
    for (npy_intp group = 0; group < ngroup; ++group) {
        nimage = std::max<npy_int64>(nimage, group_image_ptr[group] + 1);
    }
    if (!validate_image_group_metadata(
            group_starts_ptr,
            group_image_ptr,
            group_term_start_ptr,
            group_term_stop_ptr,
            group_product_start_ptr,
            group_product_stop_ptr,
            product_term_start_ptr,
            product_term_stop_ptr,
            npair,
            ngroup,
            nproduct,
            nterm,
            nimage)) {
        return nullptr;
    }
    npy_intp nfactor = 0;
    if (!validate_product_factor_ids(product_factor_id_ptr, nproduct, nfactor)) {
        return nullptr;
    }

    npy_intp dims[2] = {ng, ngroup};
    PyObject* out_obj = PyArray_ZEROS(2, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));

    const double* gvecs_ptr = double_data(gvecs.obj);
    const double* center_ptr = double_data(term_center.obj);
    const double* inv_4p_ptr = double_data(term_inv_4p.obj);
    const double* coeff_ptr = double_data(term_coeff.obj);
    const npy_int64* power_ptr = int64_data(term_power.obj);
    const bool plane_z = plane_z_int != 0;
    const bool use_precomputed_product_factors =
        nfactor > 0
        && nproduct >= 2 * nfactor
        && env_flag_enabled("PYQED_GDF_PRECOMPUTE_PRODUCT_FACTORS", true);

    npy_int64 max_power_x = 0;
    npy_int64 max_power_y = 0;
    npy_int64 max_power_z = 0;
    bool all_zero_powers = true;
    for (npy_intp term = 0; term < nterm; ++term) {
        const npy_int64 px = power_ptr[3 * term + 0];
        const npy_int64 py = power_ptr[3 * term + 1];
        const npy_int64 pz = power_ptr[3 * term + 2];
        if (px < 0 || py < 0 || pz < 0) {
            PyErr_SetString(PyExc_ValueError, "term_power contains a negative power.");
            Py_DECREF(out_obj);
            return nullptr;
        }
        if (px != 0 || py != 0 || pz != 0) {
            all_zero_powers = false;
        }
        max_power_x = std::max(max_power_x, px);
        max_power_y = std::max(max_power_y, py);
        max_power_z = std::max(max_power_z, pz);
    }
    const bool use_power_product_cache = power_product_cache_enabled(
        all_zero_powers,
        max_power_x,
        max_power_y,
        max_power_z);
    std::vector<double> product_coeff_sum(static_cast<size_t>(nproduct), 0.0);
    std::vector<npy_int64> product_factor_first_term(static_cast<size_t>(nfactor), -1);
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 term_begin = product_term_start_ptr[product];
        const npy_int64 term_end = product_term_stop_ptr[product];
        double coeff_sum = 0.0;
        for (npy_int64 term = term_begin; term < term_end; ++term) {
            if (term >= 0 && term < nterm) {
                coeff_sum += coeff_ptr[term];
            }
        }
        product_coeff_sum[static_cast<size_t>(product)] = coeff_sum;
        const npy_int64 factor_id = product_factor_id_ptr[product];
        if (factor_id >= 0 && factor_id < nfactor) {
            const size_t factor_index = static_cast<size_t>(factor_id);
            if (product_factor_first_term[factor_index] < 0) {
                product_factor_first_term[factor_index] = term_begin;
            }
        }
    }
    std::vector<double> product_factor_center_x(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_center_y(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_center_z(static_cast<size_t>(nfactor), 0.0);
    std::vector<double> product_factor_inv_4p(static_cast<size_t>(nfactor), 0.0);
    for (npy_intp factor = 0; factor < nfactor; ++factor) {
        const npy_int64 first_term =
            product_factor_first_term[static_cast<size_t>(factor)];
        if (first_term < 0 || first_term >= nterm) {
            continue;
        }
        const size_t factor_index = static_cast<size_t>(factor);
        product_factor_center_x[factor_index] = center_ptr[3 * first_term + 0];
        product_factor_center_y[factor_index] = center_ptr[3 * first_term + 1];
        product_factor_center_z[factor_index] = center_ptr[3 * first_term + 2];
        product_factor_inv_4p[factor_index] = inv_4p_ptr[first_term];
    }
    std::vector<unsigned char> group_has_power(static_cast<size_t>(ngroup), 0);
    if (!all_zero_powers) {
        for (npy_intp group = 0; group < ngroup; ++group) {
            for (npy_int64 term = group_term_start_ptr[group];
                 term < group_term_stop_ptr[group];
                 ++term) {
                if (term >= 0 && term < nterm
                    && (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
                        || power_ptr[3 * term + 2] != 0)) {
                    group_has_power[static_cast<size_t>(group)] = 1;
                    break;
                }
            }
        }
    }

    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, ng))));

    auto worker = [&](npy_intp begin_g, npy_intp end_g) {
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;
        std::vector<double> power_product_real;
        std::vector<double> power_product_imag;
        std::vector<npy_intp> product_factor_seen(
            static_cast<size_t>(nfactor),
            static_cast<npy_intp>(-1));
        std::vector<double> product_factor_real(static_cast<size_t>(nfactor));
        std::vector<double> product_factor_imag(static_cast<size_t>(nfactor));

        for (npy_intp gidx = begin_g; gidx < end_g; ++gidx) {
            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];
            const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
            if (use_precomputed_product_factors) {
                for (npy_intp factor = 0; factor < nfactor; ++factor) {
                    const npy_int64 first_term =
                        product_factor_first_term[static_cast<size_t>(factor)];
                    if (first_term < 0 || first_term >= nterm) {
                        product_factor_real[static_cast<size_t>(factor)] = 0.0;
                        product_factor_imag[static_cast<size_t>(factor)] = 0.0;
                        continue;
                    }
                    const size_t factor_index = static_cast<size_t>(factor);
                    gaussian_product_plane_factor_values(
                        product_factor_center_x[factor_index],
                        product_factor_center_y[factor_index],
                        product_factor_center_z[factor_index],
                        product_factor_inv_4p[factor_index],
                        gxv,
                        gyv,
                        gzv,
                        g2,
                        product_factor_real[factor_index],
                        product_factor_imag[factor_index]);
                }
            }
            if (!all_zero_powers) {
                fill_neg_i_power_table(gxv, max_power_x, gx_power_real, gx_power_imag);
                fill_neg_i_power_table(gyv, max_power_y, gy_power_real, gy_power_imag);
                fill_neg_i_power_table(gzv, max_power_z, gz_power_real, gz_power_imag);
                if (use_power_product_cache) {
                    fill_power_product_table(
                        max_power_x,
                        max_power_y,
                        max_power_z,
                        gx_power_real,
                        gx_power_imag,
                        gy_power_real,
                        gy_power_imag,
                        gz_power_real,
                        gz_power_imag,
                        power_product_real,
                        power_product_imag);
                }
            }

            const npy_intp product_factor_current_stamp = gidx;
            for (npy_intp group = 0; group < ngroup; ++group) {
                const npy_int64 product_begin = group_product_start_ptr[group];
                const npy_int64 product_end = group_product_stop_ptr[group];
                double image_sum_real = 0.0;
                double image_sum_imag = 0.0;
                if (group_has_power[static_cast<size_t>(group)] == 0) {
                    for (npy_int64 product = product_begin; product < product_end; ++product) {
                        const npy_int64 term = product_term_start_ptr[product];
                        if (term < 0 || term >= nterm) {
                            continue;
                        }
                        const npy_int64 factor_id = product_factor_id_ptr[product];
                        double factor_real = 0.0;
                        double factor_imag = 0.0;
                        const bool valid_factor =
                            factor_id >= 0
                            && static_cast<size_t>(factor_id) < product_factor_real.size();
                        if (use_precomputed_product_factors && valid_factor) {
                            factor_real = product_factor_real[static_cast<size_t>(factor_id)];
                            factor_imag = product_factor_imag[static_cast<size_t>(factor_id)];
                        } else if (valid_factor) {
                            const size_t factor_index = static_cast<size_t>(factor_id);
                            if (product_factor_seen[factor_index]
                                != product_factor_current_stamp) {
                                gaussian_product_plane_factor(
                                    term,
                                    center_ptr,
                                    inv_4p_ptr,
                                    gxv,
                                    gyv,
                                    gzv,
                                    g2,
                                    product_factor_real[factor_index],
                                    product_factor_imag[factor_index]);
                                product_factor_seen[factor_index] =
                                    product_factor_current_stamp;
                            }
                            factor_real = product_factor_real[factor_index];
                            factor_imag = product_factor_imag[factor_index];
                        } else {
                            gaussian_product_plane_factor(
                                term,
                                center_ptr,
                                inv_4p_ptr,
                                gxv,
                                gyv,
                                gzv,
                                g2,
                                factor_real,
                                factor_imag);
                        }
                        const double coeff_sum = product_coeff_sum[static_cast<size_t>(product)];
                        image_sum_real += factor_real * coeff_sum;
                        image_sum_imag += factor_imag * coeff_sum;
                    }
                } else {
                    for (npy_int64 product = product_begin; product < product_end; ++product) {
                        if (product < 0 || product >= nproduct) {
                            continue;
                        }
                        accumulate_product_group(
                            product_term_start_ptr[product],
                            product_term_stop_ptr[product],
                            nterm,
                            center_ptr,
                            inv_4p_ptr,
                            coeff_ptr,
                            power_ptr,
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            plane_z,
                            false,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            image_sum_real,
                            image_sum_imag,
                            product_factor_id_ptr[product],
                            use_precomputed_product_factors ? nullptr : &product_factor_seen,
                            &product_factor_real,
                            &product_factor_imag,
                            nullptr,
                            nullptr,
                            use_power_product_cache ? &power_product_real : nullptr,
                            use_power_product_cache ? &power_product_imag : nullptr,
                            max_power_x,
                            max_power_y,
                            max_power_z,
                            product_factor_current_stamp);
                    }
                }
                const npy_intp out_index = gidx * ngroup + group;
                store_complex(out_ptr[out_index], image_sum_real, image_sum_imag);
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, ng);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (ng * tid) / nthreads;
            const npy_intp end = (ng * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* periodic_pair_ft_product_sum_many(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* pair_p_obj = nullptr;
    PyObject* pair_q_obj = nullptr;
    PyObject* factor_center_obj = nullptr;
    PyObject* factor_inv_4p_obj = nullptr;
    PyObject* product_image_obj = nullptr;
    PyObject* product_factor_id_obj = nullptr;
    PyObject* product_entry_start_obj = nullptr;
    PyObject* product_entry_stop_obj = nullptr;
    PyObject* entry_pair_obj = nullptr;
    PyObject* entry_coeff_obj = nullptr;
    PyObject* entry_power_obj = nullptr;
    PyObject* phases_obj = nullptr;
    int nleft = 0;
    int nright = 0;
    int plane_z_int = 0;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOiiOOOOOOOOOOp|i",
            &gvecs_obj,
            &pair_p_obj,
            &pair_q_obj,
            &nleft,
            &nright,
            &factor_center_obj,
            &factor_inv_4p_obj,
            &product_image_obj,
            &product_factor_id_obj,
            &product_entry_start_obj,
            &product_entry_stop_obj,
            &entry_pair_obj,
            &entry_coeff_obj,
            &entry_power_obj,
            &phases_obj,
            &plane_z_int,
            &explicit_threads)) {
        return nullptr;
    }
    if (nleft < 0 || nright < 0) {
        PyErr_SetString(PyExc_ValueError, "nleft and nright must be non-negative.");
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_p(pair_p_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_q(pair_q_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef factor_center(factor_center_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef factor_inv_4p(factor_inv_4p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_image(product_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_factor_id(product_factor_id_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_entry_start(product_entry_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_entry_stop(product_entry_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef entry_pair(entry_pair_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef entry_coeff(entry_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef entry_power(entry_power_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef phases(phases_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !pair_p || !pair_q || !factor_center || !factor_inv_4p
        || !product_image || !product_factor_id || !product_entry_start
        || !product_entry_stop || !entry_pair || !entry_coeff || !entry_power
        || !phases) {
        return nullptr;
    }

    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(pair_p.obj, 1, "pair_p")
        || !expect_ndim(pair_q.obj, 1, "pair_q")
        || !expect_ndim(factor_center.obj, 2, "factor_center")
        || !expect_last_dim3(factor_center.obj, "factor_center")
        || !expect_ndim(factor_inv_4p.obj, 1, "factor_inv_4p")
        || !expect_ndim(product_image.obj, 1, "product_image")
        || !expect_ndim(product_factor_id.obj, 1, "product_factor_id")
        || !expect_ndim(product_entry_start.obj, 1, "product_entry_start")
        || !expect_ndim(product_entry_stop.obj, 1, "product_entry_stop")
        || !expect_ndim(entry_pair.obj, 1, "entry_pair")
        || !expect_ndim(entry_coeff.obj, 1, "entry_coeff")
        || !expect_ndim(entry_power.obj, 2, "entry_power")
        || !expect_last_dim3(entry_power.obj, "entry_power")
        || !expect_ndim(phases.obj, 2, "phases")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_p.obj, 0);
    const npy_intp nfactor = PyArray_DIM(factor_center.obj, 0);
    const npy_intp nproduct = PyArray_DIM(product_image.obj, 0);
    const npy_intp nentry = PyArray_DIM(entry_pair.obj, 0);
    const npy_intp nphase = PyArray_DIM(phases.obj, 0);
    const npy_intp nimage = PyArray_DIM(phases.obj, 1);

    if (PyArray_DIM(pair_q.obj, 0) != npair) {
        PyErr_SetString(PyExc_ValueError, "pair_p and pair_q must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(factor_inv_4p.obj, 0) != nfactor) {
        PyErr_SetString(PyExc_ValueError, "factor_center and factor_inv_4p are inconsistent.");
        return nullptr;
    }
    if (PyArray_DIM(product_factor_id.obj, 0) != nproduct
        || PyArray_DIM(product_entry_start.obj, 0) != nproduct
        || PyArray_DIM(product_entry_stop.obj, 0) != nproduct) {
        PyErr_SetString(PyExc_ValueError, "product arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(entry_coeff.obj, 0) != nentry
        || PyArray_DIM(entry_power.obj, 0) != nentry) {
        PyErr_SetString(PyExc_ValueError, "entry arrays must have the same length.");
        return nullptr;
    }

    const npy_int64* product_image_ptr = int64_data(product_image.obj);
    const npy_int64* product_factor_id_ptr = int64_data(product_factor_id.obj);
    const npy_int64* product_entry_start_ptr = int64_data(product_entry_start.obj);
    const npy_int64* product_entry_stop_ptr = int64_data(product_entry_stop.obj);
    const npy_int64* entry_pair_ptr = int64_data(entry_pair.obj);
    const npy_int64* entry_power_ptr = int64_data(entry_power.obj);
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 image = product_image_ptr[product];
        const npy_int64 factor = product_factor_id_ptr[product];
        const npy_int64 begin = product_entry_start_ptr[product];
        const npy_int64 end = product_entry_stop_ptr[product];
        if (image < 0 || image >= nimage || factor < 0 || factor >= nfactor) {
            PyErr_SetString(PyExc_ValueError, "product image or factor id is out of range.");
            return nullptr;
        }
        if (begin < 0 || end < begin || end > nentry) {
            PyErr_SetString(PyExc_ValueError, "invalid product entry span.");
            return nullptr;
        }
    }
    for (npy_intp entry = 0; entry < nentry; ++entry) {
        const npy_int64 pair_idx = entry_pair_ptr[entry];
        if (pair_idx < 0 || pair_idx >= npair) {
            PyErr_SetString(PyExc_ValueError, "entry_pair contains an out-of-range pair.");
            return nullptr;
        }
    }

    npy_intp dims[4] = {
        nphase,
        ng,
        static_cast<npy_intp>(nleft),
        static_cast<npy_intp>(nright),
    };
    PyObject* out_obj = PyArray_ZEROS(4, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));

    const double* gvecs_ptr = double_data(gvecs.obj);
    const npy_int64* pair_p_ptr = int64_data(pair_p.obj);
    const npy_int64* pair_q_ptr = int64_data(pair_q.obj);
    const double* factor_center_ptr = double_data(factor_center.obj);
    const double* factor_inv_4p_ptr = double_data(factor_inv_4p.obj);
    const double* entry_coeff_ptr = double_data(entry_coeff.obj);
    const npy_complex128* phase_ptr = complex_data(phases.obj);
    const bool plane_z = plane_z_int != 0;

    npy_int64 max_power_x = 0;
    npy_int64 max_power_y = 0;
    npy_int64 max_power_z = 0;
    bool all_zero_powers = true;
    for (npy_intp entry = 0; entry < nentry; ++entry) {
        const npy_int64 px = entry_power_ptr[3 * entry + 0];
        const npy_int64 py = entry_power_ptr[3 * entry + 1];
        const npy_int64 pz = entry_power_ptr[3 * entry + 2];
        if (px < 0 || py < 0 || pz < 0) {
            PyErr_SetString(PyExc_ValueError, "entry_power contains a negative power.");
            Py_DECREF(out_obj);
            return nullptr;
        }
        if (px != 0 || py != 0 || pz != 0) {
            all_zero_powers = false;
        }
        max_power_x = std::max(max_power_x, px);
        max_power_y = std::max(max_power_y, py);
        max_power_z = std::max(max_power_z, pz);
    }
    const bool use_power_product_cache = power_product_cache_enabled(
        all_zero_powers,
        max_power_x,
        max_power_y,
        max_power_z);

    std::vector<double> phase_image_real(static_cast<size_t>(nimage * nphase));
    std::vector<double> phase_image_imag(static_cast<size_t>(nimage * nphase));
    for (npy_intp image = 0; image < nimage; ++image) {
        const size_t image_offset = static_cast<size_t>(image * nphase);
        for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
            const double* phase_value =
                reinterpret_cast<const double*>(&phase_ptr[iphase * nimage + image]);
            const size_t phase_index = image_offset + static_cast<size_t>(iphase);
            phase_image_real[phase_index] = phase_value[0];
            phase_image_imag[phase_index] = phase_value[1];
        }
    }

    const npy_intp pair_stride =
        static_cast<npy_intp>(nleft) * static_cast<npy_intp>(nright);
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(requested_threads, static_cast<long>(std::max<npy_intp>(1, ng))));

    auto worker = [&](npy_intp begin_g, npy_intp end_g) {
        std::vector<double> factor_real(static_cast<size_t>(nfactor));
        std::vector<double> factor_imag(static_cast<size_t>(nfactor));
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;
        std::vector<double> power_product_real;
        std::vector<double> power_product_imag;

        for (npy_intp gidx = begin_g; gidx < end_g; ++gidx) {
            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];
            const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;

            for (npy_intp factor = 0; factor < nfactor; ++factor) {
                const double cx = factor_center_ptr[3 * factor + 0];
                const double cy = factor_center_ptr[3 * factor + 1];
                const double cz = factor_center_ptr[3 * factor + 2];
                const double phase_arg = gxv * cx + gyv * cy + gzv * cz;
                const double scale = std::exp(-g2 * factor_inv_4p_ptr[factor]);
                double sin_phase = 0.0;
                double cos_phase = 0.0;
                sincos_values(phase_arg, sin_phase, cos_phase);
                factor_real[static_cast<size_t>(factor)] = scale * cos_phase;
                factor_imag[static_cast<size_t>(factor)] = -scale * sin_phase;
            }

            if (!all_zero_powers) {
                fill_neg_i_power_table(gxv, max_power_x, gx_power_real, gx_power_imag);
                fill_neg_i_power_table(gyv, max_power_y, gy_power_real, gy_power_imag);
                fill_neg_i_power_table(gzv, max_power_z, gz_power_real, gz_power_imag);
                if (use_power_product_cache) {
                    fill_power_product_table(
                        max_power_x,
                        max_power_y,
                        max_power_z,
                        gx_power_real,
                        gx_power_imag,
                        gy_power_real,
                        gy_power_imag,
                        gz_power_real,
                        gz_power_imag,
                        power_product_real,
                        power_product_imag);
                }
            }

            for (npy_intp product = 0; product < nproduct; ++product) {
                const npy_int64 image = product_image_ptr[product];
                const npy_int64 factor = product_factor_id_ptr[product];
                const double base_real = factor_real[static_cast<size_t>(factor)];
                const double base_imag = factor_imag[static_cast<size_t>(factor)];
                if (base_real == 0.0 && base_imag == 0.0) {
                    continue;
                }
                const size_t phase_offset = static_cast<size_t>(image * nphase);
                const npy_int64 entry_begin = product_entry_start_ptr[product];
                const npy_int64 entry_end = product_entry_stop_ptr[product];
                for (npy_int64 entry = entry_begin; entry < entry_end; ++entry) {
                    const npy_int64 pair_idx = entry_pair_ptr[entry];
                    const npy_int64 pidx = pair_p_ptr[pair_idx];
                    const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
                    if (pidx < 0 || pidx >= nleft || qlocal < 0 || qlocal >= nright) {
                        continue;
                    }
                    const npy_int64 px = entry_power_ptr[3 * entry + 0];
                    const npy_int64 py = entry_power_ptr[3 * entry + 1];
                    const npy_int64 pz = entry_power_ptr[3 * entry + 2];
                    if (plane_z && pz != 0) {
                        continue;
                    }
                    double poly_real = 1.0;
                    double poly_imag = 0.0;
                    if (!all_zero_powers) {
                        if (use_power_product_cache) {
                            const size_t ny = static_cast<size_t>(max_power_y + 1);
                            const size_t nz = static_cast<size_t>(max_power_z + 1);
                            const size_t power_index =
                                (static_cast<size_t>(px) * ny + static_cast<size_t>(py)) * nz
                                + static_cast<size_t>(pz);
                            poly_real = power_product_real[power_index];
                            poly_imag = power_product_imag[power_index];
                        } else {
                            multiply_power_factor(
                                poly_real, poly_imag, gx_power_real, gx_power_imag, px);
                            multiply_power_factor(
                                poly_real, poly_imag, gy_power_real, gy_power_imag, py);
                            multiply_power_factor(
                                poly_real, poly_imag, gz_power_real, gz_power_imag, pz);
                        }
                    }
                    const double coeff = entry_coeff_ptr[entry];
                    double term_real = 0.0;
                    double term_imag = 0.0;
                    const double coeff_poly_real = coeff * poly_real;
                    const double coeff_poly_imag = coeff * poly_imag;
                    term_real =
                        base_real * coeff_poly_real - base_imag * coeff_poly_imag;
                    term_imag =
                        base_real * coeff_poly_imag + base_imag * coeff_poly_real;

                    const npy_intp pair_offset =
                        static_cast<npy_intp>(pidx) * static_cast<npy_intp>(nright)
                        + static_cast<npy_intp>(qlocal);
                    for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                        const size_t phase_index = phase_offset + static_cast<size_t>(iphase);
                        const double phase_real = phase_image_real[phase_index];
                        const double phase_imag = phase_image_imag[phase_index];
                        const double out_real =
                            phase_real * term_real - phase_imag * term_imag;
                        const double out_imag =
                            phase_real * term_imag + phase_imag * term_real;
                        const npy_intp out_index =
                            (iphase * ng + gidx) * pair_stride + pair_offset;
                        add_complex(out_ptr[out_index], out_real, out_imag);
                    }
                }
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, ng);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (ng * tid) / nthreads;
            const npy_intp end = (ng * (tid + 1)) / nthreads;
            threads.emplace_back(worker, begin, end);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyObject* periodic_pair_ft_primitive_contract_many(PyObject*, PyObject* args) {
    PyObject* gvecs_obj = nullptr;
    PyObject* pair_p_obj = nullptr;
    PyObject* pair_q_obj = nullptr;
    PyObject* starts_obj = nullptr;
    PyObject* term_image_obj = nullptr;
    PyObject* term_center_obj = nullptr;
    PyObject* term_inv_4p_obj = nullptr;
    PyObject* term_coeff_obj = nullptr;
    PyObject* term_power_obj = nullptr;
    PyObject* pair_group_starts_obj = nullptr;
    PyObject* group_image_obj = nullptr;
    PyObject* group_term_start_obj = nullptr;
    PyObject* group_term_stop_obj = nullptr;
    PyObject* group_product_start_obj = nullptr;
    PyObject* group_product_stop_obj = nullptr;
    PyObject* product_term_start_obj = nullptr;
    PyObject* product_term_stop_obj = nullptr;
    PyObject* product_factor_id_obj = nullptr;
    PyObject* phases_obj = nullptr;
    PyObject* weighted_aux_obj = nullptr;
    int nleft = 0;
    int nright = 0;
    int plane_z_int = 0;
    int explicit_threads = 0;

    if (!PyArg_ParseTuple(
            args,
            "OOOiiOOOOOOOOOOOOOOOOOp|i",
            &gvecs_obj,
            &pair_p_obj,
            &pair_q_obj,
            &nleft,
            &nright,
            &starts_obj,
            &term_image_obj,
            &term_center_obj,
            &term_inv_4p_obj,
            &term_coeff_obj,
            &term_power_obj,
            &pair_group_starts_obj,
            &group_image_obj,
            &group_term_start_obj,
            &group_term_stop_obj,
            &group_product_start_obj,
            &group_product_stop_obj,
            &product_term_start_obj,
            &product_term_stop_obj,
            &product_factor_id_obj,
            &phases_obj,
            &weighted_aux_obj,
            &plane_z_int,
            &explicit_threads)) {
        return nullptr;
    }
    if (nleft < 0 || nright < 0) {
        PyErr_SetString(PyExc_ValueError, "nleft and nright must be non-negative.");
        return nullptr;
    }

    ArrayRef gvecs(gvecs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_p(pair_p_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_q(pair_q_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef starts(starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_image(term_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_center(term_center_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_inv_4p(term_inv_4p_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_coeff(term_coeff_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    ArrayRef term_power(term_power_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef pair_group_starts(pair_group_starts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_image(group_image_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_start(group_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_term_stop(group_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_start(group_product_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef group_product_stop(group_product_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_start(product_term_start_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_term_stop(product_term_stop_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef product_factor_id(product_factor_id_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    ArrayRef phases(phases_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    ArrayRef weighted_aux(weighted_aux_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (!gvecs || !pair_p || !pair_q || !starts || !term_image || !term_center
        || !term_inv_4p || !term_coeff || !term_power || !pair_group_starts
        || !group_image || !group_term_start || !group_term_stop
        || !group_product_start || !group_product_stop || !product_term_start
        || !product_term_stop || !product_factor_id || !phases || !weighted_aux) {
        return nullptr;
    }

    if (!expect_ndim(gvecs.obj, 2, "gvecs") || !expect_last_dim3(gvecs.obj, "gvecs")
        || !expect_ndim(pair_p.obj, 1, "pair_p")
        || !expect_ndim(pair_q.obj, 1, "pair_q")
        || !expect_ndim(starts.obj, 1, "pair_term_starts")
        || !expect_ndim(term_image.obj, 1, "term_image")
        || !expect_ndim(term_center.obj, 2, "term_center")
        || !expect_last_dim3(term_center.obj, "term_center")
        || !expect_ndim(term_inv_4p.obj, 1, "term_inv_4p")
        || !expect_ndim(term_coeff.obj, 1, "term_coeff")
        || !expect_ndim(term_power.obj, 2, "term_power")
        || !expect_last_dim3(term_power.obj, "term_power")
        || !expect_ndim(pair_group_starts.obj, 1, "pair_image_group_starts")
        || !expect_ndim(group_image.obj, 1, "image_group_image")
        || !expect_ndim(group_term_start.obj, 1, "image_group_term_start")
        || !expect_ndim(group_term_stop.obj, 1, "image_group_term_stop")
        || !expect_ndim(group_product_start.obj, 1, "image_group_product_start")
        || !expect_ndim(group_product_stop.obj, 1, "image_group_product_stop")
        || !expect_ndim(product_term_start.obj, 1, "product_group_term_start")
        || !expect_ndim(product_term_stop.obj, 1, "product_group_term_stop")
        || !expect_ndim(product_factor_id.obj, 1, "product_group_factor_id")
        || !expect_ndim(phases.obj, 2, "phases")
        || !expect_ndim(weighted_aux.obj, 2, "weighted_aux")) {
        return nullptr;
    }

    const npy_intp ng = PyArray_DIM(gvecs.obj, 0);
    const npy_intp npair = PyArray_DIM(pair_p.obj, 0);
    const npy_intp nterm = PyArray_DIM(term_image.obj, 0);
    const npy_intp nphase = PyArray_DIM(phases.obj, 0);
    const npy_intp nimage = PyArray_DIM(phases.obj, 1);
    const npy_intp naux = PyArray_DIM(weighted_aux.obj, 1);
    const npy_intp ngroup = PyArray_DIM(group_image.obj, 0);
    const npy_intp nproduct = PyArray_DIM(product_term_start.obj, 0);
    if (PyArray_DIM(pair_q.obj, 0) != npair) {
        PyErr_SetString(PyExc_ValueError, "pair_p and pair_q must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(starts.obj, 0) != npair + 1) {
        PyErr_SetString(PyExc_ValueError, "pair_term_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(term_center.obj, 0) != nterm
        || PyArray_DIM(term_inv_4p.obj, 0) != nterm
        || PyArray_DIM(term_coeff.obj, 0) != nterm
        || PyArray_DIM(term_power.obj, 0) != nterm) {
        PyErr_SetString(PyExc_ValueError, "primitive term arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(pair_group_starts.obj, 0) != npair + 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "pair_image_group_starts must have length npair + 1.");
        return nullptr;
    }
    if (PyArray_DIM(group_term_start.obj, 0) != ngroup
        || PyArray_DIM(group_term_stop.obj, 0) != ngroup
        || PyArray_DIM(group_product_start.obj, 0) != ngroup
        || PyArray_DIM(group_product_stop.obj, 0) != ngroup) {
        PyErr_SetString(PyExc_ValueError, "image group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_term_stop.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "primitive-product group arrays must have the same length.");
        return nullptr;
    }
    if (PyArray_DIM(product_factor_id.obj, 0) != nproduct) {
        PyErr_SetString(
            PyExc_ValueError,
            "product_group_factor_id must have length nproduct.");
        return nullptr;
    }
    if (PyArray_DIM(weighted_aux.obj, 0) != ng) {
        PyErr_SetString(PyExc_ValueError, "weighted_aux must have shape (ng, naux).");
        return nullptr;
    }

    const npy_int64* group_starts_ptr = int64_data(pair_group_starts.obj);
    const npy_int64* group_image_ptr = int64_data(group_image.obj);
    const npy_int64* group_term_start_ptr = int64_data(group_term_start.obj);
    const npy_int64* group_term_stop_ptr = int64_data(group_term_stop.obj);
    const npy_int64* group_product_start_ptr = int64_data(group_product_start.obj);
    const npy_int64* group_product_stop_ptr = int64_data(group_product_stop.obj);
    const npy_int64* product_term_start_ptr = int64_data(product_term_start.obj);
    const npy_int64* product_term_stop_ptr = int64_data(product_term_stop.obj);
    const npy_int64* product_factor_id_ptr = int64_data(product_factor_id.obj);
    if (!validate_image_group_metadata(
            group_starts_ptr,
            group_image_ptr,
            group_term_start_ptr,
            group_term_stop_ptr,
            group_product_start_ptr,
            group_product_stop_ptr,
            product_term_start_ptr,
            product_term_stop_ptr,
            npair,
            ngroup,
            nproduct,
            nterm,
            nimage)) {
        return nullptr;
    }
    npy_intp nfactor = 0;
    if (!validate_product_factor_ids(product_factor_id_ptr, nproduct, nfactor)) {
        return nullptr;
    }

    npy_intp dims[4] = {
        nphase,
        naux,
        static_cast<npy_intp>(nleft),
        static_cast<npy_intp>(nright),
    };
    PyObject* out_obj = PyArray_ZEROS(4, dims, NPY_COMPLEX128, 0);
    if (out_obj == nullptr) {
        return nullptr;
    }
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(out_obj);

    const double* gvecs_ptr = double_data(gvecs.obj);
    const npy_int64* pair_p_ptr = int64_data(pair_p.obj);
    const npy_int64* pair_q_ptr = int64_data(pair_q.obj);
    const npy_int64* starts_ptr = int64_data(starts.obj);
    const npy_int64* image_ptr = int64_data(term_image.obj);
    const double* center_ptr = double_data(term_center.obj);
    const double* inv_4p_ptr = double_data(term_inv_4p.obj);
    const double* coeff_ptr = double_data(term_coeff.obj);
    const npy_int64* power_ptr = int64_data(term_power.obj);
    const npy_complex128* phase_ptr = complex_data(phases.obj);
    const npy_complex128* weighted_aux_ptr = complex_data(weighted_aux.obj);
    npy_complex128* out_ptr = reinterpret_cast<npy_complex128*>(PyArray_DATA(out));
    const bool plane_z = plane_z_int != 0;
    const bool use_precomputed_product_factors =
        nfactor > 0
        && nproduct >= 2 * nfactor
        && env_flag_enabled("PYQED_GDF_PRECOMPUTE_PRODUCT_FACTORS", true)
        && env_flag_enabled("PYQED_GDF_CONTRACT_PRODUCT_FACTORS", false);
    const npy_intp total_pair_g = npair * ng;
    const long requested_threads = selected_thread_count(explicit_threads);
    const long nthreads = std::max<long>(
        1,
        std::min<long>(
            requested_threads,
            static_cast<long>(std::max<npy_intp>(1, total_pair_g))));
    const bool use_pair_power_bounds =
        env_flag_enabled("PYQED_GDF_PAIR_POWER_BOUNDS", false);
    const char* heavy_first_env = std::getenv("PYQED_GDF_PAIR_HEAVY_FIRST");
    bool use_heavy_first = nthreads > 1 && npair > 1;
    if (heavy_first_env != nullptr && heavy_first_env[0] != '\0') {
        use_heavy_first = env_flag_enabled("PYQED_GDF_PAIR_HEAVY_FIRST", use_heavy_first);
    }
    std::vector<npy_intp> pair_work_order;
    if (use_heavy_first) {
        pair_work_order.resize(static_cast<size_t>(npair));
        for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
            pair_work_order[static_cast<size_t>(pair_idx)] = pair_idx;
        }
        std::sort(
            pair_work_order.begin(),
            pair_work_order.end(),
            [&](npy_intp left, npy_intp right) {
                const npy_int64 left_terms = starts_ptr[left + 1] - starts_ptr[left];
                const npy_int64 right_terms = starts_ptr[right + 1] - starts_ptr[right];
                if (left_terms != right_terms) {
                    return left_terms > right_terms;
                }
                return left < right;
            });
    }
    npy_int64 max_power_x = 0;
    npy_int64 max_power_y = 0;
    npy_int64 max_power_z = 0;
    bool all_zero_powers = true;
    for (npy_intp term = 0; term < nterm; ++term) {
        if (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
            || power_ptr[3 * term + 2] != 0) {
            all_zero_powers = false;
        }
        max_power_x = std::max(max_power_x, std::max<npy_int64>(0, power_ptr[3 * term + 0]));
        max_power_y = std::max(max_power_y, std::max<npy_int64>(0, power_ptr[3 * term + 1]));
        max_power_z = std::max(max_power_z, std::max<npy_int64>(0, power_ptr[3 * term + 2]));
    }
    const bool use_power_product_cache = power_product_cache_enabled(
        all_zero_powers,
        max_power_x,
        max_power_y,
        max_power_z);
    std::vector<double> product_coeff_sum(static_cast<size_t>(nproduct), 0.0);
    for (npy_intp product = 0; product < nproduct; ++product) {
        const npy_int64 term_begin = product_term_start_ptr[product];
        const npy_int64 term_end = product_term_stop_ptr[product];
        double coeff_sum = 0.0;
        for (npy_int64 term = term_begin; term < term_end; ++term) {
            if (term >= 0 && term < nterm) {
                coeff_sum += coeff_ptr[term];
            }
        }
        product_coeff_sum[static_cast<size_t>(product)] = coeff_sum;
    }
    std::vector<unsigned char> pair_has_power(static_cast<size_t>(npair), 0);
    std::vector<npy_int64> pair_max_power_x(static_cast<size_t>(npair), 0);
    std::vector<npy_int64> pair_max_power_y(static_cast<size_t>(npair), 0);
    std::vector<npy_int64> pair_max_power_z(static_cast<size_t>(npair), 0);
    if (!all_zero_powers && use_pair_power_bounds) {
        for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
            npy_int64 term = starts_ptr[pair_idx];
            const npy_int64 term_stop = starts_ptr[pair_idx + 1];
            while (term < term_stop) {
                if (term >= 0 && term < nterm) {
                    const npy_int64 px = std::max<npy_int64>(0, power_ptr[3 * term + 0]);
                    const npy_int64 py = std::max<npy_int64>(0, power_ptr[3 * term + 1]);
                    const npy_int64 pz = std::max<npy_int64>(0, power_ptr[3 * term + 2]);
                    if (px != 0 || py != 0 || pz != 0) {
                        pair_has_power[static_cast<size_t>(pair_idx)] = 1;
                    }
                    pair_max_power_x[static_cast<size_t>(pair_idx)] = std::max(
                        pair_max_power_x[static_cast<size_t>(pair_idx)], px);
                    pair_max_power_y[static_cast<size_t>(pair_idx)] = std::max(
                        pair_max_power_y[static_cast<size_t>(pair_idx)], py);
                    pair_max_power_z[static_cast<size_t>(pair_idx)] = std::max(
                        pair_max_power_z[static_cast<size_t>(pair_idx)], pz);
                }
                ++term;
            }
        }
    } else if (!all_zero_powers) {
        for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
            npy_int64 term = starts_ptr[pair_idx];
            const npy_int64 term_stop = starts_ptr[pair_idx + 1];
            while (term < term_stop) {
                if (term >= 0 && term < nterm
                    && (power_ptr[3 * term + 0] != 0 || power_ptr[3 * term + 1] != 0
                        || power_ptr[3 * term + 2] != 0)) {
                    pair_has_power[static_cast<size_t>(pair_idx)] = 1;
                    break;
                }
                ++term;
            }
        }
    }
    npy_int64 max_terms_per_pair = 0;
    for (npy_intp pair_idx = 0; pair_idx < npair; ++pair_idx) {
        max_terms_per_pair = std::max(
            max_terms_per_pair,
            starts_ptr[pair_idx + 1] - starts_ptr[pair_idx]);
    }
    const bool use_scale_cache =
        nimage > 1
        && max_terms_per_pair >= 16
        && env_flag_enabled("PYQED_GDF_PAIR_SCALE_CACHE", false);

    auto worker = [&](npy_intp begin, npy_intp end) {
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;
        std::vector<double> local_real(static_cast<size_t>(nphase * naux));
        std::vector<double> local_imag(static_cast<size_t>(nphase * naux));
        std::vector<double> value_real(static_cast<size_t>(nphase));
        std::vector<double> value_imag(static_cast<size_t>(nphase));
        std::vector<npy_intp> product_factor_seen(
            static_cast<size_t>(nfactor),
            static_cast<npy_intp>(-1));
        std::vector<double> product_factor_real(static_cast<size_t>(nfactor));
        std::vector<double> product_factor_imag(static_cast<size_t>(nfactor));
        std::vector<double> power_product_real;
        std::vector<double> power_product_imag;
        std::vector<double> scale_cache_inv_4p;
        std::vector<double> scale_cache_values;
        if (use_scale_cache) {
            scale_cache_inv_4p.reserve(64);
            scale_cache_values.reserve(64);
        }

        for (npy_intp pair_idx = begin; pair_idx < end; ++pair_idx) {
            const npy_int64 pidx = pair_p_ptr[pair_idx];
            const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
            if (pidx < 0 || pidx >= nleft || qlocal < 0 || qlocal >= nright) {
                continue;
            }
            const bool pair_needs_power =
                pair_has_power[static_cast<size_t>(pair_idx)] != 0;
            const npy_int64 pair_power_x =
                use_pair_power_bounds ? pair_max_power_x[static_cast<size_t>(pair_idx)] : max_power_x;
            const npy_int64 pair_power_y =
                use_pair_power_bounds ? pair_max_power_y[static_cast<size_t>(pair_idx)] : max_power_y;
            const npy_int64 pair_power_z =
                use_pair_power_bounds ? pair_max_power_z[static_cast<size_t>(pair_idx)] : max_power_z;

            std::fill(local_real.begin(), local_real.end(), 0.0);
            std::fill(local_imag.begin(), local_imag.end(), 0.0);

            if (nphase == 2) {
                for (npy_intp gidx = 0; gidx < ng; ++gidx) {
                    double value0_real = 0.0;
                    double value0_imag = 0.0;
                    double value1_real = 0.0;
                    double value1_imag = 0.0;

                    const double gxv = gvecs_ptr[3 * gidx + 0];
                    const double gyv = gvecs_ptr[3 * gidx + 1];
                    const double gzv = gvecs_ptr[3 * gidx + 2];
                    const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
                    if (use_scale_cache) {
                        scale_cache_inv_4p.clear();
                        scale_cache_values.clear();
                    }
                    if (pair_power_x > 0) {
                        fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                    }
                    if (pair_power_y > 0) {
                        fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                    }
                    if (pair_power_z > 0) {
                        fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                    }
                    if (pair_needs_power && use_power_product_cache) {
                        fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                        fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                        fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                        fill_power_product_table(
                            pair_power_x,
                            pair_power_y,
                            pair_power_z,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            power_product_real,
                            power_product_imag);
                    }

                    auto accumulate_image_phase2 =
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                            const double* phase0 =
                                reinterpret_cast<const double*>(&phase_ptr[image]);
                            const double* phase1 =
                                reinterpret_cast<const double*>(&phase_ptr[nimage + image]);
                            value0_real += phase0[0] * image_sum_real - phase0[1] * image_sum_imag;
                            value0_imag += phase0[0] * image_sum_imag + phase0[1] * image_sum_real;
                            value1_real += phase1[0] * image_sum_real - phase1[1] * image_sum_imag;
                            value1_imag += phase1[0] * image_sum_imag + phase1[1] * image_sum_real;
                        };
                    const npy_intp product_factor_current_stamp = pair_idx * ng + gidx;
                    if (!pair_needs_power) {
                        for_each_image_group_sum_zero_power(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            center_ptr,
                            inv_4p_ptr,
                            product_coeff_sum.data(),
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            accumulate_image_phase2,
                            use_precomputed_product_factors ? &product_factor_seen : nullptr,
                            use_precomputed_product_factors ? &product_factor_real : nullptr,
                            use_precomputed_product_factors ? &product_factor_imag : nullptr,
                            product_factor_current_stamp);
                    } else {
                        for_each_image_group_sum(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_term_start_ptr,
                            group_term_stop_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_term_stop_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            image_ptr,
                            center_ptr,
                            inv_4p_ptr,
                            coeff_ptr,
                            power_ptr,
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            plane_z,
                            false,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            accumulate_image_phase2,
                            use_precomputed_product_factors ? &product_factor_seen : nullptr,
                            use_precomputed_product_factors ? &product_factor_real : nullptr,
                            use_precomputed_product_factors ? &product_factor_imag : nullptr,
                            use_scale_cache ? &scale_cache_inv_4p : nullptr,
                            use_scale_cache ? &scale_cache_values : nullptr,
                            use_power_product_cache ? &power_product_real : nullptr,
                            use_power_product_cache ? &power_product_imag : nullptr,
                            pair_power_x,
                            pair_power_y,
                            pair_power_z,
                            product_factor_current_stamp);
                    }

                    for (npy_intp aux = 0; aux < naux; ++aux) {
                        const double* weighted =
                            reinterpret_cast<const double*>(&weighted_aux_ptr[gidx * naux + aux]);
                        const size_t local0 = static_cast<size_t>(aux);
                        const size_t local1 = static_cast<size_t>(naux + aux);
                        local_real[local0] += weighted[0] * value0_real - weighted[1] * value0_imag;
                        local_imag[local0] += weighted[0] * value0_imag + weighted[1] * value0_real;
                        local_real[local1] += weighted[0] * value1_real - weighted[1] * value1_imag;
                        local_imag[local1] += weighted[0] * value1_imag + weighted[1] * value1_real;
                    }
                }

                for (npy_intp aux = 0; aux < naux; ++aux) {
                    const npy_intp out_index0 =
                        ((aux * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    const npy_intp out_index1 =
                        (((naux + aux) * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    const size_t local0 = static_cast<size_t>(aux);
                    const size_t local1 = static_cast<size_t>(naux + aux);
                    store_complex(out_ptr[out_index0], local_real[local0], local_imag[local0]);
                    store_complex(out_ptr[out_index1], local_real[local1], local_imag[local1]);
                }
                continue;
            }

            if ((nphase > 0 && nphase <= 4) || (nphase >= 8 && nphase <= 32)) {
                for (npy_intp gidx = 0; gidx < ng; ++gidx) {
                    double value_real_small[32] = {};
                    double value_imag_small[32] = {};

                    const double gxv = gvecs_ptr[3 * gidx + 0];
                    const double gyv = gvecs_ptr[3 * gidx + 1];
                    const double gzv = gvecs_ptr[3 * gidx + 2];
                    const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
                    if (use_scale_cache) {
                        scale_cache_inv_4p.clear();
                        scale_cache_values.clear();
                    }
                    if (pair_power_x > 0) {
                        fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                    }
                    if (pair_power_y > 0) {
                        fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                    }
                    if (pair_power_z > 0) {
                        fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                    }
                    if (pair_needs_power && use_power_product_cache) {
                        fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                        fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                        fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                        fill_power_product_table(
                            pair_power_x,
                            pair_power_y,
                            pair_power_z,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            power_product_real,
                            power_product_imag);
                    }

                    auto accumulate_image_small =
                        [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                        for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                            const double* image_phase =
                                reinterpret_cast<const double*>(&phase_ptr[iphase * nimage + image]);
                            value_real_small[iphase] +=
                                image_phase[0] * image_sum_real - image_phase[1] * image_sum_imag;
                            value_imag_small[iphase] +=
                                image_phase[0] * image_sum_imag + image_phase[1] * image_sum_real;
                        }
                        };
                    const npy_intp product_factor_current_stamp = pair_idx * ng + gidx;
                    if (!pair_needs_power) {
                        for_each_image_group_sum_zero_power(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            center_ptr,
                            inv_4p_ptr,
                            product_coeff_sum.data(),
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            accumulate_image_small,
                            use_precomputed_product_factors ? &product_factor_seen : nullptr,
                            use_precomputed_product_factors ? &product_factor_real : nullptr,
                            use_precomputed_product_factors ? &product_factor_imag : nullptr,
                            product_factor_current_stamp);
                    } else {
                        for_each_image_group_sum(
                            pair_idx,
                            group_starts_ptr,
                            group_image_ptr,
                            group_term_start_ptr,
                            group_term_stop_ptr,
                            group_product_start_ptr,
                            group_product_stop_ptr,
                            product_term_start_ptr,
                            product_term_stop_ptr,
                            product_factor_id_ptr,
                            ngroup,
                            nproduct,
                            nterm,
                            image_ptr,
                            center_ptr,
                            inv_4p_ptr,
                            coeff_ptr,
                            power_ptr,
                            gxv,
                            gyv,
                            gzv,
                            g2,
                            plane_z,
                            false,
                            gx_power_real,
                            gx_power_imag,
                            gy_power_real,
                            gy_power_imag,
                            gz_power_real,
                            gz_power_imag,
                            accumulate_image_small,
                            use_precomputed_product_factors ? &product_factor_seen : nullptr,
                            use_precomputed_product_factors ? &product_factor_real : nullptr,
                            use_precomputed_product_factors ? &product_factor_imag : nullptr,
                            use_scale_cache ? &scale_cache_inv_4p : nullptr,
                            use_scale_cache ? &scale_cache_values : nullptr,
                            use_power_product_cache ? &power_product_real : nullptr,
                            use_power_product_cache ? &power_product_imag : nullptr,
                            pair_power_x,
                            pair_power_y,
                            pair_power_z,
                            product_factor_current_stamp);
                    }

                    for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                        const double pair_real = value_real_small[iphase];
                        const double pair_imag = value_imag_small[iphase];
                        for (npy_intp aux = 0; aux < naux; ++aux) {
                            const double* weighted =
                                reinterpret_cast<const double*>(&weighted_aux_ptr[gidx * naux + aux]);
                            const size_t local =
                                static_cast<size_t>(iphase * naux + aux);
                            local_real[local] +=
                                weighted[0] * pair_real - weighted[1] * pair_imag;
                            local_imag[local] +=
                                weighted[0] * pair_imag + weighted[1] * pair_real;
                        }
                    }
                }

                for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                    for (npy_intp aux = 0; aux < naux; ++aux) {
                        const npy_intp out_index =
                            (((iphase * naux + aux) * static_cast<npy_intp>(nleft) + pidx)
                             * static_cast<npy_intp>(nright))
                            + qlocal;
                        const size_t local = static_cast<size_t>(iphase * naux + aux);
                        store_complex(out_ptr[out_index], local_real[local], local_imag[local]);
                    }
                }
                continue;
            }

            for (npy_intp gidx = 0; gidx < ng; ++gidx) {
                std::fill(value_real.begin(), value_real.end(), 0.0);
                std::fill(value_imag.begin(), value_imag.end(), 0.0);

                const double gxv = gvecs_ptr[3 * gidx + 0];
                const double gyv = gvecs_ptr[3 * gidx + 1];
                const double gzv = gvecs_ptr[3 * gidx + 2];
                const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
                if (use_scale_cache) {
                    scale_cache_inv_4p.clear();
                    scale_cache_values.clear();
                }
                if (pair_power_x > 0) {
                    fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                }
                if (pair_power_y > 0) {
                    fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                }
                if (pair_power_z > 0) {
                    fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                }
                if (pair_needs_power && use_power_product_cache) {
                    fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                    fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                    fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                    fill_power_product_table(
                        pair_power_x,
                        pair_power_y,
                        pair_power_z,
                        gx_power_real,
                        gx_power_imag,
                        gy_power_real,
                        gy_power_imag,
                        gz_power_real,
                        gz_power_imag,
                        power_product_real,
                        power_product_imag);
                }

                auto accumulate_image_general =
                    [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                    for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                        const double* image_phase =
                            reinterpret_cast<const double*>(&phase_ptr[iphase * nimage + image]);
                        value_real[static_cast<size_t>(iphase)] +=
                            image_phase[0] * image_sum_real - image_phase[1] * image_sum_imag;
                        value_imag[static_cast<size_t>(iphase)] +=
                            image_phase[0] * image_sum_imag + image_phase[1] * image_sum_real;
                    }
                    };
                const npy_intp product_factor_current_stamp = pair_idx * ng + gidx;
                if (!pair_needs_power) {
                    for_each_image_group_sum_zero_power(
                        pair_idx,
                        group_starts_ptr,
                        group_image_ptr,
                        group_product_start_ptr,
                        group_product_stop_ptr,
                        product_term_start_ptr,
                        product_factor_id_ptr,
                        ngroup,
                        nproduct,
                        nterm,
                        center_ptr,
                        inv_4p_ptr,
                        product_coeff_sum.data(),
                        gxv,
                        gyv,
                        gzv,
                        g2,
                        accumulate_image_general,
                        use_precomputed_product_factors ? &product_factor_seen : nullptr,
                        use_precomputed_product_factors ? &product_factor_real : nullptr,
                        use_precomputed_product_factors ? &product_factor_imag : nullptr,
                        product_factor_current_stamp);
                } else {
                    for_each_image_group_sum(
                        pair_idx,
                        group_starts_ptr,
                        group_image_ptr,
                        group_term_start_ptr,
                        group_term_stop_ptr,
                        group_product_start_ptr,
                        group_product_stop_ptr,
                        product_term_start_ptr,
                        product_term_stop_ptr,
                        product_factor_id_ptr,
                        ngroup,
                        nproduct,
                        nterm,
                        image_ptr,
                        center_ptr,
                        inv_4p_ptr,
                        coeff_ptr,
                        power_ptr,
                        gxv,
                        gyv,
                        gzv,
                        g2,
                        plane_z,
                        false,
                        gx_power_real,
                        gx_power_imag,
                        gy_power_real,
                        gy_power_imag,
                        gz_power_real,
                        gz_power_imag,
                        accumulate_image_general,
                        use_precomputed_product_factors ? &product_factor_seen : nullptr,
                        use_precomputed_product_factors ? &product_factor_real : nullptr,
                        use_precomputed_product_factors ? &product_factor_imag : nullptr,
                        use_scale_cache ? &scale_cache_inv_4p : nullptr,
                        use_scale_cache ? &scale_cache_values : nullptr,
                        use_power_product_cache ? &power_product_real : nullptr,
                        use_power_product_cache ? &power_product_imag : nullptr,
                        pair_power_x,
                        pair_power_y,
                        pair_power_z,
                        product_factor_current_stamp);
                }

                for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                    const double pair_real = value_real[static_cast<size_t>(iphase)];
                    const double pair_imag = value_imag[static_cast<size_t>(iphase)];
                    for (npy_intp aux = 0; aux < naux; ++aux) {
                        const double* weighted =
                            reinterpret_cast<const double*>(&weighted_aux_ptr[gidx * naux + aux]);
                        const size_t local = static_cast<size_t>(iphase * naux + aux);
                        local_real[local] += weighted[0] * pair_real - weighted[1] * pair_imag;
                        local_imag[local] += weighted[0] * pair_imag + weighted[1] * pair_real;
                    }
                }
            }

            for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                for (npy_intp aux = 0; aux < naux; ++aux) {
                    const npy_intp out_index =
                        (((iphase * naux + aux) * static_cast<npy_intp>(nleft) + pidx)
                         * static_cast<npy_intp>(nright))
                        + qlocal;
                    const size_t local = static_cast<size_t>(iphase * naux + aux);
                    store_complex(out_ptr[out_index], local_real[local], local_imag[local]);
                }
            }
        }
    };

    const npy_intp out_size =
        nphase * naux * static_cast<npy_intp>(nleft) * static_cast<npy_intp>(nright);
    const ContractThreadStrategy thread_strategy = contract_thread_strategy();
    const bool pair_g_memory_ok =
        out_size > 0
        && (static_cast<unsigned long long>(out_size)
            * static_cast<unsigned long long>(nthreads)
            * static_cast<unsigned long long>(2 * sizeof(double))
            <= static_cast<unsigned long long>(512) * 1024ULL * 1024ULL);
    const bool auto_pair_g_threads = nthreads > npair;
    const bool use_pair_g_threads =
        nthreads > 1
        && pair_g_memory_ok
        && thread_strategy != ContractThreadStrategy::Pair
        && (thread_strategy == ContractThreadStrategy::PairG || auto_pair_g_threads);

    auto worker_pair_g = [&](
                             npy_intp begin,
                             npy_intp end,
                             std::vector<double>& partial_real,
                             std::vector<double>& partial_imag) {
        std::vector<double> gx_power_real;
        std::vector<double> gx_power_imag;
        std::vector<double> gy_power_real;
        std::vector<double> gy_power_imag;
        std::vector<double> gz_power_real;
        std::vector<double> gz_power_imag;
        std::vector<double> value_real(static_cast<size_t>(nphase));
        std::vector<double> value_imag(static_cast<size_t>(nphase));
        std::vector<npy_intp> product_factor_seen(
            static_cast<size_t>(nfactor),
            static_cast<npy_intp>(-1));
        std::vector<double> product_factor_real(static_cast<size_t>(nfactor));
        std::vector<double> product_factor_imag(static_cast<size_t>(nfactor));
        std::vector<double> power_product_real;
        std::vector<double> power_product_imag;
        std::vector<double> scale_cache_inv_4p;
        std::vector<double> scale_cache_values;
        if (use_scale_cache) {
            scale_cache_inv_4p.reserve(64);
            scale_cache_values.reserve(64);
        }

        for (npy_intp flat = begin; flat < end; ++flat) {
            const npy_intp pair_idx = flat / ng;
            const npy_intp gidx = flat - pair_idx * ng;
            const npy_int64 pidx = pair_p_ptr[pair_idx];
            const npy_int64 qlocal = pair_q_ptr[pair_idx] - nleft;
            if (pidx < 0 || pidx >= nleft || qlocal < 0 || qlocal >= nright) {
                continue;
            }
            const bool pair_needs_power =
                pair_has_power[static_cast<size_t>(pair_idx)] != 0;
            const npy_int64 pair_power_x =
                use_pair_power_bounds ? pair_max_power_x[static_cast<size_t>(pair_idx)] : max_power_x;
            const npy_int64 pair_power_y =
                use_pair_power_bounds ? pair_max_power_y[static_cast<size_t>(pair_idx)] : max_power_y;
            const npy_int64 pair_power_z =
                use_pair_power_bounds ? pair_max_power_z[static_cast<size_t>(pair_idx)] : max_power_z;

            std::fill(value_real.begin(), value_real.end(), 0.0);
            std::fill(value_imag.begin(), value_imag.end(), 0.0);

            const double gxv = gvecs_ptr[3 * gidx + 0];
            const double gyv = gvecs_ptr[3 * gidx + 1];
            const double gzv = gvecs_ptr[3 * gidx + 2];
            const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
            if (use_scale_cache) {
                scale_cache_inv_4p.clear();
                scale_cache_values.clear();
            }
            if (pair_power_x > 0) {
                fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
            }
            if (pair_power_y > 0) {
                fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
            }
            if (pair_power_z > 0) {
                fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
            }
            if (pair_needs_power && use_power_product_cache) {
                fill_neg_i_power_table(gxv, pair_power_x, gx_power_real, gx_power_imag);
                fill_neg_i_power_table(gyv, pair_power_y, gy_power_real, gy_power_imag);
                fill_neg_i_power_table(gzv, pair_power_z, gz_power_real, gz_power_imag);
                fill_power_product_table(
                    pair_power_x,
                    pair_power_y,
                    pair_power_z,
                    gx_power_real,
                    gx_power_imag,
                    gy_power_real,
                    gy_power_imag,
                    gz_power_real,
                    gz_power_imag,
                    power_product_real,
                    power_product_imag);
            }

            auto accumulate_image_pair_g =
                [&](npy_int64 image, double image_sum_real, double image_sum_imag) {
                for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                    const double* image_phase =
                        reinterpret_cast<const double*>(&phase_ptr[iphase * nimage + image]);
                    value_real[static_cast<size_t>(iphase)] +=
                        image_phase[0] * image_sum_real - image_phase[1] * image_sum_imag;
                    value_imag[static_cast<size_t>(iphase)] +=
                        image_phase[0] * image_sum_imag + image_phase[1] * image_sum_real;
                }
                };
            const npy_intp product_factor_current_stamp = pair_idx * ng + gidx;
            if (!pair_needs_power) {
                for_each_image_group_sum_zero_power(
                    pair_idx,
                    group_starts_ptr,
                    group_image_ptr,
                    group_product_start_ptr,
                    group_product_stop_ptr,
                    product_term_start_ptr,
                    product_factor_id_ptr,
                    ngroup,
                    nproduct,
                    nterm,
                    center_ptr,
                    inv_4p_ptr,
                    product_coeff_sum.data(),
                    gxv,
                    gyv,
                    gzv,
                    g2,
                    accumulate_image_pair_g,
                    use_precomputed_product_factors ? &product_factor_seen : nullptr,
                    use_precomputed_product_factors ? &product_factor_real : nullptr,
                    use_precomputed_product_factors ? &product_factor_imag : nullptr,
                    product_factor_current_stamp);
            } else {
                for_each_image_group_sum(
                    pair_idx,
                    group_starts_ptr,
                    group_image_ptr,
                    group_term_start_ptr,
                    group_term_stop_ptr,
                    group_product_start_ptr,
                    group_product_stop_ptr,
                    product_term_start_ptr,
                    product_term_stop_ptr,
                    product_factor_id_ptr,
                    ngroup,
                    nproduct,
                    nterm,
                    image_ptr,
                    center_ptr,
                    inv_4p_ptr,
                    coeff_ptr,
                    power_ptr,
                    gxv,
                    gyv,
                    gzv,
                    g2,
                    plane_z,
                    false,
                    gx_power_real,
                    gx_power_imag,
                    gy_power_real,
                    gy_power_imag,
                    gz_power_real,
                    gz_power_imag,
                    accumulate_image_pair_g,
                    use_precomputed_product_factors ? &product_factor_seen : nullptr,
                    use_precomputed_product_factors ? &product_factor_real : nullptr,
                    use_precomputed_product_factors ? &product_factor_imag : nullptr,
                    use_scale_cache ? &scale_cache_inv_4p : nullptr,
                    use_scale_cache ? &scale_cache_values : nullptr,
                    use_power_product_cache ? &power_product_real : nullptr,
                    use_power_product_cache ? &power_product_imag : nullptr,
                    pair_power_x,
                    pair_power_y,
                    pair_power_z,
                    product_factor_current_stamp);
            }

            for (npy_intp iphase = 0; iphase < nphase; ++iphase) {
                const double pair_real = value_real[static_cast<size_t>(iphase)];
                const double pair_imag = value_imag[static_cast<size_t>(iphase)];
                for (npy_intp aux = 0; aux < naux; ++aux) {
                    const double* weighted =
                        reinterpret_cast<const double*>(&weighted_aux_ptr[gidx * naux + aux]);
                    const size_t out_index =
                        static_cast<size_t>(
                            (((iphase * naux + aux) * static_cast<npy_intp>(nleft) + pidx)
                             * static_cast<npy_intp>(nright))
                            + qlocal);
                    partial_real[out_index] += weighted[0] * pair_real - weighted[1] * pair_imag;
                    partial_imag[out_index] += weighted[0] * pair_imag + weighted[1] * pair_real;
                }
            }
        }
    };

    Py_BEGIN_ALLOW_THREADS
    if (nthreads == 1) {
        worker(0, npair);
    } else if (use_pair_g_threads) {
        std::vector<std::vector<double>> partial_real;
        std::vector<std::vector<double>> partial_imag;
        partial_real.reserve(static_cast<size_t>(nthreads));
        partial_imag.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            partial_real.emplace_back(static_cast<size_t>(out_size), 0.0);
            partial_imag.emplace_back(static_cast<size_t>(out_size), 0.0);
        }
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(nthreads));
        for (long tid = 0; tid < nthreads; ++tid) {
            const npy_intp begin = (total_pair_g * tid) / nthreads;
            const npy_intp end = (total_pair_g * (tid + 1)) / nthreads;
            threads.emplace_back([&, begin, end, tid]() {
                worker_pair_g(
                    begin,
                    end,
                    partial_real[static_cast<size_t>(tid)],
                    partial_imag[static_cast<size_t>(tid)]);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (npy_intp index = 0; index < out_size; ++index) {
            double real = 0.0;
            double imag = 0.0;
            for (long tid = 0; tid < nthreads; ++tid) {
                real += partial_real[static_cast<size_t>(tid)][static_cast<size_t>(index)];
                imag += partial_imag[static_cast<size_t>(tid)][static_cast<size_t>(index)];
            }
            store_complex(out_ptr[index], real, imag);
        }
    } else {
        const long pair_threads = std::max<long>(
            1,
            std::min<long>(nthreads, static_cast<long>(std::max<npy_intp>(1, npair))));
        std::atomic<npy_intp> next_pair(0);
        auto dynamic_pair_worker = [&]() {
            while (true) {
                const npy_intp work_idx = next_pair.fetch_add(1, std::memory_order_relaxed);
                if (work_idx >= npair) {
                    break;
                }
                const npy_intp pair_idx = use_heavy_first
                    ? pair_work_order[static_cast<size_t>(work_idx)]
                    : work_idx;
                worker(pair_idx, pair_idx + 1);
            }
        };
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(pair_threads));
        for (long tid = 0; tid < pair_threads; ++tid) {
            threads.emplace_back(dynamic_pair_worker);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    return out_obj;
}

PyMethodDef methods[] = {
    {
        "gaussian_ft_batch",
        gaussian_ft_batch,
        METH_VARARGS,
        "Build one-center Cartesian Gaussian Fourier transforms for many G vectors.",
    },
    {
        "cartesian_shell_transform",
        cartesian_shell_transform,
        METH_VARARGS,
        "Apply shell-block Cartesian-to-auxiliary transforms to Gaussian FT rows.",
    },
    {
        "periodic_pair_ft_primitive_sum",
        periodic_pair_ft_primitive_sum,
        METH_VARARGS,
        "Build periodic AO-pair Fourier sums from packed primitive terms.",
    },
    {
        "periodic_pair_ft_primitive_sum_many",
        periodic_pair_ft_primitive_sum_many,
        METH_VARARGS,
        "Build periodic AO-pair Fourier sums for multiple image-phase rows.",
    },
    {
        "periodic_pair_ft_primitive_image_group_sum",
        periodic_pair_ft_primitive_image_group_sum,
        METH_VARARGS,
        "Build image-group-resolved periodic AO-pair Fourier sums.",
    },
    {
        "periodic_pair_ft_product_sum_many",
        periodic_pair_ft_product_sum_many,
        METH_VARARGS,
        "Build product-driven periodic AO-pair Fourier sums for multiple phase rows.",
    },
    {
        "periodic_pair_ft_primitive_contract_many",
        periodic_pair_ft_primitive_contract_many,
        METH_VARARGS,
        "Build contracted periodic AO-pair Fourier sums for multiple image-phase rows.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_gdf_cpp",
    "C++ kernels for periodic GDF construction.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__gdf_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
