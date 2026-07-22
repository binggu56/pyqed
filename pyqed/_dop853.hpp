#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace pyqed::dop853 {

constexpr int n_stages = 12;
constexpr int n_stages_extended = 16;
constexpr int interpolator_power = 7;

using Matrix16 = std::array<std::array<double, n_stages_extended>, n_stages_extended>;
using DenseMatrix = std::array<std::array<double, n_stages_extended>, interpolator_power - 3>;

// Dormand-Prince 8(5,3) coefficients from Hairer's DOP853 implementation.
inline constexpr std::array<double, n_stages_extended> c = {
    0.0,
    0.526001519587677318785587544488e-01,
    0.789002279381515978178381316732e-01,
    0.118350341907227396726757197510,
    0.281649658092772603273242802490,
    0.333333333333333333333333333333,
    0.25,
    0.307692307692307692307692307692,
    0.651282051282051282051282051282,
    0.6,
    0.857142857142857142857142857142,
    1.0,
    1.0,
    0.1,
    0.2,
    0.777777777777777777777777777778,
};

constexpr Matrix16 make_a() {
    Matrix16 a{};
    a[1][0] = 5.26001519587677318785587544488e-2;

    a[2][0] = 1.97250569845378994544595329183e-2;
    a[2][1] = 5.91751709536136983633785987549e-2;

    a[3][0] = 2.95875854768068491816892993775e-2;
    a[3][2] = 8.87627564304205475450678981324e-2;

    a[4][0] = 2.41365134159266685502369798665e-1;
    a[4][2] = -8.84549479328286085344864962717e-1;
    a[4][3] = 9.24834003261792003115737966543e-1;

    a[5][0] = 3.7037037037037037037037037037e-2;
    a[5][3] = 1.70828608729473871279604482173e-1;
    a[5][4] = 1.25467687566822425016691814123e-1;

    a[6][0] = 3.7109375e-2;
    a[6][3] = 1.70252211019544039314978060272e-1;
    a[6][4] = 6.02165389804559606850219397283e-2;
    a[6][5] = -1.7578125e-2;

    a[7][0] = 3.70920001185047927108779319836e-2;
    a[7][3] = 1.70383925712239993810214054705e-1;
    a[7][4] = 1.07262030446373284651809199168e-1;
    a[7][5] = -1.53194377486244017527936158236e-2;
    a[7][6] = 8.27378916381402288758473766002e-3;

    a[8][0] = 6.24110958716075717114429577812e-1;
    a[8][3] = -3.36089262944694129406857109825;
    a[8][4] = -8.68219346841726006818189891453e-1;
    a[8][5] = 2.75920996994467083049415600797e1;
    a[8][6] = 2.01540675504778934086186788979e1;
    a[8][7] = -4.34898841810699588477366255144e1;

    a[9][0] = 4.77662536438264365890433908527e-1;
    a[9][3] = -2.48811461997166764192642586468;
    a[9][4] = -5.90290826836842996371446475743e-1;
    a[9][5] = 2.12300514481811942347288949897e1;
    a[9][6] = 1.52792336328824235832596922938e1;
    a[9][7] = -3.32882109689848629194453265587e1;
    a[9][8] = -2.03312017085086261358222928593e-2;

    a[10][0] = -9.3714243008598732571704021658e-1;
    a[10][3] = 5.18637242884406370830023853209;
    a[10][4] = 1.09143734899672957818500254654;
    a[10][5] = -8.14978701074692612513997267357;
    a[10][6] = -1.85200656599969598641566180701e1;
    a[10][7] = 2.27394870993505042818970056734e1;
    a[10][8] = 2.49360555267965238987089396762;
    a[10][9] = -3.0467644718982195003823669022;

    a[11][0] = 2.27331014751653820792359768449;
    a[11][3] = -1.05344954667372501984066689879e1;
    a[11][4] = -2.00087205822486249909675718444;
    a[11][5] = -1.79589318631187989172765950534e1;
    a[11][6] = 2.79488845294199600508499808837e1;
    a[11][7] = -2.85899827713502369474065508674;
    a[11][8] = -8.87285693353062954433549289258;
    a[11][9] = 1.23605671757943030647266201528e1;
    a[11][10] = 6.43392746015763530355970484046e-1;

    a[12][0] = 5.42937341165687622380535766363e-2;
    a[12][5] = 4.45031289275240888144113950566;
    a[12][6] = 1.89151789931450038304281599044;
    a[12][7] = -5.8012039600105847814672114227;
    a[12][8] = 3.1116436695781989440891606237e-1;
    a[12][9] = -1.52160949662516078556178806805e-1;
    a[12][10] = 2.01365400804030348374776537501e-1;
    a[12][11] = 4.47106157277725905176885569043e-2;

    a[13][0] = 5.61675022830479523392909219681e-2;
    a[13][6] = 2.53500210216624811088794765333e-1;
    a[13][7] = -2.46239037470802489917441475441e-1;
    a[13][8] = -1.24191423263816360469010140626e-1;
    a[13][9] = 1.5329179827876569731206322685e-1;
    a[13][10] = 8.20105229563468988491666602057e-3;
    a[13][11] = 7.56789766054569976138603589584e-3;
    a[13][12] = -8.298e-3;

    a[14][0] = 3.18346481635021405060768473261e-2;
    a[14][5] = 2.83009096723667755288322961402e-2;
    a[14][6] = 5.35419883074385676223797384372e-2;
    a[14][7] = -5.49237485713909884646569340306e-2;
    a[14][10] = -1.08347328697249322858509316994e-4;
    a[14][11] = 3.82571090835658412954920192323e-4;
    a[14][12] = -3.40465008687404560802977114492e-4;
    a[14][13] = 1.41312443674632500278074618366e-1;

    a[15][0] = -4.28896301583791923408573538692e-1;
    a[15][5] = -4.69762141536116384314449447206;
    a[15][6] = 7.68342119606259904184240953878;
    a[15][7] = 4.06898981839711007970213554331;
    a[15][8] = 3.56727187455281109270669543021e-1;
    a[15][12] = -1.39902416515901462129418009734e-3;
    a[15][13] = 2.9475147891527723389556272149;
    a[15][14] = -9.15095847217987001081870187138;
    return a;
}

inline constexpr Matrix16 a = make_a();

constexpr std::array<double, n_stages + 1> make_e3() {
    std::array<double, n_stages + 1> e{};
    for (int i = 0; i < n_stages; ++i) {
        e[i] = a[12][i];
    }
    e[0] -= 0.244094488188976377952755905512;
    e[8] -= 0.733846688281611857341361741547;
    e[11] -= 0.220588235294117647058823529412e-1;
    return e;
}

inline constexpr auto e3 = make_e3();
inline constexpr std::array<double, n_stages + 1> e5 = {
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e+1,
    -0.4957589496572501915214079952,
    0.1664377182454986536961530415e+1,
    -0.3503288487499736816886487290,
    0.3341791187130174790297318841,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
    0.0,
};

constexpr DenseMatrix make_d() {
    DenseMatrix d{};
    d[0][0] = -0.84289382761090128651353491142e+1;
    d[0][5] = 0.56671495351937776962531783590;
    d[0][6] = -0.30689499459498916912797304727e+1;
    d[0][7] = 0.23846676565120698287728149680e+1;
    d[0][8] = 0.21170345824450282767155149946e+1;
    d[0][9] = -0.87139158377797299206789907490;
    d[0][10] = 0.22404374302607882758541771650e+1;
    d[0][11] = 0.63157877876946881815570249290;
    d[0][12] = -0.88990336451333310820698117400e-1;
    d[0][13] = 0.18148505520854727256656404962e+2;
    d[0][14] = -0.91946323924783554000451984436e+1;
    d[0][15] = -0.44360363875948939664310572000e+1;

    d[1][0] = 0.10427508642579134603413151009e+2;
    d[1][5] = 0.24228349177525818288430175319e+3;
    d[1][6] = 0.16520045171727028198505394887e+3;
    d[1][7] = -0.37454675472269020279518312152e+3;
    d[1][8] = -0.22113666853125306036270938578e+2;
    d[1][9] = 0.77334326684722638389603898808e+1;
    d[1][10] = -0.30674084731089398182061213626e+2;
    d[1][11] = -0.93321305264302278729567221706e+1;
    d[1][12] = 0.15697238121770843886131091075e+2;
    d[1][13] = -0.31139403219565177677282850411e+2;
    d[1][14] = -0.93529243588444783865713862664e+1;
    d[1][15] = 0.35816841486394083752465898540e+2;

    d[2][0] = 0.19985053242002433820987653617e+2;
    d[2][5] = -0.38703730874935176555105901742e+3;
    d[2][6] = -0.18917813819516756882830838328e+3;
    d[2][7] = 0.52780815920542364900561016686e+3;
    d[2][8] = -0.11573902539959630126141871134e+2;
    d[2][9] = 0.68812326946963000169666922661e+1;
    d[2][10] = -0.10006050966910838403183860980e+1;
    d[2][11] = 0.77771377980534432092869265740;
    d[2][12] = -0.27782057523535084065932004339e+1;
    d[2][13] = -0.60196695231264120758267380846e+2;
    d[2][14] = 0.84320405506677161018159903784e+2;
    d[2][15] = 0.11992291136182789328035130030e+2;

    d[3][0] = -0.25693933462703749003312586129e+2;
    d[3][5] = -0.15418974869023643374053993627e+3;
    d[3][6] = -0.23152937917604549567536039109e+3;
    d[3][7] = 0.35763911791061412378285349910e+3;
    d[3][8] = 0.93405324183624310003907691704e+2;
    d[3][9] = -0.37458323136451633156875139351e+2;
    d[3][10] = 0.10409964950896230045147246184e+3;
    d[3][11] = 0.29840293426660503123344363579e+2;
    d[3][12] = -0.43533456590011143754432175058e+2;
    d[3][13] = 0.96324553959188282948394950600e+2;
    d[3][14] = -0.39177261675615439165231486172e+2;
    d[3][15] = -0.14972683625798562581422125276e+3;
    return d;
}

inline constexpr DenseMatrix d = make_d();

using complex128 = std::complex<double>;

struct Stats {
    bool success = false;
    std::string message;
    std::int64_t nfev = 0;
    std::int64_t n_steps = 0;
    std::int64_t n_rejected = 0;
};

constexpr std::size_t parallel_vector_min_size = 16384;

class ThreadPool {
public:
    explicit ThreadPool(int threads)
        : thread_count_(threads < 1 ? 1 : static_cast<std::size_t>(threads)) {
        if (thread_count_ <= 1) {
            return;
        }
        workers_.reserve(thread_count_ - 1);
        for (std::size_t worker = 1; worker < thread_count_; ++worker) {
            workers_.emplace_back([this, worker]() { worker_loop(worker); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t thread_count() const {
        return thread_count_;
    }

    std::size_t worker_count(std::size_t size) const {
        if (thread_count_ <= 1 || size == 0) {
            return 1;
        }
        return std::min<std::size_t>(thread_count_, size);
    }

    template <typename Function>
    void for_each(std::size_t size, Function&& function) {
        const std::size_t worker_count = this->worker_count(size);
        if (worker_count <= 1) {
            function(0, size);
            return;
        }

        const std::size_t chunk = (size + worker_count - 1) / worker_count;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_workers_ = worker_count - 1;
            finished_workers_ = 0;
            task_ = [&](std::size_t worker) {
                const std::size_t begin = worker * chunk;
                const std::size_t end = std::min(size, begin + chunk);
                if (begin < end) {
                    function(begin, end);
                }
            };
            ++generation_;
        }

        start_cv_.notify_all();
        function(0, std::min(size, chunk));

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() { return finished_workers_ == active_workers_; });
        task_ = nullptr;
    }

private:
    void worker_loop(std::size_t worker) {
        std::size_t seen_generation = 0;
        while (true) {
            std::function<void(std::size_t)> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(
                    lock,
                    [&]() { return stop_ || generation_ != seen_generation; });
                if (stop_) {
                    return;
                }
                seen_generation = generation_;
                if (worker > active_workers_) {
                    continue;
                }
                task = task_;
            }

            task(worker);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++finished_workers_;
                if (finished_workers_ == active_workers_) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::size_t thread_count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::function<void(std::size_t)> task_;
    std::size_t active_workers_ = 0;
    std::size_t finished_workers_ = 0;
    std::size_t generation_ = 0;
    bool stop_ = false;
};

class ParallelContext {
public:
    explicit ParallelContext(int threads, ThreadPool* pool = nullptr)
        : threads_(threads < 1 ? 1 : static_cast<std::size_t>(threads)),
          pool_(pool) {}

    bool enabled(std::size_t size) const {
        return max_threads() > 1 && size >= parallel_vector_min_size;
    }

    std::size_t worker_count(std::size_t size) const {
        return enabled(size) ? std::min<std::size_t>(max_threads(), size) : 1;
    }

    template <typename Function>
    void for_each(std::size_t size, Function&& function) const {
        if (!enabled(size)) {
            function(0, size);
            return;
        }

        if (pool_ != nullptr) {
            pool_->for_each(size, std::forward<Function>(function));
            return;
        }

        const std::size_t worker_count = this->worker_count(size);
        const std::size_t chunk = (size + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        workers.reserve(worker_count - 1);

        try {
            for (std::size_t worker = 1; worker < worker_count; ++worker) {
                const std::size_t begin = worker * chunk;
                const std::size_t end = std::min(size, begin + chunk);
                if (begin >= end) {
                    break;
                }
                workers.emplace_back(function, begin, end);
            }
            function(0, std::min(size, chunk));
            for (auto& worker : workers) {
                worker.join();
            }
        } catch (...) {
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            function(0, size);
        }
    }

private:
    std::size_t max_threads() const {
        return pool_ == nullptr ? threads_ : pool_->thread_count();
    }

    std::size_t threads_;
    ThreadPool* pool_;
};

inline double scaled_rms_norm(
    const complex128* values,
    const complex128* scale_state,
    std::size_t size,
    double rtol,
    double atol,
    const ParallelContext& parallel) {
    if (!parallel.enabled(size)) {
        long double sum = 0.0L;
        for (std::size_t i = 0; i < size; ++i) {
            const double scale = atol + rtol * std::abs(scale_state[i]);
            sum += static_cast<long double>(std::norm(values[i] / scale));
        }
        return std::sqrt(static_cast<double>(sum / static_cast<long double>(size)));
    }

    const std::size_t worker_count = parallel.worker_count(size);
    const std::size_t chunk = (size + worker_count - 1) / worker_count;
    std::vector<long double> partials(worker_count, 0.0L);
    parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
        long double local = 0.0L;
        for (std::size_t i = begin; i < end; ++i) {
            const double scale = atol + rtol * std::abs(scale_state[i]);
            local += static_cast<long double>(std::norm(values[i] / scale));
        }
        partials[begin / chunk] = local;
    });
    long double sum = 0.0L;
    for (long double partial : partials) {
        sum += partial;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(size)));
}

inline void combine_stages(
    complex128* destination,
    const complex128* state,
    const complex128* stages,
    std::size_t size,
    double h,
    const std::array<double, n_stages_extended>& coefficients,
    int stage_count,
    const ParallelContext& parallel) {
    parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            destination[i] = state[i];
        }
        for (int stage = 0; stage < stage_count; ++stage) {
            const double coefficient = coefficients[stage];
            if (coefficient == 0.0) {
                continue;
            }
            const complex128* source = stages + static_cast<std::size_t>(stage) * size;
            const double factor = h * coefficient;
            for (std::size_t i = begin; i < end; ++i) {
                destination[i] += factor * source[i];
            }
        }
    });
}

inline double scaled_difference_rms(
    const complex128* left,
    const complex128* right,
    const complex128* reference,
    std::size_t size,
    double rtol,
    double atol,
    const ParallelContext& parallel) {
    if (!parallel.enabled(size)) {
        long double sum = 0.0L;
        for (std::size_t i = 0; i < size; ++i) {
            const double scale = atol + rtol * std::abs(reference[i]);
            sum += static_cast<long double>(std::norm((left[i] - right[i]) / scale));
        }
        return std::sqrt(static_cast<double>(sum / static_cast<long double>(size)));
    }

    const std::size_t worker_count = parallel.worker_count(size);
    const std::size_t chunk = (size + worker_count - 1) / worker_count;
    std::vector<long double> partials(worker_count, 0.0L);
    parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
        long double local = 0.0L;
        for (std::size_t i = begin; i < end; ++i) {
            const double scale = atol + rtol * std::abs(reference[i]);
            local += static_cast<long double>(std::norm((left[i] - right[i]) / scale));
        }
        partials[begin / chunk] = local;
    });
    long double sum = 0.0L;
    for (long double partial : partials) {
        sum += partial;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(size)));
}

inline void embedded_error_sums(
    const complex128* state,
    const complex128* trial_state,
    const complex128* error5,
    const complex128* error3,
    std::size_t size,
    double rtol,
    double atol,
    const ParallelContext& parallel,
    long double& err5_norm2,
    long double& err3_norm2) {
    if (!parallel.enabled(size)) {
        err5_norm2 = 0.0L;
        err3_norm2 = 0.0L;
        for (std::size_t i = 0; i < size; ++i) {
            const double scale = atol + rtol * std::max(std::abs(state[i]), std::abs(trial_state[i]));
            err5_norm2 += static_cast<long double>(std::norm(error5[i] / scale));
            err3_norm2 += static_cast<long double>(std::norm(error3[i] / scale));
        }
        return;
    }

    const std::size_t worker_count = parallel.worker_count(size);
    const std::size_t chunk = (size + worker_count - 1) / worker_count;
    std::vector<long double> partial5(worker_count, 0.0L);
    std::vector<long double> partial3(worker_count, 0.0L);
    parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
        long double local5 = 0.0L;
        long double local3 = 0.0L;
        for (std::size_t i = begin; i < end; ++i) {
            const double scale = atol + rtol * std::max(std::abs(state[i]), std::abs(trial_state[i]));
            local5 += static_cast<long double>(std::norm(error5[i] / scale));
            local3 += static_cast<long double>(std::norm(error3[i] / scale));
        }
        const std::size_t worker = begin / chunk;
        partial5[worker] = local5;
        partial3[worker] = local3;
    });

    err5_norm2 = 0.0L;
    err3_norm2 = 0.0L;
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        err5_norm2 += partial5[worker];
        err3_norm2 += partial3[worker];
    }
}

inline complex128 dense_output_component(
    std::size_t index,
    double x,
    double h,
    const complex128* y_old,
    const complex128* y_new,
    const complex128* stages,
    std::size_t size) {
    const complex128 delta = y_new[index] - y_old[index];
    const complex128 f_old = stages[index];
    const complex128 f_new = stages[12 * size + index];
    std::array<complex128, interpolator_power> f{};
    f[0] = delta;
    f[1] = h * f_old - delta;
    f[2] = 2.0 * delta - h * (f_new + f_old);
    for (int row = 0; row < interpolator_power - 3; ++row) {
        complex128 value(0.0, 0.0);
        for (int stage = 0; stage < n_stages_extended; ++stage) {
            value += d[row][stage] * stages[stage * size + index];
        }
        f[row + 3] = h * value;
    }

    complex128 value(0.0, 0.0);
    for (int i = 0; i < interpolator_power; ++i) {
        value += f[interpolator_power - 1 - i];
        value *= (i % 2 == 0) ? x : (1.0 - x);
    }
    return y_old[index] + value;
}

class OutputView {
public:
    static OutputView direct(const complex128* state) {
        return OutputView(state, nullptr, nullptr, nullptr, 0, 0.0, 0.0);
    }

    static OutputView interpolated(
        const complex128* y_old,
        const complex128* y_new,
        const complex128* stages,
        std::size_t size,
        double x,
        double h) {
        return OutputView(nullptr, y_old, y_new, stages, size, x, h);
    }

    complex128 operator[](std::size_t index) const {
        if (state_ != nullptr) {
            return state_[index];
        }
        return dense_output_component(index, x_, h_, y_old_, y_new_, stages_, size_);
    }

private:
    OutputView(
        const complex128* state,
        const complex128* y_old,
        const complex128* y_new,
        const complex128* stages,
        std::size_t size,
        double x,
        double h)
        : state_(state),
          y_old_(y_old),
          y_new_(y_new),
          stages_(stages),
          size_(size),
          x_(x),
          h_(h) {}

    const complex128* state_;
    const complex128* y_old_;
    const complex128* y_new_;
    const complex128* stages_;
    std::size_t size_;
    double x_;
    double h_;
};

template <typename Rhs, typename Observer>
Stats integrate(
    Rhs& rhs,
    complex128* state,
    std::size_t size,
    const double* t_eval,
    std::size_t n_times,
    Observer&& observer,
    double rtol,
    double atol,
    int threads = 1,
    ThreadPool* pool = nullptr,
    bool accepted_step_output = false) {
    Stats stats;
    if (size == 0 || n_times == 0) {
        stats.message = "DOP853 requires a non-empty state and output grid";
        return stats;
    }
    ParallelContext parallel(threads, pool);
    const double t_bound = t_eval[n_times - 1];
    double t = t_eval[0];
    observer(0, t, OutputView::direct(state));
    if (n_times == 1) {
        stats.success = true;
        stats.message = "success";
        return stats;
    }

    if (size > std::numeric_limits<std::size_t>::max() /
                   static_cast<std::size_t>(n_stages_extended)) {
        stats.message = "DOP853 workspace size overflow";
        return stats;
    }

    std::vector<complex128> stages;
    std::vector<complex128> work;
    try {
        stages.resize(static_cast<std::size_t>(n_stages_extended) * size);
        work.resize(size);
    } catch (const std::bad_alloc&) {
        stats.message = "unable to allocate DOP853 workspace";
        return stats;
    }

    auto stage = [&](int index) {
        return stages.data() + static_cast<std::size_t>(index) * size;
    };
    auto evaluate = [&](double time, const complex128* y, complex128* out) {
        if (!rhs.evaluate(time, y, out)) {
            return false;
        }
        ++stats.nfev;
        return true;
    };

    if (!evaluate(t, state, stage(0))) {
        stats.message = "right-hand side evaluation failed";
        return stats;
    }

    const double interval = t_bound - t;
    const double max_step = interval;
    const double d0 = scaled_rms_norm(state, state, size, rtol, atol, parallel);
    const double d1 = scaled_rms_norm(stage(0), state, size, rtol, atol, parallel);
    double h0 = (d0 < 1.0e-5 || d1 < 1.0e-5) ? 1.0e-6 : 0.01 * d0 / d1;
    h0 = std::min(h0, interval);
    parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            work[i] = state[i] + h0 * stage(0)[i];
        }
    });
    if (!evaluate(t + h0, work.data(), stage(1))) {
        stats.message = "right-hand side evaluation failed";
        return stats;
    }

    const double d2 = scaled_difference_rms(stage(1), stage(0), state, size, rtol, atol, parallel) / h0;
    double h1;
    if (d1 <= 1.0e-15 && d2 <= 1.0e-15) {
        h1 = std::max(1.0e-6, h0 * 1.0e-3);
    } else {
        h1 = std::pow(0.01 / std::max(d1, d2), 1.0 / 8.0);
    }
    double h_abs = std::min({100.0 * h0, h1, interval, max_step});

    constexpr double safety = 0.9;
    constexpr double min_factor = 0.2;
    constexpr double max_factor = 10.0;
    constexpr double error_exponent = -1.0 / 8.0;
    constexpr std::int64_t max_attempts = 1000000;
    std::int64_t attempts = 0;
    std::size_t output_index = 1;
    bool previous_rejected = false;

    while (t < t_bound) {
        if (++attempts > max_attempts) {
            stats.message = "DOP853 exceeded the maximum number of step attempts";
            return stats;
        }
        const double min_step = 10.0 * std::abs(
            std::nextafter(t, std::numeric_limits<double>::infinity()) - t);
        h_abs = std::min(h_abs, max_step);
        if (h_abs < min_step) {
            stats.message = "DOP853 step size became smaller than floating-point spacing";
            return stats;
        }

        const double h = std::min(h_abs, t_bound - t);
        const double t_new = t + h;

        for (int s = 1; s < n_stages; ++s) {
            combine_stages(work.data(), state, stages.data(), size, h, a[s], s, parallel);
            if (!evaluate(t + c[s] * h, work.data(), stage(s))) {
                stats.message = "right-hand side evaluation failed";
                return stats;
            }
        }

        combine_stages(work.data(), state, stages.data(), size, h, a[12], n_stages, parallel);
        if (!evaluate(t_new, work.data(), stage(12))) {
            stats.message = "right-hand side evaluation failed";
            return stats;
        }

        // Stages 1--4 have zero error and dense-output coefficients, so stages
        // 1 and 2 can hold the embedded error estimates without extra arrays.
        parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                complex128 err5(0.0, 0.0);
                complex128 err3(0.0, 0.0);
                for (int s = 0; s <= n_stages; ++s) {
                    const complex128 value = stage(s)[i];
                    if (e5[s] != 0.0) {
                        err5 += e5[s] * value;
                    }
                    if (e3[s] != 0.0) {
                        err3 += e3[s] * value;
                    }
                }
                stage(1)[i] = err5;
                stage(2)[i] = err3;
            }
        });

        long double err5_norm2 = 0.0L;
        long double err3_norm2 = 0.0L;
        embedded_error_sums(
            state,
            work.data(),
            stage(1),
            stage(2),
            size,
            rtol,
            atol,
            parallel,
            err5_norm2,
            err3_norm2);

        double error_norm = 0.0;
        if (err5_norm2 != 0.0L || err3_norm2 != 0.0L) {
            const long double denominator =
                (err5_norm2 + 0.01L * err3_norm2) * static_cast<long double>(size);
            error_norm = std::abs(h) * static_cast<double>(err5_norm2 / std::sqrt(denominator));
        }

        if (std::isfinite(error_norm) && error_norm < 1.0) {
            double factor = error_norm == 0.0
                ? max_factor
                : std::min(max_factor, safety * std::pow(error_norm, error_exponent));
            if (previous_rejected) {
                factor = std::min(1.0, factor);
            }

            if (accepted_step_output) {
                observer(output_index, t_new, OutputView::direct(work.data()));
                ++output_index;
            } else {
                const double endpoint_tolerance =
                    32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(t_new));
                std::size_t due_end = output_index;
                while (due_end < n_times && t_eval[due_end] <= t_new + endpoint_tolerance) {
                    ++due_end;
                }
                bool needs_dense_output = false;
                for (std::size_t i = output_index; i < due_end; ++i) {
                    if (std::abs(t_eval[i] - t_new) > endpoint_tolerance) {
                        needs_dense_output = true;
                        break;
                    }
                }

                if (needs_dense_output) {
                    for (int s = 13; s < n_stages_extended; ++s) {
                        combine_stages(stage(1), state, stages.data(), size, h, a[s], s, parallel);
                        if (!evaluate(t + c[s] * h, stage(1), stage(s))) {
                            stats.message = "right-hand side evaluation failed";
                            return stats;
                        }
                    }
                }

                while (output_index < due_end) {
                    if (std::abs(t_eval[output_index] - t_new) <= endpoint_tolerance) {
                        observer(output_index, t_eval[output_index], OutputView::direct(work.data()));
                    } else {
                        const double x = (t_eval[output_index] - t) / h;
                        observer(
                            output_index,
                            t_eval[output_index],
                            OutputView::interpolated(
                                state, work.data(), stages.data(), size, x, h));
                    }
                    ++output_index;
                }
            }

            parallel.for_each(size, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    state[i] = work[i];
                    stage(0)[i] = stage(12)[i];
                }
            });
            t = t_new;
            ++stats.n_steps;
            h_abs *= factor;
            previous_rejected = false;
        } else {
            double factor = min_factor;
            if (std::isfinite(error_norm) && error_norm > 0.0) {
                factor = std::max(min_factor, safety * std::pow(error_norm, error_exponent));
            }
            h_abs *= factor;
            ++stats.n_rejected;
            previous_rejected = true;
        }
    }

    if (!accepted_step_output && output_index != n_times) {
        stats.message = "DOP853 did not reach every requested output time";
        return stats;
    }
    stats.success = true;
    stats.message = "success";
    return stats;
}

}  // namespace pyqed::dop853
