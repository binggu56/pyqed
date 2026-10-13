#pragma once
#include <algorithm>
#include <cstddef>

namespace pyqed::linalg {
// Row-contiguous right operands keep trial/output columns together.
template <typename Scalar>
inline void row_product(const Scalar *a, int stride, const double *b, Scalar *out,
                        int m, int n, int k, int right_stride=0) {
    if (!right_stride) right_stride = n;
    std::fill(out, out+std::size_t(m)*n, Scalar{});
    for (int i = 0; i < m; ++i)
        for (int q = 0; q < k; ++q) {
            const Scalar value = a[std::size_t(i)*stride+q];
            const double *row = b+std::size_t(q)*right_stride;
            Scalar *target = out+std::size_t(i)*n;
            for (int j = 0; j < n; ++j) target[j] += value*row[j];
        }
}
}
