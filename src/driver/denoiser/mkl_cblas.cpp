#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

extern "C" {

    /**
     * a is m cross k mat
     * b is k cross n mat
     * c is m cross n mat -> result of a * b
     */
    void mkl_blas_mm_mult(int32_t m, int32_t n, int32_t k, const float* a, const float* b, float* c) {
        sgemm('n', 'n', m, n, k, 1.0, a, k, b, n, 0.0, c, n);
    }
}
