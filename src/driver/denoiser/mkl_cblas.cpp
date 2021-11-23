#include <mkl.h>

extern "C" {

    /**
     * a is m cross k mat
     * b is k cross n mat
     * c is m cross n mat -> result of a * b
     */
    void mkl_blas_mm_mult(int32_t m, int32_t n, int32_t k, const float* a, int32_t lda, const float* b, int32_t ldb, float* c, const int32_t ldc) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f, a, lda,
                    b, ldb,
                    0, c, ldc);
    }
}
