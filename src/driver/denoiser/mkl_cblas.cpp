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

    /**
     * Adds the float val to n elements starting at a and saves them in res
     */
    void mkl_add_constant(int32_t n, const float* a, const float val, float* res) {
        vsLinearFrac(n, a, a, 1, val, 0, 1, res);
    }

    /**
     * Adds the first n elements in a and b elementwise and stores them in c
     */
    void mkl_add_elemwise(int32_t n, const float* a, const float* b, float* c) {
        vsAdd(n, a, b, c);
    }

    /**
     * Elementwise max(0, x) for each element in a and stores it in c
     */
    void mkl_apply_relu(int32_t n, const float* a, float* c) {
        float zero = 0;
        vsFmaxI(n, a, 1, &zero, 0, c, 1);
    }
}
