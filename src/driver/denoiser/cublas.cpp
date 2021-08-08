#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" {
    void cublas_gemm(float* d_A, float* d_B, float* d_C, int a_width, int a_height, int b_width, int device) {
        cudaSetDevice(device);
        cublasHandle_t handle;
        cublasCreate(&handle);

        float a = 1.f, b = 1.f;

        cublasSgemm(handle,
                    CUBLAS_OP_N,   // no transpose
                    CUBLAS_OP_N,   // no transpose
                    b_width,       // rows of b'
                    a_height,      // cols of a'
                    a_width,       // cols of b'
                    &a,            // alpha
                    d_B,           // B^T left matrix
                    b_width,       // b col first
                    d_A,           // A^T right matrix
                    a_width,       // a col first
                    &b,            // beta
                    d_C,           // C
                    b_width);      // c col first
        cublasDestroy(handle);
    }

}