#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

extern "C" {
    /* cublaslt supports row-major matrix multiplication. */
    void cublaslt_S_gemm(const float* d_A, const float* d_B, float* d_C, int a_width, int a_height, int b_width, int device) {
        cudaSetDevice(device);
        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasLtHandle_t handlelt = (cublasLtHandle_t) handle;

        cublasLtMatmulDesc_t descriptor;
        cublasLtMatrixLayout_t a_layout;
        cublasLtMatrixLayout_t b_layout;
        cublasLtMatrixLayout_t c_layout;

        cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F, a_height, a_width, a_width);
        cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F, a_width,  b_width, b_width);
        cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, a_height, b_width, b_width);

        cublasLtMatmulDescCreate(&descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        // Matrices are row-major
        cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
        cublasLtMatrixLayoutSetAttribute( a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) );
        cublasLtMatrixLayoutSetAttribute( b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) );
        cublasLtMatrixLayoutSetAttribute( c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) );

        float a = 1.f, b = 0.f;

        cublasLtMatmul(handlelt,
                    descriptor,     // description
                    &a,             // alpha
                    d_A,            // pointer to a
                    a_layout,       // a description
                    d_B,            // pointer to b
                    b_layout,       // b description
                    &b,             // beta
                    d_C,            // C pointer
                    c_layout,       // C layout
                    d_C,            // D pointer
                    c_layout,       // D layout
                    NULL,           // matmul algorithm, NULL = take one heuristically
                    nullptr,        // workspace*
                    0,              // workspaceSize
                    nullptr);       // stream
        cublasDestroy(handle);
    }

    /* Use identity (AB)^t = B^t A^t = C^t and that transposing only changes
       interpretation from row-major to column-major and vice versa.

       Necessary since cublas expects column-major while we use row-major. */
    void cublas_S_gemm(float* d_A, float* d_B, float* d_C, int a_width, int a_height, int b_width, int device) {
        cudaSetDevice(device);
        cublasHandle_t handle;
        cublasCreate(&handle);

        float a = 1.f, b = 0.f;

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
