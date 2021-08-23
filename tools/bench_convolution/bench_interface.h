#ifndef BENCH_INTERFACE_H
#define BENCH_INTERFACE_H
#include <anydsl_runtime.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void sres_dump(anydsl::Array<float>* in_mat, anydsl::Array<float>* flattened_kernels, float* biases, anydsl::Array<float>* out);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_INTERFACE_H */
