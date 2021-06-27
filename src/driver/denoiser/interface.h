#ifndef INTERFACE_H
#define INTERFACE_H
#include <anydsl_runtime.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void forward_denoise(anydsl::Array<float>* img, anydsl::Array<float>* alb, anydsl::Array<float>* nrm, int32_t width, int32_t height, anydsl::Array<float>* out, anydsl::Array<float>* kernels, float* biases);

#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */
