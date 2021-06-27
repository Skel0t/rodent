#ifndef DENOISE_H
#define DENOISE_H

#include <anydsl_runtime.hpp>

#include "common.h"
#include "denoiser/nn.h"
#include "denoiser/interface.h"

#ifdef OIDN
#include "OpenImageDenoise/oidn.hpp"

void denoise_nn(anydsl::Array<float>* color, anydsl::Array<float>* albedo, anydsl::Array<float>* normal, anydsl::Array<float>* out, int width, int height);
oidn::FilterRef create_filter(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height);
void denoise(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height);
#endif

#endif