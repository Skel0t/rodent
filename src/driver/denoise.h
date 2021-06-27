#ifndef DENOISE_H
#define DENOISE_H

#include <anydsl_runtime.hpp>

#include "common.h"
#include "interface.h"
#include "denoiser/nn.h"

void read_in(anydsl::Array<float>* weights, float** biases);
void denoise(anydsl::Array<float>* color, anydsl::Array<float>* albedo, anydsl::Array<float>* normal, anydsl::Array<uint8_t>* memory, anydsl::Array<float>* out, int width, int height, anydsl::Array<float>* weights, float* biases);

#endif // DENOISE_H