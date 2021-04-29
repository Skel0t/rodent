#include "OpenImageDenoise/oidn.hpp"
#include "common.h"

void denoise(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height,  uint32_t iter);
void correct(float* data, uint32_t iter, size_t width, size_t height, bool gamma);