#ifdef OIDN
#include "OpenImageDenoise/oidn.hpp"
#include "common.h"

oidn::FilterRef create_filter(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height);
void denoise(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height);
#endif