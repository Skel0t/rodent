#ifdef OIDN
#include "denoiser.h"

oidn::FilterRef create_filter(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height) {
    // Create an Intel Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  colorPtr,  oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedoPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("normal", normalPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("output", outputPtr, oidn::Format::Float3, width, height);
    filter.set("hdr", false); // image is NOT HDR
    filter.commit();

    return filter;
}

void denoise(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height) {
    // Create an Intel Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create a denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color",  colorPtr,  oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedoPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("normal", normalPtr, oidn::Format::Float3, width, height); // optional
    filter.setImage("output", outputPtr, oidn::Format::Float3, width, height);
    filter.set("hdr", false); // image is NOT HDR
    filter.commit();

    // Filter the image
    filter.execute();
}
#endif