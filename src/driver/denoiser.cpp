#include "denoiser.h"

void denoise(float* colorPtr, float* albedoPtr, float* normalPtr, float* outputPtr, size_t width, size_t height,  uint32_t iter) {
    correct(colorPtr, iter, width, height, true);
    correct(albedoPtr, iter, width, height, true);
    correct(normalPtr, iter, width, height, false);

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

void correct(float* data, uint32_t iter, size_t width, size_t height, bool gamma) {
    auto inv_gamma = 1.0f / 2.2f;
    auto inv_iter = 1.0f / iter;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto r = data[(y * width + x) * 3 + 0];
            auto g = data[(y * width + x) * 3 + 1];
            auto b = data[(y * width + x) * 3 + 2];

            if(gamma) {
                data[3 * (y * width + x) + 0] = clamp(std::pow(r * inv_iter, inv_gamma), 0.0f, 1.0f);
                data[3 * (y * width + x) + 1] = clamp(std::pow(g * inv_iter, inv_gamma), 0.0f, 1.0f);
                data[3 * (y * width + x) + 2] = clamp(std::pow(b * inv_iter, inv_gamma), 0.0f, 1.0f);
            } else {
                data[3 * (y * width + x) + 0] = clamp(r * inv_iter, 0.0f, 1.0f);
                data[3 * (y * width + x) + 1] = clamp(g * inv_iter, 0.0f, 1.0f);
                data[3 * (y * width + x) + 2] = clamp(b * inv_iter, 0.0f, 1.0f);
            }
        }
    }
}