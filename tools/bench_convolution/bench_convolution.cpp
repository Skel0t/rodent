#include <iostream>
#include <chrono>

#include <anydsl_runtime.hpp>

#include "bench_interface.h"
#include "../../src/driver/denoiser/nn_io.h"
#include "../../src/driver/denoise.h"

void bench_sresDump(int times = 1, int heatup_iterations = 0, bool correct_check = false);
void bench_denoiseDump(int times = 1, int heatup_iterations = 0, bool correct_check = false);
void bench_im2col(bool correct_check = false);

int main() {
    bench_im2col(true);

    // bench_denoiseDump(1, 0, true);

    // bench_sresDump(10, 0);
    // bench_sresDump(10, 5);
    // bench_sresDump(10, 10);

    return 0;
}

void sres_dump_wrap(anydsl::Array<float>* in_mat, anydsl::Array<float>* weights, float* biases, anydsl::Array<float>* out_mat) {
    Buffer in_mat_buf  = { (char*) in_mat->data(),   in_mat->size(),   in_mat->device()   };
    Buffer weights_buf = { (char*) weights->data(),  weights->size(),  weights->device()  };
    Buffer out_mat_buf = { (char*) out_mat->data(),  out_mat->size(),  out_mat->device()  };

    sres_dump(&in_mat_buf, &weights_buf, biases, &out_mat_buf);
}

/**
 * Only (!) benchmarking im2col + matmul + bias
 *
 * Not including activation function
 */
void bench_sresDump(int times, int heatup_iterations, bool correct_check) {
    // Constant values used in the dumped data
    const int in_channels  = 64;
    const int out_channels = 32;
    const int ksize  = 5;
    const int width  = 960;
    const int height = 540;
    const int memsize_weights = ksize * ksize * in_channels * out_channels;
    const int memsize_biases  = out_channels;
    const int size_im2col = ksize * ksize * in_channels * width * height;  // size for im2col matrix


    // Buffer for weights and biases
    anydsl::Array<float> weights(memsize_weights);
    float* biases = (float*) malloc(sizeof(float) * (memsize_biases));

    // Buffer for dumped data
    anydsl::Array<float> in_mat(width * height * in_channels);
    anydsl::Array<float> out_mat(width * height * out_channels + size_im2col);
    anydsl::Array<float> ref_mat(width * height * out_channels);

    // Read in weights and biases of neural network
    read_in_weigths_chw(weights.data(), 0, BENCH_DUMP_DIR "/dumped_data/sres_matmul/upconv1.txt", 64, 32, 5);
    read_in_biases(biases, 0, BENCH_DUMP_DIR "/dumped_data/sres_matmul/uc1bias.txt", 32);

    // Read in dumped data of hidden layers
    read_in_matrix_chw(in_mat.data(), BENCH_DUMP_DIR "/dumped_data/sres_matmul/conv4_in.txt", in_channels, height, width);
    read_in_matrix_chw(ref_mat.data(), BENCH_DUMP_DIR "/dumped_data/sres_matmul/conv4_out.txt", out_channels, height, width);

    // Heat-Up iterations
    for (int i = 0; i < heatup_iterations; i++) {
        sres_dump_wrap(&in_mat, &weights, biases, &out_mat);
    }

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    sres_dump_wrap(&in_mat, &weights, biases, &out_mat);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        for (size_t chn = 0; chn < out_channels; chn++) {
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    if (abs(out_mat.data()[size_im2col + chn * width * height + row * width + col] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-4) {
                        std::cout << "Diff at:\t" << chn << "\t" << col << "\t" << row << "\t(chn, x, y)" << std::endl;
                        std::cout << "Was:\t\t" << out_mat.data()[size_im2col + chn * width * height + row * width + col] << "\n"
                            << "Should be:\t" << ref_mat.data()[chn * width * height + row * width + col] << std::endl;
                        goto outer_break;
                    }
                }
            }
        }
        std::cout << "Correct Calculation!" << std::endl;
    }

    for (int i = 1; i < times; i++) {
        ticks = std::chrono::high_resolution_clock::now();
        sres_dump_wrap(&in_mat, &weights, biases, &out_mat);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    in_mat.release();
    out_mat.release();
    ref_mat.release();
    weights.release();
    free(biases);
}

/**
 * Benchmarks one forwarding through the denoising neural network with dumped input data
 */
void bench_denoiseDump(int times, int heatup_iterations, bool correct_check) {
    // Constant values used in the dumped data
    const int width  = 512;
    const int height = 512;
    const int channels = 3;

    // Buffer for weights and biases
    anydsl::Array<float> weights;
    float* biases;

    // Buffer for dumped data
    anydsl::Array<float> img_mat(width * height * channels);
    anydsl::Array<float> alb_mat(width * height * channels);
    anydsl::Array<float> nrm_mat(width * height * channels);
    anydsl::Array<float> ref_mat(width * height * channels);
    anydsl::Array<float> out_mat(width * height * channels);

    auto nn_memory = anydsl::Array<uint8_t>(get_necessary_mem(width, height));

    // Read in weights and biases of neural network
    read_in(&weights, &biases);

    // Read in dumped data of forwarding routine
    read_in_matrix_hwc(img_mat.data(), BENCH_DUMP_DIR "/dumped_data/denoising_fw/img_mat.txt", channels, height, width);
    read_in_matrix_hwc(alb_mat.data(), BENCH_DUMP_DIR "/dumped_data/denoising_fw/alb_mat.txt", channels, height, width);
    read_in_matrix_hwc(nrm_mat.data(), BENCH_DUMP_DIR "/dumped_data/denoising_fw/nrm_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), BENCH_DUMP_DIR "/dumped_data/denoising_fw/ref_mat.txt", channels, height, width);

    // Heat-Up iterations
    for (int i = 0; i < heatup_iterations; i++) {
        denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, biases);
    }

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, biases);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        for (size_t chn = 0; chn < channels; chn++) {
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    if (abs(out_mat.data()[row * width * channels + col * channels + chn] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-4) {
                        std::cout << "Diff at:\t" << chn << "\t" << col << "\t" << row << "\t(chn, x, y)" << std::endl;
                        std::cout << "Was:\t\t" << out_mat.data()[row * width * channels + col * channels + chn] << "\n"
                            << "Should be:\t" << ref_mat.data()[chn * width * height + row * width + col] << std::endl;
                        goto outer_break;
                    }
                }
            }
        }
        std::cout << "Correct Calculation!" << std::endl;
    }

    // std::cout << "Benchmarking loop..." << std::endl;

    for (int i = 1; i < times; i++) {
        ticks = std::chrono::high_resolution_clock::now();
        denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, biases);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    img_mat.release();
    alb_mat.release();
    nrm_mat.release();
    out_mat.release();
    ref_mat.release();
    weights.release();
    free(biases);
}

void im2col_wrap(anydsl::Array<float>* in_mat, anydsl::Array<float>* out_mat) {
    Buffer in_mat_buf  = { (char*) in_mat->data(),   in_mat->size(),   in_mat->device()   };
    Buffer out_mat_buf = { (char*) out_mat->data(),  out_mat->size(),  out_mat->device()  };

    im2col_dump(&in_mat_buf, &out_mat_buf);
}

void bench_im2col(bool correct_check) {
    const int ksize  = 3;
    const int width  = 32;
    const int height = 32;
    const int in_channels  = 70;
    const int out_channels = 70;

    anydsl::Array<float> in_mat(width * height * in_channels);
    anydsl::Array<float> out_mat(width * height * ksize * ksize * in_channels);
    anydsl::Array<float> ref_mat(width * height * ksize * ksize * in_channels);

    read_in_matrix_chw(in_mat.data(), BENCH_DUMP_DIR "/dumped_data/im2col/in_mat.txt", in_channels, height, width);
    read_in_matrix_chw(ref_mat.data(), BENCH_DUMP_DIR "/dumped_data/im2col/out_mat.txt", 1, in_channels * ksize * ksize, height * width);

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    im2col_wrap(&in_mat, &out_mat);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        for (size_t row = 0; row < height; row++) {
            for (size_t col = 0; col < width; col++) {
                if (abs(out_mat.data()[row * width + col] - ref_mat.data()[row * width + col]) > 1.0e-4) {
                    std::cout << "Diff at:\t" << col << "\t" << row << "\t(x, y)" << std::endl;
                    std::cout << "Was:\t\t" << out_mat.data()[row * width + col] << "\n"
                        << "Should be:\t" << ref_mat.data()[row * width + col] << std::endl;
                    goto outer_break;
                }
            }
        }
        std::cout << "Correct Calculation!" << std::endl;
    }
    outer_break:
    std::cout << "Average Time:\t" << (elapsed_ms) << " ms" << std::endl;

    in_mat.release();
    ref_mat.release();
    out_mat.release();
}
