#include <iostream>
#include <chrono>

#include <anydsl_runtime.hpp>

#include "bench_interface.h"
#include "../../src/driver/denoiser/nn_io.h"
#include "../../src/driver/denoise.h"

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

void bench_sresDump(int times = 1, int heatup_iterations = 0, bool correct_check = false);
void bench_denoiseDump512(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);
void bench_denoiseDump1k(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);
void bench_im2col(bool correct_check = false);

void bench_denoiseDump(int width, int height, int channels, std::string dump_dir, int times, int heat_up_iterations, bool correct_check);

template <typename T>
anydsl::Array<T> copy_to_device(int32_t dev, const T* data, size_t n) {
    anydsl::Array<T> array(dev, reinterpret_cast<T*>(anydsl_alloc(dev, n * sizeof(T))), n);
    anydsl_copy(0, data, 0, dev, array.data(), 0, sizeof(T) * n);
    return array;
}


int main() {
    bench_denoiseDump512(1, 0, true, true);
    // bench_denoiseDump1k(100, 10, false);

    // bench_sresDump(10, 10, false);

    // return 0;
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
 * Not including activation function (is "id")
 *
 * Input  Size: 960 x 540
 * Output Size: 1920 x 1080
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

void bench_denoiseDump(int width, int height, int channels, std::string dump_dir, int times, int heat_up_iterations, bool correct_check) {
    // Buffer for weights and biases
    anydsl::Array<float> weights;
    anydsl::Array<float> biases;

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
    read_in_matrix_hwc(img_mat.data(), dump_dir + "img_mat.txt", channels, height, width);
    read_in_matrix_hwc(alb_mat.data(), dump_dir + "alb_mat.txt", channels, height, width);
    read_in_matrix_hwc(nrm_mat.data(), dump_dir + "nrm_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), dump_dir + "ref_mat.txt", channels, height, width);

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
    }

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
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
        denoise(&img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
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
    nn_memory.release();
    biases.release();
}

void im2col_wrap(anydsl::Array<float>* in_mat, anydsl::Array<float>* out_mat) {
    Buffer in_mat_buf  = { (char*) in_mat->data(),   in_mat->size(),   in_mat->device()   };
    Buffer out_mat_buf = { (char*) out_mat->data(),  out_mat->size(),  out_mat->device()  };

    im2col_dump(&in_mat_buf, &out_mat_buf);
}

/**
 * Benchmarks the im2col time of a middle convolutional layer of a 512x512 image.
 */
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

void bench_denoiseDump_gpu(int width, int height, int channels, std::string dump_dir, int times, int heat_up_iterations, bool correct_check) {
    // Buffer for weights and biases
    anydsl::Array<float> weights;
    anydsl::Array<float> biases;

    // Buffer for dumped data on cpu
    anydsl::Array<float> img_mat(width * height * channels);
    anydsl::Array<float> alb_mat(width * height * channels);
    anydsl::Array<float> nrm_mat(width * height * channels);
    anydsl::Array<float> ref_mat(width * height * channels);
    anydsl::Array<float> out_mat(width * height * channels);

    // Buffer for dumped data on gpu
    anydsl::Array<float> img_mat_gpu;
    anydsl::Array<float> alb_mat_gpu;
    anydsl::Array<float> nrm_mat_gpu;
    anydsl::Array<float> out_mat_gpu;
    anydsl::Array<float> weights_gpu;
    anydsl::Array<float> biases_gpu;

    // GPU mask
    int32_t mask_dst = 0x01;

    // Allocate memory on gpu for network to use
    auto necess_mem = get_necessary_mem(width, height);
    auto nn_memory  = anydsl::Array<uint8_t>(mask_dst, reinterpret_cast<uint8_t*>(anydsl_alloc(mask_dst, necess_mem * sizeof(uint8_t))), necess_mem);

    // Read in weights and biases of neural network
    read_in(&weights, &biases);

    // Read in dumped data of forwarding routine
    read_in_matrix_hwc(img_mat.data(), dump_dir + "img_mat.txt", channels, height, width);
    read_in_matrix_hwc(alb_mat.data(), dump_dir + "alb_mat.txt", channels, height, width);
    read_in_matrix_hwc(nrm_mat.data(), dump_dir + "nrm_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), dump_dir + "ref_mat.txt", channels, height, width);

    // Copy everything over to the GPU
    img_mat_gpu = copy_to_device(mask_dst, img_mat.data(), img_mat.size());
    alb_mat_gpu = copy_to_device(mask_dst, alb_mat.data(), alb_mat.size());
    nrm_mat_gpu = copy_to_device(mask_dst, nrm_mat.data(), nrm_mat.size());
    out_mat_gpu = copy_to_device(mask_dst, out_mat.data(), out_mat.size()); // No copy necessary, but size is surely right like this
    weights_gpu = copy_to_device(mask_dst, weights.data(), weights.size());
    biases_gpu  = copy_to_device(mask_dst, biases.data(),  biases.size());

    // Release data on cpu memory that's not needed anymore
    // img_mat.release();
    // alb_mat.release();
    // nrm_mat.release();
    // weights.release();
    // biases.release();

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        denoise(&img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
    }

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    denoise(&img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        // copy result back to cpu
        anydsl_copy(mask_dst, out_mat_gpu.data(), 0, 0, out_mat.data(), 0, sizeof(float) * out_mat.size());
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
        denoise(&img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    out_mat.release();
    ref_mat.release();


    img_mat_gpu.release();
    alb_mat_gpu.release();
    nrm_mat_gpu.release();
    out_mat_gpu.release();
    weights_gpu.release();
    biases_gpu.release();
    nn_memory.release();
}

/**
 * Benchmarks one forwarding through the denoising neural network with dumped input data
 *
 * Input size: 512x512
 */
void bench_denoiseDump512(int times, int heatup_iterations, bool correct_check, bool gpu) {
    // Constant values used in the dumped data
    const int width  = 512;
    const int height = 512;
    const int channels = 3;
    const std::string dump_dir = BENCH_DUMP_DIR "/dumped_data/denoising_512x512/";

    if (gpu)
        bench_denoiseDump_gpu(width, height, channels, dump_dir, times, heatup_iterations, correct_check);
    else
        bench_denoiseDump(width, height, channels, dump_dir, times, heatup_iterations, correct_check);
}

/**
 * Benchmarks one forwarding through the denoising neural network with dumped input data
 *
 * Input size: 1024x1024
 */
void bench_denoiseDump1k(int times, int heatup_iterations, bool correct_check, bool gpu) {
    // Constant values used in the dumped data
    const int width  = 1024;
    const int height = 1024;
    const int channels = 3;
    const std::string dump_dir = BENCH_DUMP_DIR "/dumped_data/denoising_1kx1k/";

    if (gpu)
        bench_denoiseDump_gpu(width, height, channels, dump_dir, times, heatup_iterations, correct_check);
    else
        bench_denoiseDump(width, height, channels, dump_dir, times, heatup_iterations, correct_check);
}


/* Needed in artic to measure how much time is spent in specific code snippets. */
int64_t clock_us() {
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#define CPU_FREQ 4e9
    __asm__ __volatile__("xorl %%eax,%%eax \n    cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
    return __rdtsc() * int64_t(1000000) / int64_t(CPU_FREQ);
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
}
