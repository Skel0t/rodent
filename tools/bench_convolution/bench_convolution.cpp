#include <iostream>
#include <chrono>

#include <anydsl_runtime.hpp>

#include "bench_interface.h"
#include "../../src/driver/denoiser/nn_io.h"
#include "../../src/driver/denoise.h"

static inline void check_arg(int argc, char** argv, int arg, int n) {
    if (arg + n >= argc)
        error("Option '", argv[arg], "' expects ", n, " arguments, got ", argc - arg);
}

int main(int argc, char** argv) {
    std::string backend = "cpu";
    std::string bench   = "denoise512";
    size_t iterations   = 1;
    size_t warmup       = 0;
    bool correct_check  = false;

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (!strcmp(argv[i], "--backend")) {
                check_arg(argc, argv, i, 1);
                backend = argv[++i];
            } else if (!strcmp(argv[i], "--bench")) {
                check_arg(argc, argv, i, 1);
                bench = argv[++i];
            } else if (!strcmp(argv[i], "--iterations")) {
                check_arg(argc, argv, i, 1);
                iterations = strtoul(argv[++i], nullptr, 10);
            } else if (!strcmp(argv[i], "--warmup")) {
                check_arg(argc, argv, i, 1);
                warmup = strtoul(argv[++i], nullptr, 10);
            } else if (!strcmp(argv[i], "--check")) {
                correct_check = true;
            } else {
                error("Unknown option '", argv[i], "'");
            }
            continue;
        }
        error("Unexpected argument '", argv[i], "'");
    }

    if (bench == "denoise512")
        bench_denoiseDump512(iterations, warmup, backend, correct_check);
    else if (bench == "denoise1k" || bench == "denoise1024")
        bench_denoiseDump1k(iterations, warmup, backend, correct_check);
    else if (bench == "sres512")
        bench_sresDump512(iterations, warmup, backend, correct_check);
    else if (bench == "sres1k" || bench == "sres1024")
        bench_sresDump1k(iterations, warmup, backend, correct_check);

    return 0;
}

/******** DENOISING ********/

void bench_denoiseDump_cpu(std::string backend, int width, int height, int channels, std::string dump_dir, std::string net_dir, int times, int heat_up_iterations, bool correct_check) {
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
    read_in(&weights, &biases, net_dir);

    // Read in dumped data of forwarding routine
    read_in_matrix_hwc(img_mat.data(), dump_dir + "img_mat.txt", channels, height, width);
    read_in_matrix_hwc(alb_mat.data(), dump_dir + "alb_mat.txt", channels, height, width);
    read_in_matrix_hwc(nrm_mat.data(), dump_dir + "nrm_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), dump_dir + "ref_mat.txt", channels, height, width);

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        denoise(backend, &img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
    }

    reset_counters();

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    denoise(backend, &img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        for (size_t chn = 0; chn < channels; chn++) {
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    if (abs(out_mat.data()[row * width * channels + col * channels + chn] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-5) {
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

    for (int i = 1; i < times; i++) {
        ticks = std::chrono::high_resolution_clock::now();
        denoise(backend, &img_mat, &alb_mat, &nrm_mat, &nn_memory, &out_mat, width, height, &weights, &biases);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    print_counters();
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

void bench_denoiseDump_gpu(std::string backend, int width, int height, int channels, std::string dump_dir, std::string net_dir, int times, int heat_up_iterations, bool correct_check) {
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
    read_in(&weights, &biases, net_dir);

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
    img_mat.release();
    alb_mat.release();
    nrm_mat.release();
    weights.release();
    biases.release();

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        denoise(backend, &img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
    }

    reset_counters();

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    denoise(backend, &img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
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

    for (int i = 1; i < times; i++) {
        ticks = std::chrono::high_resolution_clock::now();
        denoise(backend, &img_mat_gpu, &alb_mat_gpu, &nrm_mat_gpu, &nn_memory, &out_mat_gpu, width, height, &weights_gpu, &biases_gpu);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    print_counters();
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    out_mat.release();
    ref_mat.release();

    nn_memory.release();
    img_mat_gpu.release();
    alb_mat_gpu.release();
    nrm_mat_gpu.release();
    out_mat_gpu.release();
    weights_gpu.release();
    biases_gpu.release();
}

/**
 * Benchmarks one forwarding through the denoising neural network with dumped input data
 *
 * Input size: 512x512
 * Free memory required: 0.7GB
 */
void bench_denoiseDump512(int times, int heatup_iterations, std::string backend, bool correct_check, std::string dump_dir, std::string net_dir) {
    // Constant values used in the dumped data
    const int width  = 512;
    const int height = 512;
    const int channels = 3;

    if (backend == "cuda" || backend == "cublas" || backend == "cublaslt")
        bench_denoiseDump_gpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
    else
        bench_denoiseDump_cpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
}

/**
 * Benchmarks one forwarding through the denoising neural network with dumped input data
 *
 * Input size: 1024x1024
 * Free memory required: 2.7GB
 */
void bench_denoiseDump1k(int times, int heatup_iterations, std::string backend, bool correct_check, std::string dump_dir, std::string net_dir) {
    // Constant values used in the dumped data
    const int width  = 1024;
    const int height = 1024;
    const int channels = 3;

    if (backend == "cuda" || backend == "cublas" || backend == "cublaslt")
        bench_denoiseDump_gpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
    else
        bench_denoiseDump_cpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
}

/******** SRES ********/

void read_in_sres(anydsl::Array<float>* weights, anydsl::Array<float>* biases, std::string network_path) {
    const int memsize1 = 5 * 5 *  3 * 32;
    const int memsize2 = 3 * 3 * 32 * 64;
    const int memsize3 = 3 * 3 * 64 * 64;
    const int memsize4 = 5 * 5 * 64 * 32;
    const int memsize5 = 3 * 3 * 32 * 32;
    const int memsize6 = 3 * 3 * 32 * 32;
    const int memsize7 = 3 * 3 * 32 *  3;

    const int memsize_weights = memsize1 + memsize2 + memsize3 + memsize4 +
                                memsize5 + memsize6 + memsize7;
    const int memsize_biases = 32 + 64 + 64 + 32 + 32 + 32;

    *weights = anydsl::Array<float>(memsize_weights);
    *biases  = anydsl::Array<float>(memsize_biases);

    int offset = 0;

    float* w = weights->data();
    read_in_weigths_chw(w, offset, network_path + "/conv1.txt",    3, 32, 5);
    offset += memsize1;
    read_in_weigths_chw(w, offset, network_path + "/conv2.txt",   32, 64, 3);
    offset += memsize2;
    read_in_weigths_chw(w, offset, network_path + "/conv3.txt",   64, 64, 3);
    offset += memsize3;
    read_in_weigths_chw(w, offset, network_path + "/upconv1.txt", 64, 32, 5);
    offset += memsize4;
    read_in_weigths_chw(w, offset, network_path + "/conv4.txt",   32, 32, 3);
    offset += memsize5;
    read_in_weigths_chw(w, offset, network_path + "/conv5.txt",   32, 32, 3);
    offset += memsize6;
    read_in_weigths_chw(w, offset, network_path + "/conv6.txt",   32,  3, 3);

    float* b = biases->data();
    offset = 0;
    read_in_biases(b, offset, network_path + "/c1bias.txt",  32);
    read_in_biases(b, offset, network_path + "/c2bias.txt",  64);
    read_in_biases(b, offset, network_path + "/c3bias.txt",  64);
    read_in_biases(b, offset, network_path + "/uc1bias.txt", 32);
    read_in_biases(b, offset, network_path + "/c4bias.txt",  32);
    read_in_biases(b, offset, network_path + "/c5bias.txt",  32);
}

void sres_forward_wrap(std::string denoising_backend, anydsl::Array<float>* img_mat, anydsl::Array<float>* out_mat, anydsl::Array<float>* weights,
                 anydsl::Array<float>* biases, int32_t width, int32_t height, anydsl::Array<uint8_t>* memory) {
    Buffer img_mat_buf = { (int8_t*) img_mat->data(),  img_mat->size(),  img_mat->device()  };
    Buffer weights_buf = { (int8_t*) weights->data(),  weights->size(),  weights->device()  };
    Buffer biases_buf  = { (int8_t*) biases->data(),   biases->size(),   biases->device()   };
    Buffer out_mat_buf = { (int8_t*) out_mat->data(),  out_mat->size(),  out_mat->device()  };
    Buffer memory_buf  = { (int8_t*) memory->data(),   memory->size(),   memory->device()   };

    if (denoising_backend == "cuda")
        sres_forward_cuda(&img_mat_buf, &out_mat_buf, &weights_buf, &biases_buf, width, height, &memory_buf);
    else if (denoising_backend == "cublas")
        sres_forward_cublas(&img_mat_buf, &out_mat_buf, &weights_buf, &biases_buf, width, height, &memory_buf);
    else if (denoising_backend == "cublaslt")
        sres_forward_cublaslt(&img_mat_buf, &out_mat_buf, &weights_buf, &biases_buf, width, height, &memory_buf);
    else if (denoising_backend == "cpu")
        sres_forward_cpu(&img_mat_buf, &out_mat_buf, &weights_buf, &biases_buf, width, height, &memory_buf);
    else if (denoising_backend == "oneapi")
        sres_forward_oneapi(&img_mat_buf, &out_mat_buf, &weights_buf, &biases_buf, width, height, &memory_buf);
}

void bench_sresDump_cpu(std::string backend, int width, int height, int channels, std::string dump_dir, std::string net_dir, int times, int heat_up_iterations, bool correct_check) {
    // Buffer for weights and biases
    anydsl::Array<float> weights;
    anydsl::Array<float> biases;

    // Buffer for dumped data
    anydsl::Array<float> img_mat(width * height * channels);
    anydsl::Array<float> ref_mat(4 * width * height * channels);
    anydsl::Array<float> out_mat(4 * width * height * channels);

    auto nn_memory = anydsl::Array<uint8_t>(get_sres_mem(width, height));

    // Read in weights and biases of neural network
    read_in_sres(&weights, &biases, net_dir);

    // Read in dumped data of forwarding routine
    read_in_matrix_chw(img_mat.data(), dump_dir + "img_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), dump_dir + "ref_mat.txt", channels, 2 * height, 2 * width);

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        sres_forward_wrap(backend, &img_mat, &out_mat, &weights, &biases, width, height, &nn_memory);
    }

    reset_counters();

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    sres_forward_wrap(backend, &img_mat, &out_mat, &weights, &biases, width, height, &nn_memory);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        for (size_t chn = 0; chn < channels; chn++) {
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    if (abs(out_mat.data()[chn * width * height + row * width + col] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-5) {
                        std::cout << "Diff at:\t" << chn << "\t" << col << "\t" << row << "\t(chn, x, y)" << std::endl;
                        std::cout << "Was:\t\t" << out_mat.data()[chn * width * height + row * width + col] << "\n"
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
        sres_forward_wrap(backend, &img_mat, &out_mat, &weights, &biases, width, height, &nn_memory);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    print_counters();
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    img_mat.release();
    out_mat.release();
    ref_mat.release();
    weights.release();
    biases.release();
    nn_memory.release();
}

void bench_sresDump_gpu(std::string backend, int width, int height, int channels, std::string dump_dir, std::string net_dir, int times, int heat_up_iterations, bool correct_check) {
    // GPU mask
    int32_t mask_dst = 0x01;

    // Buffer for weights and biases
    anydsl::Array<float> weights;
    anydsl::Array<float> biases;

    // Buffer for dumped data on cpu
    anydsl::Array<float> img_mat(width * height * channels);
    anydsl::Array<float> ref_mat(4 * width * height * channels);
    anydsl::Array<float> out_mat(4 * width * height * channels);

    // Buffer for dumped data on gpu
    anydsl::Array<float> img_mat_gpu;
    anydsl::Array<float> out_mat_gpu;
    anydsl::Array<float> weights_gpu;
    anydsl::Array<float> biases_gpu;

    auto necess_mem = get_sres_mem(width, height);

    // Read in weights and biases of neural network
    read_in_sres(&weights, &biases, net_dir);

    // Read in dumped data of forwarding routine
    read_in_matrix_chw(img_mat.data(), dump_dir + "img_mat.txt", channels, height, width);
    read_in_matrix_chw(ref_mat.data(), dump_dir + "ref_mat.txt", channels, 2 * height, 2 * width);

    // Copy everything over to the GPU
    img_mat_gpu = copy_to_device(mask_dst, img_mat.data(), img_mat.size());
    out_mat_gpu = copy_to_device(mask_dst, out_mat.data(), out_mat.size()); // No copy necessary, but size is surely right like this
    weights_gpu = copy_to_device(mask_dst, weights.data(), weights.size());
    biases_gpu  = copy_to_device(mask_dst, biases.data(),  biases.size());

    auto nn_memory  = anydsl::Array<uint8_t>(mask_dst, reinterpret_cast<uint8_t*>(anydsl_alloc(mask_dst, necess_mem * sizeof(uint8_t))), necess_mem);

    // Release data on cpu memory that's not needed anymore
    img_mat.release();
    weights.release();
    biases.release();

    // Heat-Up iterations
    for (int i = 0; i < heat_up_iterations; i++) {
        sres_forward_wrap(backend, &img_mat_gpu, &out_mat_gpu, &weights_gpu, &biases_gpu, width, height, &nn_memory);
    }

    reset_counters();

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    sres_forward_wrap(backend, &img_mat_gpu, &out_mat_gpu, &weights_gpu, &biases_gpu, width, height, &nn_memory);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();

    // Check if result is correct (only for the first time)
    if (correct_check) {
        anydsl_copy(mask_dst, out_mat_gpu.data(), 0, 0, out_mat.data(), 0, sizeof(float) * out_mat.size());
        for (size_t chn = 0; chn < channels; chn++) {
            for (size_t row = 0; row < height; row++) {
                for (size_t col = 0; col < width; col++) {
                    if (abs(out_mat.data()[chn * width * height + row * width + col] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-4) {
                        std::cout << "Diff at:\t" << chn << "\t" << col << "\t" << row << "\t(chn, x, y)" << std::endl;
                        std::cout << "Was:\t\t" << out_mat.data()[chn * width * height + row * width + col] << "\n"
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
                sres_forward_wrap(backend, &img_mat_gpu, &out_mat_gpu, &weights_gpu, &biases_gpu, width, height, &nn_memory);
        elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    }
outer_break:
    print_counters();
    std::cout << "Average Time:\t" << (elapsed_ms / times) << " ms" << std::endl;

    out_mat.release();
    ref_mat.release();

    nn_memory.release();
    img_mat_gpu.release();
    out_mat_gpu.release();
    weights_gpu.release();
    biases_gpu.release();
}

/**
 * Benchmarks one forwarding through a sample upsampling neural network with dumped input data
 *
 * Input size: 256x256
 * Output size: 512x512
 * Free memory required: 1.6GB
 */
void bench_sresDump512(int times, int heatup_iterations, const std::string backend, bool correct_check, std::string dump_dir, std::string net_dir) {
    const int width  = 256;
    const int height = 256;
    const int channels = 3;

    if (backend == "cuda" || backend == "cublas" || backend == "cublaslt")
        bench_sresDump_gpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
    else
        bench_sresDump_cpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
}

/**
 * Benchmarks one forwarding through a sample upsampling neural network with dumped input data
 *
 * Input size: 512x512
 * Output size: 1024x1024
 * Free memory required: 6.5GB
 */
void bench_sresDump1k(int times, int heatup_iterations, const std::string backend, bool correct_check, std::string dump_dir, std::string net_dir) {
    const int width  = 512;
    const int height = 512;
    const int channels = 3;

    if (backend == "cuda" || backend == "cublas" || backend == "cublaslt")
        bench_sresDump_gpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
    else
        bench_sresDump_cpu(backend, width, height, channels, dump_dir, net_dir, times, heatup_iterations, correct_check);
}

/******** MATMUL SRES ********/

void sres_matmul_dump_wrap(anydsl::Array<float>* in_mat, anydsl::Array<float>* weights, float* biases, anydsl::Array<float>* out_mat) {
    Buffer in_mat_buf  = { (int8_t*) in_mat->data(),   in_mat->size(),   in_mat->device()   };
    Buffer weights_buf = { (int8_t*) weights->data(),  weights->size(),  weights->device()  };
    Buffer out_mat_buf = { (int8_t*) out_mat->data(),  out_mat->size(),  out_mat->device()  };

    sres_dump(&in_mat_buf, &weights_buf, biases, &out_mat_buf);
}

/**
 * Only (!) benchmarking a sample convolution with dumped data
 *
 * Activation function used is "id"
 *
 * Input  Size: 960 x 540
 * Output Size: 1920 x 1080
 */
void bench_sres_matmulDump(int times, int heatup_iterations, bool correct_check) {
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
    int offset = 0;
    read_in_weigths_chw(weights.data(), 0, BENCH_DUMP_DIR "/dumped_data/sres_network/upconv1.txt", 64, 32, 5);
    read_in_biases(biases, offset, BENCH_DUMP_DIR "/dumped_data/sres_network/uc1bias.txt", 32);

    // Read in dumped data of hidden layers
    read_in_matrix_chw(in_mat.data(), BENCH_DUMP_DIR "/dumped_data/sres_matmul/conv4_in.txt", in_channels, height, width);
    read_in_matrix_chw(ref_mat.data(), BENCH_DUMP_DIR "/dumped_data/sres_matmul/conv4_out.txt", out_channels, height, width);

    // Heat-Up iterations
    for (int i = 0; i < heatup_iterations; i++) {
        sres_matmul_dump_wrap(&in_mat, &weights, biases, &out_mat);
    }

    // Time execution
    auto ticks = std::chrono::high_resolution_clock::now();
    sres_matmul_dump_wrap(&in_mat, &weights, biases, &out_mat);
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
        sres_matmul_dump_wrap(&in_mat, &weights, biases, &out_mat);
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

/******** IMAGE TO COLUMN ********/

void im2col_wrap(anydsl::Array<float>* in_mat, anydsl::Array<float>* out_mat) {
    Buffer in_mat_buf  = { (int8_t*) in_mat->data(),   in_mat->size(),   in_mat->device()   };
    Buffer out_mat_buf = { (int8_t*) out_mat->data(),  out_mat->size(),  out_mat->device()  };

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

/******** TESTS ********/

void matmul_cublas_test_wrap(anydsl::Array<float>* d_mat_a, anydsl::Array<float>* d_mat_b, anydsl::Array<float>* d_mat_c) {
    Buffer d_mat_a_b = { (int8_t*) d_mat_a->data(),   d_mat_a->size(),   d_mat_a->device()   };
    Buffer d_mat_b_b = { (int8_t*) d_mat_b->data(),   d_mat_b->size(),   d_mat_b->device()   };
    Buffer d_mat_c_b = { (int8_t*) d_mat_c->data(),   d_mat_c->size(),   d_mat_c->device()   };

    matmul_cublas_test(&d_mat_a_b, &d_mat_b_b, &d_mat_c_b);
}

void cublas_test() {
    float h_mat_a[] = {
        1.,    2.,     0.,
        4.,    5.,     6.,
        7.,    8.,     9.,
        10.,   11.,    12.,
        13.,   14.,    15.
    };
    float h_mat_b[] = {
        1.,     2.,     3.,     4.,
        5.,     6.,     7.,     8.,
        9.,     10.,    11.,    12.
    };
    anydsl::Array<float> h_mat_c(20);

    anydsl::Array<float> d_mat_a = copy_to_device(0x01, h_mat_a, 15);
    anydsl::Array<float> d_mat_b = copy_to_device(0x01, h_mat_b, 12);
    anydsl::Array<float> d_mat_c = copy_to_device(0x01, h_mat_c.data(), 20);

    matmul_cublas_test_wrap(&d_mat_a, &d_mat_b, &d_mat_c);

    anydsl_copy(0x01, d_mat_c.data(), 0, 0, h_mat_c.data(), 0, sizeof(float) * h_mat_c.size());

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 4; i++) {
            std::cout << h_mat_c.data()[4 * j + i] << "\t";
        }
        std::cout << std::endl;
    }

    d_mat_a.release();
    d_mat_b.release();
    d_mat_c.release();
    h_mat_c.release();
}
