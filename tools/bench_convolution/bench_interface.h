#ifndef BENCH_INTERFACE_H
#define BENCH_INTERFACE_H
#include <anydsl_runtime.hpp>

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

/* Functions that ought to be called in main for benchmarking and testing purposes. */
void bench_denoiseDump512(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);
void bench_denoiseDump1k(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);

void bench_sresDump512(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);
void bench_sresDump1k(int times = 1, int heatup_iterations = 0, bool correct_check = false, bool gpu = false);

void bench_sres_matmulDump(int times = 1, int heatup_iterations = 0, bool correct_check = false);

void bench_im2col(bool correct_check = false);

void cublas_test();
/* End of functions to be called in main. */

template <typename T>
anydsl::Array<T> copy_to_device(int32_t dev, const T* data, size_t n) {
    anydsl::Array<T> array(dev, reinterpret_cast<T*>(anydsl_alloc(dev, n * sizeof(T))), n);
    anydsl_copy(0, data, 0, dev, array.data(), 0, sizeof(T) * n);
    return array;
}

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif /* BENCH_INTERFACE_H */
