set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE FILEPATH "Compiler to use with CUDA")
cuda_compile(AILA_TRAVERSAL
    CudaTracerKernels.hpp
    kepler_dynamic_fetch.cu
    OPTIONS "-O3;-std=c++11;--expt-extended-lambda;-arch=sm_52;-I${CMAKE_CURRENT_SOURCE_DIR}/../common")

add_executable(bench_aila bench_aila.cpp ${AILA_TRAVERSAL})
target_include_directories(bench_aila PUBLIC ../../src)
target_link_libraries(bench_aila PUBLIC traversal_interface ${CUDA_LIBRARIES} ${AnyDSL_runtime_LIBRARIES})
