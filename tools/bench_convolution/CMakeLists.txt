set(NN_ART_SRCS
    ../../src/driver/denoiser/interface.art
    ../../src/driver/denoiser/iterate.art
    ../../src/driver/denoiser/kernel.art
    ../../src/driver/denoiser/matrix.art
    ../../src/driver/denoiser/nn.art
    ../../src/core/common.impala
    ../../src/core/cpu_common.impala
    ../../src/core/sort.impala
    ../../src/core/color.impala
    ../../src/core/random.impala
    ../../src/core/matrix.impala
    ../../src/core/vector.impala
    bench_convolution.art)

set(NN_CPP_SRCS
    ../../src/driver/denoiser/nn_io.h
    ../../src/driver/denoiser/nn_io.cpp
    bench_interface.h
    bench_convolution.cpp
)

anydsl_runtime_wrap(NN_OBJS
    NAME "bench_convolution"
    FRONTEND "artic"
    CLANG_FLAGS ${CLANG_FLAGS}
    FILES ${NN_ART_SRCS})

set(BLA_VENDOR Intel10_64lp)

find_package(LAPACK REQUIRED)

add_executable(bench_convolution ${INTERFACE_OBJS} ${NN_OBJS} ${NN_CPP_SRCS})
target_include_directories(bench_convolution PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(bench_convolution ${AnyDSL_runtime_LIBRARIES} ${LAPACK_LIBRARIES})
target_compile_definitions(bench_convolution PUBLIC BENCH_DUMP_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
