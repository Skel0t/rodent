# Rodent

Rodent is a BVH traversal library and renderer implemented using the AnyDSL compiler framework (https://anydsl.github.io/).

This fork integrates a denoising autoencoder into the renderer.

# Building

The dependencies for Rodent are: CMake, AnyDSL, libpng, SDL2, and optionally the Embree sources for the benchmarking tools.

Additional dependencies for the denoising are: intel oneAPI, CUDA, cuBLAS.
The build will fail if either of these is not installed.
However, the denoising can work independently without use of any of these libraries
by native implementations in AnyDSL. In fact, our own CPU implementation is
(by our own benchmarks) even faster than using oneAPI.

Once the dependencies are installed, use the following commands to build the project:

    mkdir build
    cd build
    # Set the OBJ file to use with the SCENE_FILE variable
    # By default, SCENE_FILE=../testing/cornell_box.obj
    cmake .. -DSCENE_FILE=myfile.obj
    # Optional: Create benchmarking tools for Embree and BVH extractor tools
    # cmake .. -DEMBREE_ROOT_DIR=<path to Embree sources>
    make


# Testing

This section assumes that the current directory is the build directory. To run rodent, just type:

    bin/rodent

You may want to change the initial camera parameters using the command line options `--eye`, `--dir` and `--up`. Also, if denoising should be done at the end of the rendering
process, add the `--denoise <output_path>` flag. Additionally, live denoising can
be activated by executing rodent with the `--live` flag along with the `--denoise` flag.

Run `bin/rodent --help` to get a full list of options.

Denoising in rodent can be executed with 4 different mappings that need to have different
denoising platforms defined in `src/driver/driver.cpp`:
 - CPU Mapping: `get_cpu_nn()` with `anydsl::Platform::Host`
 - oneAPI Mapping: `get_oneapi_nn()` with `anydsl::Platform::Host`
 - CUDA Mapping: `get_cuda_nn()` with `anydsl::Platform::Cuda`
 - cuBLAS Mapping: `get_cublas_nn()` with `anydsl::Platform::Cuda`

For benchmarking the forward propagation implementation we provide four pre-written
benchmarks. These can be found in `tools/bench_convolution/bench_convolution.cpp`.
The provided benchmarks each propagate real dumped data through a sample neural
network and time how long the execution takes.

All benchmarks accept 4 arguments:
 - Number of times to average over
 - Number of warmup iterations before timing
 - Boolean, whether the computed result should be compared to a precalculated reference
 - Boolean, whether the execution should be done on the GPU

Note that, when executing on the GPU, you also need to change the neural network
method mapping in the corresponding `make_nn()` function to a GPU function mapping
(`get_cuda_nn()` or `get_cublas_nn()`).
