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


# Testing and Benchmarking

This section assumes that the current directory is the build directory. To run rodent, just type:

    bin/rodent

You may want to change the initial camera parameters using the command line
options `--eye`, `--dir` and `--up`. Also, if denoising should be done at the
end of the rendering process, add the `--denoise` flag. Additionally, live
denoising can be activated by executing rodent with the `--live` flag alongside
with the `--denoise` flag. The denoising backend can be chosen with `--dback`.

Run `bin/rodent --help` to get a full list of options.

**Note:** Allowing to choose the denoising backend from command line results in high
compilation times, as the AnyDSL compiler needs to compile and partially evaluate
for all available platforms when building. It is possible to remove this, by
setting the forwarding functions in `src/driver/denoiser/interface.art` one does
not need to dummy functions which do nothing.

For benchmarking the forward propagation implementation we provide four pre-written
benchmarks. These can be found in `tools/bench_convolution/`.
The provided benchmarks each propagate real dumped data through a sample neural
network and time how long the execution takes.

All benchmarks accept 4 arguments:
 - Number of times to average over
 - Number of warmup iterations before timing
 - Backend to use
 - Boolean, whether the computed result should be compared to a precalculated reference

Run a specific benchmark with `bin/bench_convolution` and arguments:
 - `--bench benchmark` to choose a benchmark (denoise512 denoise1k sres512 sres1k)
 - `--iterations iter` to choose the number of iterations
 - `--warmup iter` to choose the number of warmup iterations before timing
 - `--backend backend` to choose the backend (cpu oneapi cuda cublas cublaslt)
 - `--check` to activate the check if the calculated result is correct

We also provide a script in `bin/bench_convolution/bench_all.sh` which by default
runs all benchmarks on all devices and prints the results in a textfile.
This script additionally measures metrics and performance counters for GPU and CPU.
For performance counters and metrics the dependencies `perf` (CPU) and `nvprof` (GPU)
are required. It might be necessary to run the script elevated for the metrics as
otherwise it might not have the required access rights to read them.

Profiling where time is spent during forwarding can be done with by setting
```
static cpu_profiling_enabled = true;
static cpu_profiling_serial  = true;
```
in `src/core/cpu_common.impala`. Currently these flags are set. When
executing Rodent, both these flagst should be set to false, as Rodent uses them
for profiling as well.
