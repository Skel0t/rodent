# Rodent

Rodent is a BVH traversal library and renderer implemented using the AnyDSL compiler framework (https://anydsl.github.io/).

This fork integrates a denoising autoencoder into the renderer.

# Building

The dependencies for Rodent are: CMake, AnyDSL, libpng, SDL2, and optionally the Embree sources for the benchmarking tools.

Additional dependencies for the denoising are: intel oneAPI, CUDA, cuBLAS.
The build will fail if either of these is not installed.
However, the denoising can work independently without the use of any of these libraries by native implementations in AnyDSL.

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

This section assumes that the current directory is the build directory. To run Rodent, just type:

    bin/rodent

You may want to change the initial camera parameters using the command line
options `--eye`, `--dir`, and `--up`. Also, if denoising should be done at the
end of the rendering process, add the `--denoise` flag. Additionally, live
denoising can be activated by executing Rodent with the `--live` flag along
with the `--denoise` flag. The denoising backend can be chosen with `--dback`.

Run `bin/rodent --help` to get a full list of options.

**Note:** Allowing to choose the denoising backend from the command line results in high
compilation times, as the AnyDSL compiler needs to compile and partially evaluate
for all available platforms when building. It is possible to remove this by
setting the forwarding functions one does not need in `src/driver/denoiser/interface.art` to dummy functions that do nothing.

# Benchmarking

For benchmarking the forward propagation implementation, we provide four pre-written
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
For performance counters and metrics, the dependencies `perf` (CPU) and `nvprof` (GPU)
are required. It might be necessary to run the script elevated for the metrics as
otherwise, it might not have the required rights to read them.

Profiling where time is spent during forwarding can be done by setting
```
static cpu_profiling_enabled = true;
static cpu_profiling_serial  = true;
```
in `src/core/cpu_common.impala`. Currently, these flags are set to `false`. When
executing Rodent on the CPU, both flags should be set to false since Rodent uses them
for profiling as well.

# Intel's Open Image Denoise

As the goal was to integrate a native denoiser into Rodent, leaving the dependency of having installed OIDN would not add much value. Furthermore, it would make the code less readable. Therefore, the integration of OIDN, which was described in my thesis, is not included in the end result anymore.

However, I left the git history in this submission such that one can fall back to the OIDN integration and test it. The corresponding commit can be reached with

```
git checkout 6aeabbcff3f406b5e93d14450bde798ff98ceedc
```

However, the CLI worked slightly differently at that commit. To denoise with OIDN, one must first set the OIDN cmake flag. Then, by adding the `--oidn` flag to Rodent's execution, live denoising should work.

To return to the actual implementation again, use

```
git checkout artic
```

# About the Segfault when executing Rodent on the CPU

When Rodent is executed on the CPU (default), a segfault happens at the very end,
when rendering is stopped.
This segfault already happens in the default version that I forked to implement
the denoiser. Thus, my code should be segfault-free and also release all memory
allocated by me. Executing Rodent with, for example, the `nvvm`
backend does not cause a segfault.
