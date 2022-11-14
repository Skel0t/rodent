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

The forward propagation of the denoising neural network was benchmarked on an AMD Ryzen 9 5900X with 3200MHz DDR4 RAM and an nVidia Geforce GTX 1080. On the same build, another neural network (a super resolution network was tested) to achieve the following results:

| Benchmark           | Backend        | Execution Time | % spent in mm | % spent in im2col
--------------------- | -------------- | -------------- | ------------- | -----------------
|denoise on 512x512   | CPU AnyDSL     | 366 ms         | 58            | 41
|denoise on 512x512   | CPU oneAPI     | 344 ms         | 54            | 44
|denoise on 512x512   | GPU AnyDSL     | 114 ms         | 87            | 12
|denoise on 512x512   | GPU cuBLAS     | 38  ms         | 64            | 34

| Benchmark           | Backend        | Execution Time | % spent in mm | % spent in im2col
--------------------- | -------------- | -------------- | ------------- | -----------------
|denoise on 1024x1024 | CPU AnyDSL     | 1467 ms        | 53            | 46
|denoise on 1024x1024 | CPU oneAPI     | 1403 ms        | 50            | 48
|denoise on 1024x1024 | GPU AnyDSL     | 475  ms        | 88            | 11
|denoise on 1024x1024 | GPU cuBLAS     | 135  ms        | 60            | 38

| Benchmark           | Backend        | Execution Time | % spent in mm | % spent in im2col
--------------------- | -------------- | -------------- | ------------- | -----------------
|S.resolution (4x) to 512 | CPU AnyDSL | 319 ms         | 51            | 48
|S.resolution (4x) to 512 | CPU oneAPI | 308 ms         | 46            | 53
|S.resolution (4x) to 512 | GPU AnyDSL | 84  ms         | 83            | 16
|S.resolution (4x) to 512 | GPU cuBLAS | 30  ms         | 57            | 42

| Benchmark           | Backend        | Execution Time | % spent in mm | % spent in im2col
--------------------- | -------------- | -------------- | ------------- | -----------------
|S.resolution (4x) to 1024 | CPU AnyDSL| 1315 ms        | 45            | 54
|S.resolution (4x) to 1024 | CPU oneAPI| 1253 ms        | 42            | 57
|S.resolution (4x) to 1024 | GPU AnyDSL| 351  ms        | 84            | 14
|S.resolution (4x) to 1024 | GPU cuBLAS| 113  ms        | 55            | 44

The results in these tables were obtained by following these steps:

1. Train a NN with the help of another library, e.g., PyTorch
2. Dump the weights into a binary, with a loop over the parameters similar to one like the following example:
```cpp
for (int i = 0; i < out_channels; i++) {
int k_nr = i * in_channels * ksize * ksize;
    for (int j = 0; j < in_channels; j++) {
        for (int y = 0; y < ksize; y++) {
            int k_row = y * ksize;
            for (int x = 0; x < ksize; x++) {
                const float val = ptr[k_nr + k_row + x + j * ksize * ksize];
                file.write((char*) &val, sizeof(float));
            }
        }
    }
}
```
3. Optional: Dump example forward propagation to later compare your result against (to check that the calculation is correct)
4. Read in the weights with the help of an appropiate function in `src/driver/nn_io.h`
5. Use the weights in a network model design like shown in `src/driver/nn.art`
6. Optional: In `src/core/cpu_common.impala`, to measure where time was spent, set:
```rust
static cpu_profiling_enabled = true;
static cpu_profiling_serial  = true;
```
7. Write a script that can time the execution over several runs, with optionally some profiling metrics and a warmup period
