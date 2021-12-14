# Rodent

Rodent is a BVH traversal library and renderer implemented using the AnyDSL compiler framework (https://anydsl.github.io/).

This fork integrates a denoising autoencoder into the renderer.

**Note**: This branch is the preliminary version that integrates OIDN.
Therefore, the code might not be as cleaned up as in the final version on the artic branch.

# Building

The dependencies for Rodent are: CMake, AnyDSL, libpng, SDL2, and optionally the Embree sources for the benchmarking tools.

Additional dependencies for the denoising are: OIDN.
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
options `--eye`, `--dir`, and `--up`. Also, to activate denoising, add the `--oidn` flag.

Run `bin/rodent --help` to get a full list of options.


# Intel's Open Image Denoise

This branch is the preliminary version that integrates OIDN.

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
