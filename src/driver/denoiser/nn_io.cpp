#include <iostream>
#include <fstream>
#include <stdexcept>

#include "nn_io.h"

void read_in_weigths_hwc(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        for (int i = 0; i < out_channels; i++) {
            int k_nr = i * in_channels * ksize * ksize;
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize * in_channels;
                    for (int x = 0; x < ksize; x++) {
                        f >> buffer[offset + k_nr + k_row + x * in_channels + j];
                    }
                }
            }
        }
    }
}

void read_in_weigths_chw(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        for (int i = 0; i < out_channels; i++) {
            int k_nr = i * in_channels * ksize * ksize;
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize;
                    for (int x = 0; x < ksize; x++) {
                        f >> buffer[offset + k_nr + k_row + x + j * ksize * ksize];
                    }
                }
            }
        }
    }
}

void read_in_biases(float* buffer, int offset, std::string path, int out_channels) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        for (int i = 0; i < out_channels; i++) {
            f >> buffer[offset + i];
        }
    }
}

float* read_in_biases2(std::string path, int out_channels) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        float* ptr = (float*) malloc(sizeof(float) * out_channels);

        for (int i = 0; i < out_channels; i++) {
            f >> ptr[i];
        }
        float x;
        return ptr;
    }
}

void read_in_matrix_chw(float* buffer, std::string path, int channels, int rows, int cols) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        for (int chn = 0; chn < channels; chn++) {
            int chn_off = chn * rows * cols;
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    f >> buffer[chn_off + row * cols + col];
                }
            }
        }
    }
}
