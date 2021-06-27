#include <iostream>
#include <fstream>

#include "interface.h"
#include "nn.h"

void read_in_weigths_hwc(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
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
        std::cout << "Couldn't open" << path << std::endl;
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
        return nullptr;
    } else {
        float* ptr = (float*) malloc(sizeof(float) * out_channels);

        for (int i = 0; i < out_channels; i++) {
            f >> ptr[i];
        }
        float x;
        return ptr;
    }
}