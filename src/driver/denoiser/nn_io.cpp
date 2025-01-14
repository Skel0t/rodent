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
    // std::cout << "read weights: " << path << " with offset " << offset << std::endl;
}

void read_in_weigths_bytes_chw(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize) {
    std::ifstream f;
    f.open(path, std::ios::binary);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        float val;
        for (int i = 0; i < out_channels; i++) {
            int k_nr = i * in_channels * ksize * ksize;
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize;
                    for (int x = 0; x < ksize; x++) {
                        f.read((char*) &val, sizeof(float));
                        buffer[offset + k_nr + k_row + x + j * ksize * ksize] = val;
                    }
                }
            }
        }
    }
    f.close();
    // std::cout << "read weights: " << path << " with offset " << offset << std::endl;
}

void read_in_biases_bytes(float* buffer, int& offset, std::string path, int out_channels) {
    std::ifstream f;
    f.open(path, std::ios::binary);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        float val;
        for (int i = 0; i < out_channels; i++) {
            f.read((char*) &val, sizeof(float));
            buffer[offset + i] = val;
        }
    }
    offset += out_channels;
    f.close();
    // std::cout << "read biases: " << path << " with offset " << offset << std::endl;
}

void read_in_biases(float* buffer, int& offset, std::string path, int out_channels) {
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
    offset += out_channels;
    // std::cout << "read biases: " << path << " with offset " << offset << std::endl;
}

void read_in_matrix_bytes_hwc(float* buffer, std::string path, int channels, int rows, int cols) {
    std::ifstream f;
    f.open(path, std::ios::binary);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        float val;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                for (int chn = 0; chn < channels; chn++) {
                    f.read((char*) &val, sizeof(float));
                    buffer[r * cols * channels + c * channels + chn] = val;
                }
            }
        }
    }
    f.close();
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

void read_in_matrix_hwc(float* buffer, std::string path, int channels, int rows, int cols) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
        throw std::invalid_argument(path);
    } else {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                for (int chn = 0; chn < channels; chn++) {
                    f >> buffer[row * cols * channels + col * channels + chn];
                }
            }
        }
    }
}

extern "C" {
    void dump_mat_binary(const char* file_name, const float* ptr, int32_t rows, int32_t cols, int32_t channels) {
        std::ofstream file(file_name, std::ios::binary);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                for (int chn = 0; chn < channels; chn++) {
                    const float val = ptr[((r * cols + c) * channels + chn)];
                    file.write((char*) &val, sizeof(float));
                }
            }
        }
    }
}
