#ifndef NN_IO_H
#define NN_IO_H

void read_in_weigths_chw(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize);
void read_in_weigths_hwc(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize);
void read_in_matrix_chw(float* buffer, std::string path, int channels, int rows, int cols);
void read_in_matrix_hwc(float* buffer, std::string path, int channels, int rows, int cols);
void read_in_biases(float* buffer, int& offset, std::string path, int out_channels);
void read_in_weigths_bytes_chw(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize);
void read_in_biases_bytes(float* buffer, int& offset, std::string path, int out_channels);
void read_in_matrix_bytes_hwc(float* buffer, std::string path, int channels, int rows, int cols);

#endif /* NN_H */
