#ifndef NN_H
#define NN_H

void read_in_weigths_chw(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize);
void read_in_weigths_hwc(float* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize);
void read_in_biases(float* buffer, int offset, std::string path, int out_channels);
float* read_in_biases2(std::string path, int out_channels);

#endif /* NN_H */