#include "denoise.h"

void read_in(anydsl::Array<float>* weights, float** biases) {
    const int memsize1  = 3 * 3 *  9 * 12;
    const int memsize2  = 3 * 3 * 12 * 12;
    const int memsize3  = 3 * 3 * 12 * 16;
    const int memsize4  = 3 * 3 * 16 * 32;
    const int memsize5  = 3 * 3 * 32 * 64;
    const int memsize6  = 3 * 3 * 64 * 70;
    const int memsize7  = 3 * 3 * 70 * 70;
    const int memsize8  = 3 * 3 * 102* 92;
    const int memsize9  = 3 * 3 * 92 * 92;
    const int memsize10 = 3 * 3 * 108* 70;
    const int memsize11 = 3 * 3 * 70 * 70;
    const int memsize12 = 3 * 3 * 82 * 64;
    const int memsize13 = 3 * 3 * 64 * 64;
    const int memsize14 = 3 * 3 * 73 * 32;
    const int memsize15 = 3 * 3 * 32 * 16;
    const int memsize16 = 3 * 3 * 16 * 3;

    const int memsize_biases = 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32 + 16 + 3;
    const int memsize_weights = (memsize1 + memsize2  + memsize3  + memsize4  + memsize5  + memsize6  + memsize7  + memsize8 +
                                memsize9  + memsize10 + memsize11 + memsize12 + memsize13 + memsize14 + memsize15 + memsize16);

    // Buffer for all convolution weights
    *weights = anydsl::Array<float>(sizeof(float) * memsize_weights);

    *biases = (float*) malloc(sizeof(float) * (memsize_biases));

    int offset = 0;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv1.txt",  9, 12, 3);
    offset += memsize1;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv2.txt", 12, 12, 3);
    offset += memsize2;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv3.txt", 12, 16, 3);
    offset += memsize3;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv4.txt", 16, 32, 3);
    offset += memsize4;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv5.txt", 32, 64, 3);
    offset += memsize5;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv6.txt", 64, 70, 3);
    offset += memsize6;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv7.txt", 70, 70, 3);
    offset += memsize7;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv8.txt", 102, 92, 3);
    offset += memsize8;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv9.txt", 92, 92, 3);
    offset += memsize9;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv10.txt", 108, 70, 3);
    offset += memsize10;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv11.txt", 70, 70, 3);
    offset += memsize11;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv12.txt", 82, 64, 3);
    offset += memsize12;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv13.txt", 64, 64, 3);
    offset += memsize13;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv14.txt", 73, 32, 3);
    offset += memsize14;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv15.txt", 32, 16, 3);
    offset += memsize15;
    read_in_weigths_chw(weights->data(), offset, "/home/woshi/Documents/mcdenoise/network/conv16.txt", 16, 3, 3);

    read_in_biases(*biases, 0, "/home/woshi/Documents/mcdenoise/network/bias1.txt",  12);
    read_in_biases(*biases, 12, "/home/woshi/Documents/mcdenoise/network/bias2.txt",  12);
    read_in_biases(*biases, 12 + 12, "/home/woshi/Documents/mcdenoise/network/bias3.txt",  16);
    read_in_biases(*biases, 12 + 12 + 16, "/home/woshi/Documents/mcdenoise/network/bias4.txt", 32);
    read_in_biases(*biases, 12 + 12 + 16 + 32, "/home/woshi/Documents/mcdenoise/network/bias5.txt",  64);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64, "/home/woshi/Documents/mcdenoise/network/bias6.txt",  70);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70, "/home/woshi/Documents/mcdenoise/network/bias7.txt",  70);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70, "/home/woshi/Documents/mcdenoise/network/bias8.txt",  92);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92, "/home/woshi/Documents/mcdenoise/network/bias9.txt",  92);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92, "/home/woshi/Documents/mcdenoise/network/bias10.txt",  70);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70, "/home/woshi/Documents/mcdenoise/network/bias11.txt",  70);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70, "/home/woshi/Documents/mcdenoise/network/bias12.txt",  64);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64, "/home/woshi/Documents/mcdenoise/network/bias13.txt",  64);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64, "/home/woshi/Documents/mcdenoise/network/bias14.txt",  32);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32, "/home/woshi/Documents/mcdenoise/network/bias15.txt",  16);
    read_in_biases(*biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32 + 16, "/home/woshi/Documents/mcdenoise/network/bias16.txt",  3);
}

void denoise(anydsl::Array<float>* color, anydsl::Array<float>* albedo, anydsl::Array<float>* normal, anydsl::Array<uint8_t>* memory, anydsl::Array<float>* out, int width, int height, anydsl::Array<float>* weights, float* biases) {
    Buffer color_buf   = { (char*) color->data(),   color->size(),   color->device()   };
    Buffer albedo_buf  = { (char*) albedo->data(),  albedo->size(),  albedo->device()  };
    Buffer normal_buf  = { (char*) normal->data(),  normal->size(),  normal->device()  };
    Buffer memory_buf  = { (char*) memory->data(),  memory->size(),  memory->device()  };
    Buffer out_buf     = { (char*) out->data(),     out->size(),     out->device()     };
    Buffer weights_buf = { (char*) weights->data(), weights->size(), weights->device() };

    forward_denoise(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, biases);
}