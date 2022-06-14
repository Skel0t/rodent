#include "denoise.h"

void read_in(anydsl::Array<float>* weights, anydsl::Array<float>* biases, std::string network_path) {
    if (network_path.back() == '/') {
        network_path = network_path.substr(0, network_path.size() - 1);
    }
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

    const int memsize_biases = 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 +
                               70 + 64 + 64 + 32 + 16 + 3;
    const int memsize_weights = (memsize1  + memsize2  + memsize3  + memsize4  +
                                 memsize5  + memsize6  + memsize7  + memsize8  +
                                 memsize9  + memsize10 + memsize11 + memsize12 +
                                 memsize13 + memsize14 + memsize15 + memsize16);

    // Buffer for all convolution weights
    *weights = anydsl::Array<float>(memsize_weights);
    *biases  = anydsl::Array<float>(memsize_biases);

    int offset = 0;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv1.txt",  9, 12, 3);
    offset += memsize1;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv2.txt", 12, 12, 3);
    offset += memsize2;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv3.txt", 12, 16, 3);
    offset += memsize3;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv4.txt", 16, 32, 3);
    offset += memsize4;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv5.txt", 32, 64, 3);
    offset += memsize5;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv6.txt", 64, 70, 3);
    offset += memsize6;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv7.txt", 70, 70, 3);
    offset += memsize7;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv8.txt", 102, 92, 3);
    offset += memsize8;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv9.txt", 92, 92, 3);
    offset += memsize9;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv10.txt", 108, 70, 3);
    offset += memsize10;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv11.txt", 70, 70, 3);
    offset += memsize11;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv12.txt", 82, 64, 3);
    offset += memsize12;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv13.txt", 64, 64, 3);
    offset += memsize13;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv14.txt", 73, 32, 3);
    offset += memsize14;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv15.txt", 32, 16, 3);
    offset += memsize15;
    read_in_weigths_chw(weights->data(), offset, network_path + "/conv16.txt", 16, 3, 3);

    offset = 0;
    read_in_biases(biases->data(), offset, network_path + "/bias1.txt",  12);
    read_in_biases(biases->data(), offset, network_path + "/bias2.txt",  12);
    read_in_biases(biases->data(), offset, network_path + "/bias3.txt",  16);
    read_in_biases(biases->data(), offset, network_path + "/bias4.txt",  32);
    read_in_biases(biases->data(), offset, network_path + "/bias5.txt",  64);
    read_in_biases(biases->data(), offset, network_path + "/bias6.txt",  70);
    read_in_biases(biases->data(), offset, network_path + "/bias7.txt",  70);
    read_in_biases(biases->data(), offset, network_path + "/bias8.txt",  92);
    read_in_biases(biases->data(), offset, network_path + "/bias9.txt",  92);
    read_in_biases(biases->data(), offset, network_path + "/bias10.txt", 70);
    read_in_biases(biases->data(), offset, network_path + "/bias11.txt", 70);
    read_in_biases(biases->data(), offset, network_path + "/bias12.txt", 64);
    read_in_biases(biases->data(), offset, network_path + "/bias13.txt", 64);
    read_in_biases(biases->data(), offset, network_path + "/bias14.txt", 32);
    read_in_biases(biases->data(), offset, network_path + "/bias15.txt", 16);
    read_in_biases(biases->data(), offset, network_path + "/bias16.txt",  3);
}

void read_in_oidn(anydsl::Array<float>* weights, anydsl::Array<float>* biases, std::string network_path) {
    if (network_path.back() == '/') {
        network_path = network_path.substr(0, network_path.size() - 1);
    }

    const int ic = 9;
    const int ec1 = 32;
    const int ec2 = 48;
    const int ec3 = 64;
    const int ec4 = 80;
    const int ec5 = 96;
    const int dc4 = 112;
    const int dc3 = 96;
    const int dc2 = 64;
    const int dc1a = 64;
    const int dc1b = 32;
    const int oc = 3;

    const int memsize1  = 3 * 3 * ic * ec1;  // ic -> ec1
    const int memsize2  = 3 * 3 * ec1 * ec1;  // ec1 -> ec1
    const int memsize3  = 3 * 3 * ec1 * ec2;  // ec1 -> ec2
    const int memsize4  = 3 * 3 * ec2 * ec3;  // ec2 -> ec3
    const int memsize5  = 3 * 3 * ec3 * ec4;  // ec3 -> ec4
    const int memsize6  = 3 * 3 * ec4 * ec5;  // ec4 -> ec5
    const int memsize7  = 3 * 3 * ec5 * ec5;  // ec5 -> ec5
    const int memsize8  = 3 * 3 * (ec5+ec3) * dc4;  // ec5+ec3 -> dc4
    const int memsize9  = 3 * 3 * dc4 * dc4;  // dc4 -> dc4
    const int memsize10 = 3 * 3 * (dc4+ec2)* dc3;  // dc4+ec2 -> dc3
    const int memsize11 = 3 * 3 * dc3 * dc3;  // dc3 -> dc3
    const int memsize12 = 3 * 3 * (dc3+ec1)* dc2;  // dc3+ec1 -> dc2
    const int memsize13 = 3 * 3 * dc2 * dc2;  // dc2 -> dc2
    const int memsize14 = 3 * 3 * (dc2+ic) * dc1a;  // dc2+ic -> dc1a
    const int memsize15 = 3 * 3 * dc1a * dc1b;  // dc1a -> dc1b
    const int memsize16 = 3 * 3 * dc1b * oc;   // dc1b -> oc

    const int memsize_biases = ec1 + ec1 + ec2 + ec3 + ec4 + ec5 + ec5 + dc4 + dc4 + dc3 + dc3 + dc2 + dc2 + dc1a + dc1b + oc;
    const int memsize_weights = (memsize1  + memsize2  + memsize3  + memsize4  +
                                 memsize5  + memsize6  + memsize7  + memsize8  +
                                 memsize9  + memsize10 + memsize11 + memsize12 +
                                 memsize13 + memsize14 + memsize15 + memsize16);

    // Buffer for all convolution weights
    *weights = anydsl::Array<float>(memsize_weights);
    *biases  = anydsl::Array<float>(memsize_biases);

    int offset = 0;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv0.weight.txt", ic, ec1, 3);
    offset += memsize1;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv1.weight.txt", ec1, ec1, 3);
    offset += memsize2;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv2.weight.txt", ec1, ec2, 3);
    offset += memsize3;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv3.weight.txt", ec2, ec3, 3);
    offset += memsize4;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv4.weight.txt", ec3, ec4, 3);
    offset += memsize5;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv5a.weight.txt", ec4, ec5, 3);
    offset += memsize6;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/enc_conv5b.weight.txt", ec5, ec5, 3);
    offset += memsize7;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv4a.weight.txt", (ec5+ec3), dc4, 3);
    offset += memsize8;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv4b.weight.txt", dc4, dc4, 3);
    offset += memsize9;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv3a.weight.txt", (dc4+ec2), dc3, 3);
    offset += memsize10;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv3b.weight.txt", dc3, dc3, 3);
    offset += memsize11;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv2a.weight.txt", (dc3+ec1), dc2, 3);
    offset += memsize12;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv2b.weight.txt", dc2, dc2, 3);
    offset += memsize13;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv1a.weight.txt", (dc2+ic), dc1a, 3);
    offset += memsize14;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv1b.weight.txt", dc1a, dc1b, 3);
    offset += memsize15;
    read_in_weigths_bytes_chw(weights->data(), offset, network_path + "/oidn_dump/dec_conv0.weight.txt", dc1b, oc, 3);

    // std::cout << weights->data()[offset] << " " << weights->data()[offset + memsize16 -1] << " " << weights->data()[memsize1+1] << std::endl;

    offset = 0;
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv0.bias.txt",  ec1);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv1.bias.txt",  ec1);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv2.bias.txt",  ec2);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv3.bias.txt", ec3);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv4.bias.txt",  ec4);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv5a.bias.txt",  ec5);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/enc_conv5b.bias.txt",  ec5);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv4a.bias.txt",  dc4);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv4b.bias.txt",  dc4);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv4a.bias.txt",  dc3);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv3b.bias.txt",  dc3);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv2a.bias.txt",  dc2);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv2b.bias.txt",  dc2);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv1a.bias.txt",  dc1a);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv1b.bias.txt",  dc1b);
    read_in_biases_bytes(biases->data(), offset, network_path + "/oidn_dump/dec_conv0.bias.txt",  oc);

    // std::cout << biases->data()[0] << " " << biases->data()[1] << " " << biases->data()[ec1] << std::endl;
}

void denoise(std::string denoising_backend, anydsl::Array<float>* color, anydsl::Array<float>* albedo, anydsl::Array<float>* normal,
             anydsl::Array<uint8_t>* memory, anydsl::Array<float>* out, int width, int height,
             anydsl::Array<float>* weights, anydsl::Array<float>* biases, NetID net_id) {
    Buffer color_buf   = { (int8_t*) color->data(),   color->size()  * (long) sizeof(float),   color->device()   };
    Buffer albedo_buf  = { (int8_t*) albedo->data(),  albedo->size() * (long) sizeof(float),   albedo->device()  };
    Buffer normal_buf  = { (int8_t*) normal->data(),  normal->size() * (long) sizeof(float),   normal->device()  };
    Buffer memory_buf  = { (int8_t*) memory->data(),  memory->size() * (long) sizeof(uint8_t), memory->device()  };
    Buffer out_buf     = { (int8_t*) out->data(),     out->size()    * (long) sizeof(float),   out->device()     };
    Buffer weights_buf = { (int8_t*) weights->data(), weights->size()* (long) sizeof(float),   weights->device() };
    Buffer biases_buf  = { (int8_t*) biases->data(),  biases->size() * (long) sizeof(float),   biases->device()  };

    if (denoising_backend == "cuda")
        forward_denoise_cuda(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, &biases_buf, net_id);
    else if (denoising_backend == "cublas")
        forward_denoise_cublas(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, &biases_buf, net_id);
    else if (denoising_backend == "cublaslt")
        forward_denoise_cublaslt(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, &biases_buf, net_id);
    else if (denoising_backend == "cpu")
        forward_denoise_cpu(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, &biases_buf, net_id);
    else if (denoising_backend == "oneapi")
        forward_denoise_oneapi(&color_buf, &albedo_buf, &normal_buf, &memory_buf, width, height, &out_buf, &weights_buf, &biases_buf, net_id);

}
