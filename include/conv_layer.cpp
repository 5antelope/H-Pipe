#include <string>

#include "Halide.h"
#include "conv_layer.h"

using namespace Halide;

// TODO: how to pass parameters
Convolutional::Convolutional(string _name,
        int _pad, int _stride, string _order) { // define input/output and stride, pad..
    std::cout << "CONV LAYER" << std::endl;
    set_name(_name);

    set_pad(_pad);
    set_stride(_stride);
    set_order(_order);
    std::cout << "CONV LAYER CONSTRUCTED" << std::endl;
}

void Convolutional::load_weight(Image<float> _weight) {
    set_weight(_weight);
}

void Convolutional::load_bias(Image<float> _bias) {
    set_bias(_bias);
}

Halide::Func
Convolutional::run(Halide::Func input, int input_width, int input_height, int input_channels, int input_num) {
    std::cout << "RUN" << std::endl;
    // assume width = height for filter
    int kernel_size = weight.width();

    printf("NUM: (%d - %d + 2*%d) / %d + 1\n", input_width, kernel_size, pad, stride);
    int output_width = (input_width  - kernel_size + 2 * pad) / stride + 1;
    int output_height = (input_height - kernel_size + 2 * pad) / stride + 1;
    // output channel should be number of filters?
    int output_channels = weight.extent(3);
    int output_num = input_num;

    std::cout << "GOING TO SET CONV LAYER" << std::endl;

    set_output_width(output_width);
    set_output_height(output_height);
    set_output_channels(output_channels);
    set_output_num(output_num);

    std::cout << "SET CONV LAYER" << std::endl;

    /* Clamped at boundary */
    Halide::Func clamped = Halide::BoundaryConditions::constant_exterior(input, 0.f, 0, input_width, 0, input_height);

    std::cout << "BOUNDED" << std::endl;

    RDom r(0, kernel_size, 0, kernel_size, 0, input_channels);

    data(x, y, z, n) = sum(weight(r.x, r.y, r.z, z) *
             clamped(x * stride + r.x - pad, y * stride + r.y - pad, r.z, n));

    data(x, y, z, n) += bias(z);

    std::cout << "READY TO PARALLEL" << std::endl;

    data.parallel(z);

    return data;

}

