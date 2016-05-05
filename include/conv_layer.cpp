#include <string>

#include "Halide.h"
#include "layers.h"
#include "conv_layer.h"

using namespace Halide;

// TODO: how to pass parameters
Convolutional::Convolutional(string _name,
        const caffe2::TensorProto *tensor, // define filter
        const caffe2::OperatorDef *ops) { // define input/output and stride, pad..
    set_name(_name);
}

/*
 * This constructor is used for correctness test purpose
 * Pass in filter and stride, pad parameters directly for conv layer
 */
Convolutional::Convolutional(string _name,
        Image<float> _filter,
        int _stride, int _pad): filter(_filter), stride(_stride), pad(_pad) {

    set_name(_name);

    std::cout << "construct a conv layer: " << get_name()  <<std::endl;

}

Func
Convolutional::run(Func input, int input_width, int input_height, int input_channels, int input_batch) {
    // assume width = height for filter
    int kernel_size = filter.width();

    int output_width = (input_width  - kernel_size + 2 * pad) / stride + 1;
    int output_height = (input_height - kernel_size + 2 * pad) / stride + 1;
    int output_channels = num_output;
    int output_batch = input_batch;

    set_width(output_width);
    set_height(output_height);
    set_channels(output_channels);
    set_num(output_batch);

    /* Clamped at boundary */
    Func clamped = BoundaryConditions::constant_exterior(input, 0.f, 0, input_width, 0, input_height);

    RDom r(0, kernel_size, 0, kernel_size, 0, input_channels);

    data(x, y, z, n) = sum(filter(r.x, r.y, r.z, z) *
             clamped(x * stride + r.x - pad, y * stride + r.y - pad, r.z, n));

    data.parallel(z);

    return data;

}

int layer_dims() { return 4; }

int layer_extent( int i) {
    assert(i < 4);

    if (i == 0)
        return get_width();
    else if (i == 1)
        return get_height();
    else if (i == 2)
        return get_channels();
    else if (i == 3)
        return get_batch();

    // error
    return -1;
}
