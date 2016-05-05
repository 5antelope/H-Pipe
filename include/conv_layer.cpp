#include <string>

#include "Halide.h"
#include "layers.h"
#include "conv_layer.h"

using namespace Halide;

// TODO: how to pass parameters
Convolutional::Convolutional(string _name,
        const caffe2::TensorProto *_tensor, // define filter
        const caffe2::OperatorDef *_op) { // define input/output and stride, pad..
    set_name(_name);

    set_tensor(_tensor);
    set_op(_op);

    set_output_num(tensor.dims(0));
}

Func
Convolutional::run(Func input, int input_width, int input_height, int input_channels, int input_num) {
    // assume width = height for filter
    int kernel_size = weight.width();

    int output_width = (input_width  - kernel_size + 2 * pad) / stride + 1;
    int output_height = (input_height - kernel_size + 2 * pad) / stride + 1;
    // output channel should be number of filters
    int output_channels = get_output_num();
    int output_num = input_num;

    set_width(output_width);
    set_height(output_height);
    set_channels(output_channels);
    set_output_num(output_num);

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
        return -1;
    else if (i == 1)
        return -1;
    else if (i == 2)
        return -1;
    else if (i == 3)
        return -1;

    // error
    return -1;
}
