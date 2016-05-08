#include "lrn_layer.h"

LRN::LRN(Layer* in, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f): Layer(in) {

    Func clamped = BoundaryConditions::constant_exterior(in->forward, 0.0f, 0, in->out_dim_size(0), 0, in->out_dim_size(1), 0, in->out_dim_size(2));
    Func activation;
    Func normalizer;

    input_width = in->out_dim_size(0);
    input_height = in->out_dim_size(1);
    intput_channel = in->out_dim_size(2);
    input_num  = in->out_dim_size(3);

    RDom r(-region_x / 2, region_x / 2 + 1, -region_y / 2, region_y / 2 + 1, -region_z / 2, region_z / 2 + 1);

    Expr val = clamped(x + r.x, y + r.y, z + r.z, n);

    activation(x, y, z, n) = sum(val * val);
    normalizer(x, y, z ,n) = fast_pow(1.0f + (alpha / (region_x * region_y * region_z)) * activation(x, y, z, n), beta);
    forward(x, y, z, n) = clamped(x, y, z, n) / normalizer(x, y, z, n);
}

void LRN::back_propagate(Func dout) {
    std::cout<< "NOT IMPLEMENTED YET" <<std::endl;
}

int LRN::out_dims() { return 4; }

int LRN::out_dim_size(int i) {
    assert(i < 4);
    int size = 0;
    if (i == 0)
      size = input_width;
    else if (i == 1)
      size = input_height;
    else if (i == 2)
      size = intput_channel;
    else if (i == 3)
      size = input_num;
    return size;
}
