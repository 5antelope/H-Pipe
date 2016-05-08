#ifndef LRN_LAYER_H
#define LRN_LAYER_H

#include "layers.h"
#include "Halide.h"

class LRN: public Layer {
public:
    Halide::Var x, y, z, n;

    int input_width, input_height, intput_channel, input_num;

    LRN(Layer* in, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f);

    void back_propagate(Halide::Func dout);

    int out_dims();

    int out_dim_size(int i);
};

#endif