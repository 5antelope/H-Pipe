#ifndef FLAT_LAYER_H
#define FLAT_LAYER_H

#include "layers.h"
#include "Halide.h"

class Flatten: public Layer {
    public:
        Halide::Var x, y, z, n;
        int out_width;
        int num_samples;

        Flatten(Layer *in, int schedule = 1);

        void back_propagate(Halide::Func dout);

        int out_dims();

        int out_dim_size( int i);
};

#endif