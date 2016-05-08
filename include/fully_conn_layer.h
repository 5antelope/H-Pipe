#ifndef FULLY_CONN_LAYER_H
#define FULLY_CONN_LAYER_H

#include "layers.h"
#include "Halide.h"

class FC: public Layer {
public:
	Halide::Var x, y, z, n;

	FC(Layer* in, int schedule = true);

    void back_propagate(Halide::Func dout);

	int out_dims();

	int out_dim_size(int i);
};

#endif