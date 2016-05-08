#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layers.h"
#include "Halide.h"

class ReLU : public Layer {
public:
  Halide::Var x, y, z, w;
  int vec_len = 8;

  ReLU(Layer *in, int schedule = 0);

  void back_propagate(Halide::Func dout);

  int out_dims();

  int out_dim_size(int i);
};

#endif