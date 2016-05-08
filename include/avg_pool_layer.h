#ifndef AVG_POOL_LAYER_H
#define AVG_POOL_LAYER_H

#include "layers.h"
#include "Halide.h"

class AvgPooling : public Layer {
public:
  // number of color channels in input in_c
  // height and width of the input in_h, in_w
  int num_samples, in_ch, in_h, in_w;

  // height and width of the pool
  // stride at which the pooling is applied
  int p_h, p_w, stride;
  Halide::Var x, y, z, n;
  
  // parameters for scheduling
  Halide::Var par;
  int vec_len;

  AvgPooling(int _p_w, int _p_h, int _stride, Layer *in, int schedule = 1);

  void back_propagate(Halide::Func dout);

  int out_dims();

  int out_dim_size(int i);
};

#endif