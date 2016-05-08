#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include "layers.h"
#include "Halide.h"

class DataLayer : public Layer {

public:
  int in_w;
  int in_h;
  int in_ch;
  int num_samples;

  Halide::Var x, y, z, n;

  DataLayer(int _in_w, int _in_h, int _in_ch, int _num_samples,
            Halide::Func data);

  // Nothing to propagate
  void back_propagate(Halide::Func dout);

  int out_dims();

  int out_dim_size(int i);

};

#endif