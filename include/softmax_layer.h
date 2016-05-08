#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layers.h"
#include "Halide.h"

class SoftMax: public Layer {
public:
  Var in_dim, n;
  int num_classes, num_samples;

  // Expects 2-dimensional input layer (num_classes x num_samples)
  SoftMax(Layer *in, int schedule = 1);

  void back_propagate(Func labels);

  // Returns a halide function that computes softmax loss given
  // the correct labels for each sample
  Func loss(Func labels);

  int out_dims();

  int out_dim_size(int i);
};

#endif