#ifndef RELU_LAYER_H
#define RELU_LAYER_H

class ReLU : public Layer {
public:
  Var x, y, z, w;
  int vec_len = 8;

  ReLU(Layer *in, int schedule);

  void back_propagate(Func dout);

  int out_dims();

  int out_dim_size(int i);
};

#endif