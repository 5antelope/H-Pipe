#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

class Concat: public Layer {

  int in_w, in_h, in_channel, in_num;

public:
  Var x, y, z, n;

  // Concat does NOT inheritant Layer, since it has
  // different constructor - a list of layers
  Concat(std::vector<Layer*> inputs);

  void back_propagate(Func dout);

  int out_dims();

  int out_dim_size(int i);
  
};

#endif