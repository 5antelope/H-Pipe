#include "concat_layer.h"

class Concat: public Layer {

  int in_w, in_h, in_channel, in_num;

public:
  Var x, y, z, n;

  // Concat does NOT inheritant Layer, since it has
  // different constructor - a list of layers
  Concat(std::vector<Layer*> inputs): Layer(inputs[0]) {
    // layers may only be concatenated along one axis
    // other axes must have the same dimensions
    int offset = 0;
    forward(x, y, z, n) = 0.0f;

    in_w = inputs[0]->out_dim_size(0);
    in_h = inputs[0]->out_dim_size(1);
    in_channel = 0;
    in_num = inputs[0]->out_dim_size(3);

    // concat over channel dimension only
    for (Layer* input : inputs) {
      Halide::RDom r(0, input->out_dim_size(0)); // x dimension
      forward(x, y, offset + r, n) = input->forward(x, y, r, n);
      in_channel += input->out_dim_size(2);
    }

  }

  void back_propagate(Func dout) {
    std::cout << "NOT IMPLEMENTED YET" << std::endl;
  }

  int out_dims() { return 4; }

  int out_dim_size(int i) {
    assert(i < 4);

    int size = 0;
    if (i == 0)
      size = in_w;
    else if (i == 1)
      size = in_h;
    else if (i == 2)
      size = in_channel;
    else if (i == 3)
      size = in_num;
    return size;
  }
};
