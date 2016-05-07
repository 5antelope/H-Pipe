#include "Halide.h"

class Concat: public Layer {

public:
  Var x, y, z, n;

  // Concat does NOT inheritant Layer, since it has
  // different constructor - a list of layers
  Concat(std::vector<Layer*> inputs): Layer(inputs[0]) {
    // layers may only be concatenated along one axis
    // other axes must have the same dimensions
    int offset = 0;
    forward(x, y, z, n) = 0.0f;

    // concat over channel dimension only
    for (Layer* input : inputs) {
      Halide::RDom r(0, input.z);
      forward(x, y, offset + r, n) = input->forward(x, y, r, n);
      offset += input.z;
    }

  }
};