#include "Halide.h"
#include <map>

using namespace std;
using namespace Halide;

class Layer {
public:
  //Default Constructor...should support other popular input
  Layer(Layer *in) {
    // The first layer in the pipeline does not have an input layer
    if (in) {
      // Get the halide function that computes values
      // of the input layer
      assert(in->forward.defined());

      // Record the input layer
      in_layer = in;
    }
  }

  // Layer that serves as an input to the current layer
  Layer *in_layer;

  // Number of output dimensions
  virtual int out_dims() = 0;

  // Size of output dimension i, 0 <= i < out_dims()
  virtual int out_dim_size(int i) = 0;

  // Storage for layer parameters
  std::vector<Image < float>> params;
  std::vector<Image < float>> param_grads;
  std::vector<Image < float>> params_cache;
  // Halide function that computes the output of the layer
  Func forward;
  // Vector of halide functions which compute the gradients
  // with respect to layer parameters
  std::vector<Func> f_param_grads;
  // Halide function which computes gradient with respect
  // to layer input
  Func f_in_grad;

  // Defines the functions which compute gradient of the objective
  // function with respective to parameters and input. Given a function
  // which computes the derivate of the objective with respect to layer
  // outputs. Does this recursively for the input layer.
  virtual void back_propagate(Func dforward) = 0;

  virtual ~Layer() { };
};
