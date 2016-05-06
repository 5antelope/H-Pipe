#include "Halide.h"

class ReLU : public Layer {
public:
  Var x, y, z, w;
  int vec_len = 8;

  ReLU(Layer *in, int schedule = 0) : Layer(in) {
    Func in_f = in_layer->forward;
    // Define forward
    switch (in_layer->out_dims()) {
      case 1:
        forward(x) = max(0, in_f(x));
        // schedule
        if (schedule) {
          //forward.compute_root();
          //forward.vectorize(x, vec_len);
        }
        break;
      case 2:
        forward(x, y) = max(0, in_f(x, y));
        // schedule
        if (schedule) {
          //forward.compute_root();
          //forward.vectorize(x, vec_len);
          //forward.parallel(y);
        }
        break;
      case 3:
        forward(x, y, z) = max(0, in_f(x, y, z));
        // schedule
        if (schedule) {
          //forward.compute_root();
          //forward.vectorize(x, vec_len);
          //forward.parallel(z);
        }
        break;
      case 4:
        forward(x, y, z, w) = max(0, in_f(x, y, z, w));
        // schedule
        if (schedule) {
          //forward.compute_root();
          //forward.vectorize(x, vec_len);
          //forward.parallel(w);
        }
        break;
      default:
        assert(0);
    }

  }

  void back_propagate(Func dout) {
    assert(dout.defined());
    if (!f_in_grad.defined()) {
      Func in_f = in_layer->forward;
      switch (in_layer->out_dims()) {
        case 1:
          f_in_grad(x) = dout(x) * select(in_f(x) > 0, 1, 0);
          break;
        case 2:
          f_in_grad(x, y) = dout(x, y) *
                            select(in_f(x, y) > 0, 1, 0);
          break;
        case 3:
          f_in_grad(x, y, z) = dout(x, y, z) *
                               select(in_f(x, y, z) > 0, 1, 0);
          break;
        case 4:
          f_in_grad(x, y, z, w) = dout(x, y, z, w) *
                                  select(in_f(x, y, z, w) > 0, 1, 0);
          break;
        default:
          assert(0);
      }
      in_layer->back_propagate(f_in_grad);
    }
  }

  int out_dims() { return in_layer->out_dims(); }

  int out_dim_size(int i) {
    return in_layer->out_dim_size(i);
  }
};
