#include "Halide.h"

class Convolutional : public Layer {
public:
  Var x, y, z, n;

  // number of channels, height and width of the input to the layer
  int num_samples, in_ch, in_h, in_w;

  // number of filters, filter height, filter width, padding and stride
  int num_f, f_h, f_w, pad, stride;

  float reg;

  Func forward_clamp;

  // parameters for scheduling
  Var y_t, z_t, par;

  int o_block_size = 16;
  int y_block_size = 32;
  int vec_len = 8;

  Convolutional(int _num_f, int _f_w, int _f_h, int _pad, int _stride,
                float _reg, Layer *in, int schedule = true) : Layer(in) {

    assert(in_layer->out_dims() == 4);

    num_samples = in_layer->out_dim_size(3);
    in_ch = in_layer->out_dim_size(2);
    in_h = in_layer->out_dim_size(1);
    in_w = in_layer->out_dim_size(0);
    reg = _reg;

    printf("ASSERT (%d + 2 * %d - %d) mod %d\n", in_h, _pad, _f_h, _stride);
    // assert((in_h + 2 * _pad - _f_h) % _stride == 0);
    // assert((in_w + 2 * _pad - _f_w) % _stride == 0);

    num_f = _num_f;
    f_h = _f_h;
    f_w = _f_w;
    pad = _pad;
    stride = _stride;

    // Boundary condition
    // This creates a padded input and avoids checking boundary
    // conditions while computing the actual convolution
    forward_clamp = BoundaryConditions::constant_exterior(
        in_layer->forward, 0.f,
        0, in_w,
        0, in_h);

    // Create parameters
    Image<float> W(f_w, f_h, in_ch, num_f), b(num_f);
    params.push_back(W);
    params.push_back(b);

    // Define forward
    RDom r(0, f_w, 0, f_h, 0, in_ch);

    // Initialize to bias
    forward(x, y, z, n) = b(z);
    forward(x, y, z, n) += W(r.x, r.y, r.z, z) *
                           forward_clamp(x * stride + r.x - pad,
                            y * stride + r.y - pad,
                            r.z,
                            n);

    if (schedule) {
      forward.update().reorder(y, x, r.z);
      // blocking spatially with vectorization
      // forward_clamp.compute_at(f_simple, n);
      forward.compute_root();
      forward.fuse(z, n, par).parallel(par);
      forward.update().reorder(x, y, r.z);
      forward.update().split(y, y, y_t, y_block_size);
      forward.update().split(z, z, z_t, o_block_size);
      forward.update().reorder(y_t, z_t, y, r.z, z);
      forward.update().vectorize(x, vec_len);
      forward.update().fuse(z, n, par).parallel(par);
      //forward.update().fuse(y, par, par).parallel(par);
      forward.update().unroll(r.x);
      forward.update().unroll(r.y);
      // There are performance implications to this and seems to
      // be incompatible with some schedules. Have to investigate
      // this more closely.
      //forward_clamp.compute_at(forward, n);
      forward_clamp.compute_at(forward, z_t);
    }

  }

  void back_propagate(Func dout) {
    assert(dout.defined());
    if (!f_in_grad.defined()) {
      Func dW, db;

      int out_w = this->out_dim_size(0);
      int out_h = this->out_dim_size(1);

      Image<float> W = params[0];
      Image<float> b = params[1];

      RDom r1(0, out_w, 0, out_h, 0, num_samples);

      // intialize to regularized weights
      dW(x, y, z, n) = cast(dout.output_types()[0],
                            reg * W(x, y, z, n));
      dW(x, y, z, n) += dout(r1.x, r1.y, n, r1.z) *
                        forward_clamp
                (r1.x * stride + x - pad,
                                   r1.y * stride + y - pad,
                                   z, r1.z);

      f_param_grads.push_back(dW);

      // intialize to zero
      db(x) = cast(dout.output_types()[0], 0);
      db(x) += dout(r1.x, r1.y, x, r1.z);

      f_param_grads.push_back(db);

      RDom r2(0, num_f);
      // intialize to zero
      f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);
      f_in_grad(x, y, z, n) += dout(x, y, r2.x, n) * W(x, y, z, r2.x);

      // Create storage for gradients and caching params
      Image<float> W_grad(f_w, f_h, in_ch, num_f);
      param_grads.push_back(W_grad);
      Image<float> W_cache(f_w, f_h, in_ch, num_f);
      params_cache.push_back(W_cache);

      Image<float> b_grad(num_f);
      param_grads.push_back(b_grad);
      Image<float> b_cache(num_f);
      params_cache.push_back(b_cache);

      in_layer->back_propagate(f_in_grad);
    }
  }

  int out_dims() { return 4; }

  int out_dim_size(int i) {
    assert(i < 4);
    int size = 0;
    if (i == 0)
      size = (1 + (in_w + 2 * pad - f_w) / stride);
    else if (i == 1)
      size = (1 + (in_h + 2 * pad - f_h) / stride);
    else if (i == 2)
      size = num_f;
    else if (i == 3)
      size = num_samples;
    return size;
  }
};
