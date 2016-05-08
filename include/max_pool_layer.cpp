#include "max_pool_layer.h"

MaxPooling::MaxPooling(int _p_w, int _p_h, int _stride, Layer *in,
           int schedule) : Layer(in) {
  vec_len = 8;

  assert(in_layer->out_dims() == 4);

  num_samples = in_layer->out_dim_size(3);
  in_ch = in_layer->out_dim_size(2);
  in_h = in_layer->out_dim_size(1);
  in_w = in_layer->out_dim_size(0);

  // assert((in_h - _p_h) % _stride == 0);
  // assert((in_w - _p_w) % _stride == 0);

  p_w = _p_w;
  p_h = _p_h;
  stride = _stride;

  // Define forward
  Func in_f = in_layer->forward;
  RDom r(0, p_w, 0, p_h);
  forward(x, y, z, n) = maximum(in_f(x * stride + r.x,
                                     y * stride + r.y,
                                     z, n));

  if (schedule) {
    forward.vectorize(x, vec_len);
    forward.compute_root().fuse(z, n, par).parallel(par);
  }

}

void MaxPooling::back_propagate(Func dout) {
  assert(dout.defined());
  if (!f_in_grad.defined()) {
    Func in_f = in_layer->forward;
    Func pool_argmax;
    RDom r1(0, p_w, 0, p_h);
    pool_argmax(x, y, z, n) = argmax(in_f(x * stride + r1.x,
                                          y * stride + r1.y,
                                          z, n));

    pool_argmax.compute_root();
    RDom r2(0, this->out_dim_size(0), 0, this->out_dim_size(1));
    f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);

    Expr x_bin = clamp(r2.x * stride +
                       pool_argmax(r2.x, r2.y, z, n)[0], 0, in_w);
    Expr y_bin = clamp(r2.y * stride +
                       pool_argmax(r2.x, r2.y, z, n)[1], 0, in_h);

    f_in_grad(x_bin, y_bin, z, n) += dout(r2.x, r2.y, z, n);
    in_layer->back_propagate(f_in_grad);
  }
}

int MaxPooling::out_dims() { return 4; }

int MaxPooling::out_dim_size(int i) {
  assert(i < 4);
  int size = 0;
  if (i == 0)
    size = 1 + ((in_w - p_w) / stride);
  else if (i == 1)
    size = 1 + ((in_h - p_h) / stride);
  else if (i == 2)
    size = in_layer->out_dim_size(2);
  else if (i == 3)
    size = num_samples;
  return size;
}
