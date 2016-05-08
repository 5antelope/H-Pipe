#include "avg_pool_layer.h"

AvgPooling::AvgPooling(int _p_w, int _p_h, int _stride, Layer *in,
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
  forward(x, y, z, n) = sum(in_f(x * stride + r.x,
                                     y * stride + r.y,
                                     z, n)) / (p_w * p_h);

  if (schedule) {
    forward.vectorize(x, vec_len);
    forward.compute_root().fuse(z, n, par).parallel(par);
  }

}

void AvgPooling::back_propagate(Func dout) {
  std::cout << "NOT IMPLEMENTED YET" << std::endl;
  return;
}

int AvgPooling::out_dims() { return 4; }

int AvgPooling::out_dim_size(int i) {
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
