#include "softmax_layer.h"

// Expects 2-dimensional input layer (num_classes x num_samples)
SoftMax::SoftMax(Layer *in, int schedule): Layer(in) {
  assert(in->out_dims() == 2);

  //in_f: in_layer's forward function; in_layer = in if in is defined.
  Func in_f = in_layer->forward;

  num_classes = in->out_dim_size(0);
  num_samples = in->out_dim_size(1);

  // Define forward
  // the softmax function, or normalized exponential
  Func exp_max, expo, normalizer;
  RDom r(0, num_classes);
  exp_max(n) = maximum(in_f(r.x, n));
  expo(in_dim, n) = exp(in_f(in_dim, n) - exp_max(n));
  normalizer(n) = cast(in_f.output_types()[0], 0);
  normalizer(n) += expo(r.x, n);
  forward(in_dim, n) = expo(in_dim, n) / normalizer(n);

  if (schedule) {
    // Local schedule
    exp_max.compute_at(forward, n);
    expo.compute_at(forward, n);
    normalizer.compute_at(forward, n);
    forward.compute_root().parallel(n);
  }
}

void SoftMax::back_propagate(Func labels) {
  if (!f_in_grad.defined()) {
    assert(labels.defined());
    assert(forward.defined());

    Expr label = clamp(labels(n), 0, num_classes - 1);
    Expr t = (forward(in_dim, n) - 1) / num_samples;
    Expr f = (forward(in_dim, n) / num_samples);
    f_in_grad(in_dim, n) = select(in_dim == label, t, f);
    in_layer->back_propagate(f_in_grad);
  }
}

// Returns a halide function that computes softmax loss given
// the correct labels for each sample
Func SoftMax::loss(Func labels) {
  // Should loss be a layer?

  // Check if labels is defined
  assert(labels.defined());
  // Check if the dimensions make sense
  assert(labels.dimensions() == 1);
  // TODO Figure out if there is a scalar type
  Var x;
  Func loss_p;
  RDom r(0, num_samples);
  loss_p(x) = cast(forward.output_types()[0], 0);
  // The clamp is necessary. Otherwise, halide will assume that the
  // label can be anything during bounds inference.
  loss_p(0) += -log(forward(clamp(labels(r.x), 0, num_classes - 1),
                            r.x)) / num_samples;
  return loss_p;
}

int SoftMax::out_dims() { return 2; }

int SoftMax::out_dim_size(int i) {
  assert(i < 2);
  int size = 0;
  if (i == 0)
    size = num_classes;
  else if (i == 1)
    size = num_samples;
  return size;
}
