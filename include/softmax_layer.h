class SoftMax: public Layer {
public:
  Var in_dim, n;
  int num_classes, num_samples;

  // Expects 2-dimensional input layer (num_classes x num_samples)
  SoftMax(Layer *in, int schedule = 1): Layer(in);

  void back_propagate(Func labels);

  // Returns a halide function that computes softmax loss given
  // the correct labels for each sample
  Func loss(Func labels);

  int out_dims();

  int out_dim_size(int i);
};
