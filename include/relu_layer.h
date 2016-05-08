class ReLU : public Layer {
public:
  Var x, y, z, w;
  int vec_len = 8;

  ReLU(Layer *in, int schedule = 0) : Layer(in);

  void back_propagate(Func dout);

  int out_dims();

  int out_dim_size(int i);
};
