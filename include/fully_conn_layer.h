class FC: public Layer {
public:
	Var x, y, z, n;

	FC(Layer* in, int schedule = true): Layer(in);

    void back_propagate(Func dout);

	int out_dims();

	int out_dim_size(int i);
};
