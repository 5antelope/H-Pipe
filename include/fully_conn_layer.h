#ifndef FULLY_CONN_LAYER_H
#define FULLY_CONN_LAYER_H

class FC: public Layer {
public:
	Var x, y, z, n;

	FC(Layer* in, int schedule);

    void back_propagate(Func dout);

	int out_dims();

	int out_dim_size(int i);
};

#endif