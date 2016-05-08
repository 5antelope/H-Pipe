#include "Halide.h"

class FC: public Layer {
public:
	Var x, y, z, n;

	FC(Layer* in, int schedule = true): Layer(in) {
		RDom r(0, in->out_dim_size(0));

		Image<float> w = params[0];
		Image<float> b = params[1];

		forward(x, y, z, n) = sum(w(x, r.x) * in->forward(r.x, y, z, n));
		forward(x, y, z, n) += b(x);
	}

	int out_dims() { return 4; }

	int out_dim_size(int i) {
		assert(i < 4);

		int size = 0;
		if (i == 0)
		  size = in_layer->out_dim_size(0);
		else if (i == 1)
		  size = in_layer->out_dim_size(1);
		else if (i == 2)
		  size = in_layer->out_dim_size(2);
		else if (i == 3)
		  size = in_layer->out_dim_size(3);
		return size;
	}
};
