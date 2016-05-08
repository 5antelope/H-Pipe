#include "fully_conn_layer.h"

FC::FC(Layer* in, int schedule = true): Layer(in) {
	RDom r(0, in->out_dim_size(0));

	Image<float> w = params[0];
	Image<float> b = params[1];

	forward(x, y, z, n) = sum(w(x, r.x) * in->forward(r.x, y, z, n));
	forward(x, y, z, n) += b(x);
}

void FC::back_propagate(Func dout) {
    std::cout << "NOT IMPLEMENTED YET" << std::endl;
}

int FC::out_dims() { return 4; }

int FC::out_dim_size(int i) {
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
