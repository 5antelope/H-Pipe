#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
//#include "utils.h"

int main(int argc, char **argv) {

	// Google logging needed for parts that were extracted from
	// caffe

	// Network structure
	// data - conv - reLU - pool - fc - softmax

	std::vector<Layer*> network;
	float reg = 0.001;

	// Description of the neural network

	int N = 64; // number of samples/batch_size
	int d_w = 32; // data width
	int d_h = 32; // data height
	int ch = 3; // number of channels

	Image<float> data(d_w, d_h, ch, N);
	Image<int> labels(N);


	DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
	network.push_back(d_layer);
	printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
												d_layer->out_dim_size(1),
												d_layer->out_dim_size(2),
												d_layer->out_dim_size(3));
	int n_f = 32; // number of filters
	int f_w = 7;  // filter width
	int f_h = 7;  // filter height
	int pad = (f_w-1)/2; // padding required to handle boundaries
	int stride = 1; // stride at which the filter evaluated

	Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
											  stride, reg, d_layer);
	network.push_back(conv);
	printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
												conv->out_dim_size(1),
												conv->out_dim_size(2),
												conv->out_dim_size(3));


	printf("test pass");        
	

	return 0;       
	

}
