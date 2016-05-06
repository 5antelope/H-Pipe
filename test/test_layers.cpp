#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
//#include "utils.h"

int main(int argc, char **argv) {

	std::vector<Layer*> network;
	float reg = 0.001;
	// caffe
	int n_f = 32; // number of filters
	int f_w = 7;  // filter width
	int f_h = 7;  // filter height
	int pad = (f_w-1)/2; // padding required to handle boundaries
	int stride = 1; // stride at which the filter evaluated

	Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
											  stride, reg, NULL);
	network.push_back(conv);
	printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
												conv->out_dim_size(1),
												conv->out_dim_size(2),
												conv->out_dim_size(3));

	ReLU * relu = new ReLU(conv);
	network.push_back(relu);

	int p_w = 2; // pooling width
	int p_h = 2; // pooling height
	int p_stride = 2; // pooling stride

	MaxPooling * pool = new MaxPooling(p_w, p_h, p_stride, relu);
	network.push_back(pool);
	printf("pool out size %d x %d x %d x %d\n", pool->out_dim_size(0),
												pool->out_dim_size(1),
												pool->out_dim_size(2),
												pool->out_dim_size(3));

	
	SoftMax * softm = new SoftMax(NULL);
	network.push_back(softm);
	printf("softm out size %d x %d\n", softm->out_dim_size(0),
									   softm->out_dim_size(1));

	printf("test pass");        
	

}
