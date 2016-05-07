#include "layers.h"
#include "data_layer.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "max_pool_layer.h"
#include "flat_layer.h"
#include "tensor2image.h"

#include "Halide.h"
#include "halide_image_io.h"

#include <stdio.h>
#include <sys/time.h>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

using namespace Halide;
using namespace Halide::Tools;
using ::google::protobuf::Message;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);

    /***** Parse parameters from protobuf  *****/
    string net_path="/home/yangwu/git/H-Pipe/src/inception_net.pb";
    string tensors_path="/home/yangwu/git/H-Pipe/src/inception_tensors.pb";

    std::fstream net_input(net_path, ios::in | ios::binary);
    std::fstream tensors_input(tensors_path, ios::in | ios::binary);

    // net is the network definition.
    caffe2::NetDef netDef;
    // tensors contain the parameter tensors.
    caffe2::TensorProtos tensor;

    netDef.ParseFromIstream(&net_input);
    tensor.ParseFromIstream(&tensors_input);

	// Network structure for test
	// data - conv - reLU - pool - fc - softmax

	std::vector<Layer*> network;
	float reg = 0.001;


    Halide::Image<float> data_origin = load_image("/home/yangwu/git/H-Pipe/dog.png");

    /***** DATA LAYER  *****/
	int d_w = 32; // data width
	int d_h = 32; // data height
	int ch  = 3; // number of channels
	int N   = 1; // number of samples

	Image<int> labels(N);

    Func data = BoundaryConditions::constant_exterior(data_origin, 0.f, 0, d_w, 0, d_h);

	DataLayer * d_layer = new DataLayer(d_h, d_w, ch, N, data);
	network.push_back(d_layer);
	printf("data out size %d x %d x %d x %d\n", d_layer->out_dim_size(0),
												d_layer->out_dim_size(1),
												d_layer->out_dim_size(2),
												d_layer->out_dim_size(3));

    /***** CONV LAYER  *****/

    // conv2d0 layer
    const caffe2::OperatorDef& conv_op = netDef.op(0);

    const caffe2::TensorProto& conv2d0_w = tensor.protos(0);
    Halide::Image<float> conv_k = LoadImageFromTensor(conv2d0_w);
    const caffe2::TensorProto& con2d0_b = tensor.protos(1);
    Halide::Image<float> conv_b = LoadImageFromTensor(conv2d0_b);

    /***** Set up weights and bias for layers *****/
	int n_f = conv2d0_w.dims(0);  // number of filters
	int f_w = conv2d0_w.dims(1);  // filter width
	int f_h = conv2d0_w.dims(2);  // filter height

	int pad = conv_op.arg(2).i(); // padding required to handle boundaries
	int stride = conv_op.arg(0).i(); // stride at which the filter evaluated

	Convolutional * conv  = new Convolutional(n_f, f_w, f_h, pad,
											  stride, reg, d_layer);
    conv->params[0] = conv_k;
    conv->params[1] = conv_b;

	network.push_back(conv);
	printf("conv out size %d x %d x %d x %d\n", conv->out_dim_size(0),
												conv->out_dim_size(1),
												conv->out_dim_size(2),
												conv->out_dim_size(3));

    /***** RELU  *****/
    // Rectified Linear Unit just normalize the data
    ReLU * relu = new ReLU(conv);
    network.push_back(relu);

    /***** MAX POOL  *****/
    int p_w = 2; // pooling width
    int p_h = 2; // pooling height
    int p_stride = 2; // pooling stride

    MaxPooling * pool = new MaxPooling(p_w, p_h, p_stride, relu);
    network.push_back(pool);
    printf("pool out size %d x %d x %d x %d\n", pool->out_dim_size(0),
                                                pool->out_dim_size(1),
                                                pool->out_dim_size(2),
                                                pool->out_dim_size(3));

    Flatten * flatten = new Flatten(pool);
    network.push_back(flatten);
    printf("flatten out size %d x %d\n", flatten->out_dim_size(0),
                                         flatten->out_dim_size(1));

    SoftMax * softm = new SoftMax(flatten);
    network.push_back(softm);
    printf("softm out size %d x %d\n", softm->out_dim_size(0),
                                       softm->out_dim_size(1));

    int C = 10; // number of classes

    Image<float> scores(C, N);

	printf("construct network pass\n");


	printf("test pass\n");

	return 0;


}
