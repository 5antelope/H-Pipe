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

#include <map>
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

    // key: name of network
    // value: output of network
    map<string, Func> net_output;

	std::vector<Layer*> network;
	float reg = 0.001;

    Halide::Image<float> data_origin = load_image("/home/yangwu/git/H-Pipe/dog.png");

    /***** DATA LAYER  *****/
	int d_w = 32; // data width
	int d_h = 32; // data height
	int ch  = 3; // number of channels
	int N   = 1; // number of samples

    // scale the image for testing purpose
    Func data = BoundaryConditions::constant_exterior(data_origin, 0.f, 0, d_w, 0, d_h);

	DataLayer * data_layer = new DataLayer(d_h, d_w, ch, N, data);
    net_output["input"] = data_layer;

    int tensor_idx= 0;

    for (int net_idx=0; net_idx<netDef.op_size(); net_idx++) {
        // iterate all inputs, check in net_output.
        // if not found, read from tensors.
        // define/save output with name to net_output for other functions
        const caffe2::OperatorDef op_def = netDef.op(net_idx);

        if (op_def.type() == "Conv") {
        }
        else if (op_def.type() == "Relu") {
        }
        else if (op_def.type() == "MaxPool") {
        }
        else if (op_def.type() == "LRN") {
        }
        else if (op_def.type() == "DepthConcat") {
        }
        else if (op_def.type() == "AveragePool") {
        }
        else if (op_def.type() == "FC") {
        }
        else if (op_def.type() == "Softmax") {
        }
        else {
            std::cout<< "ENCOUNTER SOME LAYER DOES NOT IMPLEMENTED YET" << std::endl;
            return 0;
        }

        for (int input_idx=0; input_idx<op_def.input_size(); input_idx++){
            string name = op_def.input(input_idx);
            if (net_output.find(name) != net_output.end()) {
                // input already defined
                continue;
            }

            // do we need to iterate from beginning every time?
            for (tensor_idx=0; tensor_idx<tensor.protos_size(); tensor_idx++) {
                if (tensor.protos(tensor_idx).name() == name) {
                    const caffe2::TensorProto& p = tensor.protos(tensor_idx);
                    Image<float> filter = LoadImageFromTensor(p);
                    net_output[name] = filter;
                    break;
                }
           }
        }
    }

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

    Flatten * flatten = new Flatten(pool);
    network.push_back(flatten);

    SoftMax * softm = new SoftMax(flatten);
    network.push_back(softm);

    int C = 10; // number of classes

    Image<float> scores(C, N);

	printf("construct network pass\n");

	printf("test pass\n");

	return 0;


}
