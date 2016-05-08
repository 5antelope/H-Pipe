#include "net_builder.h"
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
    // value: layer of network
    map<string, Layer*> net_output;
    // key: name of tensor
    // value: tensor protobuf
    map<string, const caffe2::TensorProto> net_tensor;

	std::vector<Layer*> network;
	float reg = 0.001;

    Halide::Image<float> data_origin = load_image("/home/yangwu/git/H-Pipe/dog.png");

    /********** DATA LAYER  **********/

	int d_w = 32; // data width
	int d_h = 32; // data height
	int ch  = 3; // number of channels
	int N   = 1; // number of samples

    // scale the image for testing purpose
    Func data = BoundaryConditions::constant_exterior(data_origin, 0.f, 0, d_w, 0, d_h);

	DataLayer * data_layer = new DataLayer(d_h, d_w, ch, N, data);
    net_output["input"] = data_layer;

    /********** DEFINED LAYERS **********/

    int tensor_idx= 0;

    for (int net_idx=0; net_idx<netDef.op_size(); net_idx++) {
        // iterate all inputs, check in net_output.
        // if not found, read from tensors.
        // define/save output with name to net_output for other functions

        const caffe2::OperatorDef op_def = netDef.op(net_idx);

        Layer* input;

        if (op_def.type() == "Conv") {

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            string w_name = op_def.input(1);
            string b_name = op_def.input(2);

            int count = 0;
            // do we need to iterate from beginning every time?
            for (tensor_idx=0; tensor_idx<tensor.protos_size(); tensor_idx++) {
                if (tensor.protos(tensor_idx).name() == w_name) {
                    const caffe2::TensorProto& t1 = tensor.protos(tensor_idx);
                    net_tensor[w_name] = t1;
                    count++;
                    continue;
                }
                else if (tensor.protos(tensor_idx).name() == b_name) {
                    const caffe2::TensorProto& t2 = tensor.protos(tensor_idx);
                    net_tensor[b_name] = t2;
                    count++;
                    continue;
                }

                if (count == 2) {
                    Layer* l = build_conv(net_tensor[w_name], net_tensor[b_name], op_def, input);

                    // store output layer to map
                    net_output[output_name] = l;
                    break;
                }
           }

        }
        else if (op_def.type() == "Relu") {

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            Layer* l = build_relu(input);

            // store output layer to map
            net_output[output_name] = l;
        }
        else if (op_def.type() == "MaxPool") {

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            Layer* l = build_maxpool(op_def, input);

            // store output layer to map
            net_output[output_name] = l;
        }
        else if (op_def.type() == "AveragePool") {
            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            Layer* l = build_avgpool(op_def, input);

            // store output layer to map
            net_output[output_name] = l;
        }
        else if (op_def.type() == "LRN") {

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            Layer* l = build_lrn(op_def, input);

            // store output layer to map
            net_output[output_name] = l;
        }
        else if (op_def.type() == "DepthConcat") {

            string output_name = op_def.output(0);

            std::vector<Layer*> inputs;

            for (int i=0; i<op_def.input_size(); i++) {
                string input_name = op_def.input(i);

                if (net_output.find(input_name) == net_output.end()) {
                    std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                    return -1;
                }

                inputs.push_back(net_output[input_name]);

            }

            Layer* l = build_concat(inputs);

            // store output layer to map
            net_output[output_name] = l;
        }
        else if (op_def.type() == "FC") {
            const caffe2::TensorProto* t1, t2;

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            string w_name = op_def.input(1);
            string b_name = op_def.input(2);

            int count = 0;
            // do we need to iterate from beginning every time?
            for (tensor_idx=0; tensor_idx<tensor.protos_size(); tensor_idx++) {
                if (tensor.protos(tensor_idx).name() == w_name) {
                    t1 = tensor.protos(tensor_idx);
                    net_tensor[w_name] = t1;
                    count++;
                    continue;
                }
                else if (tensor.protos(tensor_idx).name() == b_name) {
                    t2 = tensor.protos(tensor_idx);
                    net_tensor[b_name] = t2;
                    count++;
                    continue;
                }

                if (count == 2)
                    break;
           }

           Layer* l = build_fc(t1, t2, input);

           // store output layer to map
           net_output[output_name] = l;
        }
        else if (op_def.type() == "Softmax") {

            string input_name = op_def.input(0);
            string output_name = op_def.output(0);

            if (net_output.find(input_name) == net_output.end()) {
                // input layer MUST already exist
                std::cout << "LAYER INPUT IS NOT READY" << std::endl;
                return -1;
            }

            input = net_output[input_name];

            Layer* l = build_softmax(input);

            // store output layer to map
            net_output[output_name] = l;
        }
        else {
            std::cout<< "ENCOUNTER SOME LAYER DOES NOT IMPLEMENTED YET" << std::endl;
            return -1;
        }
    }

	printf("test pass\n");

	return 0;


}
