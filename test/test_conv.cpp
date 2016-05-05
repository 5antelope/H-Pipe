#include "caffe2.pb.h"

#include "Halide.h"
#include "halide_image_io.h"

#include "common.h"
#include "conv_layer.h"
#include "tensor2image.h"

#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

using namespace Halide;
using namespace Halide::Tools;
using ::google::protobuf::Message;

int main(int argc, char* argv[]) {

	google::InitGoogleLogging(argv[0]);

    string net_path="/home/yangwu/git/H-Pipe/src/inception_net.pb";
    string tensors_path="/home/yangwu/git/H-Pipe/src/inception_tensors.pb";

    std::fstream net_input(net_path, ios::in | ios::binary);
    std::fstream tensors_input(tensors_path, ios::in | ios::binary);

    // net is the network definition.
    // caffe2::NetDef netDef;
    // tensors contain the parameter tensors.
    caffe2::TensorProtos tensor;

    // netDef.ParseFromIstream(&net_input);
    tensor.ParseFromIstream(&tensors_input);

    // std::cout << "Loaded operators of size: " << netDef.op_size() << std::endl;
    // std::cout << "Net name: " << netDef.name() << std::endl;

    // DO NOT name this `operator`
    // const caffe2::OperatorDef& op = netDef.op(0);

    Halide::Image<float> input = load_image("/home/yangwu/git/H-Pipe/dog.png");

    // test on conv2d0 layer
    const caffe2::TensorProto& tensor_w = tensor.protos(0);
    Halide::Image<float> kernel = LoadImageFromTensor(tensor_w);
    std::cout << "Loaded: " << tensor_w.name() << std::endl;
    std::cout << "Dims: " << kernel.dimensions() << std::endl;

    const caffe2::TensorProto& tensor_b = tensor.protos(1);
    Halide::Image<float> bias = LoadImageFromTensor(tensor_b);
    std::cout << "Loaded: " << tensor_b.name() << std::endl;
    std::cout << "Dims: " << bias.dimensions() << std::endl;

    // int _stride = op.arg(0).i();
    // int _pad = op.arg(2).i();
    // string _order = op.arg(3).s();


    // printf("stride: %d, pad: %d\n", _stride, _pad);
    // std::cout << _order << std::endl;

    // Convolutional conv = Convolutional(kernel.name(), _pad, _stride, _order);
    Convolutional conv = Convolutional(kernel.name(), 2, 2, "ORDER");

    std::cout << "CONSTRUCTED" <<std::endl;

    conv.load_weight(kernel);
    conv.load_bias(bias);

    std::cout << "READY" <<std::endl;

    // Func output = conv.run((Func)input, input.extent(0), input.extent(1), input.extent(2), input.extent(3));

    // printf("output of conv2d0 layer: %d\n", output.dimensions());

    printf("conv test pass\n");

    return 0;
}
