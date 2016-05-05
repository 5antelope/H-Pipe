#include "caffe2.pb.h"

#include "common.h"
#include "tensor2image.h"

#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

using ::google::protobuf::Message;

int main(int argc, char* argv[]) {

	google::InitGoogleLogging(argv[0]);

    string net_path="/home/yangwu/git/H-Net/data/inception_net.pb";
    string tensors_path="/home/yangwu/git/H-Net/src/inception_tensors.pb";

    std::fstream net_input(net_path, ios::in | ios::binary);
    std::fstream tensors_input(tensors_path, ios::in | ios::binary);

    // net is the network definition.
    caffe2::NetDef netDef;
    // tensors contain the parameter tensors.
    caffe2::TensorProtos tensors;

    netDef.ParseFromIstream(&net_input);

    tensors.ParseFromIstream(&tensors_input);

    std::cout << "Loaded operators of size: " << netDef.op_size() << std::endl;

    const caffe2::OperatorDef& operator = netDef.op(0);
    std::cout << "Net name: " << operator.name() << std::endl;

    Halide::Image<float> input = load_image("/home/yangwu/git/H-Pipe/dog.png");

    // test on conv2d0 layer
    Halide::Image<float> kernel = LoadImageFromTensor(&tensors.protos(0));
    Halide::Image<float> bias = LoadImageFromTensor(&tensors.protos(1));

    int _stride = operator.arg(0).i();
    int _pad = operator.arg(2).i();
    string _order =  = operator.arg(3).s();

    Convolutional conv = Convolution(kernel.name(), _pad, _stride, _order);
    conv.load_tensor();

    conv.load_weight(kernel);
    conv.load_bias(bias);

    conv.run((Func)input, input.extent(0), input.extent(1), input.extent(2), input.extent(3));

    Func output = conv->data;
    printf("output of conv2d0 layer: %d\n", output.dimensions());

    printf("conv test pass\n");

    // Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
