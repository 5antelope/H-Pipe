#include "caffe2.pb.h"
#include "common.hpp"

#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

using ::google::protobuf::Message;

int main(int argc, char* argv[]) {

	google::InitGoogleLogging(argv[0]);

    // string net_path="/home/yangwu/git/H-Net/data/inception_net.pb";
    string tensors_path="/home/yangwu/git/H-Net/data/inception_tensors.pb";

    // std::fstream net_input(net_path, ios::in | ios::binary);
    std::fstream tensors_input(tensors_path, ios::in | ios::binary);

    // net is the network definition.
    // caffe2::NetDef netDef;
    // tensors contain the parameter tensors.
    caffe2::TensorProtos tensors;

    // if (!net_input) {
    //     std::cerr << net_path << ": File not found." << std::endl;
    //     return -1;
    // }

    if (!tensors_input) {
        printf("File not found.\n");
        return -1;
    }

    netDef.ParseFromIstream(&net_input);

    tensors.ParseFromIstream(&tensors_input);

    std::cout << "Load netDef: " << netDef.name() << std::endl;
    printf("Load tensorProtos of size %d\n", tensors.protos_size());

    printf("protbuf test pass\n");

    // Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;	
}
