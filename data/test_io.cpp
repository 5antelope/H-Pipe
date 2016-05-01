#include "data.pb.h"
#include "io.hpp"
#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

int main(int argc, char* argv[]) {

    // Initialize Google's logging library.
    FLAGS_log_dir = "/home/yangwu/log";
    google::InitGoogleLogging(argv[0]);

    string net_path="/home/yangwu/git/H-Net/data/inception_net.pb";
    string tensors_path="/home/yangwu/git/H-Net/data/inception_tensors.pb";

    std::fstream net_input(net_path, ios::in | ios::binary);
    std::fstream tensors_input(tensors_path, ios::in | ios::binary);

    // net is the network definition.
    NetDef netDef;
    // tensors contain the parameter tensors.
    TensorProtos tensors;

    if (!net_input) {
        std::cerr << net_path << ": File not found." << std::endl;
        return -1;
    }

    if (!tensors_input) {
        std::cerr << tensors_path << ": File not found." << std::endl;
        return -1;
    }

    if (!netDef.ParseFromIstream(&net_input)) {
        std::cerr << "Failed to parse address book." << std::endl;
        return -1;
    }

    if (!tensors.ParseFromIstream(&tensors_input)) {
        std::cerr << "Failed to parse address book." << std::endl;
        return -1;
    }

    std::cout << "protbuf test pass" << std::endl;

    // for (int i = 0; i < tensors._size(); i++) {
    //
    // }

    // Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
