#include "data.pb.h"
#include "io.hpp"
#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

int main(int argc, char* argv[]) {

    // Initialize Google's logging library.
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
        LOG(ERROR) << net_path << ": File not found.";
        return -1;
    }

    if (!tensors_input) {
        LOG(ERROR) << tensors_path << ": File not found.";
        return -1;
    }

    if (!netDef.ParseFromIstream(&net_input)) {
        LOG(ERROR) << "Failed to parse address book.";
        return -1;
    }

    if (!netDef.ParseFromIstream(&net_input)) {
        LOG(ERROR) << "Failed to parse address book.";
        return -1;
    }

    LOG(INFO) << "protbuf test pass";

    // Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
