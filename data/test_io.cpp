#include "data.pb.h"
#include "io.hpp"
#include <string>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

int main(int argc, char* argv[]) {

    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    string path="/home/yangwu/git/H-Net/data/inception_net.pb";
    std::fstream input(path, ios::in | ios::binary);

    NetDef netDef;

    if (!input) {
        std::cout << path << ": File not found." << std::endl;
    } else if (!netDef.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse address book." << std::endl;
        return -1;
    }

    std::cout << "protbuf test pass" << std::endl;

    return 0;
}
