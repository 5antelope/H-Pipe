#include "db.hpp"
#include "data.pb.h"
#include "io.hpp"
#include <string>

int main(int argc, char* argv[]) {

    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    string path="/home/yangwu/mnist/mnist_train_lmdb";

    ifstream file;
    file.open(path);

    if (!file.is_open())
        LOG(INFO) << "image file cannot be open";

    NetDef netDef;
    netDef = netDef.ParseFromString(file);
    
    std::cout << "protbuf test pass";

    return 0;
}
