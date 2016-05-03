#include "layers.h"
#include <stdio.h>
#include <sys/time.h>
#include "data/common.hpp"
#include "data/io.hpp"
#include "halide_image_io.h"

int main(int argc, char **argv) {

    // Google logging needed for parts that were extracted from caffe
    google::InitGoogleLogging(argv[0]);

    // datalayer test /////
    Halide::Image<uint8_t> input = Tools::load_image("dog.png");

    DataLayer * data_layer = new DataLayer(input.height(), input.width(), input.channels(), 1, input);

    printf("data out size %d x %d x %d x %d\n", data_layer->out_dim_size(0),
                                                data_layer->out_dim_size(1),
                                                data_layer->out_dim_size(2),
                                                data_layer->out_dim_size(3));

    return 0;
}
