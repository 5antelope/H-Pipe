#include <stdio.h>
#include <iostream>
#include <fstream>

#include "tensor2image.h"

using namespace Halide;

Image<float>
LoadImageFromTensor(const caffe2::TensorProto *tensor) {
    int width = 1;
    int height = 1;
    int channel = 1;
    int batch = 1;

    batch = tensor->dims(0);

    if (tensor->dims_size() == 4) {
        width = tensor->dims(1);
        height = tensor->dims(2);
        channel = tensor->dims(3);
    }

    Image<float> image(width, height, channel, batch);

    int idx = 0;

    if (tensor->dims_size() == 4) {
        /* TODO: Can we parallel this? */
        for (int n=0; n<batch; n++) {
            for (int k=0; k<channel; k++) {
                for (int j=0; j<height; j++) {
                    for (int i=0; i<width; i++) {
                        image(i, j, k, n) = tensor->float_data(idx);
                        idx += 1;
                    }
                }
            }
        }
    }
    else {// dim size = 1
        for (int n=0; n<batch; n++) {
            image(1, 1, 1, n) = tensor->float_data(idx);
            idx += 1;
        }
    }

    printf("Load Halide::Image from Tensor\n");

    return image;
}
