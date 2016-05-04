#include <stdio.h>
#include <iostream>
#include <fstream>

#include "tensor2image.h"

using namespace Halide;

Image<float>
LoadImageFromTensor(const caffe2::TensorProto *tensor) {
    int width = tensor->dims(0);
    int height = tensor->dims(1);
    int channel = tensor->dims(2);
    int batch = tensor->dims(3);

    Image<float> image(width, height, channel, batch);

    /*
     * TODO: Can we parallel this?
     */
    int idx = 0;

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

    printf("Load Halide::Image from Tensor\n");

    return image;
}
