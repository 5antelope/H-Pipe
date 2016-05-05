#include <stdio.h>
#include <iostream>
#include <fstream>

#include "caffe2.pb.h"
#include "Halide.h"

/**
 * @brief LoadImageFromTensor
 *
 * @param tensor
 *
 * @return
 */
Halide::Image<float> LoadImageFromTensor(const caffe2::TensorProto& tensor);
