#include <stdio.h>
#include <iostream>
#include <fstream>

#include "common.h"
#include "Halide.h"

/**
 * @brief LoadImageFromTensor
 *
 * @param tensor
 *
 * @return
 */
Halide::Image<float> LoadImageFromTensor(const caffe2::TensorProto& tensor);
