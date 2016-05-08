#ifndef TENSOR2IMAGE_H
#define TENSOR2IMAGE_H

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
inline Halide::Image<float> LoadImageFromTensor(const caffe2::TensorProto& tensor);

#endif
