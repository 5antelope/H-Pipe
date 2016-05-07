#include "layers.h"
#include "data_layer.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "max_pool_layer.h"
#include "flat_layer.h"

#include "Halide.h"

using namespace Halide;

Layer* build_conv(const caffe2::TensorProto& w, const caffe2::TensorProto& b, const caffe2::OperatorDef& op, Layer* input);

Layer* build_relu(Layer* input);

Layer* build_maxpool(const caffe2::OperatorDef& op, Layer* input);

Layer* build_avgpool(const caffe2::OperatorDef& op, Layer* input);

Layer* build_lrn(const caffe2::OperatorDef& op, Layer* input);

Layer* build_concat(Layer* input);

Layer* build_fc(Layer* input);

Layer* build_softmax(Layer* input);