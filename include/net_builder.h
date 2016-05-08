#ifndef NET_BUILDER_H
#define NET_BUILDER_H

#include "layers.h"
#include "data_layer.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "max_pool_layer.h"
#include "avg_pool_layer.h"
#include "flat_layer.h"
#include "concat_layer.h"
#include "lrn_layer.h"
#include "fully_conn_layer.h"

#include "common.h"

#include "Halide.h"

using namespace Halide;

Layer* build_conv(const caffe2::TensorProto& w, const caffe2::TensorProto& b, const caffe2::OperatorDef& op, Layer* input);

Layer* build_relu(Layer* input);

Layer* build_maxpool(const caffe2::OperatorDef& op, Layer* input);

Layer* build_avgpool(const caffe2::OperatorDef& op, Layer* input);

Layer* build_lrn(const caffe2::OperatorDef& op, Layer* input);

Layer* build_concat(std::vector<Layer*> inputs);

Layer* build_fc(const caffe2::TensorProto& w, const caffe2::TensorProto& b, Layer* input);

Layer* build_softmax(Layer* input);

#endif