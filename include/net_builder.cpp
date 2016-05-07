#include "net_builder.h"

using namesapce Halide;

Layer*
build_conv(const caffe2::TensorProto& _w, const caffe2::TensorProto& _b, const caffe2::OperatorDef& op, Layer* input) {
	Image<float> w = LoadImageFromTensor(_w);
	Image<float> b = LoadImageFromTensor(_b);

	int num_filter    = w.dims(0);  // number of filters
	int filter_width  = w.dims(1);  // filter width
	int filter_height = w.dims(2);  // filter height


	int pad    = op.arg(2).i(); // padding required to handle boundaries
	int stride = op.arg(0).i(); // stride at which the filter evaluated

	Convolutional* conv  = new Convolutional(num_filter, num_filter, filter_height, pad, stride, d_layer);

    conv->params[0] = w;
    conv->params[1] = b;

    return conv;
}

Layer* build_relu(Layer* input) {
	ReLU* relu = new ReLU(input);
    
    return relu;
}

Layer* build_maxpool(const caffe2::OperatorDef& op, Layer* input) {
	int stride = op.arg(0).i();
	int size = op.arg(1).i();
	
	// only support square box maxpoll
	assert(size == 3);
	MaxPooling* maxpool = new MaxPooling(size, size, stride, input);

	return maxpool;
}

Layer* build_avgpool(const caffe2::OperatorDef& op, Layer* input) {
	int stride = op.arg(0).i();
	int size = op.arg(1).i();
	
	// only support square box maxpoll
	assert(size == 3);
	AvgPooling* avgpool = new AvgPooling(size, size, stride, input);

	return avgpool;
}

Layer* build_lrn(const caffe2::OperatorDef& op, Layer* input) {
	int size = op.arg(0).i();
	int alpha = op.arg(1).f();
	int beta = op.arg(2).f();
	int bias = op.arg(3).f();

	// only support square box lrn
	// assume region-z = 1 (channel)
	LRN* lrn = new LRN(size, size, 1, alpha, beta, input);

	return lrn;
}

Layer* build_softmax(Layer* input) {
	SoftMax* softmax = new SoftMax(input);

	return softmax;
}

Layer* build_concat(std::vecotr<Layer*> inputs) {
	Concat* concat = new Concat(inputs);

	return concat;
}

Layer* build_fc(const caffe2::TensorProto& _w, const caffe2::TensorProto& _b, Layer* input) {
	Image<float> w = LoadImageFromTensor(_w);
	Image<float> b = LoadImageFromTensor(_b);

	FC fc = new FC(input);

	fc->params[0] = w;
    fc->params[1] = b;

    return fc;
}
