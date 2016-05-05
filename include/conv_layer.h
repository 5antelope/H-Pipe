#include "caffe2.pb.h"

#include "Halide.h"

using namespace Halide;

class Convolutional: public Layer {

    public:
	    int stride = 0, pad = 0;

        caffe2::TensorProto tensor;
        caffe2::OperatorDef op;

	    // filter and bias for convlution.
	    // should be set from outside
	    Image<float> weight;
        Image<float> bias;

    	// TODO: how to pass parameters
        Convolutional(string _name,
                const caffe2::TensorProto* tensor, // define weight and bias
                const caffe2::OperatorDef* op); // define input/output and stride, pad..


        // run the real network
        Func run(Func, int, int, int, int);

        void set_tensor(caffe2::TensorProto _tensor) {tensor = _tensor;}
        void set_op(caffe2::OperatorDef _op) {op = _op;}

        void set_weight(Image<float> _weight) {weight = _weight;}
        void set_bias(Image<float> _bias) {bias = _bias;}
};
