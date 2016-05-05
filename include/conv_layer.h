#include "caffe2.pb.h"

#include "Halide.h"

using namespace Halide;

class Convolutional: public Layer {

	int stride, pad;

	// filter and bias for convlution.
	// should be set from outside
	Image<float> filter;
    // Image<float> bias;

    public:
    	// TODO: how to pass parameters
        Convolutional(string _name,
                const caffe2::TensorProto *tensor, // define filter
                const caffe2::OperatorDef *ops); // define input/output and stride, pad..

        /*
         * This constructor is used for correctness test purpose
         * Pass in filter and stride, pad parameters directly for conv layer
         */
        Convolutional(string _name,
                Image<float> _filter,
                int _stride, int _pad);

        Func run(Func, int, int, int, int);

        int layer_dims();

        int layer_extent(int i);
};
