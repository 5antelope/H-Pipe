#include "caffe2.pb.h"

#include "Halide.h"

using namespace Halide;

class Convolutional: public Layer {

    public:
	    int stride = 0, pad = 0;
        string order;

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

        void laod_tensor(const caffe2::TensorProto* _tensor);
        void set_weight(Image<float> _weight) {weight = _weight;}
        void set_bias(Image<float> _bias) {bias = _bias;}

        void set_stride(int _stride) {stride = _stride;}
        void set_pad(int _pad) {pad = _pad;}
        void set_order(string _order) {order = _order;}
};
