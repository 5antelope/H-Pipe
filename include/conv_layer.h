#include "Halide.h"

class Convolutional: public Layer {
	
	int pad, stride;

	// filter and bias for convlution.
	// should be set from outside
	Image<float> filter, bias;

    public:
    	// TODO: how to pass parameters
        Convolutional(string _name, int schedule=false) {

        }

        int layer_dims() { return 4; }

        int layer_extent( int i) {
            assert(i < 4);

            if (i == 0)
                return get_width();
            else if (i == 1)
                return get_height();
            else if (i == 2)
                return get_channels();
            else if (i == 3)
                return get_batch();

            // error
            return -1;
        }
};