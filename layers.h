#include "Halide.h"

using namespace Halide;

class Layer {
    public:
        Layer(Layer* in) {
            // The first layer in the pipeline does not have an input layer
            if (in) {
                // Get the halide function that computes values
                // of the input layer
                assert(in->forward.defined());

                // Record the input layer
                in_layer = in;
            }
        }

        // Layer that serves as an input to the current layer
        Layer* in_layer;

        // Number of output dimensions
        virtual  int out_dims() = 0;
        // Size of output dimension i; 0 <= i < out_dims()
        virtual  int out_dim_size(int i) = 0;

        // Storage for layer parameters
        std::vector<Image<float>> params;

        // Halide function that computes the output of the layer
        Func forward;
        
        virtual ~Layer() {};
};

/**
 * DataLayer feeds pipeline the input data
 */
class DataLayer: public Layer {
    public:
        int in_w, in_h, in_ch, num_samples;

        // x, y, z dimensions and number
        Var x, y, z, n;

        DataLayer(int _in_w, int _in_h, int _in_ch, int _num_samples,
                  Image<float> &data) : Layer(0) {

            in_w = _in_w;in_h = _in_w; in_ch = _in_ch;

            num_samples = _num_samples;

            // Define forward
            forward(x, y, z, n) = data(x, y, z, n);
        }

        int out_dims() { return 4; }

        int out_dim_size( int i) {
            assert(i < 4);

            int size = 0;
            if (i == 0)
                size = in_w;
            else if (i == 1)
                size = in_h;
            else if (i == 2)
                size = in_ch;
            else if (i == 3)
                size = num_samples;

            return size;
        }

};
