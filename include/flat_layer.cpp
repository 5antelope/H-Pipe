#include "flat_layer.h"

class Flatten: public Layer {
    public:
        Var x, y, z, n;
        int out_width;
        int num_samples;
        Flatten(Layer *in, int schedule = 1) : Layer(in) {
            assert(in->out_dims() >= 2 && in->out_dims() <= 4);
            num_samples = in_layer->out_dim_size(in_layer->out_dims() - 1);
            // Define forward
            if (in_layer->out_dims() == 2) {
                out_width = in_layer->out_dim_size(0);
                forward(x, n) = in_layer->forward(x, n);
            } else if (in_layer->out_dims() == 3) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                out_width = w * h;
                forward(x, n) = in_layer->forward(x%w, (x/w), n);
            } else if (in_layer->out_dims() == 4) {
                int w = in_layer->out_dim_size(0);
                int h = in_layer->out_dim_size(1);
                int c = in_layer->out_dim_size(2);
                out_width = w * h * c;
                forward(x, n) = in_layer->forward(x%w, (x/w)%h, x/(w*h), n);
            }
            // schedule 
            if (schedule) {
                forward.compute_root().parallel(n);
            }

        }

        void back_propagate(Func dout) {
            assert(dout.defined());
            if(!f_in_grad.defined()) {
                if(in_layer->out_dims() == 2)
                    f_in_grad(x, n) = dout(x, n);
                else if(in_layer->out_dims() == 3) {
                    int w = in_layer->out_dim_size(0);
                    f_in_grad(x, y, n) = dout(y*w + x, n);
                } else if (in_layer->out_dims() == 4) {
                    int w = in_layer->out_dim_size(0);
                    int h = in_layer->out_dim_size(1);
                    f_in_grad(x, y, z, n) = dout(z*w*h + y*w + x, n);
                }
                in_layer->back_propagate(f_in_grad);
            }
        }

        int out_dims() { return 2; }

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = out_width;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};
