#include "Halide.h"
#include <map>

using namespace std;
using namespace Halide;


class Layer {
    public:
        //Default Constructor...should support other popular input
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
        virtual int out_dims() = 0;
        // Size of output dimension i, 0 <= i < out_dims()
        virtual int out_dim_size( int i) = 0;

        // Storage for layer parameters
        std::vector<Image<float>> params;
        std::vector<Image<float>> param_grads;
        std::vector<Image<float>> params_cache;
        // Halide function that computes the output of the layer
        Func forward;
        // Vector of halide functions which compute the gradients
        // with respect to layer parameters
        std::vector<Func> f_param_grads;
        // Halide function which computes gradient with respect
        // to layer input
        Func f_in_grad;
        // Defines the functions which compute gradient of the objective
        // function with respective to parameters and input. Given a function
        // which computes the derivate of the objective with respect to layer
        // outputs. Does this recursively for the input layer.
        virtual void back_propagate(Func dforward) = 0;
        virtual ~Layer() {};
};

class SoftMax: public Layer {
    public:
        Var in_dim, n;
        int num_classes, num_samples;
        // Expects 2-dimensional input layer (num_classes x num_samples)
        SoftMax(Layer* in, int schedule = 1) : Layer(in) {
            assert(in->out_dims() == 2);

            //in_f: in_layer's forward function; in_layer = in if in is defined.
            Func in_f = in_layer->forward;

            num_classes = in->out_dim_size(0);
            num_samples = in->out_dim_size(1);

            // Define forward
            // the softmax function, or normalized exponential
            Func exp_max, expo, normalizer;
            RDom r(0, num_classes);
            exp_max(n) = maximum(in_f(r.x, n));
            expo(in_dim, n) = exp(in_f(in_dim, n) - exp_max(n));
            normalizer(n) = cast(in_f.output_types()[0], 0);
            normalizer(n) += expo(r.x, n);
            forward(in_dim, n) = expo(in_dim, n)/normalizer(n);

            if (schedule) {
                // Local schedule
                exp_max.compute_at(forward, n);
                expo.compute_at(forward, n);
                normalizer.compute_at(forward, n);
                forward.compute_root().parallel(n);
            }
        }

        void back_propagate(Func labels) {
            if (!f_in_grad.defined()) {
                assert(labels.defined());
                assert(forward.defined());

                Expr label = clamp(labels(n), 0, num_classes -1);
                Expr t = (forward(in_dim, n) - 1)/num_samples;
                Expr f = (forward(in_dim, n)/num_samples);
                f_in_grad(in_dim, n) = select(in_dim == label, t, f);
                in_layer->back_propagate(f_in_grad);
            }
        }

        // Returns a halide function that computes softmax loss given
        // the correct labels for each sample
        Func loss(Func labels) {
            // Should loss be a layer?

            // Check if labels is defined
            assert(labels.defined());
            // Check if the dimensions make sense
            assert(labels.dimensions() == 1);
            // TODO Figure out if there is a scalar type
            Var x;
            Func loss_p;
            RDom r(0, num_samples);
            loss_p(x) = cast(forward.output_types()[0], 0);
            // The clamp is necessary. Otherwise, halide will assume that the
            // label can be anything during bounds inference.
            loss_p(0) += -log(forward(clamp(labels(r.x), 0, num_classes - 1),
                        r.x))/num_samples;
            return loss_p;
        }

        int out_dims() { return 2;}

        int out_dim_size( int i) {
            assert(i < 2);
            int size = 0;
            if (i == 0)
                size = num_classes;
            else if (i == 1)
                size = num_samples;
            return size;
        }
};

class ReLU: public Layer {
        public:
                Var x, y, z, w;
                int vec_len = 8;
                ReLU(Layer* in, int schedule = 0) : Layer(in) {
                        Func in_f = in_layer->forward;
                        // Define forward
                        switch(in_layer->out_dims()) {
                                case 1:
                                        forward(x) = max(0, in_f(x));
                                        // schedule
                                        if (schedule) {
                                                //forward.compute_root();
                                                //forward.vectorize(x, vec_len);
                                        }
                                        break;
                                case 2:
                                        forward(x, y) = max(0, in_f(x, y));
                                        // schedule
                                        if (schedule) {
                                                //forward.compute_root();
                                                //forward.vectorize(x, vec_len);
                                                //forward.parallel(y);
                                        }
                                        break;
                                case 3:
                                        forward(x, y, z) = max(0, in_f(x, y, z));
                                        // schedule
                                        if (schedule) {
                                                //forward.compute_root();
                                                //forward.vectorize(x, vec_len);
                                                //forward.parallel(z);
                                        }
                                        break;
                                case 4:
                                        forward(x, y, z, w) = max(0, in_f(x, y, z, w));
                                        // schedule
                                        if (schedule) {
                                                //forward.compute_root();
                                                //forward.vectorize(x, vec_len);
                                                //forward.parallel(w);
                                        }
                                        break;
                                default:
                                        assert(0);
                        }

                }

                void back_propagate(Func dout) {
                        assert(dout.defined());
                        if (!f_in_grad.defined()) {
                                Func in_f = in_layer->forward;
                                switch(in_layer->out_dims()) {
                                        case 1:
                                                f_in_grad(x) = dout(x) * select( in_f(x) > 0, 1, 0);
                                                break;
                                        case 2:
                                                f_in_grad(x, y) = dout(x, y) *
                                                        select( in_f(x, y) > 0, 1, 0);
                                                break;
                                        case 3:
                                                f_in_grad(x, y, z) = dout(x, y, z) *
                                                        select(in_f(x, y, z) > 0, 1, 0);
                                                break;
                                        case 4:
                                                f_in_grad(x, y, z, w) = dout(x, y, z, w) *
                                                        select(in_f(x, y, z, w) > 0, 1, 0);
                                                break;
                                        default:
                                                assert(0);
                                }
                                in_layer->back_propagate(f_in_grad);
                        }
                }

                int out_dims() { return in_layer->out_dims();}

                int out_dim_size( int i) {
                        return in_layer->out_dim_size(i);
                }
};

class Convolutional: public Layer {
        public:
                Var x, y, z, n;
                // number of channels, height and width of the input to the layer
                int num_samples, in_ch, in_h, in_w;
                // number of filters, filter height, filter width, padding and stride
                int num_f, f_h, f_w, pad, stride;
                float reg;
                Func f_in_bound;
                // parameters for scheduling
                Var y_t, z_t, par;
                int o_block_size = 16;
                int y_block_size = 32;
                int vec_len = 8;
                Convolutional(int _num_f, int _f_w, int _f_h, int _pad, int _stride,
                                            float _reg, Layer* in, int schedule=true) : Layer(in) {

                        assert(in_layer->out_dims() == 4);

                        num_samples = in_layer->out_dim_size(3);
                        in_ch = in_layer->out_dim_size(2);
                        in_h = in_layer->out_dim_size(1);
                        in_w = in_layer->out_dim_size(0);
                        reg = _reg;

                        assert( (in_h + 2 * _pad - _f_h) % _stride == 0);
                        assert( (in_w + 2 * _pad - _f_w) % _stride == 0);

                        num_f = _num_f; f_h = _f_h; f_w = _f_w;
                        pad = _pad; stride = _stride;

                        // Boundary condition
                        // This creates a padded input and avoids checking boundary
                        // conditions while computing the actual convolution
                        f_in_bound = BoundaryConditions::constant_exterior(
                                                                        in_layer->forward, 0,
                                                                        0, in_w,
                                                                        0, in_h);

                        // Create parameters
                        Image<float> W(f_w, f_h, in_ch, num_f), b(num_f);
                        params.push_back(W); params.push_back(b);

                        // Define forward
                        RDom r(0, f_w, 0, f_h, 0, in_ch);
                        // Initialize to bias
                        forward(x, y, z, n) = b(z);
                        forward(x, y, z, n) += W(r.x, r.y, r.z, z) *
                                                                     f_in_bound(x*stride + r.x - pad,
                                                                                            y*stride + r.y - pad,
                                                                                            r.z, n);

                        if (schedule) {
                                forward.update().reorder(y, x, r.z);
                                // blocking spatially with vectorization
                                //f_in_bound.compute_at(f_simple, n);
                                forward.compute_root();
                                forward.fuse(z, n, par).parallel(par);
                                forward.update().reorder(x, y, r.z); 
                                forward.update().split(y, y, y_t, y_block_size);
                                forward.update().split(z, z, z_t, o_block_size);
                                forward.update().reorder(y_t, z_t, y, r.z, z); 
                                forward.update().vectorize(x, vec_len);          
                                forward.update().fuse(z, n, par).parallel(par);
                                //forward.update().fuse(y, par, par).parallel(par);
                                forward.update().unroll(r.x);
                                forward.update().unroll(r.y);
                                // There are performance implications to this and seems to
                                // be incompatible with some schedules. Have to investigate
                                // this more closely.
                                //f_in_bound.compute_at(forward, n);
                                f_in_bound.compute_at(forward, z_t);
                        }

                }

                void back_propagate(Func dout) {
                        assert(dout.defined());
                        if (!f_in_grad.defined()) {
                                Func dW, db;

                                int out_w = this->out_dim_size(0);
                                int out_h = this->out_dim_size(1);

                                Image<float> W = params[0];
                                Image<float> b = params[1];

                                RDom r1(0, out_w, 0, out_h, 0, num_samples);

                                // intialize to regularized weights
                                dW(x, y, z, n) = cast(dout.output_types()[0],
                                                                            reg * W(x, y, z, n));
                                dW(x, y, z, n) += dout(r1.x, r1.y, n, r1.z) *
                                                                             f_in_bound(r1.x*stride + x - pad,
                                                                                                    r1.y*stride + y - pad,
                                                                                                    z, r1.z);

                                f_param_grads.push_back(dW);

                                // intialize to zero
                                db(x) = cast(dout.output_types()[0], 0);
                                db(x) += dout(r1.x, r1.y, x, r1.z);

                                f_param_grads.push_back(db);

                                RDom r2(0, num_f);
                                // intialize to zero
                                f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);
                                f_in_grad(x, y, z, n) += dout(x, y, r2.x, n) * W(x, y, z, r2.x);

                                // Create storage for gradients and caching params
                                Image<float> W_grad(f_w, f_h, in_ch, num_f);
                                param_grads.push_back(W_grad);
                                Image<float> W_cache(f_w, f_h, in_ch, num_f);
                                params_cache.push_back(W_cache);

                                Image<float> b_grad(num_f);
                                param_grads.push_back(b_grad);
                                Image<float> b_cache(num_f);
                                params_cache.push_back(b_cache);

                                in_layer->back_propagate(f_in_grad);
                        }
                }

                int out_dims() { return 4; }

                int out_dim_size( int i) {
                        assert(i < 4);
                        int size = 0;
                        if (i == 0)
                                size = (1 + (in_w + 2 * pad - f_w)/stride);
                        else if (i == 1)
                                size = (1 + (in_h + 2 * pad - f_h)/stride);
                        else if (i == 2)
                                size = num_f;
                        else if (i == 3)
                                size = num_samples;
                        return size;
                }
};

class MaxPooling: public Layer {
        public:
                // number of color channels in input in_c
                // height and width of the input in_h, in_w
                int num_samples, in_ch, in_h, in_w;
                // height and width of the pool
                // stride at which the pooling is applied
                int p_h, p_w, stride;
                Var x, y, z, n;
                // parameters for scheduling
                Var par;
                int vec_len = 8;
                MaxPooling(int _p_w, int _p_h, int _stride, Layer* in, 
                                     int schedule = 1) : Layer(in) {
                        assert(in_layer->out_dims() == 4);

                        num_samples = in_layer->out_dim_size(3);
                        in_ch = in_layer->out_dim_size(2);
                        in_h = in_layer->out_dim_size(1);
                        in_w = in_layer->out_dim_size(0);

                        assert((in_h - _p_h) % _stride == 0);
                        assert((in_w - _p_w) % _stride == 0);

                        p_w = _p_w; p_h = _p_h; stride = _stride;

                        // Define forward

                        Func in_f = in_layer->forward;
                        RDom r(0, p_w, 0, p_h);
                        forward(x, y, z, n) = maximum(in_f(x * stride + r.x,
                                                                                             y * stride + r.y,
                                                                                             z, n));
                 
                        if (schedule) {
                                forward.vectorize(x, vec_len);
                                forward.compute_root().fuse(z, n, par).parallel(par);
                        }

                }

                void back_propagate(Func dout) {
                        assert(dout.defined());
                        if (!f_in_grad.defined()) {
                                Func in_f = in_layer->forward;
                                Func pool_argmax;
                                RDom r1(0, p_w, 0, p_h);
                                pool_argmax(x, y, z, n) = argmax(in_f(x * stride + r1.x,
                                                                                                            y * stride + r1.y,
                                                                                                            z, n));

                                pool_argmax.compute_root();
                                RDom r2(0, this->out_dim_size(0), 0, this->out_dim_size(1));
                                f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);

                                Expr x_bin = clamp(r2.x * stride +
                                                                     pool_argmax(r2.x, r2.y, z, n)[0], 0, in_w);
                                Expr y_bin = clamp(r2.y * stride +
                                                                     pool_argmax(r2.x, r2.y, z, n)[1], 0, in_h);

                                f_in_grad(x_bin, y_bin, z, n) += dout(r2.x, r2.y, z, n);
                                in_layer->back_propagate(f_in_grad);
                        }
                }

                int out_dims() { return 4; }

                int out_dim_size( int i) {
                        assert(i < 4);
                        int size = 0;
                        if (i == 0)
                                size = 1 + ((in_w - p_w)/stride);
                        else if (i == 1)
                                size = 1 + ((in_h - p_h)/stride);
                        else if (i == 2)
                                size = in_layer->out_dim_size(2);
                        else if (i == 3)
                                size = num_samples;
                        return size;
                }
};