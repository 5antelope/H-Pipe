#ifndef FLAT_LAYER_H
#define FLAT_LAYER_H

class Flatten: public Layer {
    public:
        Var x, y, z, n;
        int out_width;
        int num_samples;

        Flatten(Layer *in, int schedule);

        void back_propagate(Func dout);

        int out_dims();

        int out_dim_size( int i);
};

#endif