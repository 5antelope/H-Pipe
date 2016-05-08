#ifndef LRN_LAYER_H
#define LRN_LAYER_H

class LRN: public Layer {
public:
    Var x, y, z, n;

    int input_width, input_height, intput_channel, input_num;

    LRN(Layer* in, int region_x, int region_y, int region_z, float alpha, float beta);

    void back_propagate(Func dout);

    int out_dims();

    int out_dim_size(int i);
};

#endif