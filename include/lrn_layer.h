class LRN: public Layer {
public:
    Var x, y, z, n;

    int input_width, input_height, intput_channel, input_num;

    LRN(Layer* in, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f): Layer(in);

    void back_propagate(Func dout);

    int out_dims();

    int out_dim_size(int i);
};
