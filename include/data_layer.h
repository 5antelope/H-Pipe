#ifndef DATA_LAYER_H
#define DATA_LAYER_H

class DataLayer : public Layer {

public:
  int in_w;
  int in_h;
  int in_ch;
  int num_samples;

  Var x, y, z, n;

  DataLayer(int _in_w, int _in_h, int _in_ch, int _num_samples,
            Func data);

  // Nothing to propagate
  void back_propagate(Func dout);

  int out_dims();

  int out_dim_size(int i);

};

#endif