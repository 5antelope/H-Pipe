#include "Halide.h"

using namespace Halide;

class LRN: public Layer {
public:
    Var x, y, z, n;

    LRN(Layer* in, int region_x=1, int region_y=1, int region_z=1, float alpha=1.0f, float beta=5.0f): Layer(in) {

        Func clamped = BoundaryConditions::constant_exterior(in->forward, 0.0f, 0, in->out_dim_size(0), 0, in->out_dim_size(1), 0, in->out_dim_size(2));
        Func activation;
        Func normalizer;

        RDom r(-region_x / 2, region_x / 2 + 1, -region_y / 2, region_y / 2 + 1, -region_z / 2, region_z / 2 + 1);

        Expr val = clamped(x + r.x, y + r.y, z + r.z, n);

        activation(x, y, z, n) = sum(val * val);
        normalizer(x, y, z ,n) = fast_pow(1.0f + (alpha / (region_x * region_y * region_z)) * activation(x, y, z, n), beta);
        forward(x, y, z, n) = clamped(x, y, z, n) / normalizer(x, y, z, n);
  }
};
