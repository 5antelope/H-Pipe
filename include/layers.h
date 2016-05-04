#include "Halide.h"

using namespace std;
using namespace Halide;

class Layer {
    private:
        string name;
        string type;
        // dims
        int width; 
        int height; 
        int channels;
        int batch;

    public:
        // getter setter of private fields
        void set_name(string _name) {
            name = _name;
        }
        void set_type(string _type) {
            type = _type;
        }
        void set_width(int _width) {
            width = _width;
        }
        void set_height(int _height) {
            width = _width;
        }
        void set_channels(int _width) {
            width = _width;
        }
        void set_batch(int _batch) {
            batch = _batch;
        }
        string get_name() {
            return name;
        }
        string get_type() {
            return type;
        }
        int get_width() {
            return width;
        }
        int get_height() {
            return height;
        }
        int get_channels() {
            return channels;
        }
        int get_batch() {
            return batch;
        }

        // the output of layer
        Func data;

        // 4 Halide parameters for Func data
        Var x, y, z, n;

        // storage for layer parameters
        std::vector<Image<float>> params;

        // number of layer dimensions
        virtual int layer_dims() = 0;
        // size of layer in each dimensions; 0 <= i < out_dims()
        virtual int layer_extent(int i) = 0;

        virtual Func run(Func input) = data;

        Layer() {};
        ~Layer() {};
};
