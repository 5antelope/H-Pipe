#include "Halide.h"

using namespace std;
using namespace Halide;

class Layer {
    public:
        string name;
        string type;

        int input_width;
        int input_height;
        int input_channel;
        int input_num;

        int output_width;
        int output_height;
        int output_channel;
        int output_num;

        // getter setter of private fields
        void set_name(string _name) { name = _name;}

        void set_type(string _type) { type = _type;}

        void set_output_width(int _width) { output_width = _width;}

        void set_output_height(int _height) { output_height = _height;}

        void set_output_channels(int _channel) { output_channels = _channel;}

        void set_output_num(int _output_num) { output_num = _output_num;}

        string get_name() { return name;}

        string get_type() { return type;}

        int get_output_width() { return output_width; }

        int get_output_height() { return output_height;}

        int get_output_channels() { return output_channel;}

        int get_output_num() { return output_num; }

        // the output of layer
        Func data;

        // 4 Halide parameters for Func data
        Var x, y, z, n;

        // number of layer dimensions
        virtual int layer_dims() = 0;
        // size of layer in each dimensions; 0 <= i < out_dims()
        virtual int layer_extent(int i) = 0;

        virtual Func run(Func input);

        Layer() {};
        ~Layer() {};
};
