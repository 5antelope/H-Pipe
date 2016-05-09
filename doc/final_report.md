# H-Piper: A Image Pipeline Framework in Halide
\- Lei Sun, Yang Wu

## SUMMARY 

**H-Piper** is our final project for 15418.

We proposed H-Piper, a powerful, flexible, and highly-balanced framework for deep learning Nets. User can easily generate customized image pipelines by providing config files. We tested our framework on VGG and Inception Network on latedays node. With our 10 basic layers, we were able to implemente vgg network and inception networks (incompleted). The idea of our project is given network definitions and weights, H-Piper would build the network and run forward.


## BACKGROUND

Describe the algorithm, application, or system you parallelized in computer science terms. (Recall our discussion from the last day of class.) Figure(s) would be really useful here.

## APPROACH

Tell us how your implementation works. Your description should be sufficiently detailed to provide the course staff a basic understanding of your approach. Again, it might be very useful to include a figure here illustrating components of the system and/or their mapping to parallel hardware.

Since most image piplines could be presented by a combination of different layers, we implemeted a few baisc layers: *data_layer, avg_pool_layer, concat_layer, conv_layer, flat_layer, fully_conn_layer, lrn_layer, max_pool_layer, relu_layer, softmax_layer*.

Each layer has similar signature:
```c++
// `forward` is the output of this layer
Halide::Func forward(x, y, z, n);

// feed the layer with the output of the other
Layer (int...params, Layer *input);

// give output's number of dimensions
int out_dims();

// give output's size in one dimension
int out_dim_size(int i);
```
For example we want to build a network that fed a image for input and then continues with a convolution computation, we would just doing this in H-Piper:

```c++
// generate a data layer from image
Halide::Image<float> data = load_image();
DataLayer* data_layer = new DataLayer(h, w, ch, n, data);

Convolutional* conv_layer  = new Convolutional(int...params, data_layer);
... 
```
With H-Piper's basic layers, most of image pipelines become building blocks. As long as we have correct input-output mapping and parameters defined, H-Piper would do the heavy lifting in computations for you.

Also, during our development, we accidentally trigger an [assert](https://github.com/halide/Halide/blob/master/src/Buffer.cpp#L12) in Halide, before we fix the bug, our workaround was spliting the kernel filter that exceed buffer limitations into 2 kernels by 4th dimension and concat together after computation. Although it was caused by a dimension error in our implementatin, but this check actually allowed us to handle different size of filters more flexible.

## RESULTS

How successful were you at achieving your goals? We expect results sections to differ from project to project, but we expect your evaluation to be very thorough (your project evaluation is a great way to demonstrate you understood topics from this course).

## REFERENCES

[0] Going Deeper with Convolutions, C. Szegedy, W. Liu, Y. Jia and etc.

[1] PolyMage: Automatic Optimization for Image Processing Pipelines, R. Mullapudi, V. Vasista and U. Bondhugula
