# H-Piper: A Image Pipeline Framework in Halide
\- Lei Sun, Yang Wu

## SUMMARY 

**H-Piper** is our final project for 15418.

We proposed H-Piper, a powerful, flexible, and highly-balanced framework for deep learning Nets. User can easily generate customized image pipelines by providing config files. We tested our framework on VGG and Inception Network on latedays node. With our 10 basic layers, we were able to implemente vgg network and inception networks (incompleted). The idea of our project is given network definitions and weights, H-Piper would build the network and run forward.


## BACKGROUND

Describe the algorithm, application, or system you parallelized in computer science terms. (Recall our discussion from the last day of class.) Figure(s) would be really useful here.

To integrated with [PolyMage](http://drona.csa.iisc.ernet.in/~uday/publications/uday15asplos.pdf) for automatic scheduler, every layer should provide size in dimensions. However, [Halide::Func](https://github.com/halide/Halide/blob/master/src/Func.h) does not provide such information. Therefore, we extent our implementation for this dimension-based information as well.

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

We were able to run a complete VGG network with H-Piper, and explore some scheduler strategy for every layer of network:
- 1. no schedule
- 2. parallel softmax layer
- 3. parallel maxpool layer
- 4. parallel conv layer
- 5. reorder conv layer
- 6. reorder + parallel conv layer

This is our result, x-axis is the schedule we were trying and y-axis is the time cost:
![chart](https://docs.google.com/spreadsheets/d/1DVGrHmwrwSQpoLAySiYdZn9X7W8N0-TnpNACT1w4zY8/pubchart?oid=296431700&format=image)

And we have a few observations from the chart:
- Most of time in Halide pipelines are cost in computation rather in network creation. This is because in creation phase, there is no actual work being done, but define input/output of Halide::Func. And this is also a reason why fusion in Halide can be gained easily.
- Convolution layer dominates the performance of VGG, since the updates of other types of layers does not affect the performance in a similar way as convolution layer.
- HAND-TUNING IS PAINFUL. It is not true that a 'sophisticated' strategy would guarantee a better performance. Actually in the chart, we can see purely parallel would get worse performance. And also, there is no one-for-all general scheduler for layers. For example, parallel over channels seems to be a reasonable approach, but in case of fully-connected layer accross channels, this is not practical any more. In short, schedule must be defined based on filter size, input image and computation type.

## REFERENCES

[0] Going Deeper with Convolutions, C. Szegedy, W. Liu, Y. Jia and etc.

[1] PolyMage: Automatic Optimization for Image Processing Pipelines, R. Mullapudi, V. Vasista and U. Bondhugula
