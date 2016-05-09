# H-Piper: A Image Pipeline Framework in Halide
\- Lei Sun, Yang Wu

## SUMMARY 

**H-Piper** is our final project for 15418.

We proposed H-Piper, a powerful, flexible, and highly-balanced framework for deep learning Nets. User can easily generate customized image pipelines by providing config files. We tested our framework on VGG and Inception Network on latedays node. With our 10 basic layers, we were able to implemente vgg network and inception networks (incompleted). The idea of our project is given network definitions and weights, H-Piper would build the network and run forward. Not only H-Piper could be used to study the best optimization strategies for different nets, but it could also be a performance reference. We've tested H-Piper with VGG and Inception Net on Latedays clusters.


## BACKGROUND

MXNet and Caffe are pretty popular frameworks for image processing pipelines. Both frameworks are carefully hand-tuned and have outstanding performance. Hand-tuning for image pipelines could be very painful. Changes’ correctness needs to be verified everytime and the amount of code needs to be changed for the most simple reschedule is a lot.

Halide is a Domain-Specific language designed to make it easier to write high-performance image processing code on modern machines. It provides a concept of “scheduling” which allows developers to easily define he or she wants to iterate through the dataset. Below are some schedule examples.

**Image * 4 in a row**

We implemened 10 layers in H-Piper using Halide. Each layer's schedule could be defined seperately. Given a net definition, H-Piper can create the net and load the parameters using the pool of layers. Each layer could be defined it's own schedule seperately which allows users to explore optimization strategies. Also, new fused layer could be defined to reduce memory footprint. For instance, we could define a 'convpool' layer to fuse max/average pool layer after a convolutional layer. H-Piper is flexible and easy to use.

**Image * 2 Halide structure**

Hand-tuning in Halide scheduler is much easier than python or cpp. However, it's still painful and tedious when we need to account for both parallelism and locality. We've encountered an interesting paper [1] by Ravi Teja Mullapudi. It introduces a domain-specific language and compiler for image processing pipelines, PolyMage. PolyMage could generate approximate optimal schedule for Halide programs automatically. Below is the result where PolyMage is racing the Halide Experts from Google.

**Image google experts**

PolyMage is another motivaiton of us to implement a framework in Halide. Google Inception Net has 156 layers. Hand tuning inception net could be the last thing we want to do. But with the help of PolyMage, we could potentially get a schedule which is approximate optimal in few minutes. To integrated with [PolyMage](http://drona.csa.iisc.ernet.in/~uday/publications/uday15asplos.pdf) for automatic scheduler, every layer should provide size in dimensions. However, [Halide::Func](https://github.com/halide/Halide/blob/master/src/Func.h) does not provide such information. Therefore, we extent our implementation for this dimension-based information as well.



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
