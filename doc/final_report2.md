# H-Piper: A Deep Learning Framework in Halide
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

PolyMage is another motivaiton of us to implement a framework in Halide. Google Inception Net has 156 layers. Hand tuning inception net could be the last thing we want to do. But with the help of PolyMage, we could potentially get a schedule which is approximate optimal in few minutes. To integrated with PolyMage for automatic scheduler, every layer should provide size in dimensions. However, Halide::Func does not provide such information. Therefore, we extent our implementation for this dimension-based information as well. 






## APPROACH

Tell us how your implementation works. Your description should be sufficiently detailed to provide the course staff a basic understanding of your approach. Again, it might be very useful to include a figure here illustrating components of the system and/or their mapping to parallel hardware.

## RESULTS

How successful were you at achieving your goals? We expect results sections to differ from project to project, but we expect your evaluation to be very thorough (your project evaluation is a great way to demonstrate you understood topics from this course).

## REFERENCES

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. Going Deeper with Convolutions, CVPR 2015. pdf




