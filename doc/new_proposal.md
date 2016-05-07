# H-Piper: A Deep Learning Framework in Halide
\- Lei Sun, Yang Wu

### SUMMARY
**H-Piper** is our final project for 15418.

We are going to build a powerful, flexible, generic, and highly-balanced framework for deep learning Nets, in [Halide](http://halide-lang.org/). We named our framework **H-Piper**. We will embed popular neural networks like VGG, CNN, and Google Inception Net in our framework and we enable user to define own Neural Networks by privoding a simple configuration files. We will extend our input processor to support popular inputs like protobuf, json, and etc.

If time permits, we'd like to hand-tune our framework schedule (Halide's feature) and compete with Caffe and MXNet. Also, we could leverage the existing Halide Auto-Scheduler developed by Ravi to generate an automatic schedule to compete with Caffe and MXNet. 

Our framework will be extremely useful for Halide Auto-scheduler to test their performance on different Neural Net and it will be a great performance reference for users who are interested in getting higher performance.

### BACKGROUND

MXNet and Caffe are the most popular frameworks for deep learning. Both frameworks are implemented in cpp and are carefully hand-tuned. Hand-tuning in cpp could be painful. Changes' correctness needs to be verified everytime and the amount of changes for the most simple reschedule is a lot. 

Halide is a new programming language designed to make it easier to write high-performance image processing code on modern machines. It provides a concept of "scheduling" which allows developers to easily define he or she wants to iterate through the dataset. The amount to be changed is quite trivial. The correctness will be not affected heavily if only the schedule is changed. 

It's always interesting to explore the best tradeoff between parallelism and locality. With Halide, developers could explore the best tradeoff much faster with less frastration. 

[PolyMage](http://drona.csa.iisc.ernet.in/~uday/publications/uday15asplos.pdf) focus on automatically generating high-performance implementations of image processing pipelines expressed in a high-level declarative language. "Experimental results on a modern multicore system show that the performance achieved by our automatic approach is up to 1.81× better than that achieved through manual tuning in Halide, a state-of-the-art language and compiler for image processing pipelines."

### THE CHALLENGE
1. We need to figure out the scope of our framework. Surely we want to be general enough and support everything but given the fact that we only have couple of weeks to build this. It might not be as general as Caffe or MXNet.
2. We need to explore the locality possibilities between two layers in any neural network. To check if we could fuse two layres together and the compare the performance.
3. Consumer different type of inputs and provide APIs to make it easy for use to migrate to our Framework.
4. Primitives of our framework needs to be carefully designed to allow user define a pipeline or customized neural net easily.
5. There might be some difficulties to get auto-schedule from PolyMage since we don't have it in control. At the end, we might have to bear with our hand-tune performance. 

**WORKLOAD**

And we think the cache footprint would be pretty huge in convolution, although the convolution step does not have strong dependency, locality in convolution won't affect too much. We think there should be some optimization in terms of how to explore the footprint in memory and Halide makes it easy to define the way we want to handle the workload.

Also the backpropagation could be requires intensive computation and some intermeida differential result for chain rule.

**CONSTRAINTS**
<!-- Describe constraints: What are the properties of the system that make mapping the workload to it challenging? -->
The forward and backpropagation might leave more memory footprint than a machine in cluster can have. We need to find a decomposition method to reduce the affect of that.

Also, if we add padding in matrix to avoid the data shrink too quickly, there can be diversity in work load of different threads or even machines. We have to find a optimized work schedule logic to balance the load.


### RESOURCES
We are going to use GHC machines, and start form scratch in Halide. We will implement[Google Inception Net](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) first. There are some implementation in [Caffe](https://github.com/XiaoxiaoGuo/caffe-stn) and [Python](https://github.com/skaae/recurrent-spatial-transformer-code), we will try to compete their performance. 

[This Git](https://github.com/google/inception) has some helpful informations about how to implement this net.

More reference will be added if we find some useful reference paper/implementations. 

### GOALS AND DELIVERABLES
<!-- Describe the deliverables or goals of your project. -->
**PLAN TO ACHIEVE**
1. A Working Google Inception Net in Halide
2. A working deep learning framework in Halide which supports CNN, Inception Net, and VGG.


**HOPE TO ACHIEVE**

1. Automatic Schedule generated by PolyMage and the speedup observed from hand-tuned version.
2. Performance compete with Caffe, MXNet on Inception Net, VGG, CNN, and etc.

### PLATFORM CHOICE
We choose to use lateday clusters as our platform and Halide as our language.

As for Halide language. First of all, it is a DSL for image processing, which gives it advantages in this project as we are using image data set; also the tuning producure can be more easier and efficient in Halide than other languages like C++: by only defineing schedule, we can try more configures in short time. And as Professor Kayvon shows, it is usually faster to find optimal settings in Halide.

### SCHEDULE

| Timeline  | Goal to achieve | 
|:----------:|:--------------| 
| April 8th  | Understand Google Inception Net's Layers and how does they work| 
| April 15th | Scratch the framework and define primitives| 
| April 22nd | Implement the framework in Halide and test on CIFAR-10 dataset |
| April 29th | Tune different schedule to improve performance, contact PolyMage for automatic shedule, and compete with Caffe and MXNet |
| May 7th    | Wrap up implementatin and compare/analyse the performance in report |
