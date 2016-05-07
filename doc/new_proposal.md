# H-Piper: A Deep Learning Framework in Halide
\- Lei Sun, Yang Wu

### SUMMARY
**H-Piper** is our final project for 15418.

We are going to build a powerful, flexible, generic,and highly-balanced framework for deep learning Nets, in [Halide](http://halide-lang.org/). We named our framework **H-Piper**. We will embed popular neural networks like VGG, CNN, and Google Inception Net in our framework and we enable user to define own Neural Networks by privoding a simple configuration files. We will extend our input processor to support popular inputs like protobuf, json, and etc.

If time permits, we'd like to hand-tune our framework schedule (Halide's feature) and compete with Caffe and MXNet. Also, we could leverage the existing Halide Auto-Scheduler developed by Ravi to generate an automatic schedule to compete with Caffe and MXNet. 

Our framework will be extremely useful for Halide Auto-scheduler to test their performance on different Neural Net and it will be a great performance reference for users who are interested in getting higher performance.

### BACKGROUND
The traditional CNN limited by the lack of ability to be spatially invariant to the input data
in a computationally and parameter efficient manner. There are some extra work that needs to be done in steps like *MAX-POLL* layer to offset the affects from rotation, scale or shift. But that needs more deeper layers. What we want to achieve from Spatial transformer networks is a 'corrected' input after localization and transformations. 

This transform also requires a *localisation net* to train transformation parameters, which is by itself a CNN network. Therefore, this procedure involves matrix computation, convolution integral and backpropagation to turn parameters. All these steps can use parallel to speed up. 

NOTE: *This is our current ideas on some potential steps that can be benefit from parallel, might adjust later.*

### THE CHALLENGE
First of all, we need to learn theories and algorithm behind neural networks, which is new to us. CNN itself is complicated and the paper described the idea is pretty new.

Second, we need to adopt different opensource implememtation and try to find parallelism, locatlity, or less duplicated work in those implementation to see if we could implement the algorithm in the most optimized way.

Last, we will implement this in [Halide](http://halide-lang.org/). Halide's a new programming language came out in 2011/2012, there could be some un-implemented and important features for our project.

**WORKLOAD**

And we think the cache footprint would be pretty huge in convolution, although the convolution step does not have strong dependency, locality in convolution won't affect too much. We think there should be some optimization in terms of how to explore the footprint in memory. 

Also the backpropagation could be requires intensive computation and some intermeida differential result for chain rule.

**CONSTRAINTS**
<!-- Describe constraints: What are the properties of the system that make mapping the workload to it challenging? -->
The forward and backpropagation might leave more memory footprint than a machine in cluster can have. We need to find a decomposition method to reduce the affect of that.

Also, if we add padding in matrix to avoid the data shrink too quickly, there can be diversity in work load of different threads or even machines. We have to find a optimized work schedule logic to balance the load.


### RESOURCES
We are going to use GHC machines, and start form scratch in Halide. The idea comes from Google's paper: [Spatial transformer networks](http://arxiv.org/pdf/1506.02025v3.pdf). There are some implementation in [Caffe](https://github.com/XiaoxiaoGuo/caffe-stn) and [Python](https://github.com/skaae/recurrent-spatial-transformer-code), we will try to compete their performance. 

[Here](http://gitxiv.com/posts/5WTXTLuEA4Hd8W84G/spatial-transformer-networks) has some helpful links.

More reference will be added if we find some useful reference paper/implementations. 

### GOALS AND DELIVERABLES
<!-- Describe the deliverables or goals of your project. -->
**PLAN TO ACHIEVE**

We plan to implemented a *Spatial transformer networks* in Halide that improve the performance of CNN network on some well-known tasks. 

In the demo, we want to run a comparison between plain CNN and our DPSTNet on LeNet-5 dataset *(maybe)* and see if there is an improvement in speed without lose accuracy. And also, we want to show how we design and decouple the problem that fully-explored the parallel potential of the transformer problem.

**HOPE TO ACHIEVE**

If the project goes well, we could also improve CNN itself in Halide and see if there is any  components that we can make parallel to speedup.

### PLATFORM CHOICE
We choose to use GHC machines as our platform and Halide as our language.

For platform, we want our implementation to be general enough that can be plugable to everyone else's CNN application instead of stick to some hardware requirements. Also, in order to have a consistent development environment within team, we choose to jsut use GHC cluster.

As for Halide language. First of all, it is a DSL for image processing, which gives it advantages in this project as we are using image data set; also the tuning producure can be more easier and efficient in Halide than other languages like C++: by only defineing schedule, we can try more configures in short time. And as Professor Kayvon shows, it is usually faster to find optimal settings in Halide.


### SCHEDULE

| Timeline  | Goal to achieve | 
|:----------:|:--------------| 
| April 8th  | Understand of mechanism behind the transformer and implemented a serial version of Spatial transformer networks in Halide | 
| April 15th | Connect Halide implementation to following graphic piple line on LeNet-5 dataset| 
| April 22nd | Analise the dependecy of transformer and the relation of following pipelines and build a prototype of prarllel version in Halide |
| April 29th | A working parallel version of transformer and tune different schedule to improve performance |
| May 7th    | Wrap up implementatin and compare/analyse the performance in report |

NOTE: *The work in week of April 8th and 15th might to working together, since the learning curve for CNN and backpropagation is pretty steep. But should have a working implementation of pipeline in Halide by April 15th.*
