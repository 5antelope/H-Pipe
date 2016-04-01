# DPSTNet
\- Lei Sun, Yang Wu

### SUMMARY
We are going to find the optimal implementations of *Spatial transformer networks* on heterogeneous platforms with both [Halide](http://halide-lang.org/) and our manually-tuned implementation. Embedded our transformer to a CNN graphic pipeline, and compare/analyse performance on the tasks like LeNet-5 or maybe other open datasets.

### BACKGROUND
The traditional CNN limited by the lack of ability to be spatially invariant to the input data
in a computationally and parameter efficient manner. There are some extra work that needs to be done in steps like *MAX-POLL* layer to offset the affects from rotation, scale or replacement. But that needs more deeper layers. What we want to achieve from Spatial transformer networks is a 'corrected' input after localization and transformations. 

This transform also requires a *localisation net* to train transformation parameters, which is by itself a CNN network. Therefore, this procedure involves matrix computation, convolution integral and backpropagation to turn parameters. All these steps can use parallel to speed up. 

NOTE: *This is our current ideas on some potential steps that can be benefit from parallel, might adjust later.*

### THE CHALLENGE
First of all, we need to learn theories and algorithm behind neural networks, which is new to us. CNN itself is complicated and the paper described the idea is pretty new.

Second, we need to adopt different opensource implememtation and try to find parallelism, locatlity, or less duplicated work in those implementation to see if we could implement the algorithm in the most optimized way.

Last, we will implement this in [Halide](http://halide-lang.org/). Halide's 

### WORKLOAD
And we think the cache footprint would be pretty huge in convolution, although the convolution step does not have strong dependency, locality in convolution won't affect too much. We think there should be some optimization in terms of how to explore the footprint in memory. 

Also the backpropagation could be requires intensive computation and some intermeida differential result for chain rule.

#### CONSTRAINTS
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?

### RESOURCES
We are going to use GHC machines, and start form scratch in Halide. The idea comes from the paper: [Spatial transformer networks](http://arxiv.org/pdf/1506.02025v3.pdf) from Google. There are some implementation in [Caffe](https://github.com/XiaoxiaoGuo/caffe-stn) and [Python](https://github.com/skaae/recurrent-spatial-transformer-code), we will try to compete their performance. 

We will add more reference if we find some useful reference paper/implementations. 

### GOALS AND DELIVERABLES
<!-- Describe the deliverables or goals of your project. -->


This is by far the most important section of the proposal:

Separate your goals into what you PLAN TO ACHIEVE (what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule. It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)
If applicable, describe the demo you plan to show at the parallelism computation (will it be an interactive demo? will you show an output of the program that is really neat? will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.
PLATFORM CHOICE. Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?

### SCHEDULE

| Time Line  | Goal          | 
|:----------:|:--------------| 
| April 8th  | Understand of mechanism behind the transformer and implemented a serial version of Spatial transformer networks in Halide | 
| April 15th | Connect Halide implementation to following graphic piple line on LeNet-5 dataset| 
| April 22nd | Analise the dependecy of transformer and the relation of following pipelines and build a prototype of prarllel version in Halide |
| April 29th | A working parallel version of transformer and tune different schedule to improve performance |
| May 7th    | Wrap up implementatin and compare/analyse the performance in report |

NOTE: *The work in week of April 8th and 15th might to working together, since the learning curve for CNN and backpropagation is pretty steep. But should have a working implementation of pipeline in Halide by April 15th.*
