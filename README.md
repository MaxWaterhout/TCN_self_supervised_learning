# Reproducing *Time-Contrastive Networks: Self-Supervised Learning from Video*
**Authors:** Max Waterhout (5384907), Amos Yususf (4361504), Tingyu Zhang ()
***

## 1. Introduction
In this blog post, we present the results of our attempted replication and study of the 2018 paper by Pierre Sermanet et al. 
*Time-Contrastive Networks: Self-Supervised Learning from Video* [[1]](#1). This work is part of the CS4245 Seminar Computer Vision
by Deep Learning course 2021/2022 at TU Delft. 

***
<!--
This is the syntax for a figure we still need an images folder, but this is just an outline
<p align="center">
<img src="images/figure_1_paper.png" width="750" height="261" alt="Figure 1 paper">
</p>
Figure 1 -->

***
In the computer vision domain, deep neural networks have been succesfull on a big range of tasks where labels can be easily be specified by humans, like object detection/segmentation. A bigger challenge lies in applications that are difficult to label, like in the robotics domain. An example would be labeling a pouring task. How can a robot understand what important properties are while neglecting setting changes. Ideally, a robot in the real world can learn a pouring task purely from observation and understanding how to imitate this behaviour directly. In this reproduction we train a network on a pouring task that tries to learn the important features like pose and the amount of liquid in the cup while being viewpoint and setting invariant. This pouring task is learned through the use of supervised learning and representation learning. In the following we will provide an motivation for this paper, our implementation of the model using PyTorch, the results that we achieved against the benchmarks and lastly we discuss the limitations from our implementation. 

<p align="center">
<img src="images/pouring_002.gif" width="500" height="300"/> </br>
<em>fig 1. An example sequence of a pouring task</em>
</p>


## 2. Motivation
Imitation learning has already been used for learning robotic skills from demonstrations and can be split in two areas: behavioral cloning and inverse reinforcement learning. The main disadvantage of these methods is the need of a demonstration in the same context as the learner. This does not scale well with different contexts, like a changing viewpoint or an agent with a different model. In this paper the authors train a Time-Constrastive Network (TCN) on demonstrations that are diverse in embodiments, objects and backgrounds. This allows the TCN to learn the best pouring representation without labels. With this TCN network a robot can learn to link the images to the corresponding motor commands using reinforcement learning or an other method. In our blog we do not cover the reinforcement part.


## 2. Implementation

For our implementation of the TCN we only use the data of the single-view data. 

<p align="center">
<img src="images/single view TCN.png" width="360" height="261" alt="single view TCN"> </br>
<em>Fig. 1: The single-view TCN</em>
</p>

### 2.1 Framework
The framework of a TCN contains of a deep network that outputs an 32-dimensional embedding vector, see fig [1]. As an input, a sequence of preprocessed 360x640 frames of a video are putted in. In total 11 sequences of around 5 seconds (40 frames) are used for training. For every frame in a sequence, The TCN takes an anchor, positive and negative frame where it encourages the anchor and positive to be close in embedding space while distancing itself from the negative frame. This way the network learns what is common between the anchor and positive frame and different from the negative frame. In our case the negative margin range is 0.2 second (one frame). \
The actual loss is calculated with a triplet loss [[2]](#2). The formula and an illustration can be seen in fig [2]. 

<p align="center">
<img src="images/triplet loss formula.png" width="700" height="105" > </br>
</p>

<p align="center">
<img src="images/triplet_loss.png" width="600" height="161" alt="Training loss"> </br>
<em>Fig. 2: The triplet loss</em>
</p>

The main purpose of the triplet loss is to learn representations without labels and simultaneously learn meaningful features like pose for example. \ 
The deep network that is used for feature extraction is derived from the Inception architecture.  

### 2.2 Hyperparameter selection 


## 3. Results

### 3.1 Final result overview
Because there is no validation set to select the best training model, we only save models for every 200 epochs and the models reaching the new minimum loss. We trained the model for 13k iterations and the training loss is shown in Figure 1.  The zigzaging behaviour is due to the 200 epoch gap as well as the missing data betweening 2000 to 6000 epochs after one virtual machine crash.   

<p align="center">
<img src="images/tain loss.png" width="360" height="261" alt="Training loss"> </br>
<em>Fig. 3: The training loss</em>
</p>



<p align="center">
<img src="./images/accuracy.png" width="360" height="261" alt="Figure 1 paper"> </br>
<em>Fig. 4: The testing accuracy</em>
</p>

The best accuracy measured by the video alignemt using l2 distance with one frame tolerence is at the 7200 iteration. The average alignment accuracy for testing set is 80.11 percent whereas the Baseline method has an average accuracy of 71.04 percent. 

<p align="center">
<img src="./images/everything.gif" width="360" height="261"> </br>
<em>Fig. 5: Overview</em>
</p>



https://user-images.githubusercontent.com/99979529/171060214-c9998001-4c61-43a1-82c8-dca6ab182bcd.mp4













### 3.2 Reproduced figure/ table


## 4. Discussion and Limitations

### 4.1 Discussion
### 4.2 Limitations

## References
<a id="1">[1]</a> Sermanet, P., Corey, L., Chebotar Y., Hsu J., Jang E., Schaal S., Levine S., Google Brain (2018). Time-Contrastive Networks: Self-Supervised Learning from Video. <i>University of South California</i>. [https://arxiv.org/abs/1704.06888]()











