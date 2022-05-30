# Reproducing *Time-Contrastive Networks: Self-Supervised Learning from Video*
**Authors:** Max Waterhout (), Amos Yususf (4361504), Tingyu Zhang ()
***

## 1. Introduction
In this blog post, we present the results of our attempted replication and study of the 2018 paper by Pierre Sermanet et al. 
*Time-Contrastive Networks: Self-Supervised Learning from Video* [[1]](#1). This work is part of the CS4245 Seminar Computer Vision
by Deep Learning course 2021/2022 at TU Delft. 
.....



***
<!--
This is the syntax for a figure we still need an images folder, but this is just an outline
<p align="center">
<img src="images/figure_1_paper.png" width="750" height="261" alt="Figure 1 paper">
</p>
Figure 1 -->

***

In the following we will provide our results, discussion and limitations of our reproduction. In section 2 we briefly introduce our
implementation of the model using the Pytorch framework. In section 3 present our results and in section 4 we discuss these results
and the limitations encountered during this project.


## 2. Implementation

### 2.1 Framework


### 2.2 Hyperparameter selection 


## 3. Results

### 3.1 Final result overview
Because there is no validation set to select the best training model, we only save models for every 200 epochs and the models reaching the new minimum loss. We trained the model for 13k iterations and the training loss is shown in Figure 1.  The zigzaging behaviour is due to the 200 epoch gap as well as the missing data betweening 2000 to 6000 epochs after one virtual machine crash.   

<p align="center">
<img src="images/tain loss.png" width="360" height="261" alt="Training loss">
</p>



<p align="center">
<img src="./images/accuracy.png" width="360" height="261" alt="Figure 1 paper">
</p>

The best accuracy measured by the video alignemt using l2 distance with one frame tolerence is at the 7200 iteration. The average alignment accuracy for testing set is 80.11 percent whereas the Baseline method has an average accuracy of 71.04 percent. 



https://user-images.githubusercontent.com/99979529/170977368-6cdf319a-a28d-48c2-abbc-d9de61e404f8.mp4








### 3.2 Reproduced figure/ table


## 4. Discussion and Limitations

### 4.1 Discussion
### 4.2 Limitations

## References
<a id="1">[1]</a> Sermanet, P., Corey, L., Chebotar Y., Hsu J., Jang E., Schaal S., Levine S., Google Brain (2018). Time-Contrastive Networks: Self-Supervised Learning from Video. <i>University of South California</i>. [https://arxiv.org/abs/1704.06888]()











