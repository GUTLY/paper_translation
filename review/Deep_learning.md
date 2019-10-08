## 深度学习

### 原文：[Deep learning](https://creativecoding.soe.ucsc.edu/courses/cs523/slides/week3/DeepLearning_LeCun.pdf)
### 作者：Yann LeCun, Yoshua Bengio, Geoffrey Hinton

### 目录：

[TOC]

#### 摘要  Abstract

深度学习（deep learning）允许由多个处理层组成的计算模型学习具有多个抽象级别的数据表征（representation）。这些方法显著的提高了语音识别（speech recognition）、视觉对象识别（visual object recognition）、对象检测（object detection）和许多其他的领域（如药物发现和基因组学）的最佳表现。深度学习通过使用反向传播算法（backpropagation algorithm）去指示机器如何改变内部参数（internal parameters）来发现大数据集的复杂结构，这些内部参数被用于从上一层的表征中计算每一层的表征。深度卷积网络（deep convolutional nets）带来了图像（images）、视频（video）、语音（speech）和音频（audio）处理的重大突破，而递归网络（recurrent nets）则在文本（text）和语音（speech）等序列数据（sequential data）方面取得了突破。

#### 1 概述

机器学习技术驱动了现代社会的许多方面：从网络搜索到社交网络（social netwroks）的内容过滤，再到电子商务网站（e-commerce websites)的推荐。它越来越多的出现在相机和智能手机这样的消费产品中。机器学习系统被用于识别图像中的物体，将语音转换维文本，将新闻条目、帖子或产品与用户的兴趣进行匹配，并选择搜索的相关结果。这些应用越来越多的使用一种名为深度学习的技术。

传统的机器学习技术（conventional machine-learning techniques）处理原始数据的能力有限。数十年以来，构造一个模式识别（ pattern-recognition）或者机器学习系统需要详细的工程设计和相当多的领域专业知识来设计一个特征提取器（feature extractor），将原始数据（如图像的像素值）转换成合适的内部表征（internal representation）或特征向量（feature vector），使得学习子系统（通常是一个分类器）可以从中检测（detect）或分类（classify）输入中的模式（input）。

