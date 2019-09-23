## 单词在向量空间的有效估计

### 原文：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
### 作者：Tomas Mikolow, Kai Chen, Greg Corrado, Jeffrey Dean
### Google

### 目录：

[TOC]

#### 摘要

我们提出两个新的模型架构为非常大的数据集计算单词的连续向量量表征（continuous vector representations of words）。这些表征（representations）的优劣使用单词相似度(similarity)测试来衡量，研究结果与之前基于不同类型神经网络的最佳表现技术进行了比较。我们观察到在精度上的巨大的提高和更低的计算开销等。它可以在一天以内从有16亿单词的数据集中学得高质量的词向量（word vectors）。此外，我们展示了这些向量在使用测试数据集衡量句法和语义词之间的相似度时的最好的性能表现。

#### 1 简介

当前许多NLP系统和技术都把单词当作原子单位（atomic units)来处理-在单词之间没有相似度的概念，因为它们在词汇表中表示为索引(indices)。这个选择有一些很好的理由，简单、健壮，同时简单模型训练大规模数据比复杂数据训练少量数据的标现更好。其中一个例子就是颇受欢迎且如今被当作统计语言模型的N-gram模型，它使训练几乎所有可用的数据（数万亿单词）成为可能。

然而，在很多任务上简单模型有自身的局限。例如，用于语音自动识别的相关数据是有限的，模型的表现通常由高质量的手抄演讲数据（通常只有数百万单词）的大小决定。在机器学习上，现存的许多语言的语料库只包含十亿或更少的单词。因此，在这种简单扩展基础技术的情形下将不会有任何的重大进步，于是我们不得不关注于更加高级的技术。

随着近年来机器学习技术的发展，使得在非常大的数据集上训练更复杂模型成为可能，且它们通常比简单模型表现得更好。大概其中最成功的概念便是单词的分布式表征（distributed representations of words）。例如，基于语言模型的神经网络的表现的远比N-gram模型更好。

##### 1.1 本文目标

本文的主要目标是介绍一个可以被用来从数十亿的单词和数百万词汇表中学习高质量词向量的技术。据我们所知，以前尚无人提出一个成功训练超过几亿个单词，且可以训练出在50-100维的合适维度的词向量的架构。

我们使用最近提出的技术来测量所得矢量表征的质量，我们不仅希望相似的单词可以靠的更近，同时我们也希望单词有多重角度的相似性（multiple degrees of similarity）。这在较早的屈折语言（inﬂectional languages）环境中已被观察到。例如，名词可以有多个单词结尾，如果我们在原始矢量空间的子空间中搜索相似的单词，则可能找到具有相似结尾的单词。

令人惊奇的是，研究发现，词汇表征（word representations）的相似性（similarity of word representations）超越了简单的句法规则（syntactic regularities）。使用单词偏移技术（word offset technique），对词向量执行简单的算术操作。例如，向量（“King”）减去向量（“男人”）再加上向量（“女人”）的结果的向量非常接近于单词女王（Queen）的向量表征。
> 原文：vector(“King”) - vector(“Man”) + vector(“Woman”) results in a vector that is closest to the vector representation of the word Queen

本文，我们通过开发新的、可以保存单词线性规律的模型架构来尝试最大化这些向量操作的精度。我们设计了一个新的综合测试集来衡量句法和语义规律，实验结果表明，该方法具有较高的学习精度。此外，我们讨论了训练时间和训练精度如何依赖于词向量维度和测试数据的数量（训练时间和训练精度随两者的变化）。

##### 1.2 前期工作

使用连续向量来表征单词，已经有较长的历史。**[1]** 提出了一种非常流行的用于估计神经网络语言模型（neural network language model ，NNLM）的模型架构，它采用线性投影层和非线性隐藏层的前馈神经网络来共同学习词向量表征和统计语言模型。这项工作已经被许多人关注。

另一个有趣的NNLM架构由 **[13, 14]** 提出。它使用单个隐藏层的神经网络来第一次学习词向量，接着这个词向量被用来训练这个NNLM。因此，这个词向量可以从没有构造完整的NNLM中学习。再这项工作中，我们直接扩展这个架构，且仅仅关注于第一步。在这一步，词向量是从一个简单模型来学习得到的。

在稍后的研究中表明，这个词向量可以显著的提高和简化许多NLP应用 **[4, 5, 29]**。使用不同的模型架构和各种各样的语料库来对词向量进行估计，一些由此产生的词向量可用于将来的研究和比较。然而，据我们所知，这个架构训练的计算开销要比 **[13]** 所需的开销大的多，除了使用对数双线性模型某些版本，它使用了对角线权重矩阵 **[23]**。

#### 2 模型架构 

许多不同的模型被提出，用以估算连续的单词表征，包括著名的潜在语义分析（Latent Semantic Analysis，LSA) 和隐含狄利克雷分布模型（Latent Dirichlet Allocation，LDA）。本文，我们关注于从神经网络中学得的单词的分布式表征，正如先前的研究表明，在保存单词之间的线性规律上它的表现相对于LSA有极大的提高 **[20, 31]**；此外在大规模数据集上，LDA带来了相当大的计算开销。

与 **[18]** 相同，为了比较不同的模型架构，我们首先将模型的计算复杂度定义为完全训练模型所需的参数数量。接着，我们尝试最大化精确度，同时最小化计算复杂度。

对与接下来的所有模型，训练复杂度（training complexity）由以下公式计算：

![training complexity](../images/Efﬁcient_Estimation_of_Word_Representations_in_Vector_Space/no1.png)

这里的E是训练的次数（epochs），T是训练集的单词数，Q是模型架构的进一步定义。一般E的选择区间为[3, 50]，T超过十亿。所有的模型使用随机梯度下降法（stochastic gradient descent，SGD）和反向传播（backpropagation）来训练 **[26]**

**[1]**  Y. Bengio, R. Ducharme, P. Vincent. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137-1155, 2003.

**[4]** R. Collobert and J. Weston. A Uniﬁed Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. In International Conference on Machine Learning, ICML, 2008. 

**[5]** R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch. Journal of Machine Learning Research, 12:24932537, 2011

**[13]** T. Mikolov. Language Modeling for Speech Recognition in Czech, Masters thesis, Brno University of Technology, 2007. 

**[14]** T. Mikolov, J. Kopeck´y, L. Burget, O. Glembek and J. ˇCernock´y. Neural network based language models for higly inﬂective languages, In: Proc. ICASSP 2009.


**[18]** T. Mikolov, A. Deoras, D. Povey, L. Burget, J. ˇCernock´y. Strategies for Training Large Scale Neural Network Language Models, In: Proc. Automatic Speech Recognition and Understanding, 2011. 

**[20]** T. Mikolov, W.T. Yih, G. Zweig. Linguistic Regularities in Continuous Space Word Representations. NAACL HLT 2013.

**[23]** A.Mnih,G.Hinton. A Scalable Hierarchical Distributed Language Model.Advances in Neural Information Processing Systems 21, MIT Press, 2009. 

**[26]** D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning internal representations by backpropagating errors. Nature, 323:533.536, 1986. 

**[29]** J. Turian, L. Ratinov, Y. Bengio. Word Representations: A Simple and General Method for Semi-Supervised Learning. In: Proc. Association for Computational Linguistics, 2010.

**[31]** A. Zhila, W.T. Yih, C. Meek, G. Zweig, T. Mikolov. Combining Heterogeneous Models for Measuring Relational Similarity. NAACL HLT 2013. 