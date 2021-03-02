---
layout: post
title:  "OHEM 论文阅读"
date:   2021-03-02
image:  ohem.jpg
tags:   Paper
---

论文 Training Region-based Object Detectors with Online Hard Example Mining

## 1 引言

目标检测算子通常是将目标检测问题简化(reduction)为图像分类问题来进行训练, 这种简化(reduction)会带来原始分类任务中没有的新问题: 训练集标注数据和背景数据之间存在极大的不平衡状况(负样本数量远远大于正样本数量, 即背景数据大于标注数据). 因此解决训练时正负样本的比例问题成为目标检测的一大研究方向. 为解决这一问题, 通常的做法是引入 bootstrapping 方法, 或者称为困难负样本挖掘(hard negative mining), 其主要思想是通过选择检测算子的虚警目标来生成背景样本. 该策略需要一个迭代的训练算法, 通过当前数据集更新检测模型, 然后通过更新后的模型去获取新的 false positive 样本并加入到训练集中, 再通过新的数据集更新检测模型. 该方法通常的使用方式是, 训练集包含所有的目标样本以及一部分随机的背景样本.

目前深度学习的目标检测算法由于各方面原因不使用 bootstrapping 方法. 本文针对这一问题提出了一种在线的困难样本挖掘方法(OHEM, Online Hard Example Mining), 相较于标准的 Fast RCNN 算法, 它有以下优势:

- 免除了 region-based ConvNets 中的一些超参数及 heuristics.
- 它会带来 mAP 的显著提升.
- 训练集越大, 越困难其效果越明显.

## 2 Fast RCNN 回顾

Fast R-CNN 的结构如 Figure 1 所示. Fast R-CNN 的输入为输入图像以及一系列 ROI, 其主体结构可以分为两个连续的模块: 一个卷积网络(由一系列卷积层和池化层组成)以及一个　ROI 网络(包括 ROI Pooling, FC 以及 Loss 层). 在 inference 过程中, 卷积网络的作用是接收输入图像然后生成 Feature Map. 对于每个 proposal, ROI Pooling 层的作用是将 proposal 投影至 Feature Map 上, 以及得到固定长度的特征向量. 这些特征向量将送至 FC 层中, 然后得到两个输出: (1)Softmax 概率值, 判定目标为前景(哪一类)或者背景; (2)坐标回归值.

![Figure 1]({{ site.baseurl }}/images/ohem1.jpg)

选择 Fast R-CNN 作为基础检测器除了其为端到端的网络外, 其原因如下: 
- 两个基本网络模块, 卷积网络和 ROI 网络, 当然新出现的 SPP-Net, MR-CNN 都是类似的结构, 但 Fast R-CNN 更具有普适性; 
- 虽然基本结构相似, 但 Fast R-CNN 可以对卷积网络进行训练, 不像 SPP-Net, MR-CNN 必须锁定卷积网络;
- SPP-Net, MR-CNN 都需要将 ROI 网络的特征进行缓存, 然后训练 SVM 分类器, 但 Fast R-CNN 可以通过 ROI 网络自己训练分类器.

### 2.1 训练

与大多数深度网络一样, Fast R-CNN 训练采用的是 SGD 方案. 每个 ROI 的损失是分类和边界框回归损失的和. 为了共享 ROI 之间的卷积计算, SGD mini-batch 一般是分级创建. 具体为: 对于每个 mini-batch, 首先在数据中采样 N 张图像, 每张图像选取 B/N 个 ROI, 实际中 N=2, B=128 效果较好. ROI 采样过程使用了一些 heuristics.

**Foreground RoIs**. 前景样本要求样本 ROI 与 GroundTruth 之间的 IOU 大于 0.5.

**Background RoIs**. 背景样本采样一般遵循样本 ROI 与 GroundTruth 之间的 IOU 位于 [bg_lo, 0.5) 之间. Fast R-CNN 中 bg_lo 一般定为 0.1, 这样定的缘由是考虑到与 GroundTruth 有一定的 IOU 的样本为困难样本的可能性更大, 这个 heuristic 可以有效地收敛网络并提升精度, 但是其容易忽视一些背景.

**Balancing fg-bg RoIs**, 平衡前景与背景. 一般前景与背景样本的比例为 1:3.

## 3 本文方法

### 3.1 在线困难样本挖掘

Online Hard Example Mining(OHEM), 在线困难样本挖掘的大体步骤如下:
- 输入一张图像, 通过卷积网络计算得到 Feature Map.
- 将 Feature Map 和所有 ROI 输入至 ROI 网络中进行前向传播; ROI 网络中的 Loss 代表当前网络对每个 ROI 的检测性能, 将这些 ROI 按照 Loss 进行排序, 取性能最差(Loss 最大)的 B/N 个样本进行反向传播. 因为前向传播时大多数卷积运算都是共享的, 因此计算所有 ROI 并不会带来较多的额外计算量, 反向传播时只选择少量 ROI 用于更新模型, 因此耗时不会有很大变化.
- 一般情况下, ROI 之间 overlap 越大, 它们的 Loss 值也越接近; 此外, 由于 Feature Map 的分辨率问题, overlap 较大的 ROI 投影到 Feature Map 上可能是同一块区域, 从而导致 Loss 重复计数(反向传播时也就重复了一次梯度计算); 为解决该问题, 本文采用标准 NMS 对 ROI 进行去重, 即先按 Loss 进行排序, 然后进行 NMS, 最后选择 B/N 个 ROI 进行反向传播; 本文实验采用的 NMS 阈值为 0.7

上述操作无需设置 fg-bg 比例来进行数据均衡. 一旦某个类别被忽略, 它的 Loss 就会增加, 直到它有一个较高的采样几率. 这样的话, 当图像中前景比较容易获取时, 完全可以将 mini-batch sample 全部设为 bg, 反之, 如果 bg 是一些没有价值的区域(如天空, 草地之类), 则完全可以将 mini-batch sample 全部设为 fg.

### 3.2 其他细节

OHEM 的实现方式有很多, 比较直观的一种方案是对 Loss 层进行修改. Loss 层的作用是计算所有 ROI 的 Loss, 对这些 ROI 按 Loss 进行排序并选取困难样本, 然后将非困难 ROI 的 Loss 都设为 0. 但是这种方案的实现效率很低, 因为即使多数 ROI 的 Loss 已经置为 0 且不参与梯度更新, 但是 ROI 网络仍然为它们分配内存空间并参与反向传播. 为了解决这个问题, 本文在实现 ROI 网络时采用了两个副本, 其中一个副本是只读的; 只读副本只为 ROI 前向传递分配内存, 而另一个副本则为前向和后向传播分配内存. 对于 SGD 迭代, 只读副本执行前向传播并计算所有 ROI 的 Loss, 然后选择困难样本, 这些困难样本被送入另一个副本网络中进行前向和后向传播. 具体实现参见 Figure 2.

![Figure 2]({{ site.baseurl }}/images/ohem2.jpg)

## 4 实验及分析

### 4.1 实验设定

本文实验设置如下:
- 网络结构: 选择了 2 中基本网络结构, 分别为 VGG_CNN_M_1024 和 VGG16.
- 数据集: PASCAL VOC07, 采用 trainval 集进行训练, test 集进行测试.
- 除非特别说明, 其他参数设置与 Fast R-CNN 相同.

训练方式:
- SGD, 80k iter;
- 学习率: 初始学习率为 0.001, 30k 次迭代后学习率降至 0.0001
- 其他训练参数设置见 Table 1(第1~2行)

![Table 1]({{ site.baseurl }}/images/ohem3.jpg)

### 4.2 OHEM 和 heuristic sampling 对比

分组：
- Fast R-CNN + OHEM
- Fast R-CNN + heuristic(bg_lo=0.1)
- Fast R-CNN + heuristic(bg_lo=0.0)

结果: Fast R-CNN + OHEM 相较于 Fast R-CNN + heuristic(bg_lo=0.1) mAP 提升 2.4 个百分点, 相较于 Fast R-CNN + heuristic(bg_lo=0.0) mAP 提升 4.8 个百分点. 具体结果参见 Table 1.

### 4.3 鲁棒性

N = 2(图像 batch 数过低) 可能会造成梯度不稳定以及收敛缓慢, 其原因是 ROI 之间的相关性较高. Fast R-CNN 论文中认为该问题不重要,  但实际训练中会存在一定问题. 本文采用 N = 1 进行实验, 发现 mAP 仅下降 1 个点, 说明 OHEM 鲁棒性较好. 具体结果参见 Table 1.

### 4.4 为什么要使用困难样本

如果训练时采用所有 ROI 而不是采用困难样本结果会如何呢? 实际上, 采用所有 ROI 训练, 简单样本最终的 Loss 会比较低, 对梯度更新的贡献很小, 训练会自动集中在困难样本上. 为了比较采用所有 ROI 和采用困难样本的区别, 本文做了一个对比实验, 设 B = 2048, bg_lo = 0, N = {1,2}, 其他不变(一个比较大的 mini-batch), 结果如 Table 1 所示, 采用所有 ROI 比 heuristic 方法稍好, 但逊于 OHEM.

### 4.5 更好的优化

ROI Loss 变化如 Figure 3 所示.

![Figure 3]({{ site.baseurl }}/images/ohem4.jpg)

### 4.6 计算量

计算量对比如 Table 2 所示.

![Table 2]({{ site.baseurl }}/images/ohem5.jpg)

## 5 PASCAL VOC 和 MS COCO 测试结果

VOC2007 及 VOC2012 测试结果如 Table 3 及 Table 4 所示.

![Table 3]({{ site.baseurl }}/images/ohem6.jpg)

![Table 4]({{ site.baseurl }}/images/ohem7.jpg)

MSCOCO 测试结果如 Table 5 所示.

![Table 5]({{ site.baseurl }}/images/ohem9.jpg)

加入多尺度策略后的测试结果, 如 Table 5,  Table 6 所示.

![Table 6]({{ site.baseurl }}/images/ohem10.jpg)