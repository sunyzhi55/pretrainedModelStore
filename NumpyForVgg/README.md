---
date: 2025-03-14T22:29:00
tags:
  - python
  - VGG
  - numpy 
---



# VGG Pretrained Model



文件夾`./NumpyForVgg`記錄了各種網上已有的預訓練的`VGG`模型



## 説明

本项目有三个`python`文件和一个`json`文件，实现了如下的目标：

1、用`numpy`实现`VGG`模型的搭建

2、加载`VGG`在`ImageNet`上的预训练模型，输入一张图片进行推理和分类

其中：

- `layers_1.py`文件存放的是`numpy`实现的全连接和激活函数以及损失函数
- `layers_2.py`文件存放的是`numpy`实现的卷积、池化和`Flatten`函数
- `vgg_cpu.py`是主函数，定义了`VGG`模型，并且加载模型、输入图片进行预测

直接运行`vgg_cpu.py`文件即可

> 1、其中`imagenet-vgg-verydeep-19.mat`預訓練模型在對應的==release==中的
>
> `VGG_Pretrained_Model_Use_Numpy`仓库中
>
> 2、預訓練的模型下載地址：
>
> [Pretrained CNNs - MatConvNet](https://www.vlfeat.org/matconvnet/pretrained/)









