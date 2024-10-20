# SqueezeNet

## brief

SqueezeNet 在 ImageNet 上实现与 AlexNet 同等级别的精度，但参数少了 50 倍

系统架构如下：

![SqueezeNet](./firgures/1.png)

本模型训练集使用 FashionMNIST ，对网络结构做了部分调整，详见代码

训练 100 轮的情况下，准确率最高可达 90%
