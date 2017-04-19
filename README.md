# About

darkent19 by chainer [developing]
(input size 224x224)

- Classification Experiment over Cifar10

<!-- # Paper

[160224 SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

# Model

Macroarchitectural of squeezeNet

# How to run

git clone git@github.com:amazarashi/squeeze-chainer.git

cd ./squeeze-chainer

python main.py -g 1

# Inspection

### dataset
Cifar10 [(link)](https://www.cs.toronto.edu/~kriz/cifar.html)

### Result

(1) optimizer: Adam

![accuracy-adam](https://github.com/amazarashi/squeeze-chainer/blob/develop/result/adam/accuracy.png "accuracy")

![loss-adam](https://github.com/amazarashi/squeeze-chainer/blob/develop/result/adam/loss.png "loss")

(2) optimizer: MomentumSGD
  - weight decay : 1.0e-4
  - momentum : 0.9
  - schedule[default:0.1,150:0.01,225:0.001]


![accuracy-adam](https://github.com/amazarashi/squeeze-chainer/blob/develop/result/momsgd/accuracy.png "accuracy")

![loss-adam](https://github.com/amazarashi/squeeze-chainer/blob/develop/result/momsgd/loss.png "loss") -->
