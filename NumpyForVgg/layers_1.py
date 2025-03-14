# coding=utf-8
import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.d_bias = None
        self.d_weight = None
        self.output = None
        self.input = None
        self.bias = None
        self.weight = None
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.\n' % (self.num_input, self.num_output))

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):  # 前向传播计算
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        # 根据Y = W * X + B计算公式可得前向传播公式
        self.output = np.dot(input, self.weight) + self.bias
        # print(self.input.shape)
        # print(self.output.shape)
        print(f'计算量 = {(2 * self.input.shape[1] + 1) * self.output.shape[1]} FLOPS')
        return self.output

    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        # 根据公式Y = W * X + B，可知每一层的权重梯度为输入X，所以当前的权重梯度 = 上一层的梯度结果 * 输入input
        self.d_weight = np.dot(self.input.T, top_diff)
        # 根据公式Y = W * X + B，可知每一层的偏置梯度为1，所以当前的偏置梯度 = 上一层的梯度结果对每一列求和
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        # 根据公式Y = W * X + B，可知传递到下一层的梯度为权重W，所以下一层的梯度 = 上一层的梯度 * 权重W
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        # 根据梯度下降法可知权重更新公式为 W = W - lr * d(W)
        self.weight = self.weight - lr * self.d_weight
        # 根据梯度下降法可知偏置更新公式为 B = B - lr * d(B)
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):  # 参数保存
        return self.weight, self.bias


class ReLULayer(object):
    def __init__(self):
        self.input = None
        print('\tReLU layer.')

    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        # 根据ReLU的计算公式，Y = X (X > 0) : 0 (X < 0)
        # 可知 Y = max(0, X)
        output = np.maximum(0, self.input)
        print(f'计算量 = 0 FLOPS')
        return output

    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        # 根据ReLU的计算公式，Y = X (X > 0) : 0 (X < 0)
        # 可知当 X > 0时，梯度为1；当 X < 0时，梯度为0
        # 所以可知反向传播公式 = 上一层的梯度 * (输入 > 0)
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):
    def __init__(self):
        self.label_onehot = None
        self.batch_size = None
        self.prob = None
        print('\tSoftmax loss layer.')

    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        print(f'计算量 = {4 * self.prob.shape[1]} FLOPS')
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
