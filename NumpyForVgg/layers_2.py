# coding=utf-8
import numpy as np
import time


def show_matrix(mat, name):
    print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    # pass


def show_time(time, name):
    print(name + str(time))
    # pass


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (
            self.kernel_size, self.channel_in, self.channel_out))

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std,
                                       size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):  # 前向传播的计算

        start_time = time.time()
        self.input = input  # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding + self.input.shape[2],
        self.padding:self.padding + self.input.shape[3]] = self.input
        self.height_out = (height - self.kernel_size) // self.stride + 1
        self.width_out = (width - self.kernel_size) // self.stride + 1
        self.weight_reshape = np.reshape(self.weight, [-1, self.channel_out])
        self.img2col = np.zeros([self.input.shape[0] * self.height_out * self.width_out,
                                 self.channel_in * self.kernel_size * self.kernel_size])
        for idxn in range(self.input.shape[0]):
            for idxh in range(self.height_out):
                for idxw in range(self.width_out):
                    self.img2col[idxn * self.height_out * self.width_out + idxh * self.width_out + idxw,
                    :] = self.input_pad[idxn, :, idxh * self.stride:idxh * self.stride + self.kernel_size,
                         idxw * self.stride:idxw * self.stride + self.kernel_size].reshape([-1])
        output = np.dot(self.img2col, self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0], self.height_out, self.width_out, -1]).transpose([0, 3, 1, 2])
        show_matrix(self.output, 'conv out ')
        show_time(time.time() - start_time, 'conv forward time: ')
        print(
            f'计算量 = {self.input.shape[0] * self.channel_out * self.height_out * self.width_out * self.kernel_size * self.kernel_size * self.channel_in * 2} FLOPS')
        return self.output

    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))

    def forward(self, input):  # 前向传播的计算

        start_time = time.time()
        self.input = input  # [N, C, H, W]
        self.height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        self.width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        img2col = np.zeros([self.input.shape[0], self.input.shape[1], self.height_out * self.width_out,
                            self.kernel_size * self.kernel_size])
        for idxh in range(self.height_out):
            for idxw in range(self.width_out):
                img2col[:, :, idxh * self.width_out + idxw] = self.input[:, :,
                                                              idxh * self.stride:idxh * self.stride + self.kernel_size,
                                                              idxw * self.stride:idxw * self.stride + self.kernel_size].reshape(
                    [self.input.shape[0], self.input.shape[1], -1])
        self.output = np.max(img2col, axis=-1)
        self.output = np.reshape(self.output,
                                 [self.input.shape[0], self.input.shape[1], self.height_out, self.width_out])
        self.argmax = np.argmax(img2col, axis=-1)
        self.argmax = self.argmax.reshape(-1)
        self.max_index = np.zeros([self.argmax.shape[0], img2col.shape[-1]])
        self.max_index[np.arange(self.argmax.shape[0]), self.argmax] = 1.0
        self.max_index = np.reshape(self.max_index, img2col.shape)
        show_matrix(self.output, 'max pooling out ')
        show_time(time.time() - start_time, 'max pooling forward time: ')
        print(f'计算量 = 0 FLOPS')
        return self.output


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        print(f'计算量 = 0 FLOPS')
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        show_matrix(bottom_diff, 'flatten d_h ')
        return bottom_diff
