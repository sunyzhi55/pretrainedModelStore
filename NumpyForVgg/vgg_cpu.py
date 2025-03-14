# coding:utf-8
import numpy as np
from scipy.io import loadmat
import scipy
import time
import imageio
from PIL import Image
from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from layers_2 import ConvolutionalLayer, MaxPoolingLayer, FlattenLayer
import json


class VGG19(object):
    def __init__(self, param_path):
        self.image_mean = None
        self.update_layer_list = None
        self.layers = None
        self.param_path = param_path
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )

    def build_model(self):
        # TODO：定义VGG19 的网络结构
        print('Building vgg-19 model...')

        # 根据VGG的网络结构一次定义每一模块的卷积层、激活层和池化层
        # 同时各个模块之间的尺寸和维度需要对应
        self.layers = {'conv1_1': ConvolutionalLayer(3, 3, 64, 1, 1), 'relu1_1': ReLULayer(),
                       'conv1_2': ConvolutionalLayer(3, 64, 64, 1, 1), 'relu1_2': ReLULayer(),
                       'pool1': MaxPoolingLayer(2, 2), 'conv2_1': ConvolutionalLayer(3, 64, 128, 1, 1),
                       'relu2_1': ReLULayer(), 'conv2_2': ConvolutionalLayer(3, 128, 128, 1, 1), 'relu2_2': ReLULayer(),
                       'pool2': MaxPoolingLayer(2, 2), 'conv3_1': ConvolutionalLayer(3, 128, 256, 1, 1),
                       'relu3_1': ReLULayer(), 'conv3_2': ConvolutionalLayer(3, 256, 256, 1, 1), 'relu3_2': ReLULayer(),
                       'conv3_3': ConvolutionalLayer(3, 256, 256, 1, 1), 'relu3_3': ReLULayer(),
                       'conv3_4': ConvolutionalLayer(3, 256, 256, 1, 1), 'relu3_4': ReLULayer(),
                       'pool3': MaxPoolingLayer(2, 2), 'conv4_1': ConvolutionalLayer(3, 256, 512, 1, 1),
                       'relu4_1': ReLULayer(), 'conv4_2': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu4_2': ReLULayer(),
                       'conv4_3': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu4_3': ReLULayer(),
                       'conv4_4': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu4_4': ReLULayer(),
                       'pool4': MaxPoolingLayer(2, 2), 'conv5_1': ConvolutionalLayer(3, 512, 512, 1, 1),
                       'relu5_1': ReLULayer(), 'conv5_2': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu5_2': ReLULayer(),
                       'conv5_3': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu5_3': ReLULayer(),
                       'conv5_4': ConvolutionalLayer(3, 512, 512, 1, 1), 'relu5_4': ReLULayer(),
                       'pool5': MaxPoolingLayer(2, 2), 'flatten': FlattenLayer([512, 7, 7], [512 * 7 * 7]),
                       'fc6': FullyConnectedLayer(512 * 7 * 7, 4096), 'relu6': ReLULayer(),
                       'fc7': FullyConnectedLayer(4096, 4096), 'relu7': ReLULayer(),
                       'fc8': FullyConnectedLayer(4096, 1000), 'softmax': SoftmaxLossLayer()}

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer in vgg-19...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        print('Loading parameters from file ' + self.param_path)
        params = loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print('Get image mean: ' + str(self.image_mean))

        for idx in range(43):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [in_channel, height, width, out_channel]
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)
            if idx >= 37 and 'fc' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx - 1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    # def load_image(self, image_dir):
    #     print('Loading and preprocessing image from ' + image_dir)
    #     self.input_image = scipy.misc.imread(image_dir)
    #     self.input_image = scipy.misc.imresize(self.input_image, [224, 224, 3])
    #     self.input_image = np.array(self.input_image).astype(np.float32)
    #     self.input_image -= self.image_mean
    #     self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))
    #     # input dim [N, channel, height, width]
    #     self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

    # def load_image(self, image_dir):
    #     print('Loading and preprocessing image from ' + image_dir)
    #     # 使用 PIL 加载图像
    #     pil_image = Image.open(image_dir)
    #     pil_image = pil_image.convert("RGB")
    #
    #     # 使用 PIL 调整图像大小（PIL 需要的是 (width, height) 而不是 (height, width, channels)）
    #     pil_image = pil_image.resize((224, 224))
    #
    #     # 将 PIL 图像转换为 NumPy 数组
    #     self.input_image = np.array(pil_image).astype(np.float32)
    #
    #     # 如果图像是灰度图，它可能只有一个通道，您需要扩展它以匹配期望的 3 个通道
    #     # 例如: if self.input_image.ndim == 2:
    #     #     self.input_image = np.stack((self.input_image,) * 3, axis=-1)
    #
    #     # 减去均值（确保 self.image_mean 匹配图像的通道顺序和形状）
    #     self.input_image -= self.image_mean
    #
    #     # 重塑以添加批次维度（如果 self.input_image 已经是 3D 的，则可能不需要这一步）
    #     self.input_image = np.expand_dims(self.input_image, axis=0)
    #     # print('self.input_image1',self.input_image.shape)
    #
    #     # 转置以匹配 [N, channel, height, width] 的顺序
    #     self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])  # 注意 PIL 图像是 HWC，而您可能想要 CHW
    #     # print('self.input_image2',self.input_image.shape)

    def load_image(self, image_dir):
        print('Loading and preprocessing image from ' + image_dir)
        self.input_image = imageio.v2.imread(image_dir)
        # 使用Pillow来调整图像大小并转换为RGB格式
        img = Image.fromarray(self.input_image).convert('RGB')
        # img = img.resize((224, 224), Image.ANTIALIAS)
        img = img.resize((224, 224), Image.LANCZOS)
        self.input_image = np.array(img).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))
        # input dim [N, channel, height, width]
        self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

    def forward(self):
        print('Inferencing...')
        start_time = time.time()
        current = self.input_image
        for idx in range(len(self.param_layer_name)):
            start_time1 = time.time()
            print('Inferencing layer: ' + self.param_layer_name[idx])
            current = self.layers[self.param_layer_name[idx]].forward(current)
            print('Inference time: %f s \n' % (time.time() - start_time1))
        print('Inference time: %f s' % (time.time() - start_time))
        return current

    def evaluate(self):
        prob = self.forward()
        top1 = np.argmax(prob[0])
        print(f'Classification result: id = {top1}, prob = {prob[0, top1]}')

        with open('Class.json', 'r') as f:
            data = json.load(f)
            print(f'id = {top1}, category = {data[str(top1)][1]}')  # 得到ID和ID对应的类别


if __name__ == '__main__':
    vgg = VGG19(param_path=r"/path/of/imagenet-vgg-verydeep-19.mat")  # 加载预训练模型
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()

    vgg.load_image(r"/path/of/pictures")  # 加载图片
    vgg.evaluate()
