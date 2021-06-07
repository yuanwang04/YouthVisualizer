import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

"""
UW CSE 455 Final Project Youth Visualizer

This module provides functions to visualize the convolutional 
network, including show activated areas of the given layer,
using back-propagation to activate a certain channel, and 
using back-propagation to do some pseudo deep-dream. 

Yuan Wang & Jiajie Shi. 2021-6-7.
"""

to_img = transforms.ToPILImage()


class Darknet64(nn.Module):
    """
    Used in testing the visualize module
    """

    def __init__(self):
        super(Darknet64, self).__init__()  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 1000)

    def forward(self, x):
        # Input 64x64x3

        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)  # 32x32x16
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)  # 16x16x32
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2)  # 8x8x64
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2, stride=2)  # 4x4x128
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2)  # 2x2x256

        # Global average pooling across each channel
        # (Input could be 2x2x256, 4x4x256, 7x3x256, output would always be 256 length vector)
        x = F.adaptive_avg_pool2d(x, 1)  # 1x1x256
        x = torch.flatten(x, 1)  # vector 256

        x = self.fc1(x)
        return x


def get_cifar_data(dir='./cifar/'):
    """
    Prepares the CIFAR data from the folder.
    :return: The DataLoader for the training set and the test set.
    """
    transform_train = transforms.Compose([
        transforms.Resize(64),  # Takes images smaller than 64 and enlarges them
        transforms.RandomCrop(64, padding=4, padding_mode='edge'),  # Take 64x64 crops from 72x72 padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root=dir + "train", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.ImageFolder(root=dir + "test", transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return {'train': trainloader, 'test': testloader}


def tensor2img(tensor):
    """
    Converts the given tensor to PIL image. Tensor could be 2D or 3D with 3 channels.
    :return: PIL image
    """
    image = tensor.cpu().clone()
    image = image.squeeze()
    image = to_img(image)
    return image


def select_channel(tensor, channel1=0, channel2=0, channel3=0):
    """
    Converts a 3D image with multiple channels to 3 channels, used to convert the
    input tensor to RGB style.
    :param tensor:    input tensor
    :param channel1:  first channel to select from the input (R)
    :param channel2:  second channel to select from the input (G)
    :param channel3:  third channel to select from the input (B)
    :return:  a 3D tensor, with 3 channels.
    """
    channel2 = channel2 or channel1
    channel3 = channel3 or channel1
    t = torch.stack((tensor[channel1], tensor[channel2], tensor[channel3]))
    return t


def imshow(input, output):
    """
    Show the input image side-by-side with the output image
    :param input:   tensor, 2d or 3d
    :param output:  tensor, 2d or 3d
    """
    output = tensor2img(output)
    input = tensor2img(input)
    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow(input)
    fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()


def imshow_grid(inputs, width=8):
    """
    Show the inputs in a grid.
    :param inputs:  multi-dimensional tensor.
    """
    if inputs.dim() == 3:
        n = inputs.shape[0]
        height = math.ceil(n / width)
        fig = plt.figure(figsize=(width, height))
        for i, img in enumerate(inputs, 1):
            ax = fig.add_subplot(height, width, i)
            ax.axis('off')
            ax.title = plt.title(i)
            plt.imshow(tensor2img(img))
    elif inputs.dim() == 4 or 5:
        rows = inputs.shape[0]
        cols = inputs.shape[1]
        fig = plt.figure(figsize=(cols, rows))
        for row, imgs in enumerate(inputs):
            for col, img in enumerate(imgs, 1):
                ax = fig.add_subplot(cols, rows, row * cols + col)
                draw = tensor2img(img)
                plt.imshow(draw)
    else:
        raise TypeError(f'inputs has unknown shape: %d' % inputs.dim())

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def deep_dream(inputs, labels, net, criterion=None, iter=10, rate=0.1, collect_every=1):
    """
    Use back-propagation to make the input more like the label.
    Also known as deep-dream, tries to see what does the network thinks makes the image
    more like the label.
    :param inputs:        tensor, inputs to the net
    :param labels:        target labels
    :param net:           the trained neural network
    :param criterion:     the criterion function, defaulted to CrossEntropyLoss
    :param iter:          number of iterations used modifying the input
    :param rate:          the rate of modifying the input
    :param collect_every: collect the result input after every given number of iterations
    :return:              a tensor, each tensor within represents a modified version of the input.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    inputs = inputs.clone()
    res = [inputs]
    for i in range(iter):
        net.zero_grad()
        inputs.requires_grad = True  # get gradient at input
        outputs = net.forward(inputs)
        losses = criterion(outputs, labels)
        losses.backward()
        new_inputs = inputs - rate * inputs.grad  # modify the input
        inputs = new_inputs.detach()
        if i % collect_every == collect_every - 1:
            print(f'epoch %d: losses=%.3f' % (i + 1, losses.item()))
            res.append(inputs)

    return torch.stack(res)


def get_activation_inputs(inputs, to_output, epoch=10, rate=0.1, collect_every=1):
    """
    Use back-propagation to activate a certain layer/channel of the network.
    :param inputs:         the input to the network
    :param to_output:      callable, should return the output using the specified layer/channel on the input.
    :param epoch:          number of iterations to modify the input
    :param rate:           the rate of modifying the input
    :param collect_every:  collect the result input after every given number of iterations
    :return:               a tensor, each tensor within represents a modified version of the input.
    """
    inputs = inputs.clone()
    result = [inputs]
    for i in range(epoch):
        inputs.requires_grad = True
        output = to_output(inputs)
        s = output.sum()
        s.backward()
        new_inputs = inputs + rate * inputs.grad
        inputs = new_inputs.detach()
        if i % collect_every == collect_every - 1:
            print(f'epoch %d: sum=%.3f' % (i + 1, s.item()))
            result.append(inputs)

    return torch.stack(result)


def rgb_some_channel(inputs, net, layer_rgbs=None):
    """
    Show the RGB colorized version of stacking some output channel together.
    :param inputs:      the input tensor
    :param net:         the trained network, should be Darknet or CelebANet
    :param layer_rgbs:  the channels to select for each layer
    """
    if layer_rgbs is None:
        layer_rgbs = {1: (0, 1, 2)}

    def add_one_subplot(output, idx):
        plt.subplot(1, n_layers + 1, idx + 1)
        plt.imshow(tensor2img(select_channel(output[0], layer_rgbs[idx][0], layer_rgbs[idx][1], layer_rgbs[idx][2])))

    n_layers = max(layer_rgbs)
    plt.subplot(1, n_layers + 1, 1)
    plt.imshow(tensor2img(inputs[0]))

    output1 = net.conv1(inputs)
    add_one_subplot(output1, 1)
    if n_layers >= 2:
        output2 = F.max_pool2d(F.relu(net.bn1(output1)), kernel_size=2, stride=2)
        output2 = net.conv2(output2)
        add_one_subplot(output2, 2)
    if n_layers >= 3:
        output3 = F.max_pool2d(F.relu(net.bn2(output2)), kernel_size=2, stride=2)
        output3 = net.conv3(output3)
        add_one_subplot(output3, 3)
    if n_layers >= 4:
        output4 = F.max_pool2d(F.relu(net.bn3(output3)), kernel_size=2, stride=2)
        output4 = net.conv4(output4)
        add_one_subplot(output4, 4)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def get_color_input(channel=(0,), size=(400, 400)):
    """
    Returns a pure-color image, converted to tensor.
    :param channel: the channels to have value 1.0, other channels are set to 0.0
    :param size:    the size of the input
    :return: a 3D tensor, represents a pure-color image.
    """
    result = []
    for i in range(3):
        if i in channel:
            t = torch.ones(size=size)
            result.append(t)
        else:
            result.append(torch.zeros(size=size))
    result = torch.stack(result)
    return torch.stack([result])


def get_rand_input(size=(400, 400)):
    """
    Returns a random colored 4D tensor with the given size.
    """
    return torch.rand((1, 3, size[0], size[1]))


def get_inputs_from_file(file_path):
    image = Image.open(file_path)
    return torch.stack([TO_TENSOR(image)])


"""
The main function tests for some implementations in this file 
using a pre-trained Darknet64 on CIFAR dataset. 
"""
if __name__ == '__main__':
    # data = get_our_data()
    data = get_cifar_data()

    net = Darknet64()
    criterion = nn.CrossEntropyLoss()

    state = torch.load('./checkpoint/checkpoint-20.pkl')
    net.load_state_dict(state['net'])

    itr = iter(data['test'])
    # itr.__next__()
    inputs, labels = itr.__next__()

    deep_dreamed_result = deep_dream(inputs, labels, net)
    imshow_grid(deep_dreamed_result)

    # inputs = inputs.cpu().clone()
    # inputs.requires_grad = True

    # outputs = net.forward(inputs)
    # losses = criterion(outputs, labels)
    # losses.backward()
    #
    # first_layer_output = net.conv1(inputs)[0][0].reshape((1, 32, 32))  # 1 * 32 * 32
    # imshow(inputs[0], first_layer_output)

    print("done")
