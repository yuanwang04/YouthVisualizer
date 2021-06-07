import os

import numpy as np
import pandas as pd
import torch.utils.data

from visualizer import *

"""
UW CSE 455 Final Project Youth Visualizer
The module handles the CelebA data set, and some complementary functions
associated with the CelebA network. Including, getting specified layer callables. 

Yuan Wang & Jiajie Shi. 2021-6-7.
"""

CELEBA_DIR = './celebA/'
CHECKPOINT = './checkpoint/celebA_cp15.cp'
TO_TENSOR = transforms.ToTensor()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CelebADataset(torch.utils.data.Dataset):
    """
    The Dataset to load CelebA data locally.
    If selected is not empty, only images with the given ids are loaded.
    Otherwise, all images are lazily loaded and cached on loading.
    """

    def __init__(self, dir=CELEBA_DIR, label='Young', selected=[]):
        self.img_dir = os.path.join(dir, 'img_align_celeba', 'img_align_celeba')
        csv = pd.read_csv(os.path.join(dir, 'list_attr_celeba.csv'))
        if not selected:
            self.cache = {}
            self.labels = np.array(csv[label], dtype='int32')
        else:
            cache = []
            labels = []
            for idx in selected:
                image = Image.open(os.path.join(self.img_dir, f'%.6d.jpg' % (idx)))
                t = transforms.Compose([transforms.ToTensor(), ])
                cache.append(t(image))
                labels.append(csv[label][idx - 1])
            self.cache = cache
            self.labels = np.array(labels, dtype='int32')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.cache, dict) and idx not in self.cache:
            image = Image.open(os.path.join(self.img_dir, f'%.6d.jpg' % (idx + 1)))
            t = transforms.Compose([transforms.ToTensor(), ])
            self.cache[idx] = t(image)
        return self.cache[idx], self.labels[idx]


def get_celebA_data(dir=CELEBA_DIR, test_shuffle=False, test_idxs=None, test_batch_size=1):
    """
    Get the DataLoader for the celebA dataset.
    :param dir:             the root directory path
    :param test_shuffle:    boolean, shuffle test set or not
    :param test_idxs:       list of ids to use in the test set.
    :param test_batch_size: batch size of the test set.
    :return:  the DataLoader for the train and/or test set.
    """
    if test_idxs is None:
        test_idxs = []
    if not os.path.exists(dir):
        os.makedirs(dir + 'train')
        os.makedirs(dir + 'test')

    transform_train = transforms.Compose([
        transforms.Resize(64),  # Takes images smaller than 64 and enlarges them
        transforms.RandomCrop(64, padding=4, padding_mode='edge'),  # Take 64x64 crops from 72x72 padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # trainset = torchvision.datasets.CelebA(root=dir+'train', split='train', transform=transform_train,
    #                                        download=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    # testset = torchvision.datasets.CelebA(root=dir+'test', split='test', transform=transform_test,
    #                                       download=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    testset = CelebADataset(selected=test_idxs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=test_shuffle)

    return {'test': testloader}
    # return {'train': trainloader, 'test': testloader}


class CelebNet(nn.Module):
    """
    Same as Darknet64.
    """

    def __init__(self):
        super(CelebNet, self).__init__()  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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

        self.fc1 = nn.Linear(256, 2)
        self.fc2 = nn.Linear(32 * 32, 8 * 8)
        self.fc3 = nn.Linear(8 * 8, 2)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)  # 32x32x16
        x = F.avg_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)  # 16x16x32
        x = F.avg_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2)  # 8x8x64
        x = F.avg_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2, stride=2)  # 4x4x128
        x = F.avg_pool2d(F.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2)  # 2x2x256

        # Global average pooling across each channel
        # (Input could be 2x2x256, 4x4x256, 7x3x256, output would always be 256 length vector)
        x = F.adaptive_avg_pool2d(x, 1)  # 1x1x256
        x = torch.flatten(x, 1)  # vector 256

        x = self.fc1(x)
        return x


def get_layered_result(inputs, net):
    """
    Return the result at each layer.
    :param inputs:  the input tensor
    :param net:     the Darknet or the CelebAnet
    :return:        a list of output at each layer.
    """
    o1 = net.conv1(inputs)
    o2 = net.conv2(F.max_pool2d(F.relu(net.bn1(o1)), kernel_size=2, stride=2))
    o3 = net.conv3(F.max_pool2d(F.relu(net.bn2(o2)), kernel_size=2, stride=2))
    o4 = net.conv4(F.max_pool2d(F.relu(net.bn3(o3)), kernel_size=2, stride=2))

    return [o1, o2, o3, o4]


def get_layer_funcs(net, channel=None):
    """
    Returns all the callables at each layer of the given channels.
    :param net:      the Darnet or the CelebANet
    :param channel:  dict, from layer number to the wanted channel. e.g. {1: 10, 2: 20}
    :return:         dict, from layer number to the callable, {1: f1, 2: f2, ...}
    """
    if not channel:
        channel = {}

    def first_layer_func(net, channel=0):
        def temp_func(inputs):
            o1 = net.conv1(inputs)
            return o1[:, channel]

        return temp_func

    def second_layer_func(net, channel=0):
        def temp_func(inputs):
            o1 = net.conv1(inputs)
            o2 = net.conv2(F.max_pool2d(F.relu(net.bn1(o1)), kernel_size=2, stride=2))
            return o2[:, channel]

        return temp_func

    def third_layer_func(net, channel=0):
        def temp_func(inputs):
            o1 = net.conv1(inputs)
            o2 = net.conv2(F.max_pool2d(F.relu(net.bn1(o1)), kernel_size=2, stride=2))
            o3 = net.conv3(F.max_pool2d(F.relu(net.bn2(o2)), kernel_size=2, stride=2))
            return o3[:, channel]

        return temp_func

    def forth_layer_func(net, channel=0):
        def temp_func(inputs):
            o1 = net.conv1(inputs)
            o2 = net.conv2(F.max_pool2d(F.relu(net.bn1(o1)), kernel_size=2, stride=2))
            o3 = net.conv3(F.max_pool2d(F.relu(net.bn2(o2)), kernel_size=2, stride=2))
            o4 = net.conv4(F.max_pool2d(F.relu(net.bn3(o3)), kernel_size=2, stride=2))
            return o4[:, channel]

        return temp_func

    def fifth_layer_func(net, channel=0):
        def temp_func(inputs):
            o1 = net.conv1(inputs)
            o2 = net.conv2(F.avg_pool2d(F.relu(net.bn1(o1)), kernel_size=2, stride=2))
            o3 = net.conv3(F.avg_pool2d(F.relu(net.bn2(o2)), kernel_size=2, stride=2))
            o4 = net.conv4(F.avg_pool2d(F.relu(net.bn3(o3)), kernel_size=2, stride=2))
            o5 = net.conv5(F.max_pool2d(F.relu(net.bn4(o4)), kernel_size=2, stride=2))
            return o5[:, channel]

        return temp_func

    f1 = first_layer_func(net, channel[1] if 1 in channel else 0)
    f2 = second_layer_func(net, channel[2] if 2 in channel else 0)
    f3 = third_layer_func(net, channel[3] if 3 in channel else 0)
    f4 = forth_layer_func(net, channel[4] if 4 in channel else 0)
    f5 = fifth_layer_func(net, channel[5] if 5 in channel else 0)
    return {1: f1, 2: f2, 3: f3, 4: f4, 5: f5}
