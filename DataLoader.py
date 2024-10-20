# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : DataLoader.py

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


def load_dataset(batch_size: int):
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    print('DataLoader.py running')
    train_iter, test_iter = load_dataset(100)
