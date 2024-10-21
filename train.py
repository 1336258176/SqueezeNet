# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Bin Li
# @Email   : lybin1336258176@outlook.com
# @File    : train.py
import os.path

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader

import DataLoader
import SqueezeNet

epochs = 100
batch_size = 128
num_classes = 10
dropout = 0.5
lr = 0.01
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, test_iter = DataLoader.load_dataset(batch_size)
model = SqueezeNet.SqueezeNet(num_classes, dropout).to(dev)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)


def train_loop(dataloader: dataloader.DataLoader, model: nn.Module, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % batch_size == 0:
            loss, current = loss.item(), batch_idx * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: dataloader.DataLoader, model: nn.Module, loss_fn, lr_scheduler: torch.optim.lr_scheduler = None):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0., 0.
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if lr_scheduler and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(test_loss)
    elif lr_scheduler and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step()
    else:
        pass

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    print(f'running on {dev}')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_iter, model, loss_fn, optimizer)
        test_loop(test_iter, model, loss_fn, lr_scheduler)
    print('Training Done!')

    if os.path.exists(os.path.join('./model', 'SqueezeNet.pth')):
        os.remove(os.path.join('./model', 'SqueezeNet.pth'))
        print('Delete an existing model parameter.')
    else:
        os.makedirs('./model')
    torch.save(model.state_dict(), './model/SqueezeNet.pth')
    print('Model saved!')
