# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

from torch.utils.data import DataLoader, Dataset
from numpy import loadtxt


class DatasetDLBAC(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data.astype(np.float32)
        self.y_data = y_data.astype(np.int8)
        self.transform = transform
        # print('sample x_data', self.x_data[0])
        # print('sample y_data', self.y_data[0])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        image = self.x_data[index]
        image = image[..., np.newaxis]
        label = 1 if self.y_data[index] == 1 else 0

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        inputs, labels = tuple(zip(*batch))

        inputs = torch.stack(inputs, dim=0)
        labels = torch.as_tensor(labels)
        return inputs, labels


def get_loader(batch_size, x_train, x_test, y_train, y_test):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_dataset = DatasetDLBAC(x_train, y_train, transform=train_transform)
    test_dataset = DatasetDLBAC(x_test, y_test, transform=train_transform)

    img, lab = train_dataset.__getitem__(0)
    # print('Shape of Training Data: ',img.shape)
    # print(img)
    # print(type(img))
    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=8, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, collate_fn=test_dataset.collate_fn)

    return train_loader, test_loader
