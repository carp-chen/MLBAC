import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import BorderlineSMOTE
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from torch.utils.data import DataLoader, Dataset
import torchvision

class DatasetDLBAC(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data.astype(np.double)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def data_parser():
    data = pd.read_csv("dataset/train.csv", delimiter=',', usecols=range(1, 9))
    target = pd.read_csv("dataset/train.csv", delimiter=',', usecols=[0])

    # smote technique
    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    X_balanced, Y_balanced = sm.fit_resample(data, target.values.ravel())

    # dataset is highly categorical so need to perform one-hot encoding
    obj = preprocessing.OneHotEncoder()
    obj.fit(X_balanced)
    X_dummyEncode = obj.transform(X_balanced)

    selectBest_attribute = SelectKBest(chi2, k=4096)
    # fit and transforms the data
    selectBest_attribute.fit(X_dummyEncode, Y_balanced)
    modifiedData = selectBest_attribute.transform(X_dummyEncode)

    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(modifiedData, Y_balanced, test_size=0.2, random_state=42)
    # 变成numpy array
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    # reshape the array
    x_train = x_train.reshape((x_train.shape[0], 64, 64))
    x_test = x_test.reshape((x_test.shape[0], 64, 64))

    return x_train, x_test, y_train, y_test

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    #
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"])
    # train_num = len(train_dataset)
    #
    # # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    x_train, x_test, y_train, y_test = data_parser()
    batch_size = 16
    train_loader, test_loader = get_loader(batch_size, x_train, x_test, y_train, y_test)
    train_num = len(x_train)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size, shuffle=True,
    #                                            num_workers=nw)
    #
    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform["val"])
    val_num = len(x_test)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=False,
    #                                               num_workers=nw)
    #
    print("using {} records for training, {} records for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34(num_classes=2)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            input, labels = data
            # print(input.shape)
            input = input.to(torch.float32)
            optimizer.zero_grad()
            logits = net(input.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
