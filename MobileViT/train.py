import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_loader
from model import mobile_vit_small as create_model
from utils import read_split_data, train_one_epoch, evaluate


def data_parser():
    data = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=range(1, 9))
    target = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=[0])

    # smote technique
    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    ada = ADASYN(random_state=42)
    X_balanced, Y_balanced = ada.fit_resample(data, target.values.ravel())

    # dataset is highly categorical so need to perform one-hot encoding
    obj = preprocessing.OneHotEncoder()
    obj.fit(X_balanced)
    X_dummyEncode = obj.transform(X_balanced)

    selectBest_attribute = SelectKBest(chi2, k=4096)
    # fit and transforms the data
    selectBest_attribute.fit(X_dummyEncode, Y_balanced)
    modifiedData = selectBest_attribute.transform(X_dummyEncode)

    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(modifiedData, Y_balanced, test_size=0.2, random_state=100)
    # 变成numpy array
    x_train = x_train.A
    x_test = x_test.A
    # reshape the array
    x_train = x_train.reshape((x_train.shape[0], 64, 64))
    x_test = x_test.reshape((x_test.shape[0], 64, 64))

    return x_train, x_test, y_train, y_test

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    x_train, x_test, y_train, y_test = data_parser()

    batch_size = args.batch_size
    train_loader, val_loader = get_loader(batch_size, x_train, x_test, y_train, y_test)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))


    model = create_model(num_classes=args.num_classes).to(device)

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "classifier" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    #
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "classifier" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

        torch.save(model.state_dict(), "./weights/latest_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    #
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./mobilevit_xxs.pt',
    #                     help='initial weights path')
    # # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
