import os
import math
import argparse

import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_loader

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate

def data_parser():
    data = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=range(1, 9))
    target = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=[0])

    # smote technique
    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    ada = ADASYN(random_state=42)
    X_balanced, Y_balanced = sm.fit_resample(data, target.values.ravel())

    # dataset is highly categorical so need to perform one-hot encoding
    obj = preprocessing.OneHotEncoder()
    obj.fit(X_balanced)
    X_dummyEncode = obj.transform(X_balanced)

    selectBest_attribute = SelectKBest(chi2, k=50176)
    # fit and transforms the data
    selectBest_attribute.fit(X_dummyEncode, Y_balanced)
    modifiedData = selectBest_attribute.transform(X_dummyEncode)

    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(modifiedData, Y_balanced, test_size=0.2, random_state=100)
    # 变成numpy array
    x_train = x_train.A
    x_test = x_test.A
    # reshape the array
    x_train = x_train.reshape((x_train.shape[0], 224, 224))
    x_test = x_test.reshape((x_test.shape[0], 224, 224))

    return x_train, x_test, y_train, y_test

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    x_train, x_test, y_train, y_test = data_parser()

    batch_size = args.batch_size
    train_loader, val_loader = get_loader(batch_size, x_train, x_test, y_train, y_test)
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    #     "val": transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    #
    # # 实例化训练数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_dataset = MyDataSet(images_path=val_images_path,
    #                         images_class=val_images_label,
    #                         transform=data_transform["val"])

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn)
    #
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除不需要的权重
    #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #         else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    #
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

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

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    # parser.add_argument('--model-name', default='', help='create model name')
    #
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
    #                     help='initial weights path')
    # # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
