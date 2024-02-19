import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import mobile_vit_small as create_model

from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn import  metrics

from dataloader import get_loader
from tqdm import tqdm



def data_parser():
    data = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=range(1, 10))
    target = pd.read_csv("../dataset/train.csv", delimiter=',', usecols=[0])

    # smote technique
    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    ada = ADASYN(random_state=42)
    X_balanced, Y_balanced = sm.fit_resample(data, target.values.ravel())

     # Tomek Links数据清洗
    tl = TomekLinks()
    X_balanced, Y_balanced = tl.fit_resample(X_balanced, Y_balanced)

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
    x_train = x_train.A
    x_test = x_test.A
    # reshape the array
    x_train = x_train.reshape((x_train.shape[0], 64, 64))
    x_test = x_test.reshape((x_test.shape[0], 64, 64))

    return x_train, x_test, y_train, y_test

def roc_value(Y_test, prediction, fpr_score, tpr_score, mean_auc, roc_auc_value):
    """
    This function calculates fpr, tpr and AUC curve value for any algorithm
    :param Y_test: the actual values of the target class
    :param prediction: the predicted values of the target class
    :param fpr_score: the false positve rate
    :param tpr_score: the true positive rate
    :param mean_auc: the mean value of AUC curve for 10 folds
    :param roc_auc_value: each auc curve value across 10 folds
    :return: roc_auc_value: each auc curve value across 10 folds
    """

    #calculates fpr and tpr values for each model
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction)
    fpr_score.append(fpr)
    tpr_score.append(tpr)

    #calculates auc for each model
    roc_auc = metrics.auc(fpr, tpr)
    mean_auc = mean_auc + roc_auc
    roc_auc_value.append("{0:.2f}".format(roc_auc))

    return fpr_score, tpr_score, roc_auc_value, mean_auc

def plotGraph():
    """
    This function sets lables for X and Y axis
    :return: None
    """
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("AUC comparison for all models.")
    plt.grid(True)
    plt.show()

def f1Score(Y_test, prediction):
    """
    This function calculates f1-score for any algorithm
    :param Y_test: the actual values of the target class
    :param prediction: the predicted values of the target class
    """
    score = f1_score(Y_test, prediction, average='binary')
    return score


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test = data_parser()

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    batch_size = 128 # 每次预测时将多少张图片打包成一个batch
    _, val_loader = get_loader(batch_size, x_train, x_test, y_train, y_test)

    # 批量预测x_test
    real_class = []
    predict_proba = []
    predict_class = []
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        # predict class
        sample_num = 0
        data_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            inputs, labels = data
            sample_num += inputs.shape[0]

            outputs = model(inputs.to(device))
            probabilities = torch.softmax(outputs, dim=1)
            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(outputs, labels.to(device))
            accu_loss += loss

            data_loader.desc = "[valid] loss: {:.3f}, acc: {:.3f}".format(accu_loss.item() / (step + 1),
                                                                        accu_num.item() / sample_num)

            real_class.extend(labels.tolist())
            predict_proba.extend(probabilities[:, 1].tolist())
            predict_class.extend(pred_classes.tolist())


        

    # stores fpr, tpr, auc
    auc = 0
    tpr_score = []  # Fix the variable name typo
    fpr_score = []
    tpr_score = []
    roc_auc_value = []
    accuracy = 0

    # Calculate AUC, ROC, and F1-score
    fpr_score, tpr_score, roc_auc_value, auc = roc_value(real_class, predict_proba, fpr_score, tpr_score, auc, roc_auc_value)
    f1 = f1Score(real_class, predict_class)
    accuracy = sum(real_class[i] == predict_class[i] for i in range(len(real_class))) / len(real_class)

    # Print the results
    print("AUC: ", auc)
    print("ROC: ", roc_auc_value)
    print("F1-score: ", f1)
    print("ACCURACY: ", accuracy)

    


if __name__ == '__main__':
    main()
