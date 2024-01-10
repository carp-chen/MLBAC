import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import preprocessing, model_selection, metrics, linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

def data_processing():
    data = pd.read_csv("dataset/train.csv", delimiter=',', usecols=range(1, 9))
    target = pd.read_csv("dataset/train.csv", delimiter=',', usecols=[0])

    # smote technique
    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    X_balanced, Y_balanced = sm.fit_resample(data, target.values.ravel())

    # dataset is highly categorical so need to perform one-hot encoding
    obj = preprocessing.OneHotEncoder()
    obj.fit(X_balanced)
    X_dummyEncode = obj.transform(X_balanced)

    selectBest_attribute = SelectKBest(chi2, k=550)
    # fit and transforms the data
    selectBest_attribute.fit(X_dummyEncode, Y_balanced)
    modifiedData = selectBest_attribute.transform(X_dummyEncode)

    return modifiedData, Y_balanced

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

def Naive_Bayes(modifiedData, target):
    """
    This function implements Naive Bayes algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #BernoilliNB creates Niave Bayes object
    bernoilli_nb = BernoulliNB(alpha = 1)  # alpha = 1 performs laplace estimation, to handle zero frequency problem'

    #variable initialization
    random_seed = 31  #random seed value
    kFold = 10        #for 10 fold cross validation

    #stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    #performs 10 fold cross validation
    for fold in range(kFold):

        #uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(modifiedData, target, test_size=.20,
                                                                        random_state= fold * random_seed)
        #fits and predicts the values
        bernoilli_nb.fit(X_train, np.ravel(Y_train,order='C'))
        prediction = bernoilli_nb.predict_proba(X_test)[:,1]
        tempPrediction = bernoilli_nb.predict(X_test)
        #calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc, roc_auc_value)
        fscre  += f1Score(Y_test, tempPrediction)
        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for Naive Bayes: %f" % (mean_auc / kFold))
    print("F1 - Score Naive Bayes: %f" % (fscre / kFold) )
    print("Mean Accuracy Naive Bayes: %f" % (mean_accuracy / kFold) )

    #plots AUC graph for Naive Bayes
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color = 'g', label = 'Naive Bayes')
    plt.legend(loc = 'lower right')

def Logistic_Regression(data, target):
    """
    This function implements Logistic Regression algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #creates object for logistic regression
    log_regrresion = linear_model.LogisticRegression(C=3)

    # variable initialization
    random_seed = 42   #random seed value
    kFold = 10         #for 10 fold cross validation

    #stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20, random_state= fold * random_seed)

        # fits and predicts the values
        log_regrresion.fit(X_train, np.ravel(Y_train,order='C'))
        prediction = log_regrresion.predict_proba(X_test)[:,1]
        tempPrediction = log_regrresion.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc, roc_auc_value)
        fscre  += f1Score(Y_test, tempPrediction)
        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)
    print("Mean AUC for Logistic Regression: %f" % (mean_auc / kFold))
    print("F1 - Score Logistic Regression: %f" % (fscre / kFold) )
    print("Mean Accuracy Logistic Regression: %f" % (mean_accuracy / kFold) )


    # plots AUC graph for Logistic Regression
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color = 'm', label = 'Logistic Regression')
    plt.legend(loc = 'lower right')


def Random_Forest(data, target):
    """
    This function implements Random Forest algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    # creates object for random forest classifier
    random_forest = RandomForestClassifier(n_jobs=10)

    # variable initialization
    random_seed = 31  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * random_seed)

        # fits and predicts the values
        random_forest.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = random_forest.predict_proba(X_test)[:, 1]
        tempPrediction = random_forest.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, tempPrediction)

        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for Random Forest: %f" % (mean_auc / kFold))
    print("F1 - Score Random Forest: %f" % (fscre / kFold))
    print("Mean Accuracy Random Forest: %f" % (mean_accuracy / kFold))

    # plots AUC graph for Random Forest
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='r', label='Random Forest')
    plt.legend(loc='lower right')


def SVM(data, target):
    """
    This function implements SVM algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    # creates object for linear SVM
    linearSVM = LinearSVC(penalty='l1', random_state=37, max_iter=1000, dual=False, C=3)

    # initializing the variable
    random_seed = 42  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * random_seed)

        # fits and predicts the values
        linearSVM.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = linearSVM.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, prediction)

        mean_accuracy += metrics.accuracy_score(Y_test, prediction)

    print("Mean AUC for SVM: %f" % (mean_auc / kFold))
    print("F1 - Score SVM: %f" % (fscre / kFold))
    print("Mean Accuracy SVM: %f" % (mean_accuracy / kFold))

    # plots AUC graph for SVM
    max_roc = roc_auc_value.index("{0:.2f}".format(mean_auc / kFold))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='y', label='SVM')
    plt.legend(loc='lower right')


def KNN(data, target):
    """
    This function implements KNN algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    # KNN object using 7 neighbors
    kneighbor = KNeighborsClassifier(n_neighbors=7)

    # variable initialization
    ramdom_seed = 42  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * ramdom_seed)

        # fits and predicts the values
        kneighbor.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = kneighbor.predict_proba(X_test)[:, 1]
        tempPrediction = kneighbor.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, tempPrediction)

        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for KNN: %f" % (mean_auc / kFold))
    print("F1 - Score KNN: %f" % (fscre / kFold))
    print("Mean Accuracy KNN: %f" % (mean_accuracy / kFold))
    # plots AUC graph for KNN
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='b', label='KNN')
    plt.legend(loc='lower right')

def XGBoost(data, target):
    """
    This function implements XGBoost algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    # XGBoost 调参
    xgboost = XGBClassifier(
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric=['logloss', 'auc', 'error'],
        nthread=10,
        scale_pos_weight=1,
        seed=42
    )

    # variable initialization
    ramdom_seed = 42  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * ramdom_seed)

        # fits and predicts the values
        xgboost.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = xgboost.predict_proba(X_test)[:, 1]
        tempPrediction = xgboost.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, tempPrediction)

        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for XGBoost: %f" % (mean_auc / kFold))
    print("F1 - Score XGBoost: %f" % (fscre / kFold))
    print("Mean Accuracy XGBoost: %f" % (mean_accuracy / kFold))
    # plots AUC graph for XGBoost
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='pink', label='XGBoost')
    plt.legend(loc='lower right')

def lightGBM(data, target):
    """
    This function implements lightGBM algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """


    # lightGBM 调参，使用GridSearchCV
    lightgbm = lgb.LGBMClassifier(
        boosting_type='gbdt',  # 设置提升类型
        objective='binary',  # 目标函数
        num_leaves=31,  # 叶子节点数
        learning_rate=0.05,  # 学习速率
        feature_fraction=0.9,  # 建树的特征选择比例
        bagging_fraction=0.8,  # 建树的样本采样比例
        bagging_freq=5,  # k 意味着每 k 次迭代执行bagging
        verbose=-1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        is_unbalance=True,  # 用于二分类
        min_data_in_leaf=20,
        num_iterations=1000,
        n_estimators=1000,
        max_depth=8,
        max_bin=255,
        n_jobs=10,
        random_state=42
    )


    # variable initialization
    ramdom_seed = 42  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * ramdom_seed)

        # fits and predicts the values
        lightgbm.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = lightgbm.predict_proba(X_test)[:, 1]
        tempPrediction = lightgbm.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, tempPrediction)

        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for lightGBM: %f" % (mean_auc / kFold))
    print("F1 - Score lightGBM: %f" % (fscre / kFold))
    print("Mean Accuracy lightGBM: %f" % (mean_accuracy / kFold))
    # plots AUC graph for lightGBM
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='orange', label='lightGBM')
    plt.legend(loc='lower right')

def CatBoost(data, target):
    """
    This function implements CatBoost algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    # CatBoost 调参
    catboost = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        logging_level='Silent',
        thread_count=10
    )

    # variable initialization
    ramdom_seed = 42  # random seed value
    kFold = 10  # for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []
    fscre = 0
    mean_accuracy = 0
    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,
                                                                            random_state=fold * ramdom_seed)

        # fits and predicts the values
        catboost.fit(X_train, np.ravel(Y_train, order='C'))
        prediction = catboost.predict_proba(X_test)[:, 1]
        tempPrediction = catboost.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,
                                                                  roc_auc_value)
        fscre += f1Score(Y_test, tempPrediction)

        mean_accuracy += metrics.accuracy_score(Y_test, tempPrediction)

    print("Mean AUC for CatBoost: %f" % (mean_auc / kFold))
    print("F1 - Score CatBoost: %f" % (fscre / kFold))
    print("Mean Accuracy CatBoost: %f" % (mean_accuracy / kFold))
    # plots AUC graph for CatBoost
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='purple', label='CatBoost')
    plt.legend(loc='lower right')

if __name__ == '__main__':
    modifiedData, Y_balanced = data_processing()

    # start_time = time.time()
    # Naive_Bayes(modifiedData, Y_balanced)
    # print("Time Required for Naive Bayes in sec:",  (time.time() - start_time))
    # print("------------------------------------------------------------------------------")

    # start_time = time.time()
    # Logistic_Regression(modifiedData,Y_balanced)
    # print("Time Required for Logistic regression in sec: ", (time.time() - start_time))
    # print("------------------------------------------------------------------------------")

    start_time = time.time()
    Random_Forest(modifiedData, Y_balanced)
    print("Time Required for Random Forest in sec: " ,  (time.time() - start_time))
    print("------------------------------------------------------------------------------")

    # start_time = time.time()
    # KNN(modifiedData, Y_balanced)
    # print("Time Required for KNN in sec: ", (time.time() - start_time))
    # print("------------------------------------------------------------------------------")

    # start_time = time.time()
    # SVM(modifiedData, Y_balanced)
    # print("Time Required for SVM in sec: " ,(time.time() - start_time))
    # print("------------------------------------------------------------------------------")

    start_time = time.time()
    XGBoost(modifiedData, Y_balanced)
    print("Time Required for XGBoost in sec: ", (time.time() - start_time))
    print("------------------------------------------------------------------------------")

    start_time = time.time()
    lightGBM(modifiedData, Y_balanced)
    print("Time Required for lightGBM in sec: ", (time.time() - start_time))
    print("------------------------------------------------------------------------------")

    start_time = time.time()
    CatBoost(modifiedData, Y_balanced)
    print("Time Required for CatBoost in sec: ", (time.time() - start_time))
    print("------------------------------------------------------------------------------")


    plotGraph()

