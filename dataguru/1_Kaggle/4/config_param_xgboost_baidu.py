

    专栏 文章详情 

有故事
有故事 358
2018-03-28 发布
XGboost数据比赛实战之调参篇(完整流程)

这一篇博客的内容是在上一篇博客Scikit中的特征选择，XGboost进行回归预测，模型优化的实战的基础上进行调参优化的，所以在阅读本篇博客之前，请先移步看一下上一篇文章。

我前面所做的工作基本都是关于特征选择的，这里我想写的是关于XGBoost参数调整的一些小经验。之前我在网站上也看到很多相关的内容，基本是翻译自一篇英文的博客，更坑的是很多文章步骤讲的不完整，新人看了很容易一头雾水。由于本人也是一个新手，在这过程中也踩了很多大坑，希望这篇博客能够帮助到大家！下面，就进入正题吧。

首先，很幸运的是，Scikit-learn中提供了一个函数可以帮助我们更好地进行调参：

sklearn.model_selection.GridSearchCV

常用参数解读：

    estimator：所使用的分类器，如果比赛中使用的是XGBoost的话，就是生成的model。比如： model = xgb.XGBRegressor(**other_params)
    param_grid：值为字典或者列表，即需要最优化的参数的取值。比如：cv_params = {'n_estimators': [550, 575, 600, 650, 675]}
    scoring :准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。scoring参数选择如下：

这里写图片描述

具体参考地址：http://scikit-learn.org/stable/modules/model_evaluation.html

这次实战我使用的是r2这个得分函数，当然大家也可以根据自己的实际需要来选择。

调参刚开始的时候，一般要先初始化一些值：

        learning_rate: 0.1
        n_estimators: 500
        max_depth: 5
        min_child_weight: 1
        subsample: 0.8
        colsample_bytree:0.8
        gamma: 0
        reg_alpha: 0
        reg_lambda: 1

链接：XGBoost常用参数一览表

你可以按照自己的实际情况来设置初始值，上面的也只是一些经验之谈吧。

调参的时候一般按照以下顺序来进行：

1、最佳迭代次数：n_estimators

if __name__ == '__main__':
    trainFilePath = 'dataset/soccer/train.csv'
    testFilePath = 'dataset/soccer/test.csv'
    data = pd.read_csv(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)

    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

<font color=red size=4>写到这里，需要提醒大家，在代码中有一处很关键：</font>

model = xgb.XGBRegressor(**other_params)中两个*号千万不能省略！可能很多人不注意，再加上网上很多教程估计是从别人那里直接拷贝，没有运行结果，所以直接就用了 model = xgb.XGBRegressor(other_params)。<font color=red size=4>悲剧的是，如果直接这样运行的话，会报如下错误：</font>

xgboost.core.XGBoostError: b"Invalid Parameter format for max_depth expect int but value...

不信，请看链接：xgboost issue

以上是血的教训啊，自己不运行一遍代码，永远不知道会出现什么Bug！

运行后的结果为：

[Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.5min finished
每轮迭代运行结果:[mean: 0.94051, std: 0.01244, params: {'n_estimators': 400}, mean: 0.94057, std: 0.01244, params: {'n_estimators': 500}, mean: 0.94061, std: 0.01230, params: {'n_estimators': 600}, mean: 0.94060, std: 0.01223, params: {'n_estimators': 700}, mean: 0.94058, std: 0.01231, params: {'n_estimators': 800}]
参数的最佳取值：{'n_estimators': 600}
最佳模型得分:0.9406056804545407

由输出结果可知最佳迭代次数为600次。但是，我们还不能认为这是最终的结果，由于设置的间隔太大，所以，我又测试了一组参数，这次粒度小一些：

 cv_params = {'n_estimators': [550, 575, 600, 650, 675]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

运行后的结果为：

[Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.5min finished
每轮迭代运行结果:[mean: 0.94065, std: 0.01237, params: {'n_estimators': 550}, mean: 0.94064, std: 0.01234, params: {'n_estimators': 575}, mean: 0.94061, std: 0.01230, params: {'n_estimators': 600}, mean: 0.94060, std: 0.01226, params: {'n_estimators': 650}, mean: 0.94060, std: 0.01224, params: {'n_estimators': 675}]
参数的最佳取值：{'n_estimators': 550}
最佳模型得分:0.9406545392685364

果不其然，最佳迭代次数变成了550。有人可能会问，那还要不要继续缩小粒度测试下去呢？这个我觉得可以看个人情况，如果你想要更高的精度，当然是粒度越小，结果越准确，大家可以自己慢慢去调试，我在这里就不一一去做了。

2、接下来要调试的参数是min_child_weight以及max_depth：

<font color=red size=4>注意：每次调完一个参数，要把 other_params对应的参数更新为最优值。</font>

 cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

运行后的结果为：

[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  1.7min
[Parallel(n_jobs=4)]: Done 192 tasks      | elapsed: 12.3min
[Parallel(n_jobs=4)]: Done 240 out of 240 | elapsed: 17.2min finished
每轮迭代运行结果:[mean: 0.93967, std: 0.01334, params: {'min_child_weight': 1, 'max_depth': 3}, mean: 0.93826, std: 0.01202, params: {'min_child_weight': 2, 'max_depth': 3}, mean: 0.93739, std: 0.01265, params: {'min_child_weight': 3, 'max_depth': 3}, mean: 0.93827, std: 0.01285, params: {'min_child_weight': 4, 'max_depth': 3}, mean: 0.93680, std: 0.01219, params: {'min_child_weight': 5, 'max_depth': 3}, mean: 0.93640, std: 0.01231, params: {'min_child_weight': 6, 'max_depth': 3}, mean: 0.94277, std: 0.01395, params: {'min_child_weight': 1, 'max_depth': 4}, mean: 0.94261, std: 0.01173, params: {'min_child_weight': 2, 'max_depth': 4}, mean: 0.94276, std: 0.01329...]
参数的最佳取值：{'min_child_weight': 5, 'max_depth': 4}
最佳模型得分:0.94369522247392

由输出结果可知参数的最佳取值：{'min_child_weight': 5, 'max_depth': 4}。（代码输出结果被我省略了一部分，因为结果太长了，以下也是如此）

3、接着我们就开始调试参数：gamma：

cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

运行后的结果为：

[Parallel(n_jobs=4)]: Done  30 out of  30 | elapsed:  1.5min finished
每轮迭代运行结果:[mean: 0.94370, std: 0.01010, params: {'gamma': 0.1}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.2}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.3}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.4}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.5}, mean: 0.94370, std: 0.01010, params: {'gamma': 0.6}]
参数的最佳取值：{'gamma': 0.1}
最佳模型得分:0.94369522247392

由输出结果可知参数的最佳取值：{'gamma': 0.1}。

4、接着是subsample以及colsample_bytree：

cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}

运行后的结果显示参数的最佳取值：{'subsample': 0.7,'colsample_bytree': 0.7}

5、紧接着就是：reg_alpha以及reg_lambda：

 cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}

运行后的结果为：

[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.0min
[Parallel(n_jobs=4)]: Done 125 out of 125 | elapsed:  5.6min finished
每轮迭代运行结果:[mean: 0.94169, std: 0.00997, params: {'reg_alpha': 0.01, 'reg_lambda': 0.01}, mean: 0.94112, std: 0.01086, params: {'reg_alpha': 0.01, 'reg_lambda': 0.05}, mean: 0.94153, std: 0.01093, params: {'reg_alpha': 0.01, 'reg_lambda': 0.1}, mean: 0.94400, std: 0.01090, params: {'reg_alpha': 0.01, 'reg_lambda': 1}, mean: 0.93820, std: 0.01177, params: {'reg_alpha': 0.01, 'reg_lambda': 100}, mean: 0.94194, std: 0.00936, params: {'reg_alpha': 0.05, 'reg_lambda': 0.01}, mean: 0.94136, std: 0.01122, params: {'reg_alpha': 0.05, 'reg_lambda': 0.05}, mean: 0.94164, std: 0.01120...]
参数的最佳取值：{'reg_alpha': 1, 'reg_lambda': 1}
最佳模型得分:0.9441561344357595

由输出结果可知参数的最佳取值：{'reg_alpha': 1, 'reg_lambda': 1}。

6、最后就是learning_rate，一般这时候要调小学习率来测试：

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 1}

运行后的结果为：

[Parallel(n_jobs=4)]: Done  25 out of  25 | elapsed:  1.1min finished
每轮迭代运行结果:[mean: 0.93675, std: 0.01080, params: {'learning_rate': 0.01}, mean: 0.94229, std: 0.01138, params: {'learning_rate': 0.05}, mean: 0.94110, std: 0.01066, params: {'learning_rate': 0.07}, mean: 0.94416, std: 0.01037, params: {'learning_rate': 0.1}, mean: 0.93985, std: 0.01109, params: {'learning_rate': 0.2}]
参数的最佳取值：{'learning_rate': 0.1}
最佳模型得分:0.9441561344357595

由输出结果可知参数的最佳取值：{'learning_rate': 0.1}。

我们可以很清楚地看到，随着参数的调优，最佳模型得分是不断提高的，这也从另一方面验证了调优确实是起到了一定的作用。不过，我们也可以注意到，其实最佳分数并没有提升太多。提醒一点，这个分数是根据前面设置的得分函数算出来的，即：

optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)

中的scoring='r2'。在实际情境中，我们可能需要利用各种不同的得分函数来评判模型的好坏。

最后，我们把得到的最佳参数组合扔到模型里训练，就可以得到预测的结果了：

def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程，下面的参数就是刚才调试出来的最佳参数组合
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)

    # 显示重要特征
    # plot_importance(model)
    # plt.show()

好了，调参的过程到这里就基本结束了。正如我在上面提到的一样，其实调参对于模型准确率的提高有一定的帮助，但这是有限的。最重要的还是要通过数据清洗，特征选择，特征融合，模型融合等手段来进行改进！

下面我就贴出完整代码（声明一点，我的代码质量不是很好，大家参考一下思路就行）：

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from hyperopt import hp

# 加载训练数据
def featureSet(data):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    yList = data.y.values
    return XList, yList

# 加载测试数据
def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    return XList


def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程
    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)

    # 显示重要特征
    # plot_importance(model)
    # plt.show()


if __name__ == '__main__':
    trainFilePath = 'dataset/soccer/train.csv'
    testFilePath = 'dataset/soccer/test.csv'
    data = pd.read_csv(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    # 预测最终的结果
    # trainandTest(X_train, y_train, X_test)

    """
    下面部分为调试参数的代码
    """

    #
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'n_estimators': [550, 575, 600, 650, 675]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 1}
    #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

更多干货，欢迎去听我的GitChat：

这里写图片描述

你可能感兴趣的

    正则表达式 深入浅出2--从java API开始raledongjava正则表达式thinking-in-java-小记
    入门Python数据分析最好的实战项目（二）路远机器学习数据挖掘python数据分析
    为什么说java中只有值传递dmegojava
    总结了17年初到18年初百场前端面试的面试经验(含答案)yuxiaoliang面试技巧面试htmlcssjavascript
    聊聊阿里社招面试，谈谈“野生”Java程序员学习的道路阿里云云栖社区编程语言java
    Python：模拟登录以获取新浪微博OAuth的code参数值07未网页爬虫微博采集新浪微博python
    如何翻译好一篇技术文章endaphp程序员翻译
    数据挖掘实战项目——北京二手房房价分析与非数据分析数据挖掘机器学习python

7 条评论
comeonbg · 2018年07月25日

你好，博主，关于xgboostRegression模型保存，不知道有处理办法吗？我尝试了model.save_model('0001.model''),失败了

赞 回复

不应该啊？
— 有故事 作者 · 2018年07月25日
添加回复
一年肖申克 · 2018年09月10日

请问xgboost能够处理序列回归任务吗？你这个比赛好像是预测一个得分值？我在做的一个任务是预测一段时间的天气情况。谢谢

赞 回复

shi
— 有故事 作者 · 2018年09月11日
添加回复
Wizard · 1月10日

博主你好，看你的代码是单个参数调校的，为什么不将所有参数一起调校呢？除了会增加计算量的考虑外，还有其他因素吗？

赞 回复

一起调参 怎么控制变量。多个一起变 怎么知道结果变化是哪个参数引起的呢？
— 有故事 作者 · 3月23日
添加回复
LinZhu · 5月20日

博主你好，我也觉得单个参数调试是有问题的，单个参数最优不一定组合最优，应该把所有参数范围给定，然后去找最好的组合，不过这样计算量会很大（我在实验中发现按这种方法得到的结果不是最优的），希望与你讨论>_<

赞 回复
Copyright © 2011-2019 SegmentFault. 当前呈现版本 19.02.27
浙ICP备 15005796号-2   浙公网安备 33010602002000号 杭州堆栈科技有限公司版权所有

CDN 存储服务由 又拍云 赞助提供

移动版 桌面版

