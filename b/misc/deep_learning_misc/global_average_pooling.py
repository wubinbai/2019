import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from matplotlib import pyplot as plt
import numpy as np

# 为保证公平起见，使用相同的随机种子
np.random.seed(7)
batch_size = 32
# 迭代50次
epochs = 50
# 依照模型规定，图片大小被设定为224
IMAGE_SIZE = 224
# 17种花的分类
NUM_CLASSES = 17
TRAIN_PATH = '/home/yourname/Documents/tensorflow/images/17flowerclasses/train'
TEST_PATH = '/home/yourname/Documents/tensorflow/images/17flowerclasses/test'
FLOWER_CLASSES = ['Bluebell', 'ButterCup', 'ColtsFoot', 'Cowslip', 'Crocus', 'Daffodil', 'Daisy',
                  'Dandelion', 'Fritillary', 'Iris', 'LilyValley', 'Pansy', 'Snowdrop', 'Sunflower',
                  'Tigerlily', 'tulip', 'WindFlower']


def model(mode='fc'):
    if mode == 'fc':
        # FC层设定为含有512个参数的隐藏层
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='none')
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    elif mode == 'avg':
        # GAP层通过指定pooling='avg'来设定
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='avg')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    else:
        # GMP层通过指定pooling='max'来设定
        base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='max')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input=base_model.input, output=prediction)
    model.summary()
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=['accuracy'])

    # 使用数据增强
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        classes=FLOWER_CLASSES)
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                      classes=FLOWER_CLASSES)
    # 运行模型
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator)
    return history


fc_history = model('fc')
avg_history = model('avg')
max_history = model('max')


# 比较多种模型的精确度
plt.plot(fc_history.history['val_acc'])
plt.plot(avg_history.history['val_acc'])
plt.plot(max_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='lower right')
plt.grid(True)
plt.show()

# 比较多种模型的损失率
plt.plot(fc_history.history['val_loss'])
plt.plot(avg_history.history['val_loss'])
plt.plot(max_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='upper right')
plt.grid(True)
plt.show()
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten
from matplotlib import pyplot as plt
import numpy as np

# 为保证公平起见，使用相同的随机种子
np.random.seed(7)
batch_size = 32
# 迭代50次
epochs = 50
# 依照模型规定，图片大小被设定为224
IMAGE_SIZE = 224
# 17种花的分类
NUM_CLASSES = 17
TRAIN_PATH = '/home/hutao/Documents/tensorflow/images/17flowerclasses/train'
TEST_PATH = '/home/hutao/Documents/tensorflow/images/17flowerclasses/test'
FLOWER_CLASSES = ['Bluebell', 'ButterCup', 'ColtsFoot', 'Cowslip', 'Crocus', 'Daffodil', 'Daisy',
                  'Dandelion', 'Fritillary', 'Iris', 'LilyValley', 'Pansy', 'Snowdrop', 'Sunflower',
                  'Tigerlily', 'tulip', 'WindFlower']


def model(mode='fc'):
    if mode == 'fc':
        # FC层设定为含有512个参数的隐藏层
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='none')
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    elif mode == 'avg':
        # GAP层通过指定pooling='avg'来设定
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='avg')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)
    else:
        # GMP层通过指定pooling='max'来设定
        base_model = InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling='max')
        x = base_model.output
        prediction = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input=base_model.input, output=prediction)
    model.summary()
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=['accuracy'])

    # 使用数据增强
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                        classes=FLOWER_CLASSES)
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                      classes=FLOWER_CLASSES)
    # 运行模型
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=test_generator)
    return history


fc_history = model('fc')
avg_history = model('avg')
max_history = model('max')


# 比较多种模型的精确度
plt.plot(fc_history.history['val_acc'])
plt.plot(avg_history.history['val_acc'])
plt.plot(max_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='lower right')
plt.grid(True)
plt.show()

# 比较多种模型的损失率
plt.plot(fc_history.history['val_loss'])
plt.plot(avg_history.history['val_loss'])
plt.plot(max_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['FC', 'AVG', 'MAX'], loc='upper right')
plt.grid(True)
plt.show()

复制代码

　　先看看准确度的比较：

　　再看看损失度的比较：

　　很明显，在InceptionV3模型下，FC、GAP和GMP都表现很好，但可以看出GAP的表现依旧最好，其准确度普遍在90%以上，而另两种的准确度在80～90%之间。

 
四、结论

　　从本实验看出，在数据集有限的情况下，采用经典模型进行迁移学习时，GMP表现不太稳定，FC层由于训练参数过多，更易导致过拟合现象的发生，而GAP则表现稳定，优于FC层。当然具体情况具体分析，我们拿到数据集后，可以在几种方式中多训练测试，以寻求最优解决方案。

 
分类: 深度学习基础系列
好文要顶 收藏该文
可可心心
关注 - 0
粉丝 - 41
+加关注
1
0
« 上一篇： 深度学习基础系列（九）| Dropout VS Batch Normalization? 是时候放弃Dropout了
» 下一篇： Docker应用系列（六）| 如何去掉sudo及避免权限问题
posted @ 2018-11-26 11:16  可可心心  阅读(10166)  评论(0)  编辑 收藏
刷新评论刷新页面返回顶部
注册用户登录后才能发表评论，请 登录 或 注册， 访问 网站首页。
【推荐】超50万行VC++源码: 大型组态工控、电力仿真CAD与GIS源码库
【推荐】腾讯云热门云产品限时秒杀，爆款1核2G云服务器99元/年！
【推荐】阿里云双11返场来袭，热门产品低至一折等你来抢！
【推荐】防垃圾注册、薅羊毛，拦截率99%，顶象免费风控系统一键接入
【活动】京东云服务器_云主机低于1折，低价高性能产品备战双11
【优惠】七牛云采购嘉年华，云存储、CDN等云产品低至1折
相关博文：
· 深度学习基础系列（九）|DropoutVSBatchNormalization?是时候放弃Dropout了
· 基于深度学习和迁移学习的识花实践——利用VGG16的深度网络结构中的五轮卷积网络层和池化层，对每张图片得到一个4096维的特征向量，然后我们直接用这个特征向量替代原来的图片，再加若干层全连接的神经网络，对花朵数据集进行训练（属于模型迁移）
· 深度学习基础系列（七）|BatchNormalization
· 【深度学习系列】CNN模型的可视化
· 深度学习（十） GoogleNet
» 更多推荐...
阿里技术3年大合辑免费电子书一键下载
最新 IT 新闻:
· elementary OS 5.1 “Hera”发布，号称最美Linux发行版
· 中国北斗全球系统核心星座将于2019年年底部署完成
· 新西兰火山突然喷发：5死31伤 或有中国公民失联
· 德国科学家证明 X射线自由电子激光器可引发核聚变
· 三星会计欺诈案：三名高管被判两年牢狱
» 更多新闻...
公告
昵称： 可可心心
园龄： 3年10个月
粉丝： 41
关注： 0
+加关注
< 	2019年12月 	>
日 	一 	二 	三 	四 	五 	六
1 	2 	3 	4 	5 	6 	7
8 	9 	10 	11 	12 	13 	14
15 	16 	17 	18 	19 	20 	21
22 	23 	24 	25 	26 	27 	28
29 	30 	31 	1 	2 	3 	4
5 	6 	7 	8 	9 	10 	11
搜索
 
 
常用链接

    我的随笔
    我的评论
    我的参与
    最新评论
    我的标签

我的标签

    keras(5)
    tensorflow(4)
    tflite(2)
    ubuntu(2)
    zabbix(2)
    可训练参数数量(1)
    迁移学习(1)
    输出尺寸(1)
    ubuntu 18.04(1)
    Top1(1)
    更多

随笔分类

    Docker应用系列(6)
    hive
    java(1)
    keras&tensorflow(1)
    zabbix(7)
    深度学习基础系列(11)
    深度学习所遇问题集合(5)
    深度学习应用系列(4)
    深度学习源码分析系列

随笔档案

    2018年12月(2)
    2018年11月(4)
    2018年10月(4)
    2018年9月(10)
    2018年8月(6)
    2018年7月(1)
    2018年1月(1)
    2017年11月(1)
    2016年9月(1)
    2016年8月(1)
    2016年6月(3)
    2016年2月(7)

文章分类

    Android(4)
    hadoop(3)
    java(1)
    keras&tensorflow
    mongodb(4)
    mysql(1)
    netty(2)
    Security(2)

最新评论

    1. Re:Docker应用系列（一）| 构建Redis哨兵集群
    哥们你好，【三台主机上启动哨兵模式：sudo docker run -p 26379:26379 -v /data/redis_data/:/data --name redis-26379 -d do...
    --我还会怀念过去
    2. Re:深度学习应用系列（四）| 使用 TFLite Android构建自己的图像识别App
    你好，我想问一下retrain的命令是怎么放入命令行的。
    --慕辰132
    3. Re:如何用jmeter对websockt和protobuf进行压力测试
    大佬，我想问下那些插件在哪获取
    --季候海
    4. Re:深度学习应用系列（二） | 如何使用keras进行迁移学习，以训练和识别自己的图片集
    博主你好，result = model.predict(x, steps=1)显示错误 'Tensor' object has no attribute 'ndim'，请问是什么原因...
    --爬爬爬爬爬虫
    5. Re:HttpClient不同版本超时时间的设置
    多谢多谢，终于在你这里找到解决方法。
    --温酒斩华雄

阅读排行榜

    1. tensorflow 使用CPU而不使用GPU的问题解决(26364)
    2. 深度学习应用系列（一）| 在Ubuntu 18.04安装tensorflow 1.10 GPU版本(12289)
    3. 深度学习基础系列（三）| sigmoid、tanh和relu激活函数的直观解释(11616)
    4. 启动Ubuntu时出现 /dev/sda2 clean 和 /dev/sda2 recovering journal 现象的解决办法(10689)
    5. 深度学习基础系列（十）| Global Average Pooling是否可以替代全连接层？(10166)

评论排行榜

    1. 深度学习应用系列（三）| autokeras使用入门(15)
    2. 如何用jmeter对websockt和protobuf进行压力测试(11)
    3. tensorflow 使用CPU而不使用GPU的问题解决(2)
    4. 深度学习基础系列（六）| 权重初始化的选择(2)
    5. 深度学习基础系列（七）| Batch Normalization(1)

推荐排行榜

    1. 深度学习应用系列（四）| 使用 TFLite Android构建自己的图像识别App(3)
    2. 深度学习基础系列（二）| 常见的Top-1和Top-5有什么区别？(2)
    3. 深度学习应用系列（三）| autokeras使用入门(2)
    4. 深度学习基础系列（五）| 深入理解交叉熵函数及其在tensorflow和keras中的实现(2)
    5. Keras/tensorflow出现‘Loaded runtime CuDNN library: 7.0.5 but source was compiled with: 7.1.14’错误的解决办法(1)

Copyright © 2019 可可心心
Powered by .NET Core 3.1.0 on Linux
'''
