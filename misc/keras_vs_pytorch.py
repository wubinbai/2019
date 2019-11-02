下面用在 Keras 和 PyTorch 中定义的简单卷积网络来对二者进行对比：

Keras

model = Sequential()

model.add(Conv2D( 32, ( 3, 3), activation= 'relu', input_shape=( 32, 32, 3)))

model.add(MaxPool2D())

model.add(Conv2D( 16, ( 3, 3), activation= 'relu'))

model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense( 10, activation= 'softmax'))

PyTorch

classNet(nn.Module):
    def__init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 3, 32, 3)
        self.conv2 = nn.Conv2d( 32, 16, 3)
        self.fc1 = nn.Linear( 16* 6* 6, 10)
        self.pool = nn.MaxPool2d( 2, 2)
    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view( -1, 16* 6* 6)
            x = F.log_softmax(self.fc1(x), dim= -1)
            return x

model = Net()

上述代码片段显示了两个框架的些微不同。至于模型训练，它在 PyTorch 中需要大约 20 行代码，而在 Keras 中只需要一行。GPU 加速在 Keras 中可以进行隐式地处理，而 PyTorch 需要我们指定何时在 CPU 和 GPU 间迁移数据。


