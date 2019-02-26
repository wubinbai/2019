from keras.datasets import boston_housing, mnist, cifar10, imdb
(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train2, y_train2), (X_test2, y_test2) = boston_housing.load_data()
(X_train3, y_train3), (X_test3, y_test3) = cifar10.load_data()
(X_train4, y_train4), (X_test4, y_test4) = imdb.load_data(num_words=20000)
num_classes = 10




