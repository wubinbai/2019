
from keras.datasets import imdb
from keras.preprocessing import sequence


run1 = input('y or n for run1')
run2 = input('y or n for run2')
run3 = input('y or n for run3')
run4 = input('y or n for run4')
run5 = input('y or n for run5')
run6 = input('y or n for run6')
run7 = input('y or n for run7')
run8 = input('y or n for run8')
run9 = input('y or n for run9')
run10 = input('y or n for run10')
run11 = input('y or n for run11')
run12 = input('y or n for run12')

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Flatten, GRU, CuDNNGRU
from keras.optimizers import RMSprop
from keras import Model, Input

if run1 =='y':



    max_features = 10000  # number of words to consider as features
    maxlen = 500  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    #(input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)

    import numpy as np
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true


    (input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)
    # may or may not need this line: word_index=imdb.get_word_index(path="E:/ML/TextClassification/imdb_word_index.json")
    ## restore np.load for future normal usage
    #np.load = np_load_old


    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)
    # equiv. to functional api
    ######
    #input_tensor = Input(shape=(None,))
    #emb = Embedding(max_features,32)(input_tensor)
    #simplernn = SimpleRNN(32)(emb)
    #dense = Dense(1,activation='sigmoid')(simplernn)
    #output_tensor = dense
    #model_s = Model(input_tensor,output_tensor)
    #print(model_s.summary())
    ######

    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train,
                    epochs=8,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=2)

    # Let's display the training and validation loss and accuracy:
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



if run2=='y':
    
    max_features = 10000  # number of words to consider as features
    maxlen = 500  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    #(input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)

    import numpy as np
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true


    (input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)
    # may or may not need this line: word_index=imdb.get_word_index(path="E:/ML/TextClassification/imdb_word_index.json")
    ## restore np.load for future normal usage
    #np.load = np_load_old


    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)


    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=8,
                        batch_size=128,
                        validation_split=0.2)


    # In[12]:


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if run3 == 'y':

    
    ### prep


    import os

    data_dir = '/home/wb/Downloads/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values


    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    def generator(data, lookback, delay, min_index, max_index,
                  shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets



    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128


    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size



    ## finish prep

    ## model
        
    model = Sequential()
    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=5, # orig 20
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if run4 =='y':



    
    ### prep


    import os

    data_dir = '/home/wb/Downloads/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values


    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    def generator(data, lookback, delay, min_index, max_index,
                  shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets



    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128


    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size



    ## finish prep


    model = Sequential()
    model.add(CuDNNGRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=10, #orig = 20
                                  validation_data=val_gen,
                                  validation_steps=val_steps)



    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


## run 5 below
if run5 =='y':



    
    ### prep


    import os

    data_dir = '/home/wb/Downloads/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values


    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    def generator(data, lookback, delay, min_index, max_index,
                  shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets



    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128


    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size



    ## finish prep
    model = Sequential()
    model.add(GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=10,# orig 40
                              validation_data=val_gen,
                              validation_steps=val_steps)


# In[33]:


    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if run6=='y':

    ### prep


    import os

    data_dir = '/home/wb/Downloads/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values


    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    def generator(data, lookback, delay, min_index, max_index,
                  shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets



    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128


    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size



    ## finish prep


    model = Sequential()
    model.add(GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(GRU(64, activation='relu',
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)


    # Let's take a look at our results:

    # In[37]:


    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


if run7=='y':
    from keras import layers
    # Number of words to consider as features
    max_features = 10000
    # Cut texts after this number of words (among top max_features most common words)
    maxlen = 500

    #### load imdb ####
    print('Loading data...')
    #(input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)

    import numpy as np
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true


    (input_train, y_train), (input_test, y_test) = imdb.load_data(path='/home/wb/baidunetdiskdownload/imdb.npz',num_words=max_features)
    # may or may not need this line: word_index=imdb.get_word_index(path="E:/ML/TextClassification/imdb_word_index.json")
    ## restore np.load for future normal usage
    #np.load = np_load_old


    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)
    #### end of loading imdb ####




    # Load data
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = input_train
    x_test = input_test
    y_train = y_train
    y_test = y_test

    # Reverse sequences
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]

    # Pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

