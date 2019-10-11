x_train2_temp = [keras.utils.to_catogorical(eg,num_classes=10000) for eg in train_data]

x_train2 =  [i.sum(axis=0) for i in x_train2_temp]
i
