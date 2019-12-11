
# x is a 4D tensor

y = layers.Conv2D(128, 3, activation='relu')(x)
y = layers.Conv2D(128, 3, activation='relu')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

residual = layers.Conv2D(1, strides=2)(x)

y = layers.add([y, residual])
