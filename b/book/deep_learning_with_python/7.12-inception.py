from keras import layers
from keras.layers import Conv2D, AveragePooling2D
# assume x is a 4D tensor

branch_a = Conv2D(128, 1, activation='relu', strides=2)(x)
branch_b0 = Conv2D(128, 1, activation='relu')(x)
branch_b = Conv2D(128, 3, activation='relu' strides=2)(branch_b0)

branch_c0 = AveragePooling2D(3, strides=2, activation='relu')(x)
branch_c = Conv2D(128, 3 activation='relu', strides=2)(branch_c0)

branch_d = ......

output = layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis=-1)
