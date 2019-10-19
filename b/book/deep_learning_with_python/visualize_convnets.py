from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



print('first run 5.2.py')

from keras import models
from keras.models import load_model as lm
from keras.preprocessing import image 

m1=lm('cats_and_dogs_small_1.h5')
img_path = '/home/wb/Pictures/abc.png'
#'/home/wb/temp/cats_and_dogs_small/train/cats/cat.45.jpg'
img = image.load_img(img_path,target_size=(150,150)) 
img_tensor0 = image.img_to_array(img)
img_tensor1 = np.expand_dims(img_tensor0,axis=0)
img_tensor2 = img_tensor1/255
print(img_tensor2.shape)
plt.imshow(img_tensor2[0])

layer_inputs = m1.input
layer_outputs = [layer.output for layer in m1.layers[:8]]
activation_model = models.Model(inputs = layer_inputs, outputs = layer_outputs)
activations = activation_model.predict(img_tensor2)

def call_this():
    print('which layer you wanna view?')
    num = int(input('Enter the layer #: '))
    first_layer_activation = activations[num]
    plt.clf()
    for i in range(32):
        plt.matshow(first_layer_activation[0,:,:,i],cmap='viridis')

layer_names = []
for layer in m1.layers:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name, layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features//images_per_row
    display_grid = np.zeros((size*n_cols,images_per_row*size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()



