import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from GAN import GAN
import numpy as np
import matplotlib.pyplot as plt

(train_images,_),(_,_) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)
plt.show()

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


gan = GAN(BATCH_SIZE)


history = gan.fit(train_dataset,25)



x = [i for i in range(25)]
plt.plot(x,history['generator_loss'])
plt.plot(x,history['discriminator_loss'])
plt.title('change in loss over epochs')
plt.legend(['gen_loss','dis_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

noise = np.random.randn(32,100)
pred = gan.generator.predict(noise)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pred[i].squeeze(), cmap=plt.cm.binary)
plt.show()