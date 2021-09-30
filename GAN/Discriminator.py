import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Dense,LeakyReLU,Dropout,Flatten, InputLayer
from keras.models import Model

def create_discriminator():
    discriminator = keras.Sequential()
    discriminator.add(Conv2D(64,4,2,padding='same',input_shape=(28,28,1)))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128,4,2,padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Flatten())
    
    discriminator.add(Dense(1))

    return discriminator

