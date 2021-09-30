import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense,LeakyReLU,Reshape,Conv2DTranspose,BatchNormalization,InputLayer

def create_generator():
    generator = keras.Sequential()
    generator.add(Dense(7*7*256,input_shape=(100,),use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Reshape((7,7,256)))
    generator.add(Conv2DTranspose(128,5,1,padding='same',use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Conv2DTranspose(64,5,2,padding='same',use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    generator.add(Conv2DTranspose(1,5,2,padding='same',activation='tanh',use_bias=False))

    return generator