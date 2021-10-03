import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Dense,LeakyReLU,Dropout,Flatten, InputLayer
from keras.models import Model

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(64,4,2,padding='same',input_shape=(28,28,1))
        self.leaky1 = LeakyReLU()
        self.drop1 = Dropout(0.3)

        self.conv2 = Conv2D(128,4,2,padding='same')
        self.leaky2 = LeakyReLU()
        self.drop2 = Dropout(0.3)

        self.flatten = Flatten()

        self.dense = Dense(1)

    def call(self,x):
        x = self.leaky1(self.conv1(x))
        x = self.drop1(x)

        x = self.leaky2(self.conv2(x))
        x = self.drop2(x)

        x = self.flatten(x)

        x = self.dense(x)

        return x