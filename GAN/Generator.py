import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense,LeakyReLU,Reshape,Conv2DTranspose,BatchNormalization,InputLayer

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = Dense(7*7*256,input_shape=(100,),use_bias=False)
        self.batch1 = BatchNormalization()
        self.leaky1 = LeakyReLU()

        self.reshape = Reshape((7,7,256))
        self.convt1 = Conv2DTranspose(128,5,1,padding='same',use_bias=False)
        self.batch2 = BatchNormalization()
        self.leaky2 = LeakyReLU()

        self.convt2 = Conv2DTranspose(64,5,2,padding='same',use_bias=False)
        self.batch3 = BatchNormalization()
        self.leaky3 = LeakyReLU()

        self.convt3 = Conv2DTranspose(1,5,2,padding='same',use_bias=False,activation='tanh')
    
    def call(self,x):
        x = self.leaky1(self.batch1(self.dense(x)))

        x = self.reshape(x)

        x = self.leaky2(self.batch2(self.convt1(x)))
        x = self.leaky3(self.batch3(self.convt2(x)))

        x = self.convt3(x)

        return x

