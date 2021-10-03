import Discriminator as disc
import Generator as gen
import tensorflow as tf
from tensorflow import keras
from keras.losses import BinaryCrossentropy
from tqdm import tqdm

class GAN():
    def __init__(self,batch_size):
        self.generator = gen.Generator()
        self.discriminator = disc.Discriminator()
        self.loss = BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002,0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002,0.5)
        self.noise_dim = 100
        self.batch_size = batch_size

    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.loss(tf.ones_like(real_output),real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output),fake_output)
        return real_loss+fake_loss
    
    def generator_loss(self,fake_output):
        return self.loss(tf.ones_like(fake_output),fake_output)

    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.batch_size,self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_image = self.generator(noise,training=True)

            real_outputs = self.discriminator(images,training=True)
            fake_output = self.discriminator(fake_image,training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_outputs,fake_output)
        
        generator_gradients = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def fit(self,dataset,epochs):
        history = {'generator_loss':[],'discriminator_loss':[]}
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1} of {epochs}")
            for batch in tqdm(dataset):
                gen_loss, disc_loss = self.train_step(batch)
            print(f'generator loss: {gen_loss} disc_loss: {disc_loss}')
            history['generator_loss'].append(gen_loss)
            history['discriminator_loss'].append(disc_loss)

        return history