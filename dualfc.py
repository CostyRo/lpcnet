import tensorflow as tf
from tensorflow.keras.layers import Layer

class DualFC(Layer):
  def __init__(self,output_dim):
    super().__init__()
    self.output_dim = output_dim

  def build(self,input_shape):
    self.W1 = self.add_weight(
      shape=(input_shape[-1],self.output_dim),
      initializer="glorot_uniform",
      trainable=True
    )
    self.W2 = self.add_weight(
      shape=(input_shape[-1],self.output_dim),
      initializer="glorot_uniform",
      trainable=True
    )
    self.a1 = self.add_weight(
      shape=(self.output_dim,),
      initializer="ones",
      trainable=True
    )
    self.a2 = self.add_weight(
      shape=(self.output_dim,),
      initializer="ones",
      trainable=True
    )
    self.b = self.add_weight(
      shape=(self.output_dim,),
      initializer="zeros",
      trainable=True
    )
    
  def call(self,x):
    return \
    tf.multiply(self.a1,tf.tanh(tf.matmul(x,self.W1)))\
    +tf.multiply(self.a2,tf.tanh(tf.matmul(x,self.W2)))\
    +self.b