import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Conv1D,Add,GlobalAveragePooling1D

from params import *

class FrameRateNetwork(Model):
  def __init__(self,batch_size=BATCH_SIZE):
    super().__init__()
    self.batch_size = batch_size
    self.conv1 = Conv1D(
      filters=FIRST_CONV_LAYER_FILTERS_SIZE,
      kernel_size=CONVOLUTION_KERNEL_SIZE,
      padding="same",
      activation=FIRST_CONV_LAYER_ACTIVATION_FUNCTION
    )
    self.conv2 = Conv1D(
      filters=SECOND_CONV_LAYER_FILTERS_SIZE,
      kernel_size=CONVOLUTION_KERNEL_SIZE,
      padding="same",
      activation=SECOND_CONV_LAYER_ACTIVATION_FUNCTION
    )
    self.add = Add()
    self.dense1 = Dense(
      FIRST_FC_LAYER_SIZE,
      activation=FIRST_FC_LAYER_ACTIVATION_FUNCTION
    )
    self.dense2 = Dense(
      SECOND_FC_LAYER_SIZE,
      activation=SECOND_CONV_LAYER_ACTIVATION_FUNCTION
    )
    self.global_pool = GlobalAveragePooling1D()

  def call(self,input_):     # shape: [self.batch_size,N_FEATURES,1]
    f = self.conv1(input_)   # shape: [self.batch_size,N_FEATURES,FIRST_CONV_LAYER_FILTERS_SIZE]
    f = self.conv2(f)        # shape: [self.batch_size,N_FEATURES,SECOND_CONV_LAYER_FILTERS_SIZE]
    f = self.add([f,input_]) # shape: [self.batch_size,N_FEATURES,SECOND_CONV_LAYER_FILTERS_SIZE]
    f = self.dense1(f)       # shape: [self.batch_size,N_FEATURES,FIRST_FC_LAYER_SIZE]
    f = self.dense2(f)       # shape: [self.batch_size,N_FEATURES,SECOND_FC_LAYER_SIZE]
    f = self.global_pool(f)  # shape: [self.batch_size,F_SIZE]

    return f

if __name__ == "__main__":
  def test_differentiability():
    model = FrameRateNetwork(batch_size=2)
    
    with tf.GradientTape() as tape:
      input_ = tf.random.normal([2,20,1])
      output = model(input_)
      loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss,model.trainable_variables)
    assert all(g is not None for g in gradients)

  test_differentiability()
  print("All tests passed!")