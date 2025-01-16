import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate,GRU,Softmax

from utils import *
from params import *
from dualfc import DualFC

class SampleRateNetwork(Model):
  def __init__(self,batch_size=BATCH_SIZE):
    super().__init__()
    self.batch_size = batch_size
    self.concat = Concatenate()
    self.gru1 = GRU(FIRST_GRU_LAYER_SIZE,return_sequences=True)
    self.gru2 = GRU(SECOND_GRU_LAYER_SIZE,return_sequences=True)
    self.dual_fc = DualFC(DUAL_FC_LAYER_SIZE)
    self.softmax = Softmax()

  def call(self,input_):             # shape: ([self.batch_size,F_SIZE],[self.batch_size,1],[self.batch_size,1]),[self.batch_size,1])
    e_t = self.concat(input_)        # shape: [self.batch_size,131(F_SIZE+1+1+1)]
    e_t = tf.expand_dims(e_t,axis=1) # shape: [self.batch_size,1,131]
    e_t = self.gru1(e_t)             # shape: [self.batch_size,1,FIRST_GRU_LAYER_SIZE]
    e_t = self.gru2(e_t)             # shape: [self.batch_size,1,SECOND_GRU_LAYER_SIZE]
    e_t = tf.squeeze(e_t,axis=1)     # shape: [self.batch_size,SECOND_GRU_LAYER_SIZE]
    e_t = self.dual_fc(e_t)          # shape: [self.batch_size,DUAL_FC_LAYER_SIZE]
    e_t = self.softmax(e_t)          # shape: [self.batch_size,DUAL_FC_LAYER_SIZE]

    return e_t

if __name__ == "__main__":
  def test_differentiability():
    model = SampleRateNetwork(batch_size=2)
    
    with tf.GradientTape() as tape:
      f = tf.random.normal([2,128])
      p_t = tf.random.uniform([2,1])
      s_t_minus_1 = tf.zeros([2,1])
      e_t_minus_1 = tf.zeros([2,1])
      
      output = model([f,p_t,s_t_minus_1,e_t_minus_1])
      loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss,model.trainable_variables)
    assert all(g is not None for g in gradients)

  test_differentiability()
  print("All tests passed!")