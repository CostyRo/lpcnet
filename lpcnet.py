import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from utils import *
from params import *
from signal_processing import *
from frame_rate_network import FrameRateNetwork
from sample_rate_network import SampleRateNetwork

class LPCNet(Model):
  def __init__(self):
    super().__init__()
    self.frame_net = FrameRateNetwork(BATCH_SIZE*N_FRAMES)
    self.sample_net = SampleRateNetwork(BATCH_SIZE*N_FRAMES)
    self.sampling = Lambda(
      lambda logits: STE(
        tf.reduce_sum(
          tf.nn.softmax(
            (logits+(-tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits),minval=1e-10,maxval=1.0)))))/GUMBEL_SOFTMAX_TEMPERATURE,
            axis=-1
          )
          *tf.range(DUAL_FC_LAYER_SIZE,dtype=tf.float32),
          axis=-1,
          keepdims=True
        )
      )
    )

  def call(self,input_):
    frames_features,lpc_coeffs = input_                                           # shape: ([BATCH_SIZE,N_FRAMES,N_FEATURES],[BATCH_SIZE,N_FRAMES,M])
    lpc_coeffs = tf.reshape(lpc_coeffs,(BATCH_SIZE*N_FRAMES,M))                   # shape: [BATCH_SIZE*N_FRAMES,M]
    frame_features = tf.reshape(frames_features,(BATCH_SIZE*N_FRAMES,N_FEATURES)) # shape: [BATCH_SIZE*N_FRAMES,N_FEATURES]
    frame_features = tf.expand_dims(frame_features,axis=-1)                       # shape: [BATCH_SIZE*N_FRAMES,N_FEATURES,1]
    frame_net_outputs = self.frame_net(frame_features)                            # shape: [BATCH_SIZE*N_FRAMES,F_SIZE]

    def condition(t,*_):
      return t < N_SAMPLES_IN_FRAME

    @tf.function
    def body(t,previous_s_t,e_t_minus_1,probabilities):
      p_t = tf.reduce_sum(lpc_coeffs*previous_s_t,axis=1,keepdims=True)         # shape: [BATCH_SIZE*N_FRAMES,1]
      p_t = tf.numpy_function(mu_law_quantize,(p_t,MU),tf.uint8)                # shape: [BATCH_SIZE*N_FRAMES,1]
      p_t = tf.ensure_shape(tf.cast(p_t,tf.float32),[BATCH_SIZE*N_FRAMES,1])    # shape: [BATCH_SIZE*N_FRAMES,1]
      s_t_minus_1 = previous_s_t[:,-1:]                                         # shape: [BATCH_SIZE*N_FRAMES,1]
      logits = self.sample_net((frame_net_outputs,p_t,s_t_minus_1,e_t_minus_1)) # shape: [BATCH_SIZE*N_FRAMES,MU]
      e_t = self.sampling(logits)                                               # shape: [BATCH_SIZE*N_FRAMES,1]
      s_t = p_t+e_t                                                             # shape: [BATCH_SIZE*N_FRAMES,1]

      previous_s_t = tf.concat([previous_s_t[:,1:],s_t],axis=1)                 # shape: [BATCH_SIZE*N_FRAMES,M]
      probabilities = tf.tensor_scatter_nd_update(                              # shape: [BATCH_SIZE*N_FRAMES,N_SAMPLES_IN_FRAME,MU]
        probabilities,
        [[i//probabilities.shape[1],i%probabilities.shape[1]] for i in range(logits.shape[0])],
        tf.reshape(logits,[-1,probabilities.shape[2]])
      ) 
      return t+1,previous_s_t,e_t,probabilities

    _,_,_,probabilities = tf.while_loop(
      condition,
      body,
      loop_vars=(
        tf.constant(0,dtype=tf.int32),                                          # t
        tf.zeros((BATCH_SIZE*N_FRAMES,M),dtype=tf.float32),                     # previous_s_t
        tf.zeros((BATCH_SIZE*N_FRAMES,1),dtype=tf.float32),                     # e_t_minus_1
        tf.zeros((BATCH_SIZE*N_FRAMES,N_SAMPLES_IN_FRAME,256),dtype=tf.float32) # probabilities
      )
    )

    return tf.reshape(probabilities,(BATCH_SIZE,N_FRAMES*N_SAMPLES_IN_FRAME,256)) # shape: [BATCH_SIZE,N_FRAMES*N_SAMPLES_IN_FRAME,MU]

  @property
  def trainable_variables(self):
    return\
      super().trainable_variables+\
      self.frame_net.trainable_variables+\
      self.sample_net.trainable_variables

if __name__ == "__main__":
  def test_differentiability():
    model = LPCNet()

    with tf.GradientTape() as tape:
      frames_features = tf.random.normal([BATCH_SIZE,N_FRAMES,N_FEATURES])
      lpc_coeffs = tf.random.normal([BATCH_SIZE,N_FRAMES,M])
      probabilities = model((frames_features,lpc_coeffs))
      loss = tf.reduce_mean(probabilities)

    gradients = tape.gradient(loss,model.trainable_variables)
    assert all(g is not None for g in gradients),"Some gradients are None"

  test_differentiability()
  print("All tests passed!")