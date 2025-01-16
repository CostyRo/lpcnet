import tensorflow as tf

def STE(x):
  return x+tf.stop_gradient(tf.round(x)-x)