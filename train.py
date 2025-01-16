import numpy as np
import tensorflow as tf

from params import *
from lpcnet import LPCNet
from amsgrad import AMSGrad
from signal_processing import *

SIGNAL_LENGTH = N_FRAMES * 160
TRAIN_DATA_SIZE = 10 * BATCH_SIZE

def predict_with_lpc(frame,lpc_coeff):
  padded_frame = np.concatenate([np.zeros(len(lpc_coeff)),frame])
  predicted_signal = np.convolve(padded_frame,lpc_coeff)
  return predicted_signal[len(lpc_coeff)-1:len(lpc_coeff)-1+len(frame)]

def generate_synthetic_signals(size,signal_length):
  return np.random.randn(size,signal_length).astype(np.float32)

def generate_features_and_lpc(signals):
  features_list = []
  lpc_coeffs_list = []
  excitation_signals = []

  for signal in signals:
    filtered_signal = e_filter(signal)
    frames = filtered_signal.reshape(N_FRAMES,N_SAMPLES_IN_FRAME)

    lpc_coeffs = []
    features = []
    excitation = []

    for frame in frames:
      frame_ = mu_law_dequantize(mu_law_quantize(frame))
      feature = compute_features(frame_)
      lpc_coeff = compute_lpc_coefficients(frame_)
      
      predicted_signal = predict_with_lpc(frame_,lpc_coeff)

      excitation_frame = frame-predicted_signal

      features.append(feature)
      lpc_coeffs.append(lpc_coeff)
      excitation.append(mu_law_quantize(excitation_frame))

    features_list.append(np.array(features))
    lpc_coeffs_list.append(np.array(lpc_coeffs))
    excitation_signals.append(np.concatenate(excitation).astype(np.uint8))

  return (
    np.array(features_list,np.float32),
    np.array(lpc_coeffs_list,np.float32),
    np.array(excitation_signals),
  )

signals = generate_synthetic_signals(TRAIN_DATA_SIZE,SIGNAL_LENGTH)
dataset = tf.data.Dataset.from_tensor_slices(signals).map(
  lambda signal: tf.numpy_function(
    lambda s: (generate_features_and_lpc([s])),
    (signal,),
    (tf.float32,tf.float32,tf.uint8)
  )
).batch(BATCH_SIZE).shuffle(buffer_size=100).prefetch(tf.data.AUTOTUNE)

model = LPCNet()
optimizer = AMSGrad(learning_rate=ALPHA0,delta=DELTA)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

for epoch in range(1,EPOCHS+1):
  print(f"{epoch=}/{EPOCHS}")
  epoch_loss = 0

  for step,(feature_batch,lpc_coeff_batch,excitation) in enumerate(dataset,1):
    with tf.GradientTape() as tape:
      input_batch = (feature_batch,lpc_coeff_batch)

      y_pred = model(input_batch)

      loss = loss_fn(excitation,y_pred)

    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    epoch_loss += loss.numpy()
    print(f"{step=}, loss: {loss.numpy():.4f}")

  print(f"Epoch {epoch} loss: {epoch_loss/(step):.4f}")

model.save_weights("lpcnet_weights.h5")
print("Model training complete and weights saved.")