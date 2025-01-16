


import numpy as np
from scipy.fftpack import ifft,dct
from statsmodels.tsa.stattools import levinson_durbin

from params import *

def e_filter(signal,alpha=ALPHA):
  return np.insert(1-alpha*signal[:-1],0,signal[0])

def d_filter(signal,alpha=ALPHA):
  return np.insert(1/(1-alpha*signal[:-1]),0,signal[0])

def mu_law_quantize(signal,mu=MU):
  quantized = np.sign(signal)*np.log1p(mu*np.abs(signal))/np.log1p(mu)
  quantized = ((quantized+1)/2*mu+0.5).astype(np.uint8)
  return quantized

def mu_law_dequantize(quantized,mu=MU):
  quantized = 2*(quantized/mu)-1
  signal = np.sign(quantized)*(1/mu)*(np.expm1(np.abs(quantized)*np.log1p(mu)))
  return signal

def bark_filterbank(n_fft,sr,n_bands=24,fmin=0,fmax=None):
  def hz2bark(hz):
    return 6*np.arcsinh(hz/600)

  def bark2hz(bark):
    return 600*np.sinh(bark/6)

  fmax = sr/2 if fmax is None else fmax
  hz_points = bark2hz(np.linspace(hz2bark(fmin),hz2bark(fmax),n_bands+2))
  fft_freqs = np.fft.rfftfreq(n_fft,1/sr)

  filterbank = np.zeros((n_bands,n_fft//2+1))
  for i in range(n_bands):
    lower,center,upper = hz_points[i:i+3]
    idx_lower_center = (fft_freqs >= lower) & (fft_freqs <= center)
    idx_center_upper = (fft_freqs > center) & (fft_freqs <= upper)

    filterbank[i,idx_lower_center] = (fft_freqs[idx_lower_center]-lower)/(center-lower)
    filterbank[i,idx_center_upper] = (upper-fft_freqs[idx_center_upper])/(upper-center)

  return filterbank

def bark_scale_coefficients(
  signal,
  sr=SR,
  n_coeffs=N_CEPSTRUM_COEFFS,
  n_fft=N_SAMPLES_IN_FRAME,
  n_filters=N_FILTERS
):
  power_spectrum = np.abs(dct(signal,n=n_fft)[:n_fft//2+1])**2
  bark_spectrum = (bark_filterbank(n_fft,sr,n_filters)@power_spectrum).flatten()
  return dct(np.log(bark_spectrum+1e-10))[:n_coeffs]

def period_and_correlation(signal,sr=SR,max_period=80):
  correlation = np.correlate(signal,signal,mode="full")
  correlation = correlation[len(correlation)//2:]
  period = np.argmax(correlation[max_period:])+max_period

  return period/sr,np.max(correlation[max_period:])

def compute_features(signal):
  return np.concatenate(
    (
      bark_scale_coefficients(signal),
      period_and_correlation(signal)
    )
  )

def compute_lpc_coefficients(features,M=M):
  bark_coeffs = features[:18]
  psd_bark = np.concatenate((bark_coeffs,bark_coeffs[::-1][1:-1]))
  autocorrelation = np.real(ifft(psd_bark))

  _,lpc_coefficients,_,_,_ = levinson_durbin(autocorrelation,M)
  return lpc_coefficients

if __name__ == "__main__":
  def test_sine_signal():
    duration = 0.1
    t = np.linspace(0,duration,int(SR*duration))
    freq = 1000
    signal = np.sin(2*np.pi*freq*t)

    coeffs = bark_scale_coefficients(signal)

    assert\
      np.all(np.isreal(coeffs)),\
      "Coefficients are not all real"
    assert\
      coeffs[0] == max(coeffs),\
      "First coefficient is not maximum"
    assert\
      np.all(coeffs[1:] < coeffs[0]),\
      "Subsequent coefficients do not decrease accordingly"

    print("Test sine signal: PASSED")

  def test_silence():
    signal = np.zeros(SR//10)

    coeffs = bark_scale_coefficients(signal)

    assert\
      np.all(np.isfinite(coeffs)),\
      "Coefficients contain NaN or Inf values"
    assert\
      np.all(np.abs(coeffs[1:]) < 1e-5),\
      "Coefficients are not close to zero"

    print("Test silence: PASSED")

  def test_white_noise():
    signal = np.random.normal(0,1,SR//10)

    coeffs = bark_scale_coefficients(signal)

    assert\
      np.all(np.isreal(coeffs)),"Coefficients are not all real"
    assert\
      np.std(np.diff(coeffs[1:])) < 3*np.mean(np.abs(np.diff(coeffs[1:]))),\
      "Variation between coefficients is too large"
    assert\
      np.std(coeffs) > 1,\
      "Coefficients do not reflect the diversity of white noise"

    print("Test white noise: PASSED")

  def test_sweep():
    duration = 0.1
    t = np.linspace(0,duration,int(SR*duration))
    freq = np.linspace(100,4000,len(t))
    signal = np.sin(2*np.pi*freq*t)

    coeffs = bark_scale_coefficients(signal)

    assert\
      np.all(np.isreal(coeffs)),\
      "Coefficients are not all real"
    assert\
      len(np.unique(coeffs)) > N_CEPSTRUM_COEFFS//2,\
      "Coefficients do not show enough variation"
    assert\
      coeffs[0] > coeffs[-1],\
      "First coefficient is not greater than the last"

    print("Test sweep: PASSED")

  def test_period_and_correlation():
    MAX_PERIOD = 80
    freq = 100
    t = np.linspace(0,0.01,160,endpoint=False)
 
    signal = np.sin(2*np.pi*freq*t)
    period, correlation = period_and_correlation(signal,SR)
    expected_period = 1/freq
    expected_corr = np.sum(signal**2)
    assert\
      np.allclose(period,expected_period,atol=1e-3),\
      f"Expected period {expected_period}, got {period}"
    assert\
      np.allclose(correlation+MAX_PERIOD,expected_corr,atol=1e-3),\
      f"Expected correlation {expected_corr}, got {correlation+MAX_PERIOD}"

    signal = np.ones(16000)
    period,correlation = period_and_correlation(signal,SR)
    expected_corr = SR-MAX_PERIOD
    assert\
      np.allclose(period,80/SR,atol=1e-3),\
      f"Expected period {80/SR}, got {period}"
    assert\
      np.allclose(correlation,expected_corr,atol=1e-3),\
      f"Expected correlation {expected_corr}, got {correlation}"

    signal = np.random.normal(size=16000)
    period,_ = period_and_correlation(signal,SR)
    assert\
      period >= MAX_PERIOD/SR,\
      "Expected period to be at least max_period/SR"

    signal = np.zeros(16000)
    period, correlation = period_and_correlation(signal,SR)
    assert\
      np.allclose(period,MAX_PERIOD/SR,atol=1e-3),\
      f"Expected period {MAX_PERIOD/SR}, got {period}"
    assert\
      np.allclose(correlation,0,atol=1e-3),\
      f"Expected correlation 0, got {correlation}"

    print("All period and correlation tests passed!")

  def test_e_filter():
    signal = np.arange(4)+1
    output = e_filter(signal)
    expected_output = np.array([1,1-ALPHA*1,1-ALPHA*2,1-ALPHA*3])
    assert\
      np.allclose(output,expected_output),\
      f"Test failed: Expected {expected_output}, got {output}"

    print("e_filter test passed!")

  def test_d_filter():
    signal = np.array([1,1-ALPHA*1,1-ALPHA*2,1-ALPHA*3])
    expected_output = np.array([1,6.66666667,1.14613181,0.62695925])
    output = d_filter(signal)
    assert\
      np.allclose(output,expected_output),\
      f"Test failed: Expected {expected_output}, got {output}"

    print("d_filter test passed!")

  def test_mu_law_functions():
    signal = np.linspace(-1, 1, 1000)
    quantized = mu_law_quantize(signal)
    dequantized = mu_law_dequantize(quantized)
    max_diff = np.max(np.abs(signal-dequantized))

    assert\
      np.all((quantized >= 0) & (quantized <= MU)),\
      "Quantized values out of range [0,MU]"
    assert\
      max_diff < 0.05,\
      f"Dequantized signal is not close enough to the original! Max diff: {max_diff}"
  
    print("Dequantization test passed!")

  test_sine_signal()
  test_silence()
  test_white_noise()
  test_sweep()
  test_period_and_correlation()
  test_e_filter()
  test_d_filter()
  test_mu_law_functions()