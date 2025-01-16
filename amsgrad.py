from tensorflow.keras.optimizers import Adam

from params import *

class AMSGrad(Adam):
  def __init__(self, learning_rate=ALPHA0, delta=DELTA, **kwargs):
    super().__init__(learning_rate=learning_rate,amsgrad=True,**kwargs)
    self.delta = delta
    self.batch_number = 0

  def _decayed_lr(self,var_dtype):
    base_lr = super()._decayed_lr(var_dtype)
    adjusted_lr = base_lr/(1.0+self.delta*self.batch_number)
    self.batch_number+=1
    return adjusted_lr