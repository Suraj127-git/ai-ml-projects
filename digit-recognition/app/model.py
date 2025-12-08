import os
import numpy as np
import tensorflow as tf

_model = None

def _path():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, "../model/digit_model.h5"))

def load_model():
    global _model
    _model = tf.keras.models.load_model(_path())

def predict(pixels):
    arr = np.array(pixels, dtype=np.float32)
    arr = arr.reshape(1, 28, 28, 1)
    p = _model.predict(arr)[0]
    return int(np.argmax(p)), [float(x) for x in p]
