import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
N = 100
w = np.random.randn(N)
# coefficients of the autoregressive part
a = [1.0, 0.5, -0.5, -0.3]
print(w)
# y_k -> realizations.
# Find every value of y_k using previous values y_k-1 and
# gaussian white noise w_k

# lfilter used to compute difference equations
y = lfilter([1], a, w)
print("w (input): ", w[:5])
print("y (output): ", y[:5])