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
t = np.arange(len(y))

print("w (input): ", w[:5])
print("y (output): ", y[:5])

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, w, label="White Gaussian Noise (w)", color="orange")
plt.title("Input: White Gaussian Noise (w)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, y, label="Generated realization (y)", color="blue")
plt.title("Output: Generated Realization from AR(3)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# autocorrelation function calculation on both w and y.
var_w = np.var(w)
var_y = np.var(y)

acorr_w = np.correlate(w, w, mode="full")
acorr_y = np.correlate(y, y, mode="full")

# normalize the autocorrelation result using each signal's variance.
acorr_w_norm = acorr_w / var_w
acorr_y_norm = acorr_y / var_y

# shift values for the autocorrelation function
tau = np.arange(-len(w) + 1, len(w))

plt.subplot(2, 1, 1)
plt.plot(tau, acorr_w_norm, label="Autocorrelation function (w)", color="pink")
plt.title("Autocorrelation of signal w")
plt.xlabel("Shift (tau)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, y, label="Autocorrelation function (y)", color="red")
plt.title("Autocorrelation of signal y")
plt.xlabel("Shift (tau)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Build the optimal 1-step predictor of yk
# iterate every value of y to predict the current step value
# \hat{y_k} = a1*y_k-1 + a2*y_k-2 + a3*y_k-3 (P = 3)
ye = np.zeros(len(y))
e = np.zeros(len(y))

# prediction starts from P (3)
for k in range(3, len(y)):
    y_hat_k = a[0] * y[k-1] + a[1] * y[k-2] + a[2] * y[k-3]

    # Compute prediction error
    e_k = y[k] - y_hat_k

    ye[k] = y_hat_k
    e[k] = e_k
# Plot original signal and the optimal 1-step predictor in the same plot
plt.figure(figsize=(10, 6))
plt.plot(t, y, label="Original signal (y)", color="green", alpha=0.7)
plt.plot(t, ye, label="Optimal 1-step predictor (y_hat)", color="red", linestyle="--", alpha=0.7)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot the autocovariance function of the prediction error
mean_e = np.mean(e)
autocov_e = [np.mean((e[:N-h] - mean_e) * (e[h:] - mean_e)) for h in range(N)]

plt.plot(np.arange(len(autocov_e)), autocov_e, label="Autocovariance of prediction error (e)", color="red")
plt.title("Autocovariance of prediction error")
plt.xlabel("Lag (h)")
plt.ylabel("Autocovariance")
plt.grid(True)
plt.legend()

plt.show()