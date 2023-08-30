import librosa

y, sr = librosa.load(librosa.ex('libri1'))
librosa.lpc(y, order=16)

import matplotlib.pyplot as plt
import scipy
import numpy as np


y, sr = librosa.load(librosa.ex('libri1'), duration=0.020)
a = librosa.lpc(y, order=2)
b = np.hstack([[0], -1 * a[1:]])
y_hat = scipy.signal.lfilter(b, [1], y)

fig, ax = plt.subplots()
ax.plot(y)
ax.plot(y_hat, linestyle='--')
ax.legend(['y', 'y_hat'])
ax.set_title('LP Model Forward Prediction')

plt.show()

print(a)

