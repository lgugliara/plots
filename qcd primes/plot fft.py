import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# --- SETTINGS ---
N = 1024
sigma = np.sqrt(N)
frames = 1000
Q2_values = np.linspace(-np.pi, np.pi, frames)
cmap = "magma"

# --- INDEX AND DELTA ---
idx = np.arange(1, N+1)
I, J = np.meshgrid(idx, idx, indexing="ij")
Delta = np.abs(I - J)

# --- KERNEL ---
K = np.exp(- (Delta**2) / (2.0 * sigma**2))

# --- MASK: coprimi ---
W_coprime = np.fromfunction(
    lambda i, j: np.vectorize(lambda a, b: 1.0 if gcd(int(a+1), int(b+1))==1 else 0.0)(i, j),
    (N, N)
)

def fft_power(img):
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    P = np.abs(F)
    return np.log1p(P)

# --- PREP FIGURE ---
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(np.zeros((N,N)), origin="lower", cmap=cmap, vmin=0, vmax=5)
title = ax.set_title("FFT breathing", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])

def update(k):
    q2 = Q2_values[k]
    X = W_coprime * K * np.cos(q2 * Delta)
    spec = fft_power(X)
    im.set_data(spec)
    title.set_text(rf"FFT |X| (log) — coprimi mask, $Q^2={q2:.2f}$")
    return [im, title]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=12, blit=False)
plt.tight_layout()
plt.show()