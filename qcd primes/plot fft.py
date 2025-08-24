import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# --- SETTINGS ---
N = 1024
sigma = 42
frames = 256
Q2_values = np.logspace(-0.01, 0, frames) * np.pi  # come il tuo
cmap = "twilight"  # come il tuo

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

# easing per lo zoom: inizio molto zoom (finestra piccola), poi apertura
def zoom_fraction(t):
    return t**2

# --- PRECOMPUTE: per stabilizzare la scala colori ---
# Usiamo alcuni campioni per scegliere un vmax "robusto"
sample_idx = np.linspace(0, frames-1, 12, dtype=int)
vmax = 0.0
for k in sample_idx:
    q2 = Q2_values[k]
    X = W_coprime * K * np.cos(q2 * Delta)
    spec = fft_power(X)
    vmax = max(vmax, spec.max())
vmin = 0.0

# --- FIGURE ---
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(np.zeros((N,N)), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
title = ax.set_title("FFT breathing + zoom", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()

def update(k):
    q2 = Q2_values[k]
    X = W_coprime * K * np.cos(q2 * Delta)
    spec = fft_power(X)

    # crop centrale in funzione del frame
    t = k / (frames - 1)
    frac = zoom_fraction(t)  # in (0,1]
    half = int((N * frac) / 2)
    cx = cy = N // 2
    spec_crop = spec[cx-half:cx+half, cy-half:cy+half]

    im.set_data(spec_crop)
    title.set_text(rf"FFT |X| (log), $Q^2={q2:.3f}$  —  zoom: {frac*100:.1f}% lato")
    return [im, title]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
plt.tight_layout()
plt.show()