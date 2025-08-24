# Variante: fase che dipende anche da Δ = |i-j|, così compaiono onde che scorrono nello spazio

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# ------- SETTINGS -------x
N = 16
sigma = np.sqrt(N)
t_values = np.linspace(-np.pi, np.pi, int(16*np.sqrt(N)*np.pi**2))

# un ciclo completo
cmap = "twilight"  # colormap ciclica per la fase

# ------- MASKS -------
idx = np.arange(1, N+1)
I, J = np.meshgrid(idx, idx, indexing="ij")

is_prime = np.ones(N+1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(np.sqrt(N))+1):
    if is_prime[p]:
        is_prime[p*p::p] = False

P_i = is_prime[I]
P_j = is_prime[J]

W_all         = np.ones((N, N), dtype=float)
W_both_prime  = (P_i & P_j).astype(float)
W_one_prime   = ((P_i ^ P_j)).astype(float)
W_coprime     = np.fromfunction(
    lambda i, j: np.vectorize(lambda a, b: 1.0 if gcd(int(a+1), int(b+1))==1 else 0.0)(i, j),
    (N, N)
)

filters = [
    ("Base H (W=1)", W_all),
    ("Entrambi PRIMI", W_both_prime),
    ("XOR PRIMO", W_one_prime),
    ("COPRIMI (gcd=1)", W_coprime),
]

# ------- KERNEL -------
Delta = np.abs(I - J)
K = np.exp(- (Delta**2) / (2.0 * sigma**2))

def compose_grid(mats, gap=2):
    N = mats[0].shape[0]
    top = np.concatenate([mats[0], np.full((N, gap), np.nan), mats[1]], axis=1)
    bottom = np.concatenate([mats[2], np.full((N, gap), np.nan), mats[3]], axis=1)
    composite = np.concatenate([top, np.full((gap, top.shape[1]), np.nan), bottom], axis=0)
    return composite

fig = plt.figure(figsize=(7,7))
ax = plt.gca()
img = ax.imshow(np.zeros((2*N+2, 2*N+2)), origin="lower", cmap=cmap, vmin=-np.pi, vmax=np.pi)
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title(r"$\arg(H_{\mathrm{eff}})$ with $f(Q^2,\Delta)=e^{i Q^2 \Delta}$", fontsize=12, pad=10)

labels = [
    ax.text(0.02, 0.98, filters[0][0], transform=ax.transAxes, fontsize=9, va='top', ha='left', color="w"),
    ax.text(0.52, 0.98, filters[1][0], transform=ax.transAxes, fontsize=9, va='top', ha='left', color="w"),
    ax.text(0.02, 0.48, filters[2][0], transform=ax.transAxes, fontsize=9, va='top', ha='left', color="w"),
    ax.text(0.52, 0.48, filters[3][0], transform=ax.transAxes, fontsize=9, va='top', ha='left', color="w"),
]

def update_wave(k):
    Q2 = t_values[k]
    Hc = np.exp(1j * Q2 * Delta) * K
    mats = []
    for _, W in filters:
        Z = W * Hc
        phase = np.angle(Z)
        phase[W == 0] = np.nan
        mats.append(phase)
    composite = compose_grid(mats, gap=2)
    img.set_data(composite)
    title.set_text(rf"$\arg(H_{{\mathrm{{eff}}}})$ wave-phase,  $Q^2={Q2:.2f}$,  $T={(Q2/np.pi):.2f}$")
    return [img, title] + labels

ani = animation.FuncAnimation(fig, update_wave, frames=len(t_values), interval=12, blit=False, repeat=True)
plt.tight_layout()
plt.show()