import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# ------- SETTINGS -------
N = 128
sigma = 1.0 # float(N)
t_values = np.linspace(0, 2*np.pi, N)  # Q^2 real, 0..2π
cmap = "viridis"

# ------- NUMBER THEORY MASKS -------
idx = np.arange(1, N+1)
I, J = np.meshgrid(idx, idx, indexing="ij")

# sieve dei primi
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
K = np.exp(- ((I - J)**2) / (2.0 * sigma**2))

def compose_grid(mats, gap=2):
    N = mats[0].shape[0]
    top = np.concatenate([mats[0], np.full((N, gap), np.nan), mats[1]], axis=1)
    bottom = np.concatenate([mats[2], np.full((N, gap), np.nan), mats[3]], axis=1)
    composite = np.concatenate([top, np.full((gap, top.shape[1]), np.nan), bottom], axis=0)
    return composite

# ---------- ANIMATION 1: MODULUS ----------
vmax_mod = 0.0
for _, W in filters:
    vmax_mod = max(vmax_mod, np.abs(W * K).max())

fig1 = plt.figure(figsize=(8, 8))
ax1 = plt.gca()
img1 = ax1.imshow(np.zeros((2*N+2, 2*N+2)), origin="lower", cmap=cmap, vmin=-vmax_mod, vmax=vmax_mod)
ax1.set_xticks([]); ax1.set_yticks([])
title1 = ax1.set_title(r"$|H_{\mathrm{eff}}|$ with $f(Q^2)=e^{iQ^2}$ (modulus)", fontsize=13, pad=12)
labels1 = [
    ax1.text(0.02, 0.98, filters[0][0], transform=ax1.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax1.text(0.52, 0.98, filters[1][0], transform=ax1.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax1.text(0.02, 0.48, filters[2][0], transform=ax1.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax1.text(0.52, 0.48, filters[3][0], transform=ax1.transAxes, fontsize=10, va='top', ha='left', color="w"),
]

def update_mod(k):
    Q2 = t_values[k]
    Hc = np.exp(1j * Q2) * K
    mats = [np.real(W * Hc) for _, W in filters]
    composite = compose_grid(mats, gap=2)
    img1.set_data(composite)
    title1.set_text(rf"$|H_{{\mathrm{{eff}}}}|$ with $f(Q^2)=e^{{iQ^2}}$,  $Q^2={Q2:.2f}$ (modulus)")
    return [img1, title1] + labels1

ani1 = animation.FuncAnimation(fig1, update_mod, frames=len(t_values), interval=40, blit=False)
plt.tight_layout()
plt.show()

# ---------- ANIMATION 2: PHASE ----------
vmin_phase, vmax_phase = -np.pi, np.pi

fig2 = plt.figure(figsize=(8, 8))
ax2 = plt.gca()
img2 = ax2.imshow(np.zeros((2*N+2, 2*N+2)), origin="lower", cmap=cmap, vmin=vmin_phase, vmax=vmax_phase)
ax2.set_xticks([]); ax2.set_yticks([])
title2 = ax2.set_title(r"$\arg(H_{\mathrm{eff}})$ with $f(Q^2)=e^{iQ^2}$ (phase)", fontsize=13, pad=12)
labels2 = [
    ax2.text(0.02, 0.98, filters[0][0], transform=ax2.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax2.text(0.52, 0.98, filters[1][0], transform=ax2.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax2.text(0.02, 0.48, filters[2][0], transform=ax2.transAxes, fontsize=10, va='top', ha='left', color="w"),
    ax2.text(0.52, 0.48, filters[3][0], transform=ax2.transAxes, fontsize=10, va='top', ha='left', color="w"),
]

def update_phase(k):
    Q2 = t_values[k]
    Hc = np.exp(1j * Q2) * K
    mats = []
    for _, W in filters:
        Z = W * Hc
        phase = np.imag(Z)
        #phase[W == 0] = np.nan
        mats.append(phase)
    composite = compose_grid(mats, gap=2)
    img2.set_data(composite)
    title2.set_text(rf"$\arg(H_{{\mathrm{{eff}}}})$ with $f(Q^2)=e^{{iQ^2}}$,  $Q^2={Q2:.2f}$ (phase)")
    return [img2, title2] + labels2

ani2 = animation.FuncAnimation(fig2, update_phase, frames=len(t_values), interval=40, blit=False)
plt.tight_layout()
plt.show()