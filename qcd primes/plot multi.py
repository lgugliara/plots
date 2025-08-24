import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from math import gcd

# ----- COMMON SETTINGS -----
N = 64
sigma = 64.0
cmap_phase = "twilight"
cmap_mag   = "viridis"
key_Q2 = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

# Indici e Δ
idx = np.arange(1, N+1)
I, J = np.meshgrid(idx, idx, indexing="ij")
Delta = np.abs(I - J)

# Kernel
K = np.exp(-(Delta**2) / (2.0 * sigma**2))

# Maschere number-theory
is_prime = np.ones(N+1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(np.sqrt(N))+1):
    if is_prime[p]:
        is_prime[p*p::p] = False
P_i = is_prime[I]; P_j = is_prime[J]

W_all        = np.ones((N, N), float)
W_both_prime = (P_i & P_j).astype(float)
W_one_prime  = (P_i ^ P_j).astype(float)
W_coprime    = np.fromfunction(
    lambda i,j: np.vectorize(lambda a,b: 1.0 if gcd(int(a+1), int(b+1))==1 else 0.0)(i, j),
    (N, N)
)

filters = [
    ("Base H (W=1)", W_all),
    ("Entrambi PRIMI", W_both_prime),
    ("XOR PRIMO", W_one_prime),
    ("COPRIMI (gcd=1)", W_coprime),
]

def compose_quadrants(mats, gap=2):
    N = mats[0].shape[0]
    top = np.concatenate([mats[0], np.full((N, gap), np.nan), mats[1]], axis=1)
    bottom = np.concatenate([mats[2], np.full((N, gap), np.nan), mats[3]], axis=1)
    composite = np.concatenate([top, np.full((gap, top.shape[1]), np.nan), bottom], axis=0)
    return composite

# -------------------- (1) 5 FRAME CHIAVE — FASE --------------------
phase_paths = []
for q2 in key_Q2:
    Hc = np.exp(1j * q2 * Delta) * K
    mats_phase = []
    for _, W in filters:
        Z = W * Hc
        phase = np.angle(Z)
        phase[W == 0] = np.nan
        mats_phase.append(phase)
    comp = compose_quadrants(mats_phase, gap=2)

    plt.figure(figsize=(7,7))
    plt.imshow(comp, origin="lower", cmap=cmap_phase, vmin=-np.pi, vmax=np.pi)
    plt.xticks([]); plt.yticks([])
    plt.title(fr"Fase $\arg(H_{{\rm eff}})$ — $Q^2={q2:.2f}$")
    plt.tight_layout(); plt.show()

# -------------------- (2) FFT 2D della mappa (Q^2 = π/2) --------------------
q2_fft = np.pi/2
H_fft = np.exp(1j * q2_fft * Delta) * K
F = fftshift(fft2(H_fft))
F_mag = np.abs(F)
F_show = np.log1p(F_mag / (F_mag.max() + 1e-12))

# Mappa spaziale (fase) usata per FFT
plt.figure(figsize=(6,6))
plt.imshow(np.angle(H_fft), origin="lower", cmap=cmap_phase, vmin=-np.pi, vmax=np.pi)
plt.xticks([]); plt.yticks([])
plt.title(rf"Mappa (fase) per FFT — $Q^2=\pi/2$")
plt.tight_layout(); plt.show()

# Spettro 2D
plt.figure(figsize=(6,6))
plt.imshow(F_show, origin="lower", cmap=cmap_mag)
plt.xticks([]); plt.yticks([])
plt.title(r"Spettro 2D (|FFT|, log) — $Q^2=\pi/2$")
plt.tight_layout(); plt.show()

# -------------------- (3) Ampiezza con J0(Q^2 Δ) --------------------
# implementiamo j0 evitando scipy (non disponibile talvolta): serie rapida per range moderato
def j0_series(x, terms=40):
    # J0(x) = sum_{k=0}^\infty [(-1)^k (x/2)^{2k} / (k!)^2 ]
    out = np.zeros_like(x, dtype=float)
    term = np.ones_like(x, dtype=float)
    out += term
    for k in range(1, terms):
        term *= - (x**2) / (4 * k * k)
        out += term
    return out

for q2 in key_Q2:
    A = np.abs(j0_series(q2 * Delta)) * K
    mats = [W * A for _, W in filters]
    compA = compose_quadrants(mats, gap=2)

    plt.figure(figsize=(7,7))
    plt.imshow(compA, origin="lower", cmap=cmap_mag)
    plt.xticks([]); plt.yticks([])
    plt.title(fr"Ampiezza $|J_0(Q^2 \Delta)|$ — $Q^2={q2:.2f}$")
    plt.tight_layout(); plt.show()