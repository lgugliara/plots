# This script compares the truncated Dirichlet series Z_N(s) = sum_{n<=N} n^{-s}
# with an eta-series–based reference implementation of the Riemann zeta function ζ(s).
# Both are evaluated on a 2D grid over the complex plane (x ∈ [0, 1], y ∈ [-N*M, N*M]).
# The result is shown as a heatmap of the absolute error |Z_N(s) - ζ(s)|.

# NumPy-only, vectorized ζ comparison using the Dirichlet eta series as reference
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
res  = 256
N  = 8          # truncation for Z_N(s) (your "approx")
M  = res*2      # terms for eta_M(s) used as "reference" ζ (increase if you want tighter ref)
x_vals = np.linspace(0, 1, res)
y_vals = np.linspace(-M*N, M*N, res)
X, Y = np.meshgrid(x_vals, y_vals)          # shapes: (res, res)
s = X + 1j*Y

# --- Precompute n and logs for vectorization ---
nN = np.arange(1, N+1, dtype=np.float64)[:, None, None]     # shape: (N,1,1)
lognN = np.log(nN)

nM = np.arange(1, M+1, dtype=np.float64)[:, None, None]     # shape: (M,1,1)
lognM = np.log(nM)

# --- Truncated Dirichlet Z_N(s) = sum_{n<=N} n^{-s} ---
# n^{-s} = n^{-X} * exp(-i Y log n)
Z_trunc = np.add.reduce((nN**(-X)) * np.exp(-1j * Y * lognN), axis=0)

# --- Reference ζ(s) via eta_M(s) ---
# eta(s) = sum_{n>=1} (-1)^{n-1} / n^s, converges for Re(s)>0
alt_sign = ((-1.0) ** (nM - 1))  # shape (M,1,1)
eta_M = np.add.reduce(alt_sign * (nM**(-X)) * np.exp(-1j * Y * lognM), axis=0)

# ζ(s) = eta(s) / (1 - 2^{1-s})
denom = 1.0 - np.exp((1 - s) * np.log(2.0))
# Avoid blow-ups near s ≈ 1: mask tiny denominators
mask = np.abs(denom) < 1e-8
zeta_ref = np.empty_like(eta_M, dtype=np.complex128)
zeta_ref[~mask] = eta_M[~mask] / denom[~mask]
zeta_ref[mask] = np.nan + 1j*np.nan  # undefined/unstable zone

# --- Error heatmap ---
Z_error = np.abs(Z_trunc - zeta_ref)

plt.figure(figsize=(10, 8))
plt.imshow(Z_error, extent=[0, 1, -100, 100], aspect='auto',
           cmap='twilight', origin='lower')
plt.colorbar(label='Errore |Z_N(s) - ζ_ref(s)|')
plt.title(f'Errore tra $Z_{{{N}}}(s)$ (troncata) e ζ(s) (eta-based, M={M})')
plt.xlabel('Parte reale $x$')
plt.ylabel('Parte immaginaria $y$')
plt.grid(False)
plt.tight_layout()
plt.show()
