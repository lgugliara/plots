# This script compares the true Riemann zeta function ζ(s) with its truncated Dirichlet series
# approximations Z_N(s) at s = 0.5 + i y. For N = 1..512 and y ∈ [-64, 64], it plots real and
# imaginary parts side-by-side, showing how partial sums converge (or fail) to ζ(s) on the
# critical line.

import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta as mpmath_zeta
from mpmath import mp

# Set high precision for mpmath
mp.dps = 50

# Parameters
x = 0.5  # real part of s
res = 1024
zoom = 64
N_values = [1,2,4,8,16,32,64,128,256,512]
y_vals = np.linspace(-zoom, zoom, res)

# Vector of results for the true Riemann zeta function
zeta_true = np.array([complex(mpmath_zeta(complex(x, y))) for y in y_vals])
zeta_true_real = np.real(zeta_true)
zeta_true_imag = np.imag(zeta_true)

# Plot figure
fig, axs = plt.subplots(len(N_values), 2, figsize=(10, 2 * len(N_values)))

for idx, N in enumerate(N_values):
    # Partial sum approximation of zeta(s)
    zeta_approx = np.zeros_like(y_vals, dtype=np.complex128)
    for n in range(1, N + 1):
        logn = np.log(n)
        base = 1 / (n ** x)
        phase = np.exp(-1j * y_vals * logn)
        zeta_approx += base * phase

    # Real part comparison
    axs[idx, 0].plot(y_vals, np.real(zeta_approx), label=f'Re[$Z_N$], N={N}', linewidth=0.8)
    axs[idx, 0].plot(y_vals, zeta_true_real, '--', color='black', label='Re[ζ]', linewidth=0.8)
    axs[idx, 0].set_title(f'Real part: comparison $Z_{{{N}}}(s)$ vs ζ(s)')
    axs[idx, 0].legend()
    axs[idx, 0].grid(True)

    # Imaginary part comparison
    axs[idx, 1].plot(y_vals, np.imag(zeta_approx), label=f'Im[$Z_N$], N={N}', linewidth=0.8)
    axs[idx, 1].plot(y_vals, zeta_true_imag, '--', color='black', label='Im[ζ]', linewidth=0.8)
    axs[idx, 1].set_title(f'Imaginary part: comparison $Z_{{{N}}}(s)$ vs ζ(s)')
    axs[idx, 1].legend()
    axs[idx, 1].grid(True)

plt.tight_layout()
plt.show()
