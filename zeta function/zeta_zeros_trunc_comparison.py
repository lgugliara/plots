# This script compares the non-trivial zeros of the Riemann zeta function ζ(s),
# computed with high precision using mpmath, with those obtained from the truncated series Z_N(s).
# The evaluation is restricted to the critical line s = 1/2 + i y, for y ∈ [-zoom, zoom].
# It plots the real and imaginary parts of the truncated Z_N(s),
# highlighting the zeros of the real part and comparing them with the true zeros of ζ(s).

import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta as mpmath_zeta
from mpmath import mp

# Set high precision
mp.dps = 100

# Parameters
x = 0.5
res = int(1e3)
zoom = int(1e2)
y_vals = np.linspace(-zoom, zoom, res)

# Compute the Riemann zeta function along the critical line
zeta_vals = np.array([complex(mpmath_zeta(complex(x, y))) for y in y_vals])
zeta_real = np.real(zeta_vals)
zeta_imag = np.imag(zeta_vals)

# Find non-trivial zeros (sign changes in the real part)
zero_crossings = np.where(np.diff(np.sign(zeta_real)))[0]
zero_ys = y_vals[zero_crossings]

# Truncated series Z_N(s) on the same grid
N = int(1e2)
Z_approx = np.zeros_like(y_vals, dtype=np.complex128)

for n in range(1, N + 1):
    logn = np.log(n)
    base = 1 / (n ** x)
    phase = np.exp(-1j * y_vals * logn)
    Z_approx += base * phase

Z_real = np.real(Z_approx)
Z_imag = np.imag(Z_approx)

# Find approximate zeros (sign changes in the real part of Z_N)
approx_zero_crossings = np.where(np.diff(np.sign(Z_real)))[0]
approx_zero_ys = y_vals[approx_zero_crossings]

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(y_vals, Z_real, label=f'Re[$Z_{{{N}}}(1/2 + i y)$]', alpha=0.8, linewidth=0.8)
plt.plot(y_vals, Z_imag, label=f'Im[$Z_{{{N}}}(1/2 + i y)$]', alpha=0.5, linewidth=0.8)
plt.scatter(approx_zero_ys, np.zeros_like(approx_zero_ys),
            color='blue', s=30, label='Approx. zeros (Re)', zorder=5)
plt.scatter(zero_ys, np.zeros_like(zero_ys),
            color='red', marker='x', s=40, label='Exact zeros (Re)', zorder=6)
plt.axhline(0, color='black', lw=0.5)
plt.title(f'Comparison between exact zeros and zeros of $Z_{{{N}}}(s)$ on the critical line')
plt.xlabel('$y$ in $s = 1/2 + i y$')
plt.ylabel('Function value')

# Log scale on y-axis to highlight zeros
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Direct comparison of the first N zeros
print(list(zip(zero_ys[:N], approx_zero_ys[:N])))