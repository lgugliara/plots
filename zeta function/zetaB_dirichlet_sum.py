# This script explores a two-parameter extension of the Dirichlet partial sum:
# Z_N(x; y1, y2) = sum_{n=1}^N n^{-x} * exp(-i y1 log n) * exp(-i y2 log n).
# It evaluates Z_N on a 2D grid of (y1, y2), then visualizes both the modulus
# (left panel) and the phase (right panel) as heatmaps.
# This provides a view of how interference patterns emerge when introducing
# two independent imaginary directions into the construction.

import numpy as np
import matplotlib.pyplot as plt

# Parametri di base
x = 0.5  # parte reale fissa
N = 64   # numero massimo della somma parziale
res = 256  # risoluzione della griglia
zoom = 10  # fattore di zoom per la visualizzazione

# Costruzione della griglia per y1, y2
y1_vals = np.linspace(-zoom, zoom, res)
y2_vals = np.linspace(-zoom, zoom, res)
Y1, Y2 = np.meshgrid(y1_vals, y2_vals)

# Calcolo della zeta parziale estesa
Z_mod = np.zeros_like(Y1, dtype=np.float64)
Z_arg = np.zeros_like(Y1, dtype=np.float64)

for n in range(1, N+1):
    logn = np.log(n)
    base = 1 / (n**x)
    phase = np.exp(-1j * Y1 * logn) * np.exp(-1j * Y2 * logn)
    term = base * phase
    if n == 1:
        Z = term
    else:
        Z += term

Z_mod = np.real(Z)
Z_arg = np.angle(Z)

# Visualizzazione
fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Modulo
mod_plot = axs[0].imshow(Z_mod, extent=(-zoom, zoom, -zoom, zoom), origin='lower', cmap='viridis')
axs[0].set_title(f'Modulo di $Z_{N}(s)$ su $(y_1, y_2)$')
axs[0].set_xlabel('$y_1$')
axs[0].set_ylabel('$y_2$')
fig.colorbar(mod_plot, ax=axs[0], shrink=0.8)

# Argomento
arg_plot = axs[1].imshow(Z_arg, extent=(-zoom, zoom, -zoom, zoom), origin='lower', cmap='twilight')
axs[1].set_title(f'Fase di $Z_{N}(s)$ su $(y_1, y_2)$')
axs[1].set_xlabel('$y_1$')
axs[1].set_ylabel('$y_2$')
fig.colorbar(arg_plot, ax=axs[1], shrink=0.8)

plt.tight_layout()
plt.show()
