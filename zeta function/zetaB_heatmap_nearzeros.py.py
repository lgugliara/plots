# This script computes a generalized "zeta^B" function with two imaginary parameters (t_i, t_j).
# It evaluates the modulus |ζᴮ(t)| on a 2D grid and visualizes it as a heatmap.
# Additionally, it applies a threshold to highlight points where |ζᴮ(t)| is close to zero,
# overlaying these "near-zero" regions on top of the heatmap.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64
res = 1024
i_range = 32
j_range = 32
t_i_vals = np.linspace(-i_range, i_range, res)
t_j_vals = np.linspace(-j_range, j_range, res)
T_i, T_j = np.meshgrid(t_i_vals, t_j_vals)

# Initialize components
Z_real = np.zeros_like(T_i)
Z_i = np.zeros_like(T_i)
Z_j = np.zeros_like(T_i)

# Compute the zeta^B function
for n in range(1, N + 1):
    logn = np.log(n)
    phi_n = T_j * logn
    theta_n = T_i * logn

    sin_phi = np.sin(phi_n)
    cos_phi = np.cos(phi_n)
    cos_theta = np.cos(theta_n)
    sin_theta = np.sin(theta_n)

    delta_real = sin_phi * cos_theta
    delta_i = -sin_phi * sin_theta
    delta_j = -cos_phi

    factor = n ** -(1/2)

    Z_real += delta_real * factor
    Z_i += delta_i * factor
    Z_j += delta_j * factor

# Compute modulus
Z_mod = np.sqrt(Z_real**2 + Z_i**2 + Z_j**2)

# Threshold for near-zero detection
threshold = 0.5  # arbitrary threshold for "almost zero"
zero_mask = Z_mod < threshold

# Prepare the plot
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.imshow(Z_mod, extent=[-i_range, i_range, -j_range, j_range],
                    origin='lower', cmap='viridis', aspect='auto')
ax.set_title('Heatmap of |ζᴮ(t)| with overlay of near-zeros')
ax.set_xlabel('t_i (log-rotation on i)')
ax.set_ylabel('t_j (log-rotation on j)')
cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical')
cbar.set_label('|ζᴮ(t)|')

# Overlay the near-zeros
ti_zero_coords = T_i[zero_mask]
tj_zero_coords = T_j[zero_mask]
ax.scatter(ti_zero_coords, tj_zero_coords, color='white', s=0.5, label='Near-zero points')

ax.legend()
plt.tight_layout()
plt.show()
