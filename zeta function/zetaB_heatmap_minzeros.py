# This script computes the generalized "zeta^B" function with two imaginary parameters (t_i, t_j).
# It evaluates the modulus |ζᴮ(t)| on a 2D grid, visualizes it as a heatmap, and identifies 
# candidate near-zero regions. The script reports the global minimum of |ζᴮ(t)| and extracts 
# discrete t_j values where the modulus is below a chosen threshold.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64
res = 1024
zoom = 16
t_i_vals = np.linspace(-zoom, zoom, res)
t_j_vals = np.linspace(-zoom, zoom, res)
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

    factor = n ** (-1 / 3)

    Z_real += delta_real * factor
    Z_i += delta_i * factor
    Z_j += delta_j * factor

# Compute modulus
Z_mod = np.sqrt(Z_real**2 + Z_i**2 + Z_j**2)

# Plot as 2D heatmap
fig, ax = plt.subplots(figsize=(10, 10))
heatmap = ax.imshow(Z_mod, extent=[-zoom, zoom, -zoom, zoom],
                    origin='lower', cmap='twilight', aspect='equal')
ax.set_title('Heatmap of |ζᴮ(t)| over (t_i, t_j) space')
ax.set_xlabel('t_i (log-rotation on i)')
ax.set_ylabel('t_j (log-rotation on j)')
cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', shrink=0.75)
cbar.set_label('|ζᴮ(t)|')
plt.tight_layout()
plt.show()

# Find the minimum
min_val = np.min(Z_mod)
min_idx = np.unravel_index(np.argmin(Z_mod), Z_mod.shape)
t_i_min = t_i_vals[min_idx[1]]
t_j_min = t_j_vals[min_idx[0]]

print(min_val, t_i_min, t_j_min)

# Extract unique t_j values where Z_mod is near-zero
threshold = 3**(1/3)
unique_tj = np.unique(T_j[Z_mod < threshold])
rounded_tj = np.round(unique_tj, decimals=3)
unique_discrete_tj = np.unique(rounded_tj)

print(unique_discrete_tj[:30])  # First 30 for readability
