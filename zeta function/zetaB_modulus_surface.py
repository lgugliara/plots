# This script computes and visualizes a generalized "zeta^B" function with two
# imaginary parameters (t_i, t_j). The function is defined as a weighted series
# over n = 1..N, producing three components (real, i, j). The script evaluates
# its modulus |ζᴮ(t)| on a 2D parameter grid and renders the result as a 3D
# surface plot, showing the structure of the function in the (t_i, t_j) space.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define ranges for n and for parameters t_i and t_j
N = 256   # number of terms in the sum
res = 256  # resolution of the parameter space grid
zoom = 16  # range for t_i and t_j
t_i_vals = np.linspace(-zoom, zoom, res)
t_j_vals = np.linspace(-zoom, zoom, res)

# Create meshgrid for parameter space
T_i, T_j = np.meshgrid(t_i_vals, t_j_vals)
Z_real = np.zeros_like(T_i)
Z_i = np.zeros_like(T_i)
Z_j = np.zeros_like(T_i)

# Compute the zeta^B function over the parameter space
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

    factor = n ** (-1 / 19)

    Z_real += delta_real * factor
    Z_i += delta_i * factor
    Z_j += delta_j * factor

# Compute the modulus of the resulting vector field
Z_mod = np.sqrt(Z_real**2 + Z_i**2 + Z_j**2)

# Plot the modulus as a 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_i, T_j, Z_mod, cmap='viridis', edgecolor='none')
ax.set_xlabel('t_i (log-rotation on i)')
ax.set_ylabel('t_j (log-rotation on j)')
ax.set_zlabel('|ζᴮ(t)|')
ax.set_title('Modulus of ζᴮ(t) over (t_i, t_j) space')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()
