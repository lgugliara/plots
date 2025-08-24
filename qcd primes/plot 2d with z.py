# Re-run after state reset
import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Grid
N = 256
i = np.arange(1, N+1)
j = np.arange(1, N+1)
I, J = np.meshgrid(i, j, indexing="ij")

# Distanza radiale sul reticolo (centrata)
cx = (N+1)/2
cy = (N+1)/2
R = np.sqrt((I-cx)**2 + (J-cy)**2)

# Filtro W: coprimi su (i,j)
W = np.fromfunction(lambda a, b: np.vectorize(lambda x, y: 1.0 if gcd(int(x+1), int(y+1))==1 else 0.0)(a, b), (N, N))

# Kernel 2D: gaussiano * oscillazione radiale (tipo j0 ~ sinc)
sigma = N/4
k = 0.35 * np.pi  # frequenza radiale
K = np.exp(-(R**2)/(2*sigma**2)) * np.sinc((k/np.pi)*R)  # sinc(x) = sin(pi x)/(pi x)

H_eff = W * K

# 3D surface (single chart)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(I, J, H_eff, linewidth=0, antialiased=True)
ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("H_eff")
ax.set_title("3D surface: H_eff(i,j) = W(i,j) * Gaussian * sinc(radial)")
plt.tight_layout()
plt.show()
