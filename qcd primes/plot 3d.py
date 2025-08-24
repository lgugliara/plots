import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# -------------------- SETTINGS --------------------
N = 32                 # 3D grid size
sigma = N              # wide Gaussian like your version
k_base = -np.pi        # starting k (will animate around this)
frames = 200           # show enough steps to perceive periodicity
k_span = 2*np.pi       # how much k varies over the animation
threshold_K = 0.5        # keep points where the radial envelope is strong
point_size = 8
alpha_pts = 0.2
cmap = "plasma"

# -------------------- 3D GRID --------------------
x = np.arange(1, N+1)
y = np.arange(1, N+1)
z = np.arange(1, N+1)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

c = (N+1)/2.0
R = np.sqrt((X-c)**2 + (Y-c)**2 + (Z-c)**2)

# Radial Gaussian envelope (static)
K_env = np.exp(-R**2/(2.0*sigma**2))

# -------------------- PRIME MASKS --------------------
# primes up to N
is_prime = np.ones(N+1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(np.sqrt(N))+1):
    if is_prime[p]:
        is_prime[p*p::p] = False

Pi = is_prime[X]
Pj = is_prime[Y]
Pk = is_prime[Z]

# 3 masks:
W_allprime = (Pi & Pj & Pk)                  # all three indices are prime
W_xorprime = (Pi ^ Pj ^ Pk) & ~(Pi & Pj & Pk) # exactly one is prime (odd parity, but exclude 3-prime case)
# gcd(i,j,k)=1 mask
def gcd3(a,b,c):
    return gcd(gcd(int(a), int(b)), int(c))
W_coprime = np.fromfunction(lambda i,j,k: np.vectorize(lambda a,b,c: 1 if gcd3(a+1,b+1,c+1)==1 else 0)(i,j,k),
                            (N,N,N), dtype=int).astype(bool)

masks = [
    ("All Primes", W_allprime),
    ("XOR Prime (exactly 1)", W_xorprime),
    ("Coprime gcd(i,j,k)=1", W_coprime),
]

paths = []

for title_mask, W in masks:
    # fixed subset of points (keep constant across frames)
    keep = (W) & (K_env > threshold_K)
    xs, ys, zs = X[keep], Y[keep], Z[keep]
    R_keep = R[keep]
    K_keep = K_env[keep]

    # initial values
    k0 = k_base
    vals0 = K_keep * np.cos(k0 * R_keep)

    # figure
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    pts = ax.scatter(xs, ys, zs, c=vals0, cmap=cmap, s=point_size, alpha=alpha_pts, vmin=-1, vmax=1)
    cb = fig.colorbar(pts, ax=ax, pad=0.02, label="H_eff")
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("k")
    ax.set_title(f"3D H_eff with '{title_mask}' — k anim")
    ax.view_init(elev=18, azim=38)
    plt.tight_layout()

    # animator
    def update(f):
        k_val = k_base + (f/ (frames-1)) * k_span
        vals = K_keep * np.cos(k_val * R_keep)
        pts.set_array(vals)  # update colors
        ax.set_title(f"3D H_eff — {title_mask} — k={k_val:.2f}")
        return [pts]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=12, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()