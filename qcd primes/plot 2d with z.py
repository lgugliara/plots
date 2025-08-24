import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import gcd

# -------- SETTINGS (balanced for speed/quality) --------
N = 256                  # grid size (surface is heavy; 160 is a good compromise)
sigma = (N*0.5)**2              # Gaussian width
k0 = -np.pi             # base frequency
k_span = np.pi      # animate k over ~one full 2π cycle
frames = 120
stride = 1              # plot every 'stride' point to lighten rendering
cmap = "twilight"

# -------- 2D GRID --------
i = np.arange(1, N+1)
j = np.arange(1, N+1)
I, J = np.meshgrid(i, j, indexing="ij")

cx = (N+1)/2
cy = (N+1)/2
R = np.sqrt((I-cx)**2 + (J-cy)**2)

# -------- FILTER W: coprimi(i,j) --------
W = np.fromfunction(lambda a, b: np.vectorize(
    lambda x, y: 1.0 if gcd(int(x+1), int(y+1))==1 else 0.0)(a, b),
    (N, N))

# -------- HELPERS --------
def H_eff_from_k(kval):
    K = np.exp(-(R**2)/(2*sigma**2)) * np.sinc((kval/np.pi)*R)  # sinc(x)=sin(pi x)/(pi x)
    return W * K

# initial H
H0 = H_eff_from_k(k0)

# normalize for color mapping
vmax = np.max(np.abs(H0))
vmin = -vmax

def colors_from(H):
    # map H to 0..1 for colormap
    Hn = (H - vmin) / (vmax - vmin + 1e-12)
    return plt.cm.get_cmap(cmap)(np.clip(Hn, 0, 1))

# -------- FIGURE / SURFACE --------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# subsample for performance
Is = I[::stride, ::stride]
Js = J[::stride, ::stride]
Hs = H0[::stride, ::stride]
Cs = colors_from(Hs)

surf = ax.plot_surface(Is, Js, Hs, facecolors=Cs, linewidth=0, antialiased=True, shade=False)
mappable_for_cb = plt.cm.ScalarMappable(cmap=cmap)
mappable_for_cb.set_array([vmin, vmax])
cb = fig.colorbar(mappable_for_cb, ax=ax, pad=0.02, label="H_eff")

ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("H_eff")
ax.set_title("3D surface: H_eff(i,j) = W(i,j) * Gaussian * sinc(radial) — anim k")
ax.view_init(elev=26, azim=38)
plt.tight_layout()

# -------- ANIMATION --------
def update(frame_idx):
    kval = k0 + (frame_idx/(frames-1))*k_span
    H = H_eff_from_k(kval)
    Hs = H[::stride, ::stride]
    Cs = colors_from(Hs)

    # Recreate surface is simpler/robust than patching facecolors in-place
    ax.clear()
    ax.plot_surface(Is, Js, Hs, facecolors=Cs, linewidth=0, antialiased=True, shade=False)
    ax.set_title(f"3D surface — k={kval:.3f}")
    return ax.collections

ani = animation.FuncAnimation(fig, update, frames=frames, interval=12, blit=False, repeat=False, cache_frame_data=False)
plt.tight_layout()
plt.show()