# Animated visualization of the Dirichlet partial sums on the critical line
# Z_N(1/2 + i y) = sum_{n=1}^N n^{-1/2} * exp(-i y log n).
# Blue vectors: individual terms (length 1/sqrt(n), rotating phase -y log n).
# Red vector: resultant of all terms. The orange curve traces the tip of this
# resultant as y increases, showing the deterministic "Dirichlet polygon walk"
# that illustrates constructive and destructive interference among the terms.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
x = 0.5
N = 100
logn = np.log(np.arange(1, N + 1))
base = 1 / np.sqrt(np.arange(1, N + 1))

# Figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Angular interference of $e^{-iy \\log n}/n^{1/2}$')
ax.add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--'))

# Blue term-vectors (Line2D objects)
lines = [ax.plot([0, 0], [0, 0], color='blue', alpha=0.3, lw=0.8)[0] for _ in range(N)]

# Red resultant vector
sum_line, = ax.plot([0, 0], [0, 0], color='red', lw=2)

# Path of the red tip
path_x, path_y = [], []
path_line, = ax.plot([], [], color='orange', lw=1, alpha=0.7)

def update(frame):
    y = frame * 0.01
    phases = np.exp(-1j * y * logn)
    vectors = base * phases
    sum_v = np.sum(vectors)

    # Update each blue vector
    for line, v in zip(lines, vectors):
        line.set_data([0, np.real(v)], [0, np.imag(v)])

    # Update red resultant
    sum_line.set_data([0, np.real(sum_v)], [0, np.imag(sum_v)])

    # Update path of the red tip
    path_x.append(np.real(sum_v))
    path_y.append(np.imag(sum_v))
    path_line.set_data(path_x, path_y)

    ax.set_title(f'Angular interference at y = {y:.2f}')

    return lines + [sum_line, path_line]

ani = FuncAnimation(fig, update, interval=12, blit=False)
plt.show()
