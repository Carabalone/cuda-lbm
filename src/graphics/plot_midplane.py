import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import glob
import os
import re

parser = argparse.ArgumentParser(description="Visualize midplane 3D velocity slices.")
parser.add_argument('-t', '--timesteps', type=int, default=None,
                    help="Maximum timestep to visualize (multiple of 100). If not set, auto-detects.")
parser.add_argument('--nx', type=int, default=256, help="Number of grid points in x-direction (default: 256)")
parser.add_argument('--ny', type=int, default=128, help="Number of grid points in y-direction (default: 128)")
args = parser.parse_args()

NX = args.nx
NY = args.ny
VEL_COMPONENTS = 3

SLICE_VEL_DIR = "output/midplane_velocity"
SLICE_DENS_DIR = "output/midplane_density"

def detect_timesteps(directory, prefix="velocity_", suffix=".bin"):
    files = glob.glob(os.path.join(directory, f"{prefix}*.bin"))
    timesteps = []
    for f in files:
        m = re.search(rf"{prefix}(\d+){suffix}", os.path.basename(f))
        if m:
            timesteps.append(int(m.group(1)))
    timesteps = sorted(timesteps)
    return timesteps

timesteps = detect_timesteps(SLICE_VEL_DIR)
if not timesteps:
    raise RuntimeError(f"No velocity slice files found in {SLICE_VEL_DIR}")

if args.timesteps is not None:
    max_t = args.timesteps
    timesteps = [t for t in timesteps if t <= max_t]
else:
    max_t = max(timesteps)

frame_interval = 100
frames = [t for t in timesteps if t % frame_interval == 0]

print(f"Visualizing {len(frames)} frames, from t={frames[0]} to t={frames[-1]}")
print(f"Grid size: NX={NX}, NY={NY}")

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((NY, NX)), origin='lower', extent=[0, NX, 0, NY], interpolation='bilinear',
                cmap='viridis', vmin=0, vmax=0.05)
plt.colorbar(cax)
ax.set_title('Midplane Velocity Magnitude')

def update(frame):
    print(f"updating frame {frame}", end="\r")
    filename = os.path.join(SLICE_VEL_DIR, f"velocity_{frame}.bin")
    try:
        u = np.fromfile(filename, dtype=np.float32)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return cax,

    expected_elements = NX * NY * VEL_COMPONENTS
    if u.size != expected_elements:
        print(f"Unexpected data size in {filename}: got {u.size}, expected {expected_elements}")
        return cax,

    u = u.reshape((NY, NX, VEL_COMPONENTS))
    U = u[:, :, 0]
    V = u[:, :, 1]
    W = u[:, :, 2]

    velocity_magnitude = np.sqrt(U**2 + V**2 + W**2)

    vmin = 0.0
    vmax = 0.08 
    cax.set_array(velocity_magnitude)
    cax.set_clim(vmin, vmax)
    ax.set_title(f'Midplane Velocity Magnitude at Step {frame}')
    return cax,

ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
fps = 1000 / 100

ani.save('midplane_velocity_magnitude.mp4', writer='ffmpeg', fps=fps, dpi=200)
plt.show()