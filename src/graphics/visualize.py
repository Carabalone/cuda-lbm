import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

steps = 10000
NX, NY = 129, 129

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((NY, NX)), origin='lower', extent=[0, NX, 0, NY], cmap='jet', vmin=0, vmax=1)
plt.colorbar(cax)
ax.set_title('Velocity Magnitude Over Time')

def update(frame):
    print("updating")
    filename = f'output/velocity/velocity_{frame}.bin'
    try:
        u = np.fromfile(filename, dtype=np.float32)
        print(f"loaded {filename}")
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return cax,

    expected_elements = NX * NY * 2
    if u.size != expected_elements:
        print(f"Unexpected data size in {filename}: got {u.size}, expected {expected_elements}")
        return cax,

    # Reshape the flat array into a (NY, NX, 2) array.
    # Note: The simulation writes data with node ordering: node = y * NX + x.
    #       Thus, we reshape with NY rows and NX columns.
    u = u.reshape((NY, NX, 2))
    
    U = u[:, :, 0]
    V = u[:, :, 1]

    velocity_magnitude = np.sqrt(U**2 + V**2)

    vmin = np.min(velocity_magnitude)
    vmin = -0.001
    vmax = np.max(velocity_magnitude)
    vmax = 0.12

    cax.set_array(velocity_magnitude)
    cax.set_clim(vmin, vmax)
    ax.set_title(f'Velocity Magnitude at Step {frame}')
    return cax,

ani = animation.FuncAnimation(fig, update, frames=range(0, steps, 100), 
                              blit=True, interval=100)
fps = 1000 / 100 

ani.save('velocity_magnitude.mp4', writer='ffmpeg', fps=fps)

plt.show()
