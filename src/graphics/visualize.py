import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

steps = 9900
NX, NY = 400, 200  # Grid size

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((NY, NX)), cmap='jet', vmin=0, vmax=1)
plt.colorbar(cax)
ax.set_title('Velocity Magnitude Over Time')

def update(frame):
    print("updating")
    # Construct the filename. Adjust the extension to match your saved files.
    filename = f'output/velocity/velocity_{frame}.bin'
    try:
        # Read the binary file. We assume float32 was used to write the data.
        u = np.fromfile(filename, dtype=np.float32)
        print(f"loaded {filename}")
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return cax,

    # Check that the file has the expected number of elements:
    expected_elements = NX * NY * 2
    if u.size != expected_elements:
        print(f"Unexpected data size in {filename}: got {u.size}, expected {expected_elements}")
        return cax,

    # Reshape the flat array into a (NY, NX, 2) array.
    # Note: The simulation writes data with node ordering: node = y * NX + x.
    #       Thus, we reshape with NY rows and NX columns.
    u = u.reshape((NY, NX, 2))
    
    # Separate the x and y velocity components
    U = u[:, :, 0]
    V = u[:, :, 1]

    # Compute the velocity magnitude
    velocity_magnitude = np.sqrt(U**2 + V**2)

    # Optionally, update the color limits based on the data range:
    vmin = np.min(velocity_magnitude)
    vmax = np.max(velocity_magnitude)

    cax.set_array(velocity_magnitude)
    cax.set_clim(vmin, vmax)
    ax.set_title(f'Velocity Magnitude at Step {frame}')
    return cax,

# Create the animation. We update the plot every 100 timesteps.
ani = animation.FuncAnimation(fig, update, frames=range(0, steps, 100), 
                              blit=True, interval=100)
# Define frames per second (adjust as desired)
fps = 1000 / 100  # For instance, if interval=100 ms, fps = 10.

# Save the animation as a video using ffmpeg (make sure ffmpeg is installed)
ani.save('velocity_magnitude.mp4', writer='ffmpeg', fps=fps)

plt.show()
