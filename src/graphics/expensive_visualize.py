import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Simulation parameters
steps = 30000
NX, NY = 256, 128
save_interval = 100
start_frame = 100

# Cylinder parameters
x_center, y_center = 48, 64
r = 8

# Create cylinder mask
y, x = np.mgrid[0:NY, 0:NX]
cylinder_mask = (x - x_center)**2 + (y - y_center)**2 <= r**2

# Function to calculate vorticity
def compute_vorticity(u, v):
    du_dy = np.zeros_like(u)
    dv_dx = np.zeros_like(v)
    
    du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / 2.0
    dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / 2.0
    
    return dv_dx - du_dy

# Function to compute Q-criterion
def compute_q_criterion(u, v):
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    dv_dx = np.zeros_like(v)
    dv_dy = np.zeros_like(v)
    
    du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / 2.0
    du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / 2.0
    dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / 2.0
    dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / 2.0
    
    vorticity_squared = (dv_dx - du_dy)**2
    strain_squared = (du_dx)**2 + (dv_dy)**2 + 0.5*(du_dy + dv_dx)**2
    
    return 0.5 * (vorticity_squared - strain_squared)

# Get available frames
def get_available_frames():
    files = []
    for i in range(start_frame, steps + 1, save_interval):
        filename = f'output/velocity/velocity_{i}.bin'
        if os.path.exists(filename):
            files.append(i)
    return sorted(files)

available_frames = get_available_frames()
print(f"Found {len(available_frames)} velocity files")

# STEP 1: Calculate time-averaged velocity field
print("Calculating time-averaged velocity field...")
sum_u = np.zeros((NY, NX))
sum_v = np.zeros((NY, NX))
count = 0

for frame_number in available_frames:
    filename = f'output/velocity/velocity_{frame_number}.bin'
    try:
        velocity_data = np.fromfile(filename, dtype=np.float32)
        velocity_data = velocity_data.reshape((NY, NX, 2))
        
        u = velocity_data[:, :, 0]
        v = velocity_data[:, :, 1]
        
        u[cylinder_mask] = 0
        v[cylinder_mask] = 0
        
        sum_u += u
        sum_v += v
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count}/{len(available_frames)} frames for averaging")
            
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        continue

if count == 0:
    print("No valid velocity fields found.")
    exit()

avg_u = sum_u / count
avg_v = sum_v / count
print(f"Averaged {count} velocity fields")

# STEP 2: Create time-averaged streamline image
print("Creating time-averaged streamline image...")
fig_stream, ax_stream = plt.subplots(figsize=(5, 3), dpi=150)
ax_stream.set_aspect('equal')
ax_stream.set_xlim(0, NX)
ax_stream.set_ylim(0, NY)

# Calculate speed for background color
avg_speed = np.sqrt(avg_u**2 + avg_v**2)
vmax = np.percentile(avg_speed, 95)

# Create background image of average speed
im_stream = ax_stream.imshow(avg_speed, origin='lower', extent=[0, NX, 0, NY], 
                          cmap='viridis', alpha=0.7, vmin=0, vmax=vmax)

# Draw cylinder
circle = plt.Circle((x_center, y_center), r, fill=True, color='darkgray', 
                   alpha=0.8, ec='black', lw=1)
ax_stream.add_artist(circle)

# Create streamplot
density = 2  # Control streamline density
stream = ax_stream.streamplot(x, y, avg_u, avg_v, 
                           density=density,
                           color='white',
                           linewidth=0.7,
                           arrowsize=0.8)

ax_stream.set_title('Time-Averaged Flow Pattern', fontsize=10)
ax_stream.set_axis_off()  # Remove axes for cleaner look

# Save figure to memory
fig_stream.tight_layout(pad=0)
fig_stream.canvas.draw()
stream_image = np.frombuffer(fig_stream.canvas.tostring_rgb(), dtype=np.uint8)
stream_image = stream_image.reshape(fig_stream.canvas.get_width_height()[::-1] + (3,))
plt.close(fig_stream)  # Close the figure to save memory
print("Time-averaged streamline image created")

# STEP 3: Set up the main animation
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
axes = axes.flatten()

# Initialize plots
images = []
for i, ax in enumerate(axes):
    ax.set_aspect('equal')
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    
    # Draw cylinder outline except in bottom right panel (reserved for streamlines)
    if i != 3:  # Not the bottom right panel
        circle = plt.Circle((x_center, y_center), r, fill=False, color='white', linewidth=1.5)
        ax.add_artist(circle)
    
    if i == 0:  # Velocity magnitude
        im = ax.imshow(np.zeros((NY, NX)), origin='lower', extent=[0, NX, 0, NY], 
                      cmap='viridis', vmin=0, vmax=0.1)
        plt.colorbar(im, ax=ax)
        ax.set_title('Velocity Magnitude')
        images.append(im)
    
    elif i == 1:  # Vorticity
        im = ax.imshow(np.zeros((NY, NX)), origin='lower', extent=[0, NX, 0, NY], 
                      cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-0.01, vcenter=0, vmax=0.01))
        plt.colorbar(im, ax=ax)
        ax.set_title('Vorticity (∇×v)')
        images.append(im)
    
    elif i == 2:  # Q-criterion
        im = ax.imshow(np.zeros((NY, NX)), origin='lower', extent=[0, NX, 0, NY], 
                      cmap='magma', vmin=0, vmax=0.0001)
        plt.colorbar(im, ax=ax)
        ax.set_title('Q-criterion (Vortex Identification)')
        images.append(im)
    
    elif i == 3:  # Static streamline image
        # Display the pre-computed streamline image
        ax.imshow(stream_image, extent=[0, NX, 0, NY])
        ax.set_title('Time-Averaged Streamlines')
        ax.set_axis_off()
        # No need to add to images list as this won't be updated

plt.tight_layout()

def update(frame):
    frame_number = available_frames[frame]
    filename = f'output/velocity/velocity_{frame_number}.bin'
    
    print(f"Processing frame {frame+1}/{len(available_frames)}: step {frame_number}")
    
    try:
        velocity_data = np.fromfile(filename, dtype=np.float32)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return images
    
    expected_elements = NX * NY * 2
    if velocity_data.size != expected_elements:
        print(f"Unexpected data size in {filename}: got {velocity_data.size}, expected {expected_elements}")
        return images
    
    # Reshape the data
    velocity_data = velocity_data.reshape((NY, NX, 2))
    
    # Extract velocity components
    u = velocity_data[:, :, 0]
    v = velocity_data[:, :, 1]
    
    # Set velocities inside the cylinder to zero
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0
    
    # Compute derived quantities
    speed = np.sqrt(u**2 + v**2)
    vorticity = compute_vorticity(u, v)
    q_criterion = compute_q_criterion(u, v)
    
    # Update only the first three plots (velocity, vorticity, Q-criterion)
    images[0].set_array(speed)
    images[1].set_array(vorticity)
    images[2].set_array(q_criterion)
    
    plt.suptitle(f'Flow Past Cylinder, Step {frame_number}', fontsize=16)
    
    return images

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(len(available_frames)),
                             interval=200, blit=False)

# Save animation
fps = 10
ani.save('cylinder_flow_analysis.mp4', writer='ffmpeg', fps=fps, dpi=150)

plt.show()