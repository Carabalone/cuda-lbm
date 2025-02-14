import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
NX, NY = 400, 200          # Domain dimensions
sub_steps = 10             # Number of particle updates per velocity file
output_video = "output/particle_sim.mp4"
# Collect velocity file names (assumed to be sorted in order)

velocity_files = sorted(
    glob.glob("output/velocity/velocity_*.bin"),
    key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[1])
)
print(velocity_files)
num_macro = len(velocity_files)
total_anim_frames = num_macro * sub_steps

# ---------------------------------------------------------
# Particle Initialization
# ---------------------------------------------------------
num_particles = 250
particle_x = np.random.rand(num_particles) * (NX - 370)
particle_y = np.random.rand(num_particles) * (NY - 1)

particle_vx = np.zeros(num_particles)
particle_vy = np.zeros(num_particles)

current_velocity = None

def load_velocity_field(filename):
    data = np.fromfile(filename, dtype=np.float32)
    expected = NY * NX * 2
    if data.size != expected:
        raise ValueError(f"File {filename} size mismatch: got {data.size}, expected {expected}")
    return data.reshape((NY, NX, 2))

# ---------------------------------------------------------
# Helper: Bilinear Interpolation Function
# ---------------------------------------------------------
def bilinear_interpolation(field, x, y):
    """
    Given a 2D field (shape: (NY, NX)), interpolate its value at float coordinates (x, y).
    x is column index, y is row index.
    """
    x = np.clip(x, 0, NX - 1)
    y = np.clip(y, 0, NY - 1)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, NX - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, NY - 1)
    dx = x - x0
    dy = y - y0

    val00 = field[y0, x0]
    val10 = field[y0, x1]
    val01 = field[y1, x0]
    val11 = field[y1, x1]
    
    val0 = val00 * (1 - dx) + val10 * dx
    val1 = val01 * (1 - dx) + val11 * dx
    return val0 * (1 - dy) + val1 * dy

# ---------------------------------------------------------
# Particle Update Function
# ---------------------------------------------------------
def update_particles(dt):
    global particle_x, particle_y, particle_vx, particle_vy, current_velocity
    
    for i in range(num_particles):
        ux = bilinear_interpolation(current_velocity[..., 0], particle_x[i], particle_y[i])
        uy = bilinear_interpolation(current_velocity[..., 1], particle_x[i], particle_y[i])

        particle_vx[i] = ux
        particle_vy[i] = uy
        
        new_x = particle_x[i] + dt * particle_vx[i]
        new_y = particle_y[i] + dt * particle_vy[i]
        
        if new_x < 0:
            new_x = -new_x
            particle_vx[i] = -particle_vx[i]
        elif new_x >= NX:
            overshoot = new_x - (NX - 1)
            new_x = NX - 1 - overshoot
            particle_vx[i] = -particle_vx[i]
        if new_y < 0:
            new_y = -new_y
            particle_vy[i] = -particle_vy[i]
        elif new_y >= NY:
            overshoot = new_y - (NY - 1)
            new_y = NY - 1 - overshoot
            particle_vy[i] = -particle_vy[i]
        
        particle_x[i] = new_x
        particle_y[i] = new_y

# ---------------------------------------------------------
# Matplotlib Animation Setup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

speed_field = np.zeros((NY, NX))
img = ax.imshow(speed_field, origin='lower', cmap='jet',
                extent=[0, NX, 0, NY], aspect='equal')
fig.colorbar(img, ax=ax, label='Flow Speed')

scatter = ax.scatter(particle_x, particle_y, s=10, c='blue')

ax.set_xlim(0, NX)
ax.set_ylim(0, NY)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Particle Advection in LBM Flow Field")

current_macro_index = -1

def animate(frame):
    """
    Animation update function.
    Each animation frame does one particle update sub-step.
    Every 'sub_steps' frames, load the next velocity field.
    """
    global current_velocity, current_macro_index
    
    # Determine which macro frame we are in:
    macro_index = frame // sub_steps
    print(f"macro_index: {macro_index} at frame: {frame}")
    sub_frame = frame % sub_steps  # not used explicitly, but available
    
    if macro_index != current_macro_index:
        current_macro_index = macro_index
        filename = velocity_files[macro_index]
        current_velocity = load_velocity_field(filename)

        speed = np.sqrt(current_velocity[..., 0]**2 + current_velocity[..., 1]**2)
        img.set_data(speed)
    
    dt = 20.0
    update_particles(dt)
    
    scatter.set_offsets(np.column_stack((particle_x, particle_y)))
    return scatter, img

print(total_anim_frames)
ani = animation.FuncAnimation(fig, animate, frames=total_anim_frames,
                              interval=5, blit=False)

fps = 60
writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
ani.save(output_video, writer=writer)

print(f"Simulation saved to {output_video}")

