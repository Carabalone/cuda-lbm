import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

def visualize_velocity(
    max_steps, nx, ny, layout, input_dir, output_file, vmin, vmax, frame_step, fps
):
    """
    Generates an animation of velocity magnitude from LBM simulation data.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    dummy_frame = np.zeros((ny, nx))
    cax = ax.imshow(
        dummy_frame,
        origin="lower",
        extent=[0, nx, 0, ny],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(cax, label="Velocity Magnitude")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(frame_idx):
        filename = os.path.join(input_dir, f"velocity_{frame_idx}.bin")
        try:
            u_flat = np.fromfile(filename, dtype=np.float32)
            # print(f"Loaded {filename}, size: {u_flat.size}", end="\r")
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping frame {frame_idx}.")
            return (cax,)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            return (cax,)

        expected_elements = nx * ny * 2
        if u_flat.size != expected_elements:
            print(
                f"Unexpected data size in {filename}: "
                f"got {u_flat.size}, expected {expected_elements}. Skipping frame."
            )
            return (cax,)

        if layout == "aos":
            try:
                u_components = u_flat.reshape((ny, nx, 2))
                U = u_components[:, :, 0]
                V = u_components[:, :, 1]
            except ValueError as e:
                print(f"Error reshaping AoS data for {filename}: {e}. Check NX/NY.")
                return (cax,)
        elif layout == "soa":
            num_nodes_per_component = nx * ny
            try:
                U_flat = u_flat[:num_nodes_per_component]
                V_flat = u_flat[num_nodes_per_component:]
                U = U_flat.reshape((ny, nx))
                V = V_flat.reshape((ny, nx))
            except ValueError as e:
                print(f"Error reshaping SoA data for {filename}: {e}. Check NX/NY.")
                return (cax,)
        else:
            print(f"Unknown data layout: {layout}. Skipping frame.")
            return (cax,)

        velocity_magnitude = np.sqrt(U**2 + V**2)
        cax.set_array(velocity_magnitude)
        cax.set_clim(vmin, vmax)
        ax.set_title(f"Velocity Magnitude at Step {frame_idx}")
        if frame_idx % (frame_step * 10) == 0 or frame_idx == 0 :
             print(f"Processed frame {frame_idx} for {filename}", end="\r")
        return (cax,)

    frames_to_process = range(0, max_steps + 1, frame_step)
    
    if not list(frames_to_process):
        print("No frames to process with the given max_steps and frame_step. Exiting.")
        plt.close(fig)
        return

    ani = animation.FuncAnimation(
        fig, update, frames=frames_to_process, blit=True, interval=(1000 / fps)
    )

    try:
        ani.save(output_file, writer="ffmpeg", fps=fps)
        print(f"\nAnimation saved to {output_file}")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print(
            "Ensure ffmpeg is installed and accessible in your system's PATH."
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 2D LBM velocity data and save as MP4."
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=5000,
        help="Maximum timestep to process (default: 5000)",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=150,
        help="Number of grid points in x-direction (default: 150)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=100,
        help="Number of grid points in y-direction (default: 100)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="aos",
        choices=["aos", "soa"],
        help="Data layout of velocity components ('aos' or 'soa') (default: aos)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output/velocity/",
        help="Directory containing the velocity .bin files (default: output/velocity/)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="velocity_magnitude.mp4",
        help="Name of the output MP4 animation file (default: velocity_magnitude.mp4)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum value for the colorbar (default: 0.0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.05,
        help="Maximum value for the colorbar (default: 0.05)",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=100,
        help="Step between frames to process (default: 100)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the output video (default: 10.0)",
    )

    args = parser.parse_args()

    visualize_velocity(
        args.steps,
        args.nx,
        args.ny,
        args.layout,
        args.input_dir,
        args.output_file,
        args.vmin,
        args.vmax,
        args.frame_step,
        args.fps,
    )