import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
import numpy as np

matplotlib.use('WebAgg')


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This is one of the most tricky parts of 3D plotting
    in matplotlib.

    Args:
      ax: a matplotlib axis object
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence a cube.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_points(csv_filepath, ax, title, color, size):
    """
    Plots 3D points from a CSV file onto a given matplotlib axis.
    Includes robust logic to ensure points are visible.
    """
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    try:
        # For debugging, let's see what's being loaded
        print(f"Attempting to plot {csv_filepath}...")
        df = pd.read_csv(csv_filepath)

        if df.empty or not all(col in df.columns for col in ["x", "y", "z"]):
            print(f"Warning: No data or missing x,y,z columns in {csv_filepath}")
            ax.text(0.5, 0.5, 0.5, "No data to plot", ha="center", va="center")
            return

        # Drop rows with non-numeric data that would cause scatter to fail
        df = df.dropna(subset=["x", "y", "z"])
        if df.empty:
            print(f"Warning: Data became empty after dropping NaN values in {csv_filepath}")
            return

        ax.scatter(df["x"], df["y"], df["z"], label=title, color=color, s=size)
        ax.legend()

        # --- Key Fix: Force axis scaling ---
        # This ensures the camera is focused on the data.
        set_axes_equal(ax)

        print(f"Successfully plotted {csv_filepath}")

    except FileNotFoundError:
        print(f"Error: File not found - {csv_filepath}", file=sys.stderr)
        ax.set_title(f"File Not Found:\n{csv_filepath}", color="red")
    except Exception as e:
        print(f"An error occurred with {csv_filepath}: {e}", file=sys.stderr)
        ax.set_title(f"Error plotting\n{csv_filepath}", color="red")


def main():
    """
    Main function to parse arguments and generate 3D point cloud plots.
    """
    parser = argparse.ArgumentParser(
        description="Plot 1 or 2 3D point clouds from CSV files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Plot a single file (will use default color/size)
  python plot_script.py init_vertices.csv

  # Plot two files side-by-side with custom titles and colors
  python plot_script.py init_vertices.csv sampled_vertices.csv \\
    --titles "Initial Cloud" "Final Cloud" \\
    --colors blue red
""",
    )
    # ... (rest of the main function is the same)
    parser.add_argument(
        "filepaths",
        metavar="FILE",
        nargs="+",
        help="Path(s) to the CSV file(s) to plot. Accepts 1 or 2 files.",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        help="Custom titles for the plots. Must match the number of files.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        help="Colors for the plots. Must match the number of files.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        help="Sizes for the points in each plot. Must match the number of files.",
    )

    args = parser.parse_args()
    num_plots = len(args.filepaths)

    if num_plots > 2:
        print(
            "Error: This script supports plotting a maximum of 2 files.",
            file=sys.stderr,
        )
        sys.exit(1)

    colors = args.colors if args.colors is not None else ["blue", "red"]
    sizes = args.sizes if args.sizes is not None else [5, 10]
    titles = args.titles

    for arg_name, arg_list in [
        ("titles", titles),
        ("colors", args.colors),
        ("sizes", args.sizes),
    ]:
        if arg_list and len(arg_list) != num_plots:
            print(
                f"Error: You provided {len(arg_list)} --{arg_name}, but {num_plots} files.",
                file=sys.stderr,
            )
            sys.exit(1)

    if num_plots == 1:
        fig = plt.figure(figsize=(8, 8))
        axes = [fig.add_subplot(111, projection="3d")]
    else:
        fig = plt.figure(figsize=(16, 8))
        axes = [
            fig.add_subplot(121, projection="3d"),
            fig.add_subplot(122, projection="3d"),
        ]

    for i, filepath in enumerate(args.filepaths):
        title = titles[i] if titles else filepath
        color = colors[i]
        size = sizes[i]
        plot_points(filepath, axes[i], title, color, size)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
