import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import numpy as np

def plot_points(csv_filepath, title, ax, color, size=5):
    """Plots 3D points from a CSV file."""
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print(f"No data in {csv_filepath}")
            return
        ax.scatter(df['x'], df['y'], df['z'], label=title, color=color, s=size)
    except FileNotFoundError:
        print(f"Error: File not found - {csv_filepath}")
    except Exception as e:
        print(f"Error reading or plotting {csv_filepath}: {e}")

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))

    # Plot initial points
    ax1 = fig.add_subplot(121, projection='3d')
    plot_points("init_vertices.csv", "Initial Points", ax1, "blue", size=2)
    ax1.set_title("Initial Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    # ax1.legend()

    # Plot final points
    ax2 = fig.add_subplot(122, projection='3d')
    plot_points("sampled_vertices.csv", "Final (Eliminated) Points", ax2, "red", size=10)
    ax2.set_title("Final Point Cloud (After Elimination)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    # ax2.legend()
    
    # Try to set the same scale for both plots for better comparison
    # This is a bit tricky as scales might be very different if one is empty
    # For simplicity, matplotlib's auto-scaling is often fine.
    # If you want to enforce same limits, you'd calculate combined min/max
    # across both datasets for x, y, z and then ax1.set_xlim(min_x, max_x), etc.

    plt.tight_layout()
    plt.show()