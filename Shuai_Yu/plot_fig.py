import numpy as np
import matplotlib.pyplot as plt
import os, glob, re

# --------------------
# Parameters
# --------------------
Ma = 0.8
NX1 = 9
NX2 = 9
dt = 0.01
t_final = 500.0

OUTPUT_DIR = f"output_Ma_{Ma}_Nx_{NX1}_dt_{dt}/"
OUTPUT_DIR_RHO = os.path.join(OUTPUT_DIR, "rho")
OUTPUT_DIR_U1  = os.path.join(OUTPUT_DIR, "u1")
OUTPUT_DIR_U2  = os.path.join(OUTPUT_DIR, "u2")
OUTPUT_DIR_T   = os.path.join(OUTPUT_DIR, "T")

def _get_filenumber(path):
    """Get file number from filename like 'rho_100.txt'."""
    m = re.search(r'_(\d+)\.txt$', os.path.basename(path))
    return int(m.group(1)) if m else -1

def _pick_frame_paths(frame='last'):
    """Return the four file paths for the specified frame (rho/u1/u2/T). frame can be 'last' or a specific integer."""
    rho_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_RHO, "rho_*.txt")), key=_get_filenumber)
    u1_files  = sorted(glob.glob(os.path.join(OUTPUT_DIR_U1 , "u1_*.txt" )), key=_get_filenumber)
    u2_files  = sorted(glob.glob(os.path.join(OUTPUT_DIR_U2 , "u2_*.txt" )), key=_get_filenumber)
    T_files   = sorted(glob.glob(os.path.join(OUTPUT_DIR_T  , "T_*.txt"  )), key=_get_filenumber)

    if not (rho_files and u1_files and u2_files and T_files):
        raise FileNotFoundError("No data files found in one or more directories.")

    # Decide which frame to use
    if frame == 'last':
        idx = min(len(rho_files), len(u1_files), len(u2_files), len(T_files)) - 1
    else:
        # Find the file that matches the frame in the file number space (more robust, avoids sparse numbering)
        def _find_by_num(files, num):
            for p in files:
                if _get_filenumber(p) == num:
                    return p
            return None

        # If any file is missing, raise error
        paths = []
        for files, tag in [(rho_files,'rho'), (u1_files,'u1'), (u2_files,'u2'), (T_files,'T')]:
            p = _find_by_num(files, frame)
            if p is None:
                raise FileNotFoundError(f"Can't find {tag}_{frame}.txt")
            paths.append(p)
        return tuple(paths)

    # Return the paths for the selected index
    return rho_files[idx], u1_files[idx], u2_files[idx], T_files[idx]

def plot_snapshot(frame='last', save_path=None, dpi=180):
    """
    Plot a snapshot of the simulation at a specific frame.
    Parameters:
      - frame: 'last' or a specific integer (e.g., 100)
      - save_path: If given, save as this PNG file; otherwise, display directly
    """
    # Get file paths
    rho_path, u1_path, u2_path, T_path = _pick_frame_paths(frame)

    # Read data
    rho = np.loadtxt(rho_path)
    u1  = np.loadtxt(u1_path)
    u2  = np.loadtxt(u2_path)
    T   = np.loadtxt(T_path)
    speed = np.sqrt(u1**2 + u2**2)

    # Draw the meshgrid
    x = np.linspace(0, 1, NX1)
    y = np.linspace(0, 1, NX2)
    X, Y = np.meshgrid(x, y)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"2D Compressible Lid-Driven Cavity, Ma = {Ma}, t* = {t_final}", fontsize=16)

    im_rho   = axes[0,0].imshow(rho, extent=[0,1,0,1], origin='lower', cmap='viridis')
    im_T     = axes[0,1].imshow(T,   extent=[0,1,0,1], origin='lower', cmap='inferno')
    im_speed = axes[1,0].imshow(speed, extent=[0,1,0,1], origin='lower', cmap='plasma')

    # Plot the streamlines
    axes[1,1].streamplot(X, Y, u1, u2, color='black', density=1.5, linewidth=0.8)

    # Titles and axes
    axes[0,0].set_title("Density $\\rho^*$")
    axes[0,1].set_title("Temperature $T^*$")
    axes[1,0].set_title("Velocity amplitude $|\\mathbf{u}^*|$")
    axes[1,1].set_title("Velocity Streamlines")
    for ax in axes.flatten():
        ax.set_xlabel('$x_1^*$')
        ax.set_ylabel('$x_2^*$')
        ax.set_aspect('equal')

    # Create colorbars
    fig.colorbar(im_rho,   ax=axes[0,0])
    fig.colorbar(im_T,     ax=axes[0,1])
    fig.colorbar(im_speed, ax=axes[1,0])

    # Panel boundaries
    for ax in axes.flatten():
        ax.set_xlim(0,1); ax.set_ylim(0,1)

    plt.tight_layout(rect=[0,0.04,1,0.97])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved snapshot to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    
    plot_snapshot(frame='last')
