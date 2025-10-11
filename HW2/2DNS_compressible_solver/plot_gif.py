import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import sys

# ==============================================================================
# --- Simulation Parameters ---
# 【【【重要！】】】请确保这里的参数与您 C++ 代码中的 Parameters 类完全一致！
# ==============================================================================
# Grid and Time setup from your C++ code
NX1 = 8 + 1
NX2 = 8 + 1
T_FINAL = 100
dt = 0.0001
dt_sample = 1
NUM_OUTPUT_FILES = 1 + int(T_FINAL / dt_sample)

# ==============================================================================
# --- Script Configuration ---
# ==============================================================================
OUTPUT_DIR = "output"
OUTPUT_DIR_RHO = os.path.join(OUTPUT_DIR, "rho")
OUTPUT_DIR_U1 = os.path.join(OUTPUT_DIR, "u1")
OUTPUT_DIR_U2 = os.path.join(OUTPUT_DIR, "u2")
OUTPUT_DIR_T = os.path.join(OUTPUT_DIR, "T")
ANIMATION_FILENAME = "lid_driven_cavity.gif"
FPS = 15 # 动画的帧率 (Frames Per Second)

def animate_flow_field():
    """
    Scans the output directory, loads the simulation data, and creates
    an animation of the flow field evolution.
    """
    print("Starting visualization process...")

    # --- 1. Find and sort data files ---
    # glob.glob会找到所有匹配的文件，然后我们按数字顺序排序
    try:
        rho_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_RHO, "rho_*.txt")))
        u1_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_U1, "u1_*.txt")))
        u2_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_U2, "u2_*.txt")))
        T_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_T, "T_*.txt")))

        if not rho_files:
            print(f"Error: No data files found in '{OUTPUT_DIR}' directory.")
            print("Please run the C++ simulation first.")
            sys.exit(1)
            
        num_frames = len(rho_files)
        print(f"Found {num_frames} data snapshots to animate.")

    except Exception as e:
        print(f"Error while scanning for files: {e}")
        sys.exit(1)

    # --- 2. Setup the plot grid ---
    # 创建一个 2x2 的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("2D Compressible Lid-Driven Cavity Flow (Re=100, Ma=0.025)", fontsize=16)

    # 创建用于绘图的坐标网格
    x = np.linspace(0, 1, NX1)
    y = np.linspace(0, 1, NX2)
    X, Y = np.meshgrid(x, y)
    
    # 计算每个快照之间的时间间隔
    time_per_snapshot = T_FINAL / (num_frames - 1) if num_frames > 1 else 0

    # --- 3. Define the animation update function ---
    # 这个函数会在每一帧被调用
    def update(frame_index):
        # 清空所有子图，为绘制新一帧做准备
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        current_time = frame_index * time_per_snapshot
        print(f"Processing frame {frame_index+1}/{num_frames} (t* = {current_time:.3f})")

        # 加载当前帧的数据
        try:
            rho = np.loadtxt(rho_files[frame_index])
            u1 = np.loadtxt(u1_files[frame_index])
            u2 = np.loadtxt(u2_files[frame_index])
            T = np.loadtxt(T_files[frame_index])
        except Exception as e:
            print(f"\nError loading data for frame {frame_index}: {e}")
            return # Skip this frame if data is corrupted

        # --- 2. 【核心修改】初始化绘图元素 (只执行一次!) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_text = fig.suptitle("", fontsize=16) # 创建一个之后可以更新的标题

    x = np.linspace(0, 1, NX1)
    y = np.linspace(0, 1, NX2)
    X, Y = np.meshgrid(x, y)
    
    time_per_snapshot = T_FINAL / (num_frames - 1) if num_frames > 1 else 0

    # 加载第一帧的数据用于初始化
    rho_0 = np.loadtxt(rho_files[0])
    u1_0 = np.loadtxt(u1_files[0])
    u2_0 = np.loadtxt(u2_files[0])
    T_0 = np.loadtxt(T_files[0])
    speed_0 = np.sqrt(u1_0**2 + u2_0**2)

    # 创建所有绘图“艺术家” (Artist)
    # 使用 imshow 替代 contourf，因为它支持 set_data()
    im_rho = axes[0, 0].imshow(rho_0, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', animated=True)
    im_T = axes[0, 1].imshow(T_0, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', animated=True)
    im_speed = axes[1, 0].imshow(speed_0, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', animated=True)
    
    # 流线图比较特殊，最稳健的方法仍然是每帧重绘，但我们只清空它自己
    axes[1, 1].streamplot(X, Y, u1_0, u2_0, color='black', density=1.5, linewidth=0.8)

    # 【重要】只创建一次颜色条
    cbar_rho = fig.colorbar(im_rho, ax=axes[0, 0], label='$\\rho^*$')
    cbar_T = fig.colorbar(im_T, ax=axes[0, 1], label='$T^*$')
    cbar_speed = fig.colorbar(im_speed, ax=axes[1, 0], label='$|\\mathbf{u}^*|$')
    
    # 设置所有固定的属性 (标题，标签等)
    axes[0, 0].set_title("Density Field")
    axes[0, 1].set_title("Temperature Field")
    axes[1, 0].set_title("Speed Magnitude")
    axes[1, 1].set_title("Velocity Streamlines")
    for ax in axes.flatten():
        ax.set_xlabel('$x_1^*$')
        ax.set_ylabel('$x_2^*$')
        ax.set_aspect('equal')

    # --- 3. 定义动画更新函数 (现在变得非常简洁) ---
    def update(frame_index):
        current_time = frame_index * time_per_snapshot
        print(f"正在处理第 {frame_index+1}/{num_frames} 帧 (t* = {current_time:.3f})")

        # 加载新一帧的数据
        rho = np.loadtxt(rho_files[frame_index])
        u1 = np.loadtxt(u1_files[frame_index])
        u2 = np.loadtxt(u2_files[frame_index])
        T = np.loadtxt(T_files[frame_index])
        speed = np.sqrt(u1**2 + u2**2)
        
        # 【核心修改】只更新数据，不重新创建任何东西
        im_rho.set_data(rho)
        im_T.set_data(T)
        im_speed.set_data(speed)
        
        # 更新颜色条的范围，以匹配当前帧的数据范围
        im_rho.set_clim(vmin=rho.min(), vmax=rho.max())
        im_T.set_clim(vmin=T.min(), vmax=T.max())
        im_speed.set_clim(vmin=speed.min(), vmax=speed.max())

        # 对于流线图，我们仍然采用“清空再重画”的策略，但这只影响它自己
        ax_stream = axes[1, 1]
        ax_stream.clear() # 只清空这一个子图
        ax_stream.streamplot(X, Y, u1, u2, color='black', density=1.5, linewidth=0.8)
        # 清空后需要重新设置固定的属性
        ax_stream.set_title("Velocity Streamlines")
        ax_stream.set_xlabel('$x_1^*$')
        ax_stream.set_ylabel('$x_2^*$')
        ax_stream.set_xlim([0, 1])
        ax_stream.set_ylim([0, 1])
        ax_stream.set_aspect('equal')

        # 更新主标题
        time_text.set_text(f"2D Compressible Lid-Driven Cavity Flow (t* = {current_time:.3f})")
        
        # 返回被修改过的艺术家列表
        return [im_rho, im_T, im_speed, time_text]

    # --- 4. Create and save the animation ---
    print("\nCreating animation... This may take a few minutes.")
    # FuncAnimation 会调用 update 函数来生成动画的每一帧
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/FPS)

    # 保存为 GIF 文件
    try:
        ani.save(ANIMATION_FILENAME, writer='pillow', fps=FPS)
        print(f"\nAnimation successfully saved as '{ANIMATION_FILENAME}'")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure you have 'pillow' installed (`pip install pillow`)")
        print("You might also need to install ffmpeg for saving as .mp4.")

    # plt.show() # 如果你想在屏幕上实时观看动画，可以取消这行注释

if __name__ == '__main__':
    animate_flow_field()