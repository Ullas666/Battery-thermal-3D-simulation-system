import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters (MUST MATCH battery_train_3d.py)
MODES = 12  # High Accuracy Setting
WIDTH = 32  # High Accuracy Setting
GRID_SIZE = 32
T_END = 0.5

# Physics Constants
ALPHA = 0.01
BASE_DEG_RATE = 0.05
THERMAL_SENSITIVITY = 0.4


# ---------------------------------------------------------
# 2. Model Architecture (Matches Training)
# ---------------------------------------------------------
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes, width):
        super(FNO3d, self).__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, modes, modes, modes)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.conv1 = SpectralConv3d(self.width, self.width, modes, modes, modes)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.conv2 = SpectralConv3d(self.width, self.width, modes, modes, modes)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.conv3 = SpectralConv3d(self.width, self.width, modes, modes, modes)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.conv0(x);
        x2 = self.w0(x);
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x);
        x2 = self.w1(x);
        x = F.gelu(x1 + x2)
        x1 = self.conv2(x);
        x2 = self.w2(x);
        x = F.gelu(x1 + x2)
        x1 = self.conv3(x);
        x2 = self.w3(x);
        x = x1 + x2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x);
        x = F.gelu(x);
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).permute(0, 4, 1, 2, 3).to(device)


# ---------------------------------------------------------
# 3. Load Model
# ---------------------------------------------------------
print("[INIT] Loading 3D Battery Model...")
if not os.path.exists('battery_fno_3d_model.pth'):
    print("[ERROR] Model file 'battery_fno_3d_model.pth' not found.")
    print("        Please run 'battery_train_3d.py' first.")
    sys.exit(1)

model = FNO3d(MODES, WIDTH).to(device)
try:
    model.load_state_dict(torch.load('battery_fno_3d_model.pth', map_location=device))
    print("[SUCCESS] Model loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load model. Mismatch dimensions?")
    print(f"        Ensure MODES={MODES} and WIDTH={WIDTH} in both scripts.")
    print(f"        System Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 4. Interactive Loop
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("      3D BATTERY DIAGNOSTIC ENGINE")
print("      (Manual, Gallery & Dynamic Modes)")
print("=" * 50)


def visualize_gallery():
    print("\n[GALLERY] Generating representative samples...")
    scenarios = [
        {"name": "Center Core Failure", "hot": 0.95, "pos": [0.5, 0.5, 0.5]},
        {"name": "Corner Impact", "hot": 0.9, "pos": [0.1, 0.1, 0.1]},
        {"name": "Surface Cooling Loss", "hot": 0.8, "pos": [0.5, 0.5, 0.9]},
    ]
    for i, scen in enumerate(scenarios):
        print(f" > Visualizing Sample {i + 1}: {scen['name']}")
        cx, cy, cz = scen['pos'];
        hot = scen['hot']
        s = GRID_SIZE;
        x = np.linspace(0, 1.0, s);
        y = np.linspace(0, 1.0, s);
        z = np.linspace(0, 1.0, s)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        input_vol = hot * np.exp(-20.0 * r2)
        input_tensor = torch.tensor(input_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)

        pred_soh = np.clip(np.nan_to_num(prediction[0, 1].cpu().numpy(), nan=1.0), 0.0, 1.0)
        min_soh_idx = np.unravel_index(np.argmin(pred_soh), pred_soh.shape)
        critical_z = min_soh_idx[2]

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(input_vol[:, :, critical_z].T, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
        ax1.set_title('Input Heat')

        ax2 = fig.add_subplot(1, 3, 2)
        im = ax2.imshow(pred_soh[:, :, critical_z].T, cmap='RdYlGn', origin='lower', extent=[0, 1, 0, 1], vmin=0.8,
                        vmax=1.0)
        ax2.set_title('Pred SOH')
        plt.colorbar(im, ax=ax2)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        x_idx, y_idx, z_idx = np.indices((s, s, s))

        # SMART PEELING LOGIC: Only show worst 10% of points
        min_val = np.min(pred_soh)
        threshold = min_val + 0.05
        mask = pred_soh < threshold

        if np.any(mask):
            xs, ys, zs = x_idx[mask], y_idx[mask], z_idx[mask]
            vals = pred_soh[mask]
            ax3.scatter(xs, ys, zs, c=vals, cmap='RdYlGn', alpha=0.3, marker='o', vmin=0.8, vmax=1.0)
        ax3.set_title("3D Fault Core")

        plt.suptitle(f"Sample: {scen['name']}")
        plt.show()


# --- DYNAMIC SIMULATION (Robust Animation) ---
def simulate_dynamic_run():
    print("\n[DYNAMIC] Simulating a moving thermal hotspot...")
    try:
        print("Define Start Point (0.0 - 1.0):")
        sx = float(input(" > Start X [0.2]: ") or 0.2)
        sy = float(input(" > Start Y [0.2]: ") or 0.2)
        sz = float(input(" > Start Z [0.1]: ") or 0.1)
        print("Define End Point (0.0 - 1.0):")
        ex = float(input(" > End X [0.8]: ") or 0.8)
        ey = float(input(" > End Y [0.8]: ") or 0.8)
        ez = float(input(" > End Z [0.9]: ") or 0.9)
        n_steps = int(input(" > Number of Steps [20]: ") or 20)
    except ValueError:
        sx, sy, sz = 0.2, 0.2, 0.1;
        ex, ey, ez = 0.8, 0.8, 0.9;
        n_steps = 20

    path_x = np.linspace(sx, ex, n_steps)
    path_y = np.linspace(sy, ey, n_steps)
    path_z = np.linspace(sz, ez, n_steps)
    trajectory = list(zip(path_x, path_y, path_z))

    print("[INFO] Playing simulation... (Close window to exit)")

    # Environment Check
    is_notebook = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    if is_notebook: from IPython.display import clear_output, display

    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    s = GRID_SIZE
    # Initialize plots
    dummy = np.zeros((GRID_SIZE, GRID_SIZE))
    im1 = ax1.imshow(dummy, cmap='RdYlGn', origin='lower', extent=[0, 1, 0, 1], vmin=0.5, vmax=1.0)
    cbar = plt.colorbar(im1, ax=ax1, label='SOH')
    line_path, = ax1.plot([], [], 'b--', alpha=0.3, label='Path')
    marker_curr, = ax1.plot([], [], 'cx', markersize=12, markeredgewidth=3, label='Current')
    ax1.legend(loc='upper right')

    ax2.set_xlim(0, s);
    ax2.set_ylim(0, s);
    ax2.set_zlim(0, s)
    ax2.set_xlabel('X');
    ax2.set_ylabel('Y');
    ax2.set_zlabel('Z')
    ax2.set_title('3D Fault View')

    # Draw static box and path
    edges = [[[0, s], [0, 0], [0, 0]], [[0, 0], [0, s], [0, 0]], [[0, 0], [0, 0], [0, s]], [[s, s], [0, s], [0, 0]],
             [[s, s], [0, 0], [0, s]], [[0, s], [s, s], [0, 0]], [[0, 0], [s, s], [0, s]], [[0, s], [0, 0], [s, s]],
             [[0, s], [s, s], [s, s]], [[s, s], [0, s], [s, s]], [[s, s], [s, s], [0, s]]]
    for e in edges: ax2.plot(e[0], e[1], e[2], 'k-', alpha=0.1)
    ax2.plot(np.array(path_x) * s, np.array(path_y) * s, np.array(path_z) * s, 'b-', alpha=0.5)

    scatter_collection = None
    marker_3d_curr, = ax2.plot([], [], [], 'rx', markersize=8)

    X, Y, Z = np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s), np.linspace(0, 1, s), indexing='ij')

    if not is_notebook:
        plt.show(block=False);
        plt.pause(0.1)

    try:
        for step in range(n_steps):
            cx, cy, cz = trajectory[step]
            hot = 0.95
            r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            # Safe input generation
            input_vol = hot * np.exp(-20.0 * r2)

            input_tensor = torch.tensor(input_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor)
            pred_soh = np.clip(np.nan_to_num(prediction[0, 1].cpu().numpy(), nan=1.0), 0.0, 1.0)

            # Debug Stats
            min_soh = np.min(pred_soh)
            print(f"Step {step + 1}: Pos({cx:.2f},{cy:.2f},{cz:.2f}) | Min SOH: {min_soh:.3f}")

            # 2D Update
            min_idx = np.unravel_index(np.argmin(pred_soh), pred_soh.shape)
            critical_z = min_idx[2]
            im1.set_data(pred_soh[:, :, critical_z].T)
            ax1.set_title(f'2D Critical Slice (Z={critical_z}/{s})')
            line_path.set_data(path_x, path_y)
            marker_curr.set_data([cx], [cy])

            # 3D Update
            marker_3d_curr.set_data([cx * s], [cy * s])
            marker_3d_curr.set_3d_properties([cz * s])

            if scatter_collection:
                scatter_collection.remove()
                scatter_collection = None

            x_idx, y_idx, z_idx = np.indices((s, s, s))

            # --- SMART PEELING LOGIC ---
            # If healthy (>0.98), show nothing. If damaged, show worst 5%
            if min_soh < 0.98:
                threshold = min_soh + 0.06
                mask = pred_soh < threshold
                if np.any(mask):
                    xs, ys, zs = x_idx[mask], y_idx[mask], z_idx[mask]
                    vals = pred_soh[mask]
                    # Downsample for speed
                    if len(xs) > 2000:
                        xs, ys, zs, vals = xs[::3], ys[::3], zs[::3], vals[::3]

                    scatter_collection = ax2.scatter(xs, ys, zs, c=vals, cmap='RdYlGn', alpha=0.3, marker='o', vmin=0.8,
                                                     vmax=1.0)

            if is_notebook:
                clear_output(wait=True)
                display(fig)
                time.sleep(0.1)
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.05)

        print("[DONE] Simulation finished.")
        if not is_notebook:
            plt.show(block=True)

    except KeyboardInterrupt:
        print("\n[STOP] Stopped.")

    plt.ioff()


while True:
    cmd = input("\n> Press ENTER for manual diag, 'samples' for gallery, 'dynamic' for moving sim, or 'exit': ")

    if cmd.lower() == 'samples':
        visualize_gallery()
        continue

    if cmd.lower() == 'dynamic':
        simulate_dynamic_run()
        continue

    if cmd.lower() in ['exit', 'quit', 'q']:
        break

    # [Manual Mode]
    print("\n--- 3D INPUT SIMULATION (Manual) ---")
    try:
        in_hot = float(input(" > Hotspot Temp (0.0-1.0) [0.9]: ") or 0.9)
        in_cx = float(input(" > Pos X (0-1) [0.5]: ") or 0.5)
        in_cy = float(input(" > Pos Y (0-1) [0.5]: ") or 0.5)
        in_cz = float(input(" > Pos Z (0-1) [0.5]: ") or 0.5)
    except ValueError:
        in_hot, in_cx, in_cy, in_cz = 0.9, 0.5, 0.5, 0.5

    s = GRID_SIZE
    x = np.linspace(0, 1.0, s);
    y = np.linspace(0, 1.0, s);
    z = np.linspace(0, 1.0, s)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r2 = (X - in_cx) ** 2 + (Y - in_cy) ** 2 + (Z - in_cz) ** 2

    # Matches training
    input_vol = in_hot * np.exp(-20.0 * r2)

    input_tensor = torch.tensor(input_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    pred_soh = np.clip(np.nan_to_num(prediction[0, 1].cpu().numpy()), 0.0, 1.0)
    min_soh = np.min(pred_soh)
    print(f"\n[RESULTS] Min SOH: {min_soh * 100:.2f}%")

    min_idx = np.unravel_index(np.argmin(pred_soh), pred_soh.shape)

    fig = plt.figure(figsize=(15, 6))

    # 2D Slice
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(pred_soh[:, :, min_idx[2]].T, cmap='RdYlGn', origin='lower', extent=[0, 1, 0, 1], vmin=0.8,
                     vmax=1.0)
    plt.colorbar(im1, ax=ax1, label='SOH')
    ax1.set_title(f'Manual Slice (Z={min_idx[2]}/{s})')
    ax1.set_xlabel('X');
    ax1.set_ylabel('Y')
    ax1.plot(in_cx, in_cy, 'cx', markersize=12, markeredgewidth=3, label='Input Center')
    ax1.legend()

    # 3D View
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    x_idx, y_idx, z_idx = np.indices((s, s, s))
    edges = [[[0, s], [0, 0], [0, 0]], [[0, 0], [0, s], [0, 0]], [[0, 0], [0, 0], [0, s]], [[s, s], [0, s], [0, 0]],
             [[s, s], [0, 0], [0, s]], [[0, s], [s, s], [0, 0]], [[0, 0], [s, s], [0, s]], [[0, s], [0, 0], [s, s]],
             [[0, s], [s, s], [s, s]], [[s, s], [0, s], [s, s]], [[s, s], [s, s], [0, s]]]
    for e in edges: ax2.plot(e[0], e[1], e[2], 'k-', alpha=0.1)

    min_val = np.min(pred_soh)
    threshold = min_val + 0.06
    mask = pred_soh < threshold
    if np.any(mask):
        xs, ys, zs = x_idx[mask], y_idx[mask], z_idx[mask]
        vals = pred_soh[mask]
        p = ax2.scatter(xs, ys, zs, c=vals, cmap='RdYlGn', marker='o', alpha=0.3, vmin=0.8, vmax=1.0)
        fig.colorbar(p, ax=ax2, label='SOH')
    ax2.set_title(f'3D Fault View')
    ax2.set_xlabel('X');
    ax2.set_ylabel('Y');
    ax2.set_zlabel('Z')
    ax2.set_xlim(0, s);
    ax2.set_ylim(0, s);
    ax2.set_zlim(0, s)

    print("[INFO] Displaying results. Close window to return.")
    plt.show(block=True)
    plt.close()