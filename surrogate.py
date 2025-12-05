import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{'SUCCESS' if torch.cuda.is_available() else 'INFO'}] Using device: {device}")

# 2. Physics & Data Parameters (High Accuracy)
MODES = 12      # Fourier Modes
WIDTH = 32      # Network Width
ALPHA = 0.01    # Thermal Diffusivity
L = 1.0         # Length
T_END = 0.5     # Simulation Time
NUM_SAMPLES = 1000 # Number of training samples
GRID_SIZE = 32    # 32x32x32 Voxel Grid

BASE_DEG_RATE = 0.05
THERMAL_SENSITIVITY = 0.3

# ---------------------------------------------------------
# 3. Model Architecture (3D FNO)
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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
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

        x1 = self.conv0(x); x2 = self.w0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x); x = F.gelu(x); x = self.fc2(x)
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
# 4. Data Generation (3D Gaussian Diffusion)
# ---------------------------------------------------------
def generate_data_3d(num_samples, s):
    print(f"[INFO] Generating {num_samples} 3D physics samples ({s}x{s}x{s})...")
    x = np.linspace(0, L, s)
    y = np.linspace(0, L, s)
    z = np.linspace(0, L, s)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    inputs = []
    outputs = []

    for i in range(num_samples):
        # Center of hotspot
        cx, cy, cz = np.random.rand(3) * L
        r2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2

        # Initial State
        u0 = np.exp(-20.0 * r2)

        # Physics Calculation (Analytic Heat Eq)
        t_factor = 1.0 + 4 * ALPHA * T_END * 20.0
        u_final = (1.0 / (t_factor**1.5)) * np.exp(-20.0 * r2 / t_factor)

        # Degradation Logic
        thermal_stress = np.abs(u_final)
        degradation = BASE_DEG_RATE + (THERMAL_SENSITIVITY * thermal_stress**2)
        soh_final = np.clip(1.0 - degradation, 0.0, 1.0)

        inputs.append(u0)
        combined_output = np.stack([u_final, soh_final], axis=0)
        outputs.append(combined_output)

        if (i+1) % 100 == 0:
            print(f"  > Generated {i+1}/{num_samples}")

    x_train = torch.tensor(np.array(inputs), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(np.array(outputs), dtype=torch.float32)
    return x_train.to(device), y_train.to(device)

# ---------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    model = FNO3d(MODES, WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_x, train_y = generate_data_3d(NUM_SAMPLES, GRID_SIZE)

    BATCH_SIZE = 8

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"[INFO] Starting High-Accuracy 3D Training...")
    start_time = time.time()
    EPOCHS = 100

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.mse_loss(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss / len(train_loader):.6f}")

    print(f"[DONE] Training finished in {time.time() - start_time:.2f}s")

    # SAVE MODEL
    torch.save(model.state_dict(), 'battery_fno_3d_model.pth')
    print("[SAVED] High-Accuracy 3D Model weights saved to 'battery_fno_3d_model.pth'")