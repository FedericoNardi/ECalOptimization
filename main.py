from modules import generator
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

E0 = 10

# Initialize the shower generator
shower_gen = generator.Shower(E0)
shower = shower_gen()

# Original coordinates
original_x = np.linspace(-25, 25, shower.shape[0])
original_z = np.linspace(0, 500, shower.shape[1]) + 0.01

import torch.nn.functional as F

def differentiable_interpolator(R, Z, original_x, original_z, shower):
    # Normalize coordinates for grid_sample
    R_norm = 2.0 * (R - original_x.min()) / (original_x.max() - original_x.min()) - 1.0
    Z_norm = 2.0 * (Z - original_z.min()) / (original_z.max() - original_z.min()) - 1.0

    # Stack and reshape to match grid_sample's input format
    grid = torch.stack((R_norm, Z_norm), dim=-1).unsqueeze(0).unsqueeze(0)
    
    # Convert shower to a tensor and reshape
    shower_tensor = torch.tensor(shower.T, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

    # Use grid_sample for interpolation
    img_3d = F.grid_sample(shower_tensor, grid, align_corners=True).squeeze()

    return img_3d

def custom_grid(dx, dy, dz, shower, original_x, original_z):
    x_start, x_end = -25, 25  # Reduced grid range
    y_start, y_end = -25, 25
    z_start, z_end = 0, 100  # Reduced grid range
    
    x_steps = int((x_end - x_start) / torch.abs(dx))
    y_steps = int((y_end - y_start) / torch.abs(dy))
    z_steps = int((z_end - z_start) / torch.abs(dz))

    # Use a lower resolution grid
    x = torch.linspace(x_start, x_end, x_steps // 2, device=dx.device, dtype=torch.float64)
    y = torch.linspace(y_start, y_end, y_steps // 2, device=dy.device, dtype=torch.float64)
    z = torch.linspace(z_start, z_end, z_steps // 2, device=dz.device, dtype=torch.float64) + 0.01

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)

    mask = torch.abs(R) < 25
    R_masked = R[mask]
    Z_masked = Z[mask]

    # Use the differentiable interpolator
    img_3d = differentiable_interpolator(R_masked, Z_masked, original_x, original_z, shower)
    img_3d = img_3d / img_3d.sum() * E0

    Z = Z + 1497.5 + dz

    return X, Y, Z, img_3d, mask

def reco(X, Y, Z, img_3d):
    return img_3d.sum() + dx * dy * dz * torch.randn(1)

def energy_resolution_torch(dx, dy, dz):
    X, Y, Z, img_3d, mask = custom_grid(dx, dy, dz, shower, original_x, original_z)
    return torch.abs(E0 - reco(X, Y, Z, img_3d)) / E0

# Initialize tensors with requires_grad=True for optimization
dx = torch.tensor(1, requires_grad=True, dtype=torch.float64)
dy = torch.tensor(1, requires_grad=True, dtype=torch.float64)
dz = torch.tensor(5, requires_grad=True, dtype=torch.float64)

optimizer = Adam([dx, dy, dz], lr=0.1)

for i in range(1000):
    optimizer.zero_grad()
    loss = energy_resolution_torch(dx, dy, dz)
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Loss: {loss.item()}")
        print(f"dx: {dx.item()}, dy: {dy.item()}, dz: {dz.item()}")
        print("")

print(f"Final loss: {loss.item()}")
print(f"dx: {dx.item()}, dy: {dy.item()}, dz: {dz.item()}")
print(f"True dx: {1}, True dy: {1}, True dz: {5}")
