from modules import generator
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

E0 = torch.tensor(150, dtype=torch.float32)

import torch.nn.functional as F
    
# Initialize tensors with requires_grad=True for optimization
dx = torch.tensor(10, requires_grad=True, dtype=torch.float64)
dy = torch.tensor(10, requires_grad=True, dtype=torch.float64)
dz = torch.tensor(10, requires_grad=True, dtype=torch.float64)

# Generate and plot a shower
gen = generator.Shower(E0, spacing=[dx, dy, dz], DEBUG=True)
gen.plot_parameters()
shower = gen()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    gen.X.detach().flatten(), gen.Z.detach().flatten(), gen.Y.detach().flatten(), 
    c=shower.detach().flatten(), 
    cmap='Oranges',
    alpha=0.05,
    s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('img/3D_test.png')

'''
# Define the loss function
def energy_resolution_torch(dx, dy, dz):
    gen = generator.Shower(E0, spacing=[dx, dy, dz])
    shower = gen()
    
    # Calculate the energy resolution
    resolution = F.mse_loss(shower.sum(), E0)
    return resolution

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
'''