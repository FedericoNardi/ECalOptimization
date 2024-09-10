import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(r, z) in terms of r and z
def f_rz(r, z):
    return np.exp(-((r)**2 + (z)**2)/10)

# Define the grid ranges for x, y, and z
x = np.linspace(-10, 10, 25)  # X-coordinate
y = np.linspace(-10, 10, 25)  # Y-coordinate
z = np.linspace(-10, 10, 25)  # Z-coordinate

# Create the 3D Cartesian grid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Calculate the radial distance r = sqrt(x^2 + y^2) for each point in the grid
R = np.sqrt(X**2 + Y**2)

# Evaluate the function f(r, z) on the Cartesian grid
F_xyz = f_rz(R, Z)

# Plotting the 3D volume using scatter or plot_surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot to visualize points in the 3D grid
ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=F_xyz.flatten(), cmap='viridis', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.savefig('img/3D_test.png')
