from modules import generator
import numpy as np

E0 = 10

shower_gen = generator.Shower(E0)

# Calculate the shower image
shower = shower_gen()

# use an interpolator to get the shower image on the R,Z grid
from scipy.interpolate import RegularGridInterpolator

# Print a 2D image
import matplotlib.pyplot as plt
x0 = np.linspace(-25, 25, 100)
z0 = np.linspace(0, 500, 100) + 0.01

# Assuming the original shower data has dimensions corresponding to x and z axes
original_x = np.linspace(-25, 25, shower.shape[0])
original_z = np.linspace(0, 500, shower.shape[1]) + 0.01

interpolator = RegularGridInterpolator((original_x, original_z), shower.T)

# Create a meshgrid for the new interpolation points
X0, Z0 = np.meshgrid(x0, z0)
interp_points = np.array([X0.ravel(), Z0.ravel()]).T

# Interpolate the data
img2D = interpolator(interp_points).reshape(X0.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X0, Z0, c=img2D)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(shower, aspect='auto', origin='lower')
plt.savefig('img/2D_image.png')
plt.close()

# Same, but now with a rotation when evaluating the grid
theta = np.pi/6
interp_points_rot = np.array([X0.ravel(),Z0.ravel()]).T
img2D_rot = interpolator(interp_points_rot)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
X_rot = X0*np.cos(theta) - Z0*np.sin(theta)
Z_rot = X0*np.sin(theta) + Z0*np.cos(theta)
plt.scatter(X_rot, Z_rot, c=img2D_rot)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(shower, aspect='auto', origin='lower')
plt.savefig('img/2D_image_rot.png')
plt.close()

# Now move to a 3D meshgrid
x = np.linspace(-25,25,50)
y = np.linspace(-25,25,50)
z = np.linspace(0,500,50)+0.01
X, Y, Z = np.meshgrid(x, y, z)
R = np.sqrt(X**2+Y**2)
mask = np.abs(R)<25
img_3d = interpolator((R[mask],Z[mask]))
img_3d = img_3d/img_3d.sum()*E0
print('Total deposited energy: {}, Initial energy: {}, Ratio: {}'.format(np.sum(img_3d), E0, np.sum(img_3d)/E0))    
# Transparency alpha equal to inverse R normalized from 0 to 1
# import axes3d
alpha = 1/R - np.min(1/R)
alpha = alpha/np.max(alpha)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[mask], Y[mask], Z[mask], 
    c=img_3d, 
    cmap='YlOrRd', 
    alpha=0.025, #alpha[mask],
    marker='o',
    s=5)
plt.savefig('img/3D_image.png')
plt.close()

# Plot 9 2D images (without rotation) in a 3x3 grid with different shower energies
energies = [0.5, 10, 25, 50, 75, 100, 125, 150, 175]
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
x0 = np.linspace(-25, 25, 100)
z0 = np.linspace(0, 500, 100) + 0.01
X0, Z0 = np.meshgrid(x0, z0)
for i, ax in enumerate(axs.flat):
    shower_gen = generator.Shower(energies[i])
    shower = shower_gen()
    ax.scatter(X0, Z0, c=np.log(shower+1.))
    ax.set_title(f"Energy: {energies[i]}")
plt.savefig('img/2D_images_sample.png')
plt.close()


# Now insert img_3d in a custom 3D grid. Write a function defining the grid as function of spacings dx, dy, dz and insert the shower image in the grid with its true size
def custom_grid(dx, dy, dz):
    x = np.arange(-100, 100, dx)
    y = np.arange(-100, 100, dy)
    z = np.arange(0, 100, dz)+0.01
    X, Y, Z = np.meshgrid(x, y, z)
    R = np.sqrt(X**2+Y**2)
    mask = np.abs(R)<25
    img_3d = interpolator((R[mask],Z[mask]))
    img_3d = img_3d/img_3d.sum()*E0
    return X, Y, Z, img_3d, mask

dx, dy, dz = 1,1,1

X, Y, Z, img_3d, mask = custom_grid(dx, dy, dz)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[mask], Y[mask], Z[mask], 
    c=img_3d, 
    cmap='YlOrRd', 
    alpha=0.025, #alpha[mask],
    marker='o',
    s=5)
plt.savefig('img/3D_image_custom.png')
plt.close()

# Rotation angles
theta = np.pi / 6  # Rotation around the y-axis
phi = np.pi / 6    # Rotation around the x-axis

# Apply rotation around the y-axis (theta)
X_rot = X * np.cos(theta) - Z * np.sin(theta)
Z_rot = X * np.sin(theta) + Z * np.cos(theta)

# Update Z_rot after rotating around y-axis
Z_temp = Z_rot

# Apply rotation around the x-axis (phi)
Y_rot = Y * np.cos(phi) - Z_temp * np.sin(phi)
Z_rot = Y * np.sin(phi) + Z_temp * np.cos(phi)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X_rot[mask], Y_rot[mask], Z[mask], 
    c=img_3d, 
    cmap='YlOrRd', 
    alpha=0.025, #alpha[mask],
    marker='s',
    s=5)
plt.savefig('img/3D_image_custom_rot.png')
plt.close()

# Now plot it using plotly and box markers
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(
    x=X_rot[mask].ravel(), 
    y=Y_rot[mask].ravel(), 
    z=Z[mask].ravel(), 
    mode='markers',
    marker=dict(
        size=1,
        color=img_3d.ravel(),
        colorscale='YlOrRd',
        opacity=0.25
    )
)])
# save figure as jpg
fig.write_image('img/3D_image_custom_rot_plotly.png')

# Interpolate the shower in X_rot, Y_rot, Z_rot, evaluate it in X,Y,Z and plot it
from scipy.interpolate import RegularGridInterpolator
# sort the grid ascendingly according to the first axis and rearrange the rest accordingly
sort_idx = np.argsort(X_rot[mask])
X_rot = X_rot[mask][sort_idx]
Y_rot = Y_rot[mask][sort_idx]
Z_rot = Z_rot[mask][sort_idx]
img_3d = img_3d[sort_idx]

interpolator = RegularGridInterpolator((X_rot, Y_rot, Z_rot), img_3d)
rotated_shower = interpolator((X,Y,Z))
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X[mask], Y[mask], Z[mask], 
    c=rotated_shower[mask], 
    cmap='YlOrRd', 
    alpha=0.025,
    marker='o',
    s=5
)
plt.savefig('img/3D_image_custom_rot_interp.png')