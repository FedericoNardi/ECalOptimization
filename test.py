from modules import generator
from modules import bib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Initialize energy tensor
E0 = tf.constant(50.0, dtype=tf.float32)

# Initialize tensors for spacing with trainable variables for optimization
dx = tf.Variable(10.0, dtype=tf.float32) # mm
dy = tf.Variable(10.0, dtype=tf.float32) # mm
dz = tf.Variable(50.0, dtype=tf.float32) # mm

# Generate and plot a shower
#gen = generator.Shower(E0, spacing=[dx, dy, dz], DEBUG=True)
gen = generator.Shower(E0)
shower = gen()

# Overlay BIB
bib_model = bib.Model()
bib_shower = bib_model.predict(np.stack([gen.X.numpy().flatten(), gen.Z.numpy().flatten()], axis=-1)).reshape(shower.shape)
# Multiply by cell volume
bib_shower*=(dx*dy*dz)
shower += bib_shower

# Plot the 3D scatter plot of the generated shower
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

import matplotlib.colors as colors

# Use .numpy() to extract values from TensorFlow tensors
ax.scatter(
    gen.X.numpy().flatten(), gen.Z.numpy().flatten(), gen.Y.numpy().flatten(),
    c=shower.numpy().flatten(),
    cmap='viridis',
    alpha=0.05,
    norm=colors.LogNorm(),
    s=1
)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.savefig('img/3D_test.png')

# Plot a 2D projection of the shower integrating on the Y axis, and one integrating on the X axis. Use log color scale.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(shower.numpy().sum(axis=1).T, extent=(-25, 25, 0, 500), aspect='auto', cmap='viridis', norm=colors.LogNorm())
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Integrated on Y')
plt.colorbar(label='Energy (MeV)')
plt.subplot(1, 2, 2)
plt.imshow(shower.numpy().sum(axis=0).T, extent=(-25, 25, 0, 500), aspect='auto', cmap='viridis', norm=colors.LogNorm())
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Integrated on Y')
plt.colorbar(label='Energy (MeV)')
plt.savefig('img/2D_test.png')
