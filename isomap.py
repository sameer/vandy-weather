import pandas as pd

from scipy import misc
from sklearn import manifold

import matplotlib.pyplot as plt

import os

# Look pretty...
plt.style.use('ggplot')


#
# Start by creating a regular old, plain, "vanilla"
# python list.
#
samples = []
colours = []

#
# appends to your list the images
# in the /Datasets/ALOI/32_i directory.
#
directory = "Datasets/ALOI/32i/"
for fname in os.listdir(directory):
  fullname = os.path.join(directory, fname)
  img = misc.imread(fullname)
  # samples.append(  (img[::2, ::2] / 255.0).reshape(-1)  ) RESAMPLE
  samples.append( (img).reshape(-1) )
  colours.append('r')  # red colour

#
# Convert the list to a dataframe
#
df = pd.DataFrame( samples )


#
# Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)

my_isomap = iso.transform(df)


#
# Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("ISO transformation 2D")

ax.scatter(my_isomap[:,0], my_isomap[:,1], marker='.', c=colours)

#
# Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("ISO transformation 3D")

ax.scatter(my_isomap[:,0], my_isomap[:,1], my_isomap[:,2], marker='.', c=colours)

plt.show()

