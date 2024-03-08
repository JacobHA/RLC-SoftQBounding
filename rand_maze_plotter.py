# Use visualizations/visualization.py  to plot the random descs:

import numpy as np
import matplotlib.pyplot as plt
from visualizations.visualization import plot_dist
# Read in first four maps:

for i in range(4):
    desc = np.load(f'random_mazes/random_map_{i}.npy')
    desc = np.array(list(desc), dtype='c')
    plot_dist(desc, show_plot=False, figsize=(10, 10), 
              symbol_size=400,
              filename=f'visualizations/maps/random_map_{i}.png')

# Now combine them all side by side:
fig, axes = plt.subplots(1, 4, figsize=(40, 10))
for i, desc in enumerate(axes):
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    # remove bordeR:
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    
    # import image and put it in the plot:
    img = plt.imread(f'visualizations/maps/random_map_{i}.png')
    axes[i].imshow(img)

plt.savefig('visualizations/maps/random_maps.png', bbox_inches='tight', dpi=300)
# plt.show()