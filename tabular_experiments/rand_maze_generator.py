import os
import numpy
from tabular import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from visualizations.visualization import plot_dist

# Generate random maps with the following parameters:
size = 12 #50
num_mazes = 15 #30

folder_name = 'med_random_mazes'
os.makedirs('mazes/'+folder_name, exist_ok=True)

for desc_num in range(num_mazes):
    desc = generate_random_map(size, p=0.8)
    # save to a numpy file:
    numpy.save(f'mazes/{folder_name}/random_map_{desc_num}.npy', desc)

### Plot four of them and save to same folder:
for i in range(4):
    desc = np.load(f'mazes/{folder_name}/random_map_{i}.npy')
    desc = np.array(list(desc), dtype='c')
    plot_dist(desc, show_plot=False, figsize=(10, 10), 
              symbol_size=400,
              filename=f'mazes/{folder_name}/random_map_{i}.png')

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
    img = plt.imread(f'mazes/{folder_name}/random_map_{i}.png')
    axes[i].imshow(img)

plt.savefig(f'mazes/{folder_name}/random_maps.png', bbox_inches='tight', dpi=300)
