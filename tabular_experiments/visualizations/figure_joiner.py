import matplotlib.pyplot as plt

# first run the comparison_plotter.py and qplotter.py to generate the images:
import os
os.system('python visualizations/comparison_plotter.py')
os.system('python visualizations/qplotter.py')

# Join the rewards_comparison.png and Q_values.png into a single figure
# and save it as rewards_and_Q_values.png

rewards = plt.imread('visualizations/rewards_comparison.png')
Q_values = plt.imread('visualizations/Q_values.png')

fig, ax = plt.subplots(1, 2, figsize=(27, 9), gridspec_kw={'width_ratios': [1, 1.18]})

# Set aspect ratios to be equal
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

# Hide the axes
ax[0].axis('off')
ax[1].axis('off')

# Display images
ax[0].imshow(rewards)#, aspect='auto')
ax[1].imshow(Q_values)#, aspect='auto')

# Adjust tight_layout parameters
plt.tight_layout(pad=0)
# stretch out horizontally with aspect:
# plt.subplots_adjust(wspace=0.1)

plt.savefig('visualizations/rewards_and_Q_values.png', bbox_inches='tight', dpi=400)
