import numpy
from tabular import generate_random_map

# Generate a random map
size = 7
num_mazes = 30
for desc_num in range(num_mazes):
    desc = generate_random_map(size, p=0.8)
    # save to a numpy file:
    numpy.save(f'random_mazes/random_map_{desc_num}.npy', desc)