from PIL import Image
import numpy as np
import networkx as nx
import random
from scipy.ndimage import label, binary_dilation
from skimage.morphology import square

def adjacent_regions(labeled_array):
    adjacency = {}
    
    dilated = binary_dilation(labeled_array, structure=square(3))
    
    for i in range(labeled_array.shape[0]):
        for j in range(labeled_array.shape[1]):
            current_label = labeled_array[i, j]
            dilated_label = dilated[i, j]
            
            if current_label != dilated_label:
                if current_label not in adjacency:
                    adjacency[current_label] = set()
                adjacency[current_label].add(dilated_label)
                
                if dilated_label not in adjacency:
                    adjacency[dilated_label] = set()
                adjacency[dilated_label].add(current_label)

    return adjacency

print("Loading and processing image...")
# Load the image and convert to binary
img = Image.open(r"C:\colourme.jpg").convert("L")
img_array = np.array(img)
binary_array = (img_array < 128).astype(int)

# Label the white regions
labeled_array, num_features = label(binary_array == 0)

print(f"Found {num_features} distinct regions.")

print("Identifying adjacent regions...")
adjacency = adjacent_regions(labeled_array)

print("Constructing the graph...")
# Construct the graph
G = nx.Graph()
for key, neighbors in adjacency.items():
    for neighbor in neighbors:
        G.add_edge(key, neighbor)

print("Coloring the graph...")
# Color the graph
coloring = nx.coloring.greedy_color(G, strategy="largest_first")

print("Generating colors for labels...")
# Generate a random color for each label
label_colors = {}
for label in coloring.keys():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    label_colors[label] = (r, g, b)

print("Applying colors to regions...")
colored_output = np.zeros((labeled_array.shape[0], labeled_array.shape[1], 3), dtype=np.uint8)
for i in range(labeled_array.shape[0]):
    for j in range(labeled_array.shape[1]):
        if labeled_array[i, j] in label_colors:
            colored_output[i, j] = label_colors[labeled_array[i, j]]

print("Displaying the colored image...")
# Convert array to image and show
output_img = Image.fromarray(colored_output)
output_img.show()

print("Process completed.")
