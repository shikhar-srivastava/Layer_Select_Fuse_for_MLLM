import torch

# This script filters and saves specific projector weights for different strategies:
# - For 'latter' and 'former', only projector_1 weights are needed, so run this script to filter mm_projectors.1.
# - For 'all', both projector_1 and projector_2 weights are required, so run this script to filter both mm_projectors.1 and mm_projectors.2.

file_path1 = f"{your_model}/mm_projector_1.bin"
file_path2 = f"{your_model}/mm_projector_2.bin"

# Define filtering functions for specific projectors
def filter_mm_projectors_1(weights):
    """
    Filters weights to include only keys containing 'mm_projectors.1.'.
    """
    filtered_weights = {k: v for k, v in weights.items() if k.startswith('mm_projectors.1.')}
    return filtered_weights

def filter_mm_projectors_2(weights):
    """
    Filters weights to include only keys containing 'mm_projectors.2.'.
    """
    filtered_weights = {k: v for k, v in weights.items() if k.startswith('mm_projectors.2.')}
    return filtered_weights

# Load files
data1 = torch.load(file_path1, map_location='cpu')
data2 = torch.load(file_path2, map_location='cpu')

# Filter weights based on the defined functions
filtered_data1 = filter_mm_projectors_1(data1)
filtered_data2 = filter_mm_projectors_2(data2)

# Save the filtered weights to their respective paths
output_path1 = f"{your_model}/filtered_mm_projector_1.bin"
output_path2 = f"{your_model}/filtered_mm_projector_2.bin"

torch.save(filtered_data1, output_path1)
torch.save(filtered_data2, output_path2)
