import numpy as np
from scipy.ndimage import label
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_raw_image(filepath, shape):
    """
    Load a raw image file into a numpy array.
    
    Args:
        filepath (str): Path to the raw image file
        shape (tuple): Shape of the image (height, width) for 2D or (depth, height, width) for 3D
    
    Returns:
        numpy.ndarray: The loaded image data
    """
    return np.fromfile(filepath, dtype=np.uint8).reshape(shape)

def create_phase_mask(image, phase_ranges):
    """
    Create a binary mask for a specific phase based on value ranges.
    
    Args:
        image (numpy.ndarray): Input image data
        phase_ranges (list): List of (min, max) tuples defining the phase ranges
    
    Returns:
        numpy.ndarray: Binary mask where True indicates the specified phase
    """
    mask = np.zeros_like(image, dtype=bool)
    for min_val, max_val in phase_ranges:
        mask |= (image >= min_val) & (image <= max_val)
    return mask

def calculate_porosity(image, pore_ranges):
    """
    Calculate the porosity of the image based on pore ranges.
    
    Args:
        image (numpy.ndarray): Input image data
        pore_ranges (list): List of (min, max) tuples defining the pore ranges
    
    Returns:
        float: Porosity as a percentage
    """
    pore_mask = create_phase_mask(image, pore_ranges)
    return (np.sum(pore_mask) / pore_mask.size) * 100

def get_connected_pores_mask_2d(image, pore_ranges, direction):
    """
    Get a mask of connected pores in the specified direction for 2D data.
    
    Args:
        image (numpy.ndarray): Input 2D image data
        pore_ranges (list): List of (min, max) tuples defining the pore ranges
        direction (str): Direction to analyze ('x', 'y')
    
    Returns:
        numpy.ndarray: Binary mask where True indicates connected pores
    """
    # Create binary mask for pores
    pore_mask = create_phase_mask(image, pore_ranges)
    
    # Label all connected components
    labeled_array, num_features = label(pore_mask)
    
    if num_features == 0:
        return np.zeros_like(pore_mask, dtype=bool)
    
    # For 2D micromodels, we want to find pores that span across the image
    # in the specified direction
    
    if direction == 'x':
        # Check which components span from left to right
        spanning_components = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Get column indices where this component exists
            cols = np.where(np.any(component_mask, axis=0))[0]
            if len(cols) > 0:
                # Check if component spans a significant portion of the width
                span_ratio = (cols.max() - cols.min() + 1) / image.shape[1]
                # Also check if it touches both edges
                touches_left = cols.min() == 0
                touches_right = cols.max() == image.shape[1] - 1
                
                if touches_left and touches_right:
                    spanning_components.append(i)
                elif span_ratio > 0.5:  # Spans more than 50% of width
                    spanning_components.append(i)
    
    elif direction == 'y':
        # Check which components span from top to bottom
        spanning_components = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Get row indices where this component exists
            rows = np.where(np.any(component_mask, axis=1))[0]
            if len(rows) > 0:
                # Check if component spans a significant portion of the height
                span_ratio = (rows.max() - rows.min() + 1) / image.shape[0]
                # Also check if it touches both edges
                touches_top = rows.min() == 0
                touches_bottom = rows.max() == image.shape[0] - 1
                
                if touches_top and touches_bottom:
                    spanning_components.append(i)
                elif span_ratio > 0.5:  # Spans more than 50% of height
                    spanning_components.append(i)
    
    # If no spanning components found, use the largest component
    if not spanning_components:
        print(f"Warning: No components span across the image in {direction}-direction.")
        print("Using the largest connected component instead.")
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        largest_component = np.argmax(component_sizes) + 1
        spanning_components = [largest_component]
    
    # Create mask for spanning components
    connected_pores_mask = np.zeros_like(pore_mask, dtype=bool)
    for comp_id in spanning_components:
        connected_pores_mask |= (labeled_array == comp_id)
    
    return connected_pores_mask

def get_connected_pores_mask_3d(image, pore_ranges, direction):
    """
    Get a mask of connected pores in the specified direction for 3D data.
    
    Args:
        image (numpy.ndarray): Input 3D image data
        pore_ranges (list): List of (min, max) tuples defining the pore ranges
        direction (str): Direction to analyze ('x', 'y', 'z')
    
    Returns:
        numpy.ndarray: Binary mask where True indicates connected pores
    """
    # Create binary mask for pores
    pore_mask = create_phase_mask(image, pore_ranges)
    
    # Map direction to numpy axis
    # For 3D data in NumPy, the axes are ordered as (z, y, x)
    direction_map = {
        'x': 2,  # Last axis in NumPy
        'y': 1,  # Middle axis in NumPy
        'z': 0   # First axis in NumPy
    }
    axis = direction_map[direction]
    
    # Project the pore mask along the specified axis
    projection = np.any(pore_mask, axis=axis)
    
    # Label connected components
    labeled_array, num_features = label(projection)
    
    # Create a mask for the largest connected component
    if num_features > 0:
        # Find the largest component
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        largest_component = np.argmax(component_sizes) + 1
        
        # Create mask for the largest component
        largest_component_mask = (labeled_array == largest_component)
        
        # Expand the mask back to 3D
        expanded_mask = np.expand_dims(largest_component_mask, axis)
        expanded_mask = np.repeat(expanded_mask, image.shape[axis], axis=axis)
        
        # Combine with original pore mask to get only connected pores
        connected_pores_mask = expanded_mask & pore_mask
        return connected_pores_mask
    
    return np.zeros_like(pore_mask, dtype=bool)

def get_connected_pores_mask(image, pore_ranges, direction):
    """
    Get a mask of connected pores in the specified direction.
    Automatically detects 2D vs 3D data and uses appropriate method.
    
    Args:
        image (numpy.ndarray): Input image data
        pore_ranges (list): List of (min, max) tuples defining the pore ranges
        direction (str): Direction to analyze ('x', 'y', 'z')
    
    Returns:
        numpy.ndarray: Binary mask where True indicates connected pores
    """
    if len(image.shape) == 2:
        # 2D data
        if direction == 'z':
            print("Warning: z-direction specified for 2D data. Using x-direction instead.")
            direction = 'x'
        return get_connected_pores_mask_2d(image, pore_ranges, direction)
    elif len(image.shape) == 3:
        # 3D data
        return get_connected_pores_mask_3d(image, pore_ranges, direction)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

def create_modified_image(image, pore_ranges, solid_ranges, direction):
    """
    Create a new image with only connected pores, replacing unconnected pores with median solid value.
    
    Args:
        image (numpy.ndarray): Input image data
        pore_ranges (list): List of (min, max) tuples defining the pore ranges
        solid_ranges (list): List of (min, max) tuples defining the solid ranges
        direction (str): Direction to analyze ('x', 'y', 'z')
    
    Returns:
        tuple: (modified_image, unconnected_pores_mask)
    """
    # Create a copy of the original image
    modified_image = image.copy()
    
    # Get connected pores mask
    connected_pores_mask = get_connected_pores_mask(image, pore_ranges, direction)
    
    # Get solid mask
    solid_mask = create_phase_mask(image, solid_ranges)
    
    # Calculate median solid value
    median_solid = np.median(image[solid_mask])
    
    # Replace unconnected pores with median solid value
    pore_mask = create_phase_mask(image, pore_ranges)
    unconnected_pores = pore_mask & ~connected_pores_mask
    modified_image[unconnected_pores] = int(median_solid)
    
    return modified_image, unconnected_pores

def save_raw_image(image, filepath):
    """
    Save a numpy array as a raw image file.
    
    Args:
        image (numpy.ndarray): Image data to save
        filepath (str): Path to save the raw image file
    """
    image.astype(np.uint8).tofile(filepath)

def visualize_slice_2d(image, modified_image, unconnected_pores, pore_ranges, output_path):
    """
    Create a visualization for 2D data.
    """
    # Create a custom colormap: black for background, white for pores, red for unconnected pores
    colors = ['black', 'white', 'red']
    cmap = ListedColormap(colors)
    
    # Create a visualization array
    vis_array = np.zeros_like(image, dtype=np.uint8)
    pore_mask = create_phase_mask(image, pore_ranges)  # Use actual pore ranges
    vis_array[pore_mask] = 1  # Pores in white
    vis_array[unconnected_pores] = 2  # Unconnected pores in red
    
    num_unconnected = np.sum(unconnected_pores)
    
    # Create the figure - add more subplots if there are unconnected pores to show
    if num_unconnected > 0:
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original with unconnected pores highlighted
    ax1.imshow(vis_array, cmap=cmap, interpolation='nearest')
    ax1.set_title(f'Original Micromodel\n(Red = {num_unconnected} Unconnected Pores)')
    ax1.axis('off')
    
    # Plot modified
    ax2.imshow(modified_image, cmap='gray')
    ax2.set_title('Modified Micromodel\n(Unconnected Pores Removed)')
    ax2.axis('off')
    
    # If there are unconnected pores, create a zoomed view
    if num_unconnected > 0:
        # Find bounding box of unconnected pores
        rows, cols = np.where(unconnected_pores)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Add padding to show context
        padding = 20
        min_row = max(0, min_row - padding)
        max_row = min(image.shape[0], max_row + padding)
        min_col = max(0, min_col - padding)
        max_col = min(image.shape[1], max_col + padding)
        
        # Create zoomed view
        zoomed_vis = vis_array[min_row:max_row, min_col:max_col]
        ax3.imshow(zoomed_vis, cmap=cmap, interpolation='nearest')
        ax3.set_title(f'Zoomed View\n(Region: {min_row}:{max_row}, {min_col}:{max_col})')
        ax3.axis('off')
        
        # Add text showing exact pixel count
        fig.suptitle(f'Connectivity Analysis Results\nTotal pores: {np.sum(pore_mask):,}, Unconnected: {num_unconnected}, Connected: {np.sum(pore_mask) - num_unconnected:,}', 
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_slice_3d(image, modified_image, unconnected_pores, pore_ranges, slice_idx, direction, output_path):
    """
    Create a visualization of a slice before and after processing for 3D data.
    """
    # Create a custom colormap: black for background, white for pores, red for unconnected pores
    colors = ['black', 'white', 'red']
    cmap = ListedColormap(colors)
    
    # Get the appropriate slice based on direction
    direction_map = {'x': 2, 'y': 1, 'z': 0}
    axis = direction_map[direction]
    
    # Create a visualization array
    vis_array = np.zeros_like(image, dtype=np.uint8)
    vis_array[create_phase_mask(image, pore_ranges)] = 1  # Pores in white
    vis_array[unconnected_pores] = 2  # Unconnected pores in red
    
    # Take the middle slice if slice_idx is None
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original slice
    if axis == 0:  # z-slice
        slice_data = vis_array[slice_idx, :, :]
    elif axis == 1:  # y-slice
        slice_data = vis_array[:, slice_idx, :]
    else:  # x-slice
        slice_data = vis_array[:, :, slice_idx]
    
    ax1.imshow(slice_data, cmap=cmap)
    ax1.set_title(f'Original {direction}-slice {slice_idx}\n(Red = Unconnected Pores)')
    ax1.axis('off')
    
    # Plot modified slice
    if axis == 0:  # z-slice
        slice_data = modified_image[slice_idx, :, :]
    elif axis == 1:  # y-slice
        slice_data = modified_image[:, slice_idx, :]
    else:  # x-slice
        slice_data = modified_image[:, :, slice_idx]
    
    ax2.imshow(slice_data, cmap='gray')
    ax2.set_title(f'Modified {direction}-slice {slice_idx}\n(Unconnected Pores Removed)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_slice(image, modified_image, unconnected_pores, pore_ranges, slice_idx, direction, output_path):
    """
    Create a visualization before and after processing.
    Automatically detects 2D vs 3D and uses appropriate method.
    """
    if len(image.shape) == 2:
        visualize_slice_2d(image, modified_image, unconnected_pores, pore_ranges, output_path)
    else:
        visualize_slice_3d(image, modified_image, unconnected_pores, pore_ranges, slice_idx, direction, output_path)

def validate_phase_coverage(image, phase_ranges_list):
    """
    Validate that all pixels in the image are covered by at least one phase.
    
    Args:
        image (numpy.ndarray): Input image data
        phase_ranges_list (list): List of phase ranges for each phase
    
    Raises:
        ValueError: If there are unclassified pixels
    """
    # Create a mask for all classified pixels
    classified_mask = np.zeros_like(image, dtype=bool)
    for phase_ranges in phase_ranges_list:
        classified_mask |= create_phase_mask(image, phase_ranges)
    
    # Check for unclassified pixels
    unclassified = ~classified_mask
    if np.any(unclassified):
        unclassified_values = np.unique(image[unclassified])
        raise ValueError(f"Found unclassified pixels with values: {unclassified_values}")

def parse_phase_ranges(phase_str):
    """
    Parse phase ranges from string format 'min1-max1,min2-max2,...'
    """
    ranges = []
    for range_str in phase_str.split(','):
        min_val, max_val = map(int, range_str.split('-'))
        ranges.append((min_val, max_val))
    return ranges

def main():
    parser = argparse.ArgumentParser(description='Analyze connectivity of phases in a raw image.')
    parser.add_argument('filepath', help='Path to the raw image file')
    parser.add_argument('--shape', nargs='+', type=int, required=True,
                      help='Shape of the image (e.g., "256 256" for 2D or "64 256 256" for 3D)')
    parser.add_argument('--pores', type=str, required=True,
                      help='Pore phase ranges (e.g., "0-50,200-255")')
    parser.add_argument('--solids', type=str, required=True,
                      help='Solid phase ranges (e.g., "51-199")')
    parser.add_argument('--micropores', type=str,
                      help='Micropore phase ranges (e.g., "100-150")')
    parser.add_argument('--direction', type=str, choices=['x', 'y', 'z'], required=True,
                      help='Direction to analyze connectivity')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the modified image')
    parser.add_argument('--slice', type=int,
                      help='Index of the slice to visualize (optional)')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualization of before/after slices')
    
    args = parser.parse_args()
    
    # Convert shape argument to tuple
    shape = tuple(args.shape)
    
    try:
        # Load the image
        image = load_raw_image(args.filepath, shape)
        
        # Parse phase ranges
        pore_ranges = parse_phase_ranges(args.pores)
        solid_ranges = parse_phase_ranges(args.solids)
        
        # Collect all phase ranges for validation
        phase_ranges_list = [pore_ranges, solid_ranges]
        
        # Handle optional micropores
        if args.micropores:
            micropore_ranges = parse_phase_ranges(args.micropores)
            phase_ranges_list.append(micropore_ranges)
        
        # Validate that all pixels are classified
        validate_phase_coverage(image, phase_ranges_list)
        
        # Calculate initial porosity
        initial_porosity = calculate_porosity(image, pore_ranges)
        
        # Create modified image with only connected pores
        modified_image, unconnected_pores = create_modified_image(image, pore_ranges, solid_ranges, args.direction)
        
        # Calculate final porosity
        final_porosity = calculate_porosity(modified_image, pore_ranges)
        
        # Save the modified image
        save_raw_image(modified_image, args.output)
        
        # Create visualization if requested
        if args.visualize:
            vis_output = os.path.splitext(args.output)[0] + '_visualization.png'
            visualize_slice(image, modified_image, unconnected_pores, pore_ranges, args.slice, args.direction, vis_output)
            print(f"\nVisualization saved to: {vis_output}")
        
        # Print results
        print(f"\nImage Analysis:")
        print(f"Image dimensions: {len(shape)}D")
        print(f"Image shape: {shape}")
        
        # Print connectivity information
        pore_mask = create_phase_mask(image, pore_ranges)
        labeled_pores, num_components = label(pore_mask)
        if num_components > 0:
            component_sizes = np.bincount(labeled_pores.ravel())[1:]  # Skip background
            print(f"Total connected pore components: {num_components}")
            print(f"Largest component size: {max(component_sizes)} pixels")
            print(f"Smallest component size: {min(component_sizes)} pixels")
        
        print(f"\nPorosity Analysis:")
        print(f"Initial porosity: {initial_porosity:.12f}%")
        print(f"Final porosity (after removing unconnected pores): {final_porosity:.12f}%")
        print(f"Porosity reduction: {initial_porosity - final_porosity:.12f}%")
        print(f"\nModified image saved to: {args.output}")
        
        if len(shape) == 2:
            print(f"2D connectivity analysis was performed in the {args.direction}-direction")
            print("Note: For 2D micromodels, 'x' means left-right connectivity, 'y' means top-bottom connectivity")
        else:
            print(f"3D connectivity analysis was performed in the {args.direction}-direction")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 