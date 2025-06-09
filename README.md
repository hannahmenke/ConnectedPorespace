# Pore Connectivity Analysis Tool

A Python tool for analyzing connectivity of porous structures in 2D and 3D image data, designed for materials science, geology, and microfluidics applications.

## Features

- **Automatic 2D/3D detection**: Automatically handles both 2D micromodel and 3D volume data
- **Multi-phase segmentation**: Supports complex phase classifications (pores, solids, micropores)
- **Connectivity analysis**: Identifies connected vs. isolated pore networks
- **Enhanced visualization**: Creates detailed before/after comparisons with zoomed views of isolated features
- **Directional analysis**: Analyzes connectivity in specified directions (x, y, z)
- **Statistical reporting**: Provides detailed porosity and connectivity statistics

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib argparse
```

### Dependencies
- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## Usage

### Basic Usage
```bash
python analyze_connectivity.py input.raw --shape HEIGHT WIDTH --pores "0-50" --solids "51-255" --direction x --output modified.raw --visualize
```

### 2D Micromodel Example
```bash
python analyze_connectivity.py micromodel.raw \
  --shape 1200 1200 \
  --pores "0-0" \
  --solids "255-255" \
  --direction x \
  --output connected_micromodel.raw \
  --visualize
```

### 3D Rock Sample Example
```bash
python analyze_connectivity.py rock_sample.raw \
  --shape 64 256 256 \
  --pores "0-50,200-255" \
  --solids "51-199" \
  --direction z \
  --output connected_rock.raw \
  --visualize
```

## Parameters

- `filepath`: Path to the raw image file
- `--shape`: Image dimensions (e.g., "256 256" for 2D or "64 256 256" for 3D)
- `--pores`: Pore phase ranges (e.g., "0-50,200-255")
- `--solids`: Solid phase ranges (e.g., "51-199")
- `--micropores`: Optional micropore phase ranges
- `--direction`: Direction for connectivity analysis ('x', 'y', 'z')
- `--output`: Path for the modified image output
- `--visualize`: Generate before/after visualization
- `--slice`: Slice index for 3D visualization (optional)

## Algorithm Details

### 2D Analysis (Micromodels)
- Direct connected component analysis
- Identifies pore networks that span across image boundaries
- Preserves components that connect opposite edges in specified direction

### 3D Analysis (Volume Data)
- Projection-based connectivity analysis
- Projects 3D volume along specified axis
- Identifies largest connected component in projection
- Expands result back to 3D volume

## Output

### Modified Image
- Raw binary file with unconnected pores replaced by solid material
- Same format and dimensions as input

### Visualization (when --visualize is used)
- **2D**: Three-panel view with full image, modified image, and zoomed detail
- **3D**: Slice-based before/after comparison
- Red highlighting of unconnected pores
- Statistical information in titles

### Console Output
```
Image Analysis:
Image dimensions: 2D
Image shape: (1200, 1200)
Total connected pore components: 4
Largest component size: 809362 pixels
Smallest component size: 1 pixels

Porosity Analysis:
Initial porosity: 56.206041666667%
Final porosity (after removing unconnected pores): 56.205694444444%
Porosity reduction: 0.000347222222%
```

## Applications

### Materials Science
- Characterizing porous materials
- Analyzing permeability of filters and membranes
- Studying foam and cellular structures

### Geology
- Rock permeability analysis
- Reservoir characterization
- Groundwater flow studies

### Microfluidics
- Micromodel connectivity analysis
- Flow path characterization
- Device design validation

## File Format

The tool works with raw binary image files:
- **2D**: Height × Width pixels
- **3D**: Depth × Height × Width pixels
- **Data type**: uint8 (values 0-255)
- **Phase encoding**: Different grayscale values represent different material phases

## Examples

See the included example with `micromodel_heteroCircle_12000by12000_orig_BINNED10_seg.raw` - a 1200×1200 pixel micromodel demonstrating typical usage.

## License

This project is open source. Feel free to use and modify for research and educational purposes.

## Contributing

Contributions welcome! Please feel free to submit issues and enhancement requests.

## Citation

If you use this tool in your research, please cite:
```
Pore Connectivity Analysis Tool
GitHub: [Your Repository URL]
``` 