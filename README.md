# Pore Connectivity Analysis Tool

A Python tool for analyzing connectivity of porous structures in 2D and 3D image data, designed for materials science, geology, and microfluidics applications.

## Features

- **Automatic 2D/3D detection**: Automatically handles both 2D micromodel and 3D volume data
- **High-performance 3D analysis**: Uses `connected-components-3d` for fast, memory-efficient 3D connectivity analysis
- **Multi-phase segmentation**: Supports complex phase classifications (pores, solids, micropores)
- **True 3D connectivity**: Proper 3D percolation analysis with boundary spanning detection
- **Comprehensive visualization**: 
  - **2D**: Enhanced 3-panel view with zoomed isolated pore detection
  - **3D**: Professional 20×16" multi-perspective analysis with MIPs, slice views, and 3D scatter plots
- **Directional analysis**: Analyzes connectivity in specified directions (x, y, z)
- **Detailed reporting**: Pixel value distribution, connectivity statistics, and porosity analysis
- **Robust algorithms**: Multiple fallback strategies for reliable percolation detection

## Installation

### Quick Install
```bash
pip install numpy scipy matplotlib connected-components-3d
```

### For Maximum Performance
The `connected-components-3d` library provides dramatic speed improvements for 3D analysis:
```bash
# Essential for fast 3D analysis
pip install connected-components-3d

# Verify installation
python -c "import cc3d; print('cc3d version:', cc3d.__version__)"
```

### Alternative Install (slower 3D analysis)
If cc3d installation fails, the tool will fall back to scipy (10-50x slower for 3D):
```bash
pip install numpy scipy matplotlib
```

### Dependencies
- Python 3.6+
- NumPy
- SciPy  
- Matplotlib
- **connected-components-3d** (highly recommended for 10-50x faster 3D analysis)

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
- `--shape`: Image dimensions. Order matters:
  - 3D: `Z Y X` = `depth height width` (e.g., `64 256 256` means 64 slices of 256×256)
  - 2D: `Y X` = `height width` (e.g., `1200 1200`)
  - Example: For a 6-slice 1200×1200 stack, use `--shape 6 1200 1200`.
- `--pores`: Pore phase ranges (e.g., "0-50,200-255")
- `--solids`: Solid phase ranges (e.g., "51-199")
- `--micropores`: Optional micropore phase ranges
- `--direction`: Direction for connectivity analysis ('x', 'y', 'z')
- `--output`: Path for the modified image output
- `--visualize`: Generate before/after visualization
- `--slice`: Slice index for 3D visualization (optional)

## Algorithm Details

### 2D Analysis (Micromodels)
- **Direct connectivity analysis**: No approximations needed
- **Boundary spanning detection**: Identifies pore networks connecting opposite edges
- **Edge-to-edge flow analysis**: Preserves components that enable cross-sample flow
- **Intelligent fallback**: Uses largest component if no spanning networks found

### 3D Analysis (Volume Data) - Enhanced Algorithm
- **True 3D connectivity**: Uses `cc3d` for proper 3D connected component analysis (no projection artifacts)
- **Multi-criteria percolation detection**:
  - **Boundary spanning**: Components touching opposite faces in analysis direction
  - **Dimensional spanning**: Components covering >70% of sample dimension
  - **Multi-boundary touching**: Components contacting ≥3 cube faces
- **Performance optimized**: 
  - Vectorized bounding box analysis
  - Smart component filtering (analyzes only largest 100 components)
  - Memory-efficient multi-component extraction
- **Robust fallbacks**: Multiple strategies ensure reliable results for diverse pore structures

## Output

### Modified Image
- Raw binary file with unconnected pores replaced by solid material
- Same format and dimensions as input

### Visualization (when --visualize is used)

#### 2D Micromodel Visualization
- **Three-panel layout** when isolated pores detected:
  - Full micromodel with isolated pores highlighted in red
  - Modified micromodel with isolated pores removed
  - Zoomed detail view showing isolated pore locations
- **Smart scaling** with pixel count reporting
- **Professional layout** suitable for publications

#### 3D Volume Visualization (20×16" comprehensive analysis)
- **Row 1**: Maximum Intensity Projections (MIP) from 3 orthogonal views + connectivity plots + 3D scatter
- **Row 2**: Central slices in all directions + porosity distribution + 3D pore network visualization  
- **Row 3**: Modified volume slices + histogram comparison + component size distribution
- **Row 4**: Comprehensive statistical summary with quantitative metrics
- **Research-ready output** with detailed legends and annotations

### Console Output
```
Image Analysis:
Image dimensions: 3D
Image shape: (400, 400, 400)
Unique pixel values: [  0 255]
Pixel value distribution:
  Value 0: 18,232,846 pixels (28.49%)
  Value 255: 45,767,154 pixels (71.51%)
Total connected pore components: 12,240
Largest component size: 18,096,682 pixels
Smallest component size: 1 pixels

Porosity Analysis:
Initial porosity: 28.488821875000%
Final porosity (after removing unconnected pores): 28.301090625000%
Porosity reduction: 0.187731250000%

3D connectivity analysis was performed in the x-direction
```

## Performance

### Speed Improvements
- **3D Analysis**: 10-50x faster with `connected-components-3d` vs `scipy.ndimage.label`
- **Memory Efficiency**: Optimized for large volumes (tested on 400³ = 64M voxels)
- **Smart Filtering**: Analyzes only significant components (>0.1% of total pores)
- **Vectorized Operations**: Efficient bounding box and boundary detection

### Typical Processing Times
- **2D Micromodel** (1200×1200): < 5 seconds
- **3D Rock Sample** (400×400×400): 30-60 seconds with cc3d
- **Large 3D Volume** (800×800×800): 5-10 minutes with cc3d

### Memory Requirements
- **Rule of thumb**: ~8-10x the input file size for peak memory usage
- **400³ volume**: ~2-3 GB peak memory
- **Optimization**: Uses memory-mapped file loading when possible

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

## Image Shape Order

This tool reads raw files directly with NumPy using C-order: `np.fromfile(..., dtype=np.uint8).reshape(shape)`. For 3D data, the expected shape order is strictly `Z × Y × X` (depth, height, width). For 2D, it is `Y × X` (height, width).

- 3D example: `--shape 6 1200 1200` means 6 slices of size 1200×1200.
- 2D example: `--shape 1200 1200` for a single 1200×1200 image.

Why this matters: Passing the shape in a different order (e.g., `1200 1200 6`) will distort how the volume is interpreted. A common symptom is horizontal or vertical streaking/banding when viewing the output as a 2D image, often with a periodicity equal to the mistakenly short axis length (e.g., 6 pixels).

Troubleshooting shape issues:
- If you see streaks/banding, double-check that `--shape` is `Z Y X` for 3D. Swap axes if needed.
- The script prints `Image shape: (Z, Y, X)` after loading; confirm it matches expectations.
- When viewing `result.raw` in another tool, configure it as a stack of `Z` slices, each of size `Y × X`.
- If your 3D data is a repeated 2D slice along Z, you may analyze it as true 3D with `--shape Z Y X` or simply as 2D using `--shape Y X`.

## Examples

### Real Datasets Tested
1. **2D Micromodel**: `micromodel_heteroCircle_12000by12000_orig_BINNED10_seg.raw`
   - 1200×1200 pixel microfluidic device
   - Binary segmentation (0=pores, 255=solids)
   - Demonstrates 2D connectivity with tiny isolated pore detection

2. **3D Rock Sample**: `Bentheimer400-5mum_binarized.raw`
   - 400×400×400 voxel Bentheimer sandstone
   - 5µm resolution, binary segmentation
   - Shows realistic pore network percolation (28.5% porosity, 0.19% isolated)

### Usage Patterns
```bash
# Quick 2D analysis
python analyze_connectivity.py micromodel.raw --shape 1200 1200 --pores "0-0" --solids "255-255" --direction x --output result.raw

# Comprehensive 3D analysis with visualization  
python analyze_connectivity.py rock.raw --shape 400 400 400 --pores "0-0" --solids "255-255" --direction z --output result.raw --visualize

# Multi-phase rock analysis
python analyze_connectivity.py complex_rock.raw --shape 256 256 256 --pores "0-50" --solids "200-255" --micropores "51-199" --direction y --output result.raw --visualize
```

## License

This project is open source. Feel free to use and modify for research and educational purposes.

## Contributing

Contributions welcome! Please feel free to submit issues and enhancement requests.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{pore_connectivity_analysis,
  title = {Pore Connectivity Analysis Tool: High-Performance 2D/3D Percolation Analysis},
  author = {[Your Name]},
  url = {https://github.com/[username]/pore-connectivity-analysis},
  year = {2024},
  note = {Fast 3D connectivity analysis using connected-components-3d}
}
```

### Dependencies to Acknowledge
- **connected-components-3d**: Zung, William Silversmith et al. (2021) [10.5281/zenodo.5791862](https://doi.org/10.5281/zenodo.5791862)
- **NumPy**: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020) [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- **SciPy**: Virtanen, P., Gommers, R., Oliphant, T.E. et al. (2020) [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

## Troubleshooting

### Common Issues

#### "cc3d not available" Warning
If you see this warning, install connected-components-3d for faster 3D analysis:
```bash
pip install connected-components-3d
```

#### Memory Errors with Large 3D Volumes
- **Reduce size**: Crop or downsample large volumes
- **Increase memory**: Use machines with more RAM (8-16GB recommended for 400³ volumes)
- **Use chunking**: For very large volumes, consider analyzing in overlapping sections

#### "Found unclassified pixels" Error
This means your `--pores` and `--solids` ranges don't cover all pixel values:
```bash
# First, check what values are in your image (script will show unique values)
# Then adjust ranges to cover all pixels
# Example: if unique values are [0, 128, 255]
--pores "0-0" --solids "255-255" --micropores "128-128"
```

#### Slow Performance
- **Install cc3d**: Essential for fast 3D analysis
- **Check memory**: Ensure sufficient RAM (8-10x file size)
- **Reduce components**: Very fragmented volumes (>50k components) will be slower

#### Visualization Issues
- **Large files**: Visualization files can be 50-100MB for complex 3D analysis
- **Display**: Use image viewers that support high-resolution images
- **Memory**: Close other applications if visualization generation fails 
