# An Algorithmic Approach to Image-to-Pixel-Art Generation

This repository contains the code used in the dissertation project **"An Algorithmic Approach to Image-to-pixel-art generation."**

`PixelArtGenerator` is a Python application that specializes in converting regular landscape images into pixel art using a multi-stage algorithmic pipeline.

## Requirements

To run this project, you will need the following libraries:
* `cv2` (OpenCV)
* `numpy`
* `PIL` (Python Imaging Library)
* `scikit-image`

## How to Use

### Setup

1. **Update the Dataset Path:** Update the `DATASET_PATH` variable (line 12) to point to your folder containing images:

```python
DATASET_PATH = r"C:\path\to\your\images"
```
2. Install dependencies:
```python
bash
pip install opencv-python numpy pillow scikit-image
```

### Running the Script
```python
bash
python PixelArtGenerator.py
```
#### Interactive Prompts
The script will ask for three inputs:

Image Selection:

Enter an image index number to process a single image
Type 'all' or 'a' to batch process all images
Target Pixel Width (default: 256):

Controls the output resolution (lower = more pixelated)
Maintains aspect ratio automatically
Number of Colors (k) (default: 16):

Controls the color palette size
Lower values = more simplified colors; higher values = more detail
Processing Pipeline
The script processes images through 5 stages:

### Structure of Pipeline
1. Contrast Enhancement - Improves visibility of details
2. Downscaling - Reduces resolution with smooth filtering
3. Segmentation - Groups colors into distinct regions
4. Dithering - Applies artistic texture and color quantization
5. Upscaling - Scales back to original dimensions using nearest-neighbor interpolation
#### Output
Main Output: FinalOutput_{k}_{width}_{filename} - The final pixel art image
Debug Output (when debug=True): Intermediate stages saved in DEBUG_STAGES/ folder for analysis
