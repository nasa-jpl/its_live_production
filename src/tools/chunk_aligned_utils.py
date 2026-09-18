"""
Utilities used to define datacube polygons that are chunk aligned with
current ITS_LIVE granules stored in s3://its-live-data/velocity_image_pair.
"""
import logging
import numpy as np
from grid import Grid


# Chunk size to allign new datacube polygons with
CHUNK_SIZE = 512
# Pixel size in meters for the grid
PIXEL_SIZE = 120

# Original band 8 Landsat pixel size.
GRID_OFFSET = Grid.L8B8_pix/2.0

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

# See https://github.com/ASFHyP3/hyp3-autorift/blob/develop/src/hyp3_autorift/crop.py#L127
# for original code on chunk alignment calculations.
# We are using similar logic to it to determine how to align the bounds of a
# polygon to a regular grid so that the resulting cubes are properly aligned
# with the chunk size and pixel spacing.
def get_aligned_min(val, grid_spacing):
    """Align a value with the nearest grid posting less than it"""
    nearest = np.floor(val / grid_spacing) * grid_spacing
    difference = val - nearest
    pixel_misalignment = difference % PIXEL_SIZE
    padding = difference - pixel_misalignment
    return val - padding, int(padding / 120)


def get_aligned_max(val, grid_spacing):
    """Align a value with the nearest grid posting greater than it"""
    nearest = np.ceil(val / grid_spacing) * grid_spacing
    difference = nearest - val
    pixel_misalignment = difference % PIXEL_SIZE
    padding = difference - pixel_misalignment
    return val + padding, int(padding / 120)


def get_alignment_info(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    grid_spacing: int = CHUNK_SIZE * PIXEL_SIZE,
):
    """Get the bounds and additional info necessary for chunk alignment

    Args:
        x_min: cropped minimum x coordinate
        y_min: cropped minimum y coordinate
        x_max: cropped maximum x coordinate
        y_max: cropped maximum y coordinate
        grid_spacing: width/height of the chunk in the units of the product's SRS

    Returns:
        1. aligned bounds
        2. padding in pixels required to align
        3. new range of x coordinates
        4. new range of y coordinates
    """
    x_min, left_pad = get_aligned_min(x_min, grid_spacing)
    y_min, bottom_pad = get_aligned_min(y_min, grid_spacing)
    x_max, right_pad = get_aligned_max(x_max, grid_spacing)
    y_max, top_pad = get_aligned_max(y_max, grid_spacing)

    aligned_bounds = [x_min, y_min, x_max, y_max]
    aligned_padding = [left_pad, bottom_pad, right_pad, top_pad]

    x_values = np.arange(x_min, x_max + PIXEL_SIZE, PIXEL_SIZE)
    y_values = np.arange(y_min, y_max + PIXEL_SIZE, PIXEL_SIZE)[::-1]
    return aligned_bounds, aligned_padding, x_values, y_values
