"""
Unit tests for grid.py's Bounds/Grid utility classes using pytest.

Extracted from test_itscube_generation.py (now test_cube_composites.py) --
this test exercises Grid/Bounds directly and doesn't touch ITSCube or
ITSLiveComposite at all.

Authors: Masha Liukis
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grid import Bounds, Grid


# Small test polygon (Malaspina glacier area), matching the one used by the
# cube/composite generation tests.
POLYGON = [
    [-3300000, 200000],
    [-3200000, 200000],
    [-3200000, 300000],
    [-3300000, 300000],
    [-3300000, 200000]
]
CELL_SIZE = 120  # Grid cell size in kilometers


class TestGrid:
    """Tests for Grid utility class."""

    def test_grid_creation(self):
        """Test Grid utility class with test polygon bounds."""
        # Extract x and y bounds from polygon
        x_coords = [point[0] for point in POLYGON]
        y_coords = [point[1] for point in POLYGON]

        x_bounds = Bounds(x_coords)
        y_bounds = Bounds(y_coords)

        # Create grid
        grid_x, grid_y = Grid.create(x_bounds, y_bounds, CELL_SIZE)

        # Verify grid properties
        assert len(grid_x) > 0, "Grid X is empty"
        assert len(grid_y) > 0, "Grid Y is empty"

        # Verify grid spacing (should be identical within floating-point precision)
        if len(grid_x) > 1:
            x_spacing = grid_x[1] - grid_x[0]
            assert abs(x_spacing) == CELL_SIZE, (
                f"Grid X spacing {abs(x_spacing)} doesn't match "
                f"cell size {CELL_SIZE}"
            )

        if len(grid_y) > 1:
            y_spacing = grid_y[1] - grid_y[0]
            assert abs(y_spacing) == CELL_SIZE, (
                f"Grid Y spacing {abs(y_spacing)} doesn't match "
                f"cell size {CELL_SIZE}"
            )

        # Verify grid covers the polygon
        # Note: grid values are cell centers, so actual extent is ±cell_size/2
        half_cell = CELL_SIZE / 2.0
        grid_x_min_extent = grid_x.min() - half_cell
        grid_x_max_extent = grid_x.max() + half_cell
        grid_y_min_extent = grid_y.min() - half_cell
        grid_y_max_extent = grid_y.max() + half_cell

        assert grid_x_min_extent <= min(x_coords), (
            f"Grid X min extent {grid_x_min_extent} doesn't cover polygon min {min(x_coords)}"
        )
        assert grid_x_max_extent >= max(x_coords), (
            f"Grid X max extent {grid_x_max_extent} doesn't cover polygon max {max(x_coords)}"
        )
        assert grid_y_min_extent <= min(y_coords), (
            f"Grid Y min extent {grid_y_min_extent} doesn't cover polygon min {min(y_coords)}"
        )
        assert grid_y_max_extent >= max(y_coords), (
            f"Grid Y max extent {grid_y_max_extent} doesn't cover polygon max {max(y_coords)}"
        )

        print(f"✓ Grid creation verified")
        print(f"  - Grid X: {len(grid_x)} points, "
              f"centers [{grid_x.min():.0f}, {grid_x.max():.0f}], "
              f"extent [{grid_x_min_extent:.0f}, {grid_x_max_extent:.0f}]")
        print(f"  - Grid Y: {len(grid_y)} points, "
              f"centers [{grid_y.min():.0f}, {grid_y.max():.0f}], "
              f"extent [{grid_y_min_extent:.0f}, {grid_y_max_extent:.0f}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
