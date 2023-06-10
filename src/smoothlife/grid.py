from typing import Tuple

import numpy as np

from smoothlife.magic import MagicNums
from smoothlife.magic import MagicFunc

def random_init_center(arr, center_size):
    # Get the dimensions of the input array
    height, width = arr.shape

    # Calculate the indices for the center portion
    center_start_row = (height - center_size) // 2
    center_end_row = center_start_row + center_size
    center_start_col = (width - center_size) // 2
    center_end_col = center_start_col + center_size

    # Create a mask for the center portion
    center_mask = np.zeros_like(arr, dtype=bool)
    center_mask[center_start_row:center_end_row, center_start_col:center_end_col] = True

    # Generate random values for the center portion
    center_values = np.random.rand(center_size, center_size)

    # Assign the random values to the center portion
    arr[center_mask] = center_values.flatten()

    return arr


def calculating_s(cx: int, cy: int, grid: np.ndarray, o_radius: float = MagicNums.outer_radius) -> float:
    """
    Get center for the circle cx, cy and  compute m, n with radius in that grid.
    """
    i_radius = o_radius / 3
    height, width = grid.shape
    yy, xx = np.mgrid[0:height, 0:width]

    dx = np.minimum(np.abs(xx - cx), width - np.abs(xx - cx))
    dy = np.minimum(np.abs(yy - cy), height - np.abs(yy - cy))

    distance_sq = dx ** 2 + dy ** 2

    inner_circle = distance_sq <= i_radius ** 2
    outer_circle = np.logical_and(distance_sq > i_radius ** 2, distance_sq <= o_radius ** 2)

    m_inner = np.mean(grid[inner_circle])
    n_outer = np.mean(grid[outer_circle])

    # return 2 * MagicFunc.next_state(n_outer, m_inner) - 1
    return 2 * MagicFunc.next_state(m_inner, n_outer) - 1


def update_u_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray):
    for i in range(base_grid_obj.shape[0]):
        for j in range(base_grid_obj.shape[1]):
            update_grid_obj[i, j] = calculating_s(i, j, base_grid_obj)


def update_b_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray) -> np.ndarray:
    """
    Here base_grid_obj and update_grid_obj must have same shape.
    """
    base_grid_obj = base_grid_obj + MagicNums.dt * update_grid_obj

    return np.clip(base_grid_obj, 0, 1)