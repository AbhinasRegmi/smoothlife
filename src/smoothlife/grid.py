from typing import Tuple
from dataclasses import dataclass

import numpy as np

from smoothlife.magic import MagicNums
from smoothlife.magic import MagicFunc


@dataclass(frozen=True)
class GRID:
    WIDTH: int = 500
    HEIGHT: int = 500

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



# def calculating_m_n(cx: int , cy: int, grid: np.ndarray, o_radius: float = MagicNums.outer_radius):
#     """
#     optimized version for calculating m, n
#     """
#     i_radius: float = o_radius / 3.0

#     y_indices, x_indices = np.indices(grid.shape)
    
#     distance_squared = (cx - x_indices) ** 2 + (cy - y_indices) ** 2

#     inner_circle = distance_squared <= i_radius ** 2
#     outer_circle = np.logical_and(distance_squared > i_radius ** 2, distance_squared <= o_radius ** 2)


#     m_inner = np.mean(grid[inner_circle])
#     m_outer = np.mean(grid[outer_circle])

#     grid[outer_circle] = 1

#     return m_inner, m_outer


def calculating_m_n(cx: int, cy: int, grid: np.ndarray, o_radius: float = MagicNums.outer_radius) -> Tuple[float, float]:
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

    grid[inner_circle] = 1
    grid[outer_circle] = 1

    return m_inner, n_outer

def update_update_grid(base_grid_obj: np.ndarray) -> np.ndarray:
    """
    Well i tried to optimize. The best i could.
    """
    y_indices, x_indices = np.indices(base_grid_obj.shape)

    vectorized_func_for_mn = np.vectorize(calculating_m_n, signature='(),(),(m,n)->(),()')
    m_array, n_array = vectorized_func_for_mn(x_indices, y_indices, base_grid_obj)

    vectorized_func_for_s = np.vectorize(MagicFunc.next_state, signature='(),()->()')
    result_states = vectorized_func_for_s(n_array, m_array)

    return 2 * result_states - 1
    

# def update_update_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray) -> None:
#     """
#     Here base_grid_obj and update_grid_obj must have same shape.
#     """

#     for y in range(GRID.HEIGHT):
#         for x in range(GRID.WIDTH):
#             m, n = calculating_m_n(x, y, base_grid_obj)
#             res = MagicFunc.result_states(n, m)
#             update_grid_obj[y, x] = 2 * res -1

def update_base_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray) -> np.ndarray:
    """
    Here base_grid_obj and update_grid_obj must have same shape.
    """
    base_grid_obj = base_grid_obj + MagicNums.dt * update_grid_obj

    return np.clip(base_grid_obj, -1, 1)
    # return base_grid_obj