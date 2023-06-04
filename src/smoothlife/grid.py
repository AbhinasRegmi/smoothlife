from typing import Tuple
from dataclasses import dataclass

import numpy as np

from smoothlife.magic import MagicNums
from smoothlife.magic import MagicFunc


@dataclass(frozen=True)
class GRID:
    WIDTH: int = 10
    HEIGHT: int = 10


def calc_m_n(cx: int, cy: int, grid: np.ndarray, o_radius: float = MagicNums.outer_radius) -> Tuple[float, float]:
    """
    Get center for the circle cx, cy and  compute m, n with radius in that grid.
    """
    i_radius: float = o_radius / 3.0

    m_inner: float = 0.0
    area_inner: int = 0

    n_outer: float = 0.0
    area_outer: int = 0

    for dy in range(int(-o_radius), int(o_radius) + 1):
        for dx in range(int(-o_radius), int(o_radius) + 1):
            px_grid: int = (dx + cx) % GRID.WIDTH
            py_grid: int = (dy + cy) % GRID.HEIGHT

            if(dx ** 2 + dy ** 2 <= i_radius ** 2):
                m_inner += grid[py_grid, px_grid]
                area_inner += 1

            elif( dx ** 2 + dy ** 2 <= o_radius ** 2):
                n_outer += grid[py_grid, px_grid]
                area_outer += 1
    
    m_inner /= area_inner
    n_outer /= area_outer

    return m_inner, n_outer

            
def update_update_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray) -> None:
    """
    Here base_grid_obj and update_grid_obj must have same shape.
    """
    for y in range(GRID.HEIGHT):
        for x in range(GRID.WIDTH):
            m, n = calc_m_n(x, y, base_grid_obj)
            res = MagicFunc.next_state(n, m)
            update_grid_obj[y, x] = 2 * res -1

def update_base_grid(base_grid_obj: np.ndarray, update_grid_obj: np.ndarray) -> None:
    """
    Here base_grid_obj and update_grid_obj must have same shape.
    """
    for y in range(GRID.HEIGHT):
        for x in range(GRID.WIDTH):
            base_grid_obj[y, x] += MagicNums.dt * update_grid_obj[y, x]

            # clamping value to -1 to +1
            if base_grid_obj[y, x] < -1:
                base_grid_obj[y, x] = -1
            elif base_grid_obj[y, x] > 1:
                base_grid_obj[y, x] = 1