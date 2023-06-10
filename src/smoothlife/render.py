import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from smoothlife.magic import GRID
from smoothlife.grid import update_u_grid
from smoothlife.grid import update_b_grid
from smoothlife.grid import random_init_center

# base_grid: np.ndarray = random_init_center(np.zeros((GRID.HEIGHT, GRID.WIDTH)), 15)
base_grid: np.ndarray = np.random.rand(GRID.HEIGHT, GRID.WIDTH)
update_grid: np.ndarray = np.zeros((GRID.HEIGHT, GRID.WIDTH))



fig, ax = plt.subplots()
plot = ax.imshow(base_grid, vmin=-1, vmax=1)

def _update(frame):
    global base_grid
    global update_grid

    update_u_grid(base_grid, update_grid)
    base_grid = update_b_grid(base_grid, update_grid)
    plot.set_array(base_grid)
    return plot,

def now(frames: int = 1000):
    animation = animation.FuncAnimation(fig, _update, frames=range(10000), interval=0)
    plt.show()