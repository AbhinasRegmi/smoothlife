import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from smoothlife.grid import GRID
from smoothlife.grid import update_base_grid
from smoothlife.grid import update_update_grid

base_grid: np.ndarray = np.random.rand(GRID.HEIGHT, GRID.WIDTH)
update_grid: np.ndarray = np.zeros((GRID.HEIGHT, GRID.WIDTH))

fig, ax = plt.subplots()
plot = ax.imshow(base_grid, cmap='gray')

def update(frame):
    update_update_grid(base_grid, update_grid)
    update_base_grid(base_grid, update_grid)
    plot.set_array(base_grid)
    return plot,

animation = animation.FuncAnimation(fig, update, frames=range(10000), interval=200)
plt.show()