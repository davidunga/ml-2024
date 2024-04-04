from typing import Dict, Tuple, Sequence, List
from itertools import product, permutations
import numpy as np

Coord = Tuple[int, ...]


class Grid:
    """
    A multi-dim grid
    """

    def __init__(self, grid: Dict | Sequence):
        if not isinstance(grid, Dict):
            self.grid = {i: np.asarray(ax) for i, ax in enumerate(grid)}
        else:
            self.grid = {name: np.asarray(ax) for name, ax in grid.items()}
        for ax in self.grid.values():
            if np.issubdtype(ax.dtype, np.number):
                assert np.all(np.diff(ax) > 0), "Numeric values must be sorted in ascending order"
        self.shape = tuple(len(ax) for ax in self.grid.values())
        self.total_size = int(np.prod(self.shape))
        self.ndim = len(self.grid)

    @property
    def ax_names(self) -> List:
        return list(self.grid)

    def is_valid_coord(self, coord) -> bool:
        if not len(coord) == self.ndim:
            return False
        if not all(i in range(m) for i, m in zip(coord, self.shape)):
            return False
        return True

    def safe_cast_to_coord(self, coord) -> Coord:
        assert self.is_valid_coord(coord), f"Invalid coord for grid shape {self.shape}: {coord}"
        return coord if isinstance(coord, tuple) else tuple(int(i) for i in coord)

    def get_refined(self, s: int = 2):
        """
        Get a copy of the grid, up-sampled by a factor s, where s > 0 is a power of 2.
        Integer type values are not refined beyond integer values. For example,
        an integer axis [0, 2, 3] is refined to [0,1,2,3].
        Returns:
            - A new grid instance
            - A function that maps the original coordinates to the refined coordinates
        """
        assert s & (s - 1) == 0, "factor must be power of 2"

        mappings = []
        refined_grid = {}
        for name, ax in self.grid.items():
            m = len(ax)
            ticks = np.arange(m)
            refined_ticks = np.linspace(0, m - 1, s * m - 1)
            refined_ax = np.interp(refined_ticks, ticks, ax)
            if np.issubdtype(ax.dtype, int):
                refined_ax = np.unique(refined_ax.round().astype(int))
                mapping = refined_ax.searchsorted(ax)
            else:
                mapping = s * ticks

            refined_grid[name] = refined_ax
            mappings.append(mapping)

        def coord_map(coord: Coord) -> Coord:
            """ map pre-refined coordinate to refined coordinate """
            return tuple(mapping[i] for mapping, i in zip(mappings, coord))

        return Grid(refined_grid), coord_map

    def corner_coords(self) -> List[Coord]:
        return list(product(*((0, m - 1) for m in self.shape)))

    def center_coord(self) -> Coord:
        return tuple(m // 2 for m in self.shape)

    def uniform_sample_coords(self, margin: int = 0, k: int = 2) -> List[Coord]:
        """
        samples coordinates between center to each (corner - <margin>). the number of
        samples in each direction is determined by k; k = 0: only center,
        k = 1: center + corner, k = 2: center + corner + mid-point, etc.
        for a d-dim grid, returns k*(2^d) + 1 coordinates.
        """
        coords = [self.center_coord()]
        center = np.array(coords[0], float)
        for corner in self.corner_coords():
            corner = np.fromiter((c - margin if c else margin for c in corner), float)
            for s in range(1, k + 1):
                vec = (corner - center) * s / k
                coords.append(self.safe_cast_to_coord(np.round(center + vec)))
        return coords

    # ------

    def coord2dict(self, coord: Sequence[int], gridlike: bool = False) -> Dict:
        return {name: ax[i:i+1] if gridlike else ax[i]
                for i, (name, ax) in zip(coord, self.grid.items())}

    def dict2coord(self, d: Dict) -> Coord:
        tol = 100 * np.finfo(float).eps

        def _index(arr, val) -> int:
            return int(np.argmax(np.abs(arr - val) < tol))

        return tuple(_index(ax, d[name]) for name, ax in self.grid.items())

    def coords2inds(self, coords: Sequence[Sequence[int]]) -> np.ndarray[int]:
        """ convert coordinates to flat indices """
        return np.ravel_multi_index(np.asarray(coords).T, self.shape)

    def inds2coords(self, inds: Sequence[int]) -> List[Coord]:
        """ convert flat indices to coordinates """
        return list(zip(*np.unravel_index(inds, self.shape)))

    # ------

    def nhood_coords(self, center: Coord, kind: str, steps: int | Sequence[int] = 1, margin: int = 0) -> List[Coord]:
        bounds = (margin, np.array(self.shape) - margin)
        coords = [self.safe_cast_to_coord(coord) for coord in
                  get_neighborhood(center, kind=kind, steps=steps, bounds=bounds)]
        return coords

    def __iter__(self):
        """ iterate (coord, dict) pairs """
        return ((coord, self.coord2dict(coord)) for coord in self.iter_coords())

    def iter_coords(self):
        yield from (self.inds2coords([ind])[0] for ind in range(self.total_size))


def get_neighborhood(
        center: Sequence[int],
        kind: str = 'moore',
        steps: int | Sequence[int] = 1,
        bounds: Tuple = None) -> np.ndarray[int]:
    """
    - center: coordinate at the center of the neighborhood
    - kind: one of {'moore', 'vonn'},
        'moore' = Moore, full neighborhood, analogous to 8-neighborhood in 2D
        'vonn' = Von Neumann, partial neighborhood, analogous to 4-neighborhood in 2D
    - steps: step size as positive integer, or as list of step sizes for each dimension (same size as center)
    - bounds: a tuple (lb, ub), where each of lb, ub is either an integer or a list of integers, same size as center.
        a coordinate is valid if lb[i] <= coordinate[i] < ub[i] for all i.
        if lb or ub are integer, the value is replicated to all dimensions.
    Returns:
        np array of neighborhood coordinates
    """

    d = len(center)
    if isinstance(steps, int):
        steps = [steps] * d

    if kind == 'vonn':
        offsets = np.vstack((np.diag(steps), -np.diag(steps)))
    elif kind == 'moore':
        offsets = np.swapaxes(np.meshgrid(*[[-s, 0, s] for s in steps]), -1, 0).reshape(-1, d)
        offsets = offsets[np.any(offsets, axis=1)]
    else:
        raise ValueError("Unknown neighborhood kind")

    coords = np.array(center) + offsets

    if bounds:
        lb, ub = bounds
        coords = coords[np.all((coords < ub) & (coords >= lb), axis=1)]

    return coords
