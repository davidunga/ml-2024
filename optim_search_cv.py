from typing import List, Dict, Tuple, Set, Collection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid
from sklearn.metrics import get_scorer
from grid import Grid, Coord
from dataclasses import dataclass


@dataclass
class VisitItem:
    round: int  # visits round
    reason: str  # visit reason
    src: Coord = None  # source coordinate
    loss: float = None  # result loss
    msg: str = ''  # optional message
    scores: Dict[str, float] = None


class OptimSearchCV(BaseSearchCV):
    """
    Optimizer of CV-searching hyperparams
    Method:
        1. Initialize by uniformly sampling params grid
        2. Sample:
            - The direct (Von Neumann) neighborhood of best params
            - In the direction of gradient
        3. If better better params are found- repeat from 2
           If not: reduce step size and repeat from 2, or return
    """

    def __init__(self, estimator, param_grid: Dict, scales: int = 2, visualize: bool = False, **kwargs):
        """
        param_grid: parameters grid, currently only numeric values are supported
        scales: number of x2 step size scalings, must be a power of two
        """
        super().__init__(estimator, **kwargs)

        self.grid = Grid(param_grid)
        self.scales = scales
        self.visualize = visualize

        self.visit_log: Dict[Coord, VisitItem] = {}
        self.rounds_log: List[Dict] = []
        self._visited: Dict[Coord, float] = {}
        self._best_coord: Coord = None
        self.step_size = 2 ** (scales - 1)

    def _refresh(self):
        self._visited = {coord: item.loss for coord, item in self.visit_log.items() if item.loss is not None}
        self._best_coord = min(self.visited, key=self.visited.get)

    @property
    def visited(self) -> Dict[Coord, float]:
        return self._visited

    @property
    def param_grid(self):
        return self.grid.grid

    @property
    def best_coord(self) -> Coord:
        return self._best_coord

    @property
    def best_loss(self) -> float:
        return self.visited[self.best_coord]

    @property
    def best_params(self) -> Dict:
        return self.grid.coord2dict(self.best_coord)

    def _step_size_per_axs(self) -> np.ndarray[int]:
        return np.fromiter((max(1, min(self.step_size, len(ax) - 1)) if np.issubdtype(ax.dtype, np.number) else 1
                            for ax in self.grid.grid.values()), int)

    def get_queue(self) -> List[Coord]:
        """ list of non-visited coordinates from visit log """
        return [coord for coord in self.visit_log if coord not in self.visited]

    def _loss_metric_spec(self):
        metric = f"mean_test_{self.refit}"
        sgn = -1 * get_scorer(self.refit)._sign
        name = ("-" if sgn < 0 else "") + self.refit
        return metric, sgn, name

    def process_visit_result(self, result: Dict):
        loss_metric, loss_sign, _ = self._loss_metric_spec()
        losses = loss_sign * result[loss_metric]
        params_list = result['params']
        for i, (params, loss) in enumerate(zip(params_list, losses)):
            coord = self.grid.dict2coord(params)
            if self.visit_log[coord].loss is None:
                self.visit_log[coord].loss = loss
                self.visit_log[coord].scores = {k.replace('mean_test_', ''): float(v[i])
                                                for k, v in result.items() if k.startswith('mean_test_')}

        self._refresh()
        self.visit_log[self.best_coord].msg = 'NewBest'

    def get_candidates(self) -> List[Dict]:
        """ plan and return parameters to sample """
        self._plan_next_visits()
        return [self.grid.coord2dict(coord, gridlike=True) for coord in self.get_queue()]

    def _plan_next_visits(self, new_round: bool = True) -> None:

        if new_round:
            self.rounds_log.append({})

        self.rounds_log[-1]['step_size'] = self.step_size

        def _add_visit(coords: List[Coord], reason: str, src: Coord = None):
            for coord in set(coords).difference(self.visit_log):
                self.visit_log[coord] = VisitItem(round=len(self.rounds_log) - 1, reason=reason, src=src)

        # Initialize
        if not self.visit_log:
            _add_visit(self.grid.uniform_sample_coords(k=2, margin=1), reason="init")
            return

        # Von Neumann neighborhood of best point- preparation for gradient descent
        nhood = self.grid.nhood_coords(self.best_coord, 'vonn', steps=self.step_size)
        _add_visit(nhood, reason="nhood", src=self.best_coord)

        # Gradient descent:
        # computed for all points that have sufficiently sampled neighborhood
        gd = self.gradient_descent(self.visited)
        for center, (coord, est_loss) in gd.items():
            # gradient from center points to coord, est_loss = estimated loss at coord
            if est_loss < self.best_loss and coord not in self.visited:
                _add_visit([coord], reason=f"explore:loss={est_loss:2.4f}", src=center)

        if not self.get_queue() and self.step_size > 1:
            # converged (no visits were added) -> halve step size
            self.step_size //= 2
            self._plan_next_visits(new_round=False)

    def gradient_descent(self, centers: Collection[Coord]) -> Dict:
        """
        centers: coordinates to perform GD around

        returns a dict of the form {center: (coord, loss), ..}
            coord is the coordinate down the gradient from center (may be equal to center if locally converged)
            loss is the estimated loss at the coordinate
        if center does not have a sufficient neighborhood it is not included in result
        """

        gd = {}
        for center in centers:

            partial_nhood = self.grid.nhood_coords(center, 'vonn', steps=self._step_size_per_axs())
            if not set(partial_nhood).issubset(self.visited):
                continue

            # ---
            # construct loss-per-offset matrix:

            losses = np.zeros((self.grid.ndim, 3), float) + np.inf  # loss per offsets step in each axis
            losses[:, 1] = self.visited[center]  # middle column corresponds to zero offset
            directions = np.sign(np.array(partial_nhood) - center)
            for coord, direction in zip(partial_nhood, directions):
                d = np.nonzero(direction)[0]  # axis of offset
                losses[d, direction[d] + 1] = self.visited[coord]

            # ---
            # take best offset from each axis:

            best_combined_direction = np.argmin(losses, axis=1) - 1
            coord = tuple(np.array(center) + best_combined_direction * self._step_size_per_axs())
            gd[center] = (coord, float(losses.mean()))

        return gd

    def tell_visits_history(self):
        print("Visits history:")
        prev_round = -1
        for coord, item in self.visit_log.items():
            if item.round != prev_round:
                prev_round = item.round
                print(f"Round {item.round} (Step size {self.rounds_log[item.round]['step_size']}):")
            s = {'round': item.round,
                 'reason': item.reason,
                 'src': str(item.src) if item.src else "",
                 'msg': f"-> {item.msg}" if item.msg else "",
                 'loss': round(self.visited[coord], 4) if coord in self.visited else "<pending>",
                 'roc_auc': round(item.scores['roc_auc'], 4) if coord in self.visited else "<pending>",
                 'coord': coord}
            print("{coord} round:{round}, reason:{reason}{src} roc_auc={roc_auc} loss={loss} {msg}".format(**s))

    def visualize_visits(self, latest_only: bool):
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        def _draw(coords):

            colors = colormaps['terrain'](np.linspace(0, .5, 10))
            loss_cmap = colormaps['hot']
            loss_cmap.set_bad(color='blue')

            assert self.grid.ndim == 2

            loss_image = np.zeros(self.grid.shape[::-1], float) + np.nan
            for coord in coords:
                loss_image[coord[::-1]] = self.visited[coord]

            plt.figure()
            plt.imshow(loss_image, cmap=loss_cmap)
            plt.colorbar()

            ax_names = self.grid.ax_names

            ax = self.grid.grid[ax_names[0]]
            plt.xticks(ticks=np.arange(len(ax)), labels=ax)
            plt.xlabel(ax_names[0])

            ax = self.grid.grid[ax_names[1]]
            plt.yticks(ticks=np.arange(len(ax)), labels=ax)
            plt.ylabel(ax_names[1])

            for coord in coords:
                item = self.visit_log[coord]
                txt = str(item.round) + ('*' if item.msg == 'NewBest' else '')
                plt.text(coord[0], coord[1], txt, color=colors[(item.round - 1) % len(colors)])

        _, _, loss_name = self._loss_metric_spec()
        rounds_to_draw = [len(self.rounds_log) - 1] if latest_only else range(len(self.rounds_log))
        for visit_round in rounds_to_draw:
            coords = [coord for coord in self.visited if self.visit_log[coord].round <= visit_round]
            _draw(coords)
            plt.suptitle(f"Sampled loss values [{loss_name}]")
            if coords:
                best_coord = min(coords, key=lambda coord: self.visited[coord])
                best_params = self.grid.coord2dict(best_coord)
                plt.title(f"Round {visit_round}, Best: {best_coord} : {best_params}")
            else:
                plt.title(f"Round {visit_round}, [no visits]")

    def _run_search(self, evaluate_candidates):

        while True:
            candidates = self.get_candidates()
            if not candidates:
                break
            print(f"Dispatched {len(candidates)} candidates")
            result = evaluate_candidates(ParameterGrid(candidates))
            self.process_visit_result(result)
            self.tell_visits_history()
            if self.visualize:
                self.visualize_visits(latest_only=True)
                plt.show()

        print("DONE")