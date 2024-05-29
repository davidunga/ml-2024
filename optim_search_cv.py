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
        3. If better params are found- repeat from 2
           If not: halve the step size and repeat from 2, or return (if step size = 1).
    """

    def __init__(self, estimator, param_grid: Dict, scales: int = 2, nrand: int = 2, **kwargs):
        """
        param_grid: parameters grid, currently only numeric values are supported
        scales: number of step size halvings. e.g. if scale=3, step sizes are 4 -> 2 -> 1
        nrand: number of random samples in each iteration
        """
        super().__init__(estimator, **kwargs)

        self.grid = Grid(param_grid)
        self.scales = scales
        self.nrand = nrand
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
        """ coordinate->loss dict """
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
        losses = loss_sign * np.asarray(result[loss_metric])
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

        # Random additional visits
        if self.get_queue() and self.nrand:
            visit_inds = self.grid.coords2inds(list(self.visited) + self.get_queue())
            non_visited = set(range(self.grid.total_size)).difference(visit_inds)
            n_additionals = min(len(non_visited), self.nrand)
            if n_additionals:
                rand_visit_inds = np.random.default_rng(len(self.rounds_log)).choice(list(non_visited), n_additionals)
                _add_visit(self.grid.inds2coords(rand_visit_inds), reason="rand")

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
            s = {'round': item.round, 'reason': item.reason, 'src': str(item.src) if item.src else "",
                 'msg': f"-> {item.msg}" if item.msg else "", 'coord': coord,
                 'loss': round(self.visited[coord], 4) if coord in self.visited else "<pending>",
                 'roc_auc': round(item.scores['roc_auc'], 4) if coord in self.visited else "<pending>"}
            print("{coord} round:{round}, reason:{reason}{src} roc_auc={roc_auc} loss={loss} {msg}".format(**s))

    def _run_search(self, evaluate_candidates):
        while True:
            candidates = self.get_candidates()
            if not candidates:
                break
            print(f"Dispatched {len(candidates)} candidates")
            result = evaluate_candidates(ParameterGrid(candidates))
            self.process_visit_result(result)
            self.tell_visits_history()
        print("DONE")


def _show_synthetic_example():
    from xgboost import XGBClassifier

    # make synthetic loss landscape
    param_grid = {'max_depth': np.arange(1, 31), 'learning_rate': np.arange(2, 39)}
    estimator = XGBClassifier(random_state=1)
    cv = OptimSearchCV(estimator, param_grid=param_grid, refit='neg_log_loss', scales=3, nrand=0)
    h, w = cv.grid.shape
    xx, yy = np.meshgrid(range(w), range(h))
    optimal_xy = w // 4, h // 2
    gt_loss = np.sqrt((xx - optimal_xy[0]) ** 2 + (yy - optimal_xy[1]) ** 2)

    # run search
    while True:
        cands = cv.get_candidates()
        if not cands:
            break
        result = {'params': cands, 'mean_test_neg_log_loss': [gt_loss[cv.grid.dict2coord(cand)] for cand in cands]}
        cv.process_visit_result(result)

    # draw iterations
    curr_kws = {'color': 'limeGreen', 's': 50}
    past_kws = {'color': 'gray', 's': 48, 'alpha': .8}
    best_kws = {**curr_kws, 'edgecolors': 'Gold', 'lw': 2}

    rounds_to_draw = range(len(cv.rounds_log) - 1)
    pts = np.array([coord[::-1] for coord in cv.visit_log])
    is_rand_visit = np.array([cv.visit_log[coord].reason == "rand" for coord in cv.visit_log])
    visit_rounds = np.array([cv.visit_log[coord].round for coord in cv.visit_log])

    best_coord_per_round = []
    for visit_round in rounds_to_draw:
        best = [coord for coord in cv.visited
                if cv.visit_log[coord].round == visit_round and cv.visit_log[coord].msg == 'NewBest']
        best_coord_per_round.append(best[0] if len(best) else best_coord_per_round[-1])
    best_loss_per_round = [cv.visited[coord] for coord in best_coord_per_round]

    for visit_round in rounds_to_draw:
        plt.figure()
        plt.imshow(gt_loss, cmap='hot')
        plt.title(f"Iter {visit_round}, StepSize={cv.rounds_log[visit_round]['step_size']}")
        ii = visit_rounds < visit_round
        if np.any(ii):
            plt.scatter(*pts[ii].T, color='gray', alpha=.8, s=48)
        ii = visit_rounds == visit_round
        if np.any(ii):
            plt.scatter(*pts[ii].T, **curr_kws)
            plt.scatter(best_coord_per_round[visit_round][1], best_coord_per_round[visit_round][0], **best_kws)
            plt.scatter(*pts[ii & is_rand_visit].T, color='k', marker='.')

    plt.figure()
    plt.plot(best_loss_per_round, 'dodgerBlue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs optimization progress')
    plt.show()


if __name__ == "__main__":
    _show_synthetic_example()
