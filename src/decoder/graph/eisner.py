import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.DataStructures.buffer import Buffer

OpenRight = 0
OpenLeft = 1
ClosedRight = 2
ClosedLeft = 3


def eisner(n: int, sigma: np.ndarray) -> WDG:
    paths = calculate_max_tree(n, sigma)
    return create_tree(n, paths)


def calculate_max_tree(n: int, sigma: np.ndarray) -> np.ndarray:
    scores = np.zeros((4, n, n))
    paths = np.zeros((4, n, n))

    def update_cell(step_type: int):
        if score > scores[step_type, s, t]:
            scores[step_type, s, t] = score
            paths[step_type, s, t] = q

    for m in range(1, n+1):
        for s in range(0, n-m):
            t = s + m

            for q in range(s, t):
                score = scores[ClosedLeft, s, q] + scores[ClosedRight, q+1, t] + sigma[t, s]
                update_cell(OpenRight)

            for q in range(s, t):
                score = scores[ClosedLeft, s, q] + scores[ClosedRight] + sigma[s, t]
                update_cell(OpenLeft)

            for q in range(s, t):
                score = scores[ClosedRight, s, q] + scores[OpenRight, q, t]
                update_cell(ClosedRight)

            for q in range(s+1, t+1):
                score = scores[OpenLeft, s, q] + scores[ClosedLeft, q, t]
                update_cell(ClosedLeft)
    return paths


def create_tree(n, paths) -> WDG:
    tree = WDG()
    buffer = Buffer([(ClosedLeft, 0, n, paths[ClosedLeft, 0, n])])

    while buffer:
        step, s, t, q = buffer.pop()
        if s == t:
            continue
        q = step.q
        if step == OpenRight:
            tree.add_edge(t, s)
            buffer.add((ClosedLeft, s, q, paths[OpenRight, s, q]))
            buffer.add((ClosedRight, q+1, t, paths[OpenRight, q+1, t]))
        if step == OpenLeft:
            tree.add_edge(s, t)
            buffer.add((ClosedLeft, s, q, paths[OpenLeft, s, q]))
            buffer.add((ClosedRight, q+1, t, paths[OpenLeft, q+1, t]))
        if step == ClosedRight:
            buffer.add((ClosedRight, s, q, paths[ClosedRight, s, q]))
            buffer.add((OpenRight, q, t, paths[OpenRight, q, t]))
        if step == ClosedLeft:
            buffer.add((ClosedLeft, s, q, paths[ClosedLeft, s, q]))
            buffer.add((OpenLeft, q, t, paths[OpenLeft, q, t]))
    return tree
