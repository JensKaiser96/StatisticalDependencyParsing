import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.DataStructures.buffer import Buffer

OpenRight = 0
OpenLeft = 1
ClosedRight = 2
ClosedLeft = 3


def eisner(sigma: np.ndarray) -> WDG:
    n = sigma.shape[0]
    paths = calculate_max_tree(n, sigma)
    return create_tree(n, paths)


def calculate_max_tree(n: int, sigma: np.ndarray) -> np.ndarray:
    scores = np.zeros((4, n, n))
    paths = np.zeros((4, n, n), dtype=int)

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
                score = scores[ClosedLeft, s, q] + scores[ClosedRight, q+1, t] + sigma[s, t]
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
    buffer = Buffer([(ClosedLeft, 0, n-1, paths[ClosedLeft, 0, n-1])])
    no_edges = 0
    while buffer:
        #print(f"{len(buffer)=}")
        top = buffer.pop()
        step, s, t, q = top
        #print(f"{top=}")
        if s == t:
            continue
        if step == OpenRight:
            tree.add_edge(t, s)
            no_edges += 1
            if no_edges == n - 1:
                if buffer:
                    print(f"Failed to build proper tree.")
                break
            buffer.add((ClosedLeft, s, q, paths[ClosedLeft, s, q]))
            buffer.add((ClosedRight, q+1, t, paths[ClosedRight, q+1, t]))
        if step == OpenLeft:
            tree.add_edge(s, t)
            no_edges += 1
            if no_edges == n - 1:
                if buffer:
                    print(f"Failed to build proper tree.")
                break
            buffer.add((ClosedLeft, s, q, paths[ClosedLeft, s, q]))
            buffer.add((ClosedRight, q+1, t, paths[ClosedRight, q+1, t]))
        if step == ClosedRight:
            buffer.add((ClosedRight, s, q, paths[ClosedRight, s, q]))
            buffer.add((OpenRight, q, t, paths[OpenRight, q, t]))
        if step == ClosedLeft:
            buffer.add((OpenLeft, s, q, paths[OpenLeft, s, q]))
            buffer.add((ClosedLeft, q, t, paths[ClosedLeft, q, t]))
    return tree
