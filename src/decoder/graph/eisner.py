from enum import Enum

import numpy as np
from typing import List

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.DataStructures.buffer import Buffer


class StepType(Enum):
    OpenRight = 0
    OpenLeft = 1
    ClosedRight = 2
    ClosedLeft = 3




def calculate_max_tree(sentence: List[str], sigma: np.ndarray):
    n = len(sentence)
    scores_open_right = np.zeros((n, n))
    scores_open_left = np.zeros((n, n))
    scores_closed_right = np.zeros((n, n))
    scores_closed_left = np.zeros((n, n))

    path_open_right = np.zeros((n, n))
    path_open_left = np.zeros((n, n))
    path_closed_right = np.zeros((n, n))
    path_closed_left = np.zeros((n, n))

    for m in range(1, n+1):
        for s in range(0, n-m):
            t = s + m
            max_open_right = 0
            for q in range(s, t):
                cur_open_right = scores_closed_left[s, q] + scores_closed_right[q+1, t] + sigma[t, s]
                if cur_open_right > max_open_right:
                    max_open_right = cur_open_right
                    path_open_right[s, t] = q
            scores_open_right[s, t] = max_open_right

            max_open_left = 0
            for q in range(s, t):
                cur_open_left = scores_closed_left[s, q] + scores_closed_right[q+1, t] + sigma[s, t]
                if cur_open_left > max_open_left:
                    max_open_left = cur_open_left
                    path_open_left[s, t] = q
            scores_open_left[s, t] = max_open_left

            max_closed_right = 0
            for q in range(s, t):
                cur_closed_right = scores_closed_right[s, q] + scores_open_right[q, t]
                if cur_closed_right > max_closed_right:
                    max_closed_right = cur_closed_right
                    path_closed_right[s, t] = q
            scores_closed_right[s, t] = max_closed_right

            max_closed_left = 0
            for q in range(s+1, t+1):
                cur_closed_left = scores_open_left[s, q] + scores_closed_left[q, t]
                if cur_closed_left > max_closed_left:
                    max_closed_left = cur_closed_left
                    path_closed_left[s, t] = q
            scores_closed_left[s, t] = max_closed_left
    return path_open_right, path_open_left, path_closed_right, path_closed_left


def create_tree(n, path_open_right, path_open_left, path_closed_right, path_closed_left):
    tree = WDG()

    class Step:
        type_: StepType
        s: int
        t: int
        q: int

        def __init__(self, type_: StepType, s: int, t: int, q: int):
            self.type_ = type_
            self.s = s
            self.t = t
            self.q = q

    expand_buffer = Buffer([Step(StepType.ClosedLeft, 0, n, path_closed_left[0, n])])
    while expand_buffer:
        current_step = expand_buffer.pop()
        s = current_step.s
        t = current_step.t
        if s == t:
            continue
        q = current_step.q
        if current_step.type_ == StepType.OpenRight:
            tree.add_edge(t, s)
            expand_buffer.add(Step(StepType.ClosedLeft, s, q, path_open_right[s, q]))
            expand_buffer.add(Step(StepType.ClosedRight, q+1, t, path_open_right[q+1, t]))
        if current_step.type_ == StepType.OpenLeft:
            tree.add_edge(s, t)
            expand_buffer.add(Step(StepType.ClosedLeft, s, q, path_open_left[s, q]))
            expand_buffer.add(Step(StepType.ClosedRight, q+1, t, path_open_left[q+1, t]))
        if current_step.type_ == StepType.ClosedRight:
            expand_buffer.add(Step(StepType.ClosedRight, s, q, path_closed_right[s, q]))
            expand_buffer.add(Step(StepType.OpenRight, q, t, path_open_right[q, t]))
        if current_step.type_ == StepType.ClosedLeft:
            expand_buffer.add(Step(StepType.ClosedLeft, s, q, path_closed_left[s, q]))
            expand_buffer.add(Step(StepType.OpenLeft, q, t, path_open_left[q, t]))
