from typing import List

import numpy as np
from src.DataStructures.buffer import Buffer

ROOT = 0


class WeightedDirectedGraph:
    """
    represent a graph with a matrix
    N Nodes: index representation, link to object?
    N x N matrix
    values denote weight
    """
    data: np.ndarray

    def __init__(self):
        self.data = np.zeros((1, 1))
        pass

    @property
    def number_of_nodes(self) -> int:
        return len(self.data)

    @property
    def number_of_edges(self) -> int:
        return np.count_nonzero(self.data)

    @property
    def node_ids(self) -> List[int]:
        return [node_id for node_id in range(self.number_of_nodes)]

    def _heads_list(self, node_id) -> np.ndarray:
        return self.data[:, node_id]

    def _dependents_list(self, node_id) -> np.ndarray:
        return self.data[node_id, :]

    def add_edge(self, start_id: int, end_id: int, weight: float = 1):
        if not weight:
            raise ValueError("input argument 'weight' must be something else than 0.")
        # make data matrix larger if necessary
        if max(start_id, end_id) > len(self.data):
            self._expand_data_table(max(start_id, end_id))
        self.data[start_id, end_id] = weight
        return self

    def remove_edge(self, start_id: int, end_id: int):
        if not self.data[start_id, end_id]:
            raise ValueError(f"No edge from '{start_id}' to '{end_id}' in graph.")
        self.data[start_id, end_id] = 0
        return self

    def get_max_head(self, node_id: int) -> int | None:
        if not self.has_head(node_id):
            return None
        return self._heads_list(node_id).argmax()

    def has_head(self, node_id: int) -> bool:
        return bool(self._heads_list(node_id).max())

    def get_dependents(self, node_id: int) -> np.ndarray[int]:
        return np.nonzero(self._dependents_list(node_id))[0]

    def _expand_data_table(self, size_new: int):
        size_old = self.number_of_nodes
        size_diff = size_new - size_old
        if size_diff < 0:
            raise ValueError(f"Value of size_new needs to be larger than size_old. {size_new=}, {size_old=}")
        self.data = np.append(self.data, np.zeros((size_diff, size_old)), axis=0)
        self.data = np.append(self.data, np.zeros((size_new, size_diff)), axis=1)

    def has_dangling_nodes(self) -> bool:
        for node_id in self.node_ids:
            if not self.has_head(node_id):
                return True
        return False

    def is_well_formed(self) -> bool:
        return not (
                self.has_head(ROOT) or
                self.number_of_edges != self.number_of_nodes - 1 or
                self.has_dangling_nodes() or
                any([len(self._heads_list(node_id)) > 1 for node_id in self.node_ids])
        )

    def get_cycle(self) -> List[int]:
        """
        t[i,j] = [0,0,0,0,1]
        """
        t = np.zeros((self.number_of_nodes,) * 3, dtype=bool)
        # t[:] = -np.inf
        for i in self.node_ids:
            t[i, i] = self._dependents_list(i)

        changed = True
        while changed:
            for i in self.node_ids:
                for j in np.where(t[i] != 0): # indices of all possible paths from i
                    # check for new destinations
                    # all of j's possible destinations via any
                    t[j].max(axis=0)
                    t[i,j] += t[j].
                    if not (np.)








        childrens_list = {node_id: [] for node_id in self.node_ids}
        node_buffer = Buffer([ROOT])    # buffer that contains the nodes that will be visited next
        active_branch = Buffer([ROOT])  # buffer that contains every parent of the current node
        while node_buffer:
            # add childrens to current buffer
            for node_id in active_branch:
                node_buffer.pop()
                node_buffer.add_list(self.get_dependents(active_branch.top))
                new_childrens = self.get_dependents(active_branch.top)
                old_childrens = childrens_list[node_id]
                if any([child in childrens_list[node_id] for child in childrens]):
                    # found cycle
                    pass
                childrens_list[node_id] += childrens
                node_buffer.add_list(childrens)

        pass
