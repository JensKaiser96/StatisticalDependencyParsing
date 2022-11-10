import logging
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
        largest_index = max(start_id, end_id)
        if largest_index >= self.number_of_nodes:
            self._expand_data_table(largest_index + 1)
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
        if size_diff <= 0:
            raise ValueError(f"Value of size_new needs to be larger than size_old. {size_new=}, {size_old=}")
        self.data = np.append(self.data, np.zeros((size_diff, size_old)), axis=0)
        self.data = np.append(self.data, np.zeros((size_new, size_diff)), axis=1)

    def has_dangling_nodes(self) -> bool:
        for node_id in self.node_ids:
            if not self.has_head(node_id):
                return True
        return False

    def is_well_formed_tree(self) -> bool:
        return not (
                self.has_head(ROOT) or
                self.number_of_edges != self.number_of_nodes - 1 or
                self.has_dangling_nodes() or
                any([len(self._heads_list(node_id)) > 1 for node_id in self.node_ids])
        )

    def find_cycle(self):
        paths = {node_id: [[node_id, dependent] for dependent in self.get_dependents(node_id)] for node_id in self.node_ids}

        for _ in self.node_ids:   # any cycle is found after N iterations
            for node, old_paths in paths.items():
                new_paths = []
                for old_path in old_paths:
                    new_destinations = paths[old_path[-1]]
                    for new_destination in new_destinations:
                        print(f"building new path:\n{old_path}\n{new_destination}\n")
                        new_path = old_path[:-1] + new_destination
                        if node in new_path[1:]:
                            print(f"found cycle: {new_path}")
                            cycle = new_path[:new_path.index(node, 1)]
                            return cycle
                        print(f"({node}):\n{old_path}\n{new_path}")
                        new_paths.append(new_path)
                paths[node] = new_paths
        return []

