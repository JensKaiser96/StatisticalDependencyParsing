import logging
import math
from typing import List
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.patches import FancyArrowPatch, ConnectionStyle

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

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return str(self.data)

    def __eq__(self, other: "WeightedDirectedGraph"):
        return all(self.data == other.data)

    def copy(self):
        copy = WeightedDirectedGraph()
        copy.data = self.data.copy()
        return copy

    def random(self, size: int, make0root: bool = True) -> "WeightedDirectedGraph":
        if (self.data != np.zeros((1, 1))).all():
            raise ValueError("Weights are already initialized, this method is for creating a random graph from scratch")
        self.data = np.random.random((size, size))
        np.fill_diagonal(self.data, 0)
        if make0root:
            self.make_0_root()
        return self

    def draw(self):
        figure, axis = plt.subplots()

        # axis.set_aspect(1)
        # e^(2pif)

        def pos(node_id: int) -> (float, float):
            phi = 2 * math.pi / self.number_of_nodes
            x = math.cos(phi * node_id)
            y = math.sin(phi * node_id)
            return x, y

        for node_id in self.node_ids:
            plt.text(pos(node_id)[0], pos(node_id)[1], node_id,
                     verticalalignment='center', horizontalalignment='center',
                     bbox=dict(boxstyle="circle"))

            for dependent in self.get_dependents(node_id):
                axis.annotate("",
                              xy=pos(dependent), xycoords='data',
                              xytext=pos(node_id), textcoords='data',
                              arrowprops=dict(arrowstyle="->",
                                              connectionstyle=ConnectionStyle("Arc3, rad=0.2"),
                                              shrinkA=10,
                                              shrinkB=10),
                              )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        axis.axis('off')
        plt.show()
        return self

    def make_0_root(self):
        self.data[:, 0] = 0
        return self

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

    def _expand_data_table(self, size_new: int):
        size_old = self.number_of_nodes
        size_diff = size_new - size_old
        if size_diff <= 0:
            raise ValueError(f"Value of size_new needs to be larger than size_old. {size_new=}, {size_old=}")
        self.data = np.append(self.data, np.zeros((size_diff, size_old)), axis=0)
        self.data = np.append(self.data, np.zeros((size_new, size_diff)), axis=1)

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

    def remove_all_heads_but_max(self, node_id: int, cycle: List[int] = None):
        # safe index and value
        max_head_id = self.get_max_head(node_id, cycle)
        if max_head_id is None:
            return
        max_head_value = self.data[max_head_id, node_id]

        print(f"Remove all but max head:\n {node_id=}/{cycle=}\t{max_head_id=}/{max_head_value=}")

        # set all to 0
        if cycle is None:
            print(f"Non zeros: {np.nonzero(self.data[:, node_id])[0]}")
            self.data[:, node_id] = 0
        else:
            print(f"Non zeros: {np.nonzero(self.data[cycle, node_id])[0]}")
            self.data[cycle, node_id] = 0
        self.data[max_head_id, node_id] = max_head_value  # restore value at index
        return self

    def get_edge_weight(self, start_id: int, end_id: int) -> float:
        if not self.data[start_id, end_id]:
            raise ValueError(f"No edge from '{start_id}' to '{end_id}' in graph.")
        return self.data[start_id, end_id]

    def get_max_head(self, node_id: int, cycle: List[int] = None) -> int | None:
        if not self.has_head(node_id):
            return None
        if cycle:
            # returns the id out of the given cycle that has the highest edge to node_id
            return cycle[self.data[cycle, node_id].argmax()]
        else:
            return self._heads_list(node_id).argmax()

    def delete_node(self, node_id):
        """
        actually deletes a node, aka node indexes change
        """
        self.data = np.delete(self.data, node_id, node_id)
        return self

    def cut_node(self, node_id: int):
        """
        just removes all arcs from and to a node
        """
        self.data[:, node_id] = 0
        self.data[node_id, :] = 0
        return self

    def cut_nodes(self, node_ids: List[int]):
        for node_id in node_ids:
            self.cut_node(node_id)
        return self

    def has_head(self, node_id: int) -> bool:
        return bool(self._heads_list(node_id).max())

    def get_heads(self, node_id: int) -> np.ndarray[int]:
        return np.nonzero(self._heads_list(node_id))[0]

    def get_dependents(self, node_id: int) -> np.ndarray[int]:
        return np.nonzero(self._dependents_list(node_id))[0]

    def has_dangling_nodes(self) -> bool:
        for node_id in self.node_ids:
            if not self.has_head(node_id):
                return True
        return False

    def is_well_formed_tree(self) -> bool:
        root_has_head = self.has_head(ROOT)
        num_edges_vs_num_nodes = (self.number_of_edges != self.number_of_nodes - 1)
        dangling_nodes = self.has_dangling_nodes()
        more_than_one_head = any([len(self._heads_list(node_id)) > 1 for node_id in self.node_ids])

        if not (root_has_head or num_edges_vs_num_nodes or dangling_nodes or more_than_one_head):
            print(
                f"false because of:\n{root_has_head=}\n{num_edges_vs_num_nodes}\n{dangling_nodes}\n{more_than_one_head}")
            return False
        return True

    def find_cycle(self):
        paths = {node_id: [[node_id, dependent] for dependent in self.get_dependents(node_id)] for node_id in
                 self.node_ids}

        for _ in self.node_ids:  # any cycle is found after N iterations
            for node, old_paths in paths.items():
                new_paths = []
                for old_path in old_paths:
                    new_destinations = paths[old_path[-1]]
                    for new_destination in new_destinations:
                        new_path = old_path[:-1] + new_destination
                        if node in new_path[1:]:
                            cycle = new_path[:new_path.index(node, 1)]
                            print(f"Found cycle: {cycle}")
                            return cycle
                        new_paths.append(new_path)
                paths[node] = new_paths
        return []

    def cycle_weight_minus_node(self, cycle: List[int], node_id: int) -> float:
        weight = 0
        for i, current_node in enumerate(cycle):
            next_node = cycle[(i + 1) % len(cycle)]  # wrapping to 0 at the end of the list
            if next_node == node_id:
                continue
            weight += self.get_edge_weight(current_node, next_node)
        return weight

    def weight_to_cycle(self, cycle: List[int], start_id: int, end_id: int) -> float:
        return self.get_edge_weight(start_id, end_id) + self.cycle_weight_minus_node(cycle, end_id)


if __name__ == '__main__':
    wdg = WeightedDirectedGraph().random(7).draw()
