import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import ConnectionStyle

ROOT = 0


class WeightedDirectedGraph:
    """
    represent a graph with a matrix
    N Nodes: index representation, link to object?
    N x N matrix
    values denote weight
    """
    data: np.ndarray

    def __init__(self, size: int = 1, allow_negative_edges=False):
        self.allow_negative_edges = allow_negative_edges
        self.data = np.zeros((size, size))

    def __str__(self):
        return f"WeightedDirectedGraph with {self.number_of_nodes} nodes.\n{self.data}"

    def __eq__(self, other: "WeightedDirectedGraph"):
        size_diff = len(self) - len(other)
        if size_diff > 0:
            other.data = np.pad(other.data, (0, size_diff), mode="constant")
        if size_diff < 0:
            self.data = np.pad(self.data, (0, -size_diff), mode="constant")
        return (self.data == other.data).all()

    def __len__(self):
        return self.data.shape[0]

    def copy(self) -> "WeightedDirectedGraph":
        copy = WeightedDirectedGraph()
        copy.data = self.data.copy()
        return copy

    def to_nx(self) -> nx.DiGraph:
        nxg = nx.DiGraph()
        for node in self.nodes:
            for head in self.get_head_ids(node):
                nxg.add_edge(head, node, weight=self.get_edge_weight(head, node))
        return nxg

    @staticmethod
    def from_nx(nxg: nx.DiGraph) -> "WeightedDirectedGraph":
        wdg = WeightedDirectedGraph()
        for head, dependent, data in nxg.edges(data=True):
            wdg.add_edge(head, dependent, data["weight"])
        return wdg

    def nx_cle(self):
        """
        only used to verify correctness of my CLE implementation
        """
        nxg = self.to_nx()
        mst_nxg = nx.algorithms.tree.branchings.Edmonds(nxg).find_optimum(attr='weight', kind='max', style="aborescence", preserve_attrs=True)
        return WeightedDirectedGraph.from_nx(mst_nxg)

    def normalize(self):
        self.data[self.data > 0] = 1
        return self

    def count_common_edges(self, other: "WeightedDirectedGraph") -> int:
        try:
            return len(np.where(self.data * other.data > 0)[0])
        except ValueError:
            return 0

    @staticmethod
    def random(size: int, make0root: bool = True, seed: int = None) -> "WeightedDirectedGraph":
        graph = WeightedDirectedGraph()
        if seed is not None:
            np.random.seed(seed)
        if (graph.data != np.zeros((1, 1))).all():
            raise ValueError("Weights are already initialized, this method is for creating a random graph from scratch")
        graph.data = np.random.random((size, size))
        np.fill_diagonal(graph.data, 0)
        if make0root:
            graph.make_0_root()
        return graph

    def draw(self):
        figure, axis = plt.subplots()

        # axis.set_aspect(1)
        # e^(2pif)

        def pos(node_id: int) -> (float, float):
            phi = 2 * math.pi / self.number_of_nodes
            x = math.cos(phi * node_id)
            y = math.sin(phi * node_id)
            return x, y

        for node_id in self.nodes:
            plt.text(*pos(node_id), node_id,
                     verticalalignment='center', horizontalalignment='center',
                     bbox=dict(boxstyle="circle"))

            for dependent in self.get_dependent_ids(node_id):
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
        axis.axis("off")
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
    def nodes(self) -> List[int]:
        return [node_id for node_id in range(self.number_of_nodes)]

    def _heads_slice(self, node_id) -> np.ndarray:
        return self.data[:, node_id]

    def _dependents_slice(self, node_id) -> np.ndarray:
        return self.data[node_id, :]

    def _expand_data_table(self, size_new: int):
        size_old = self.number_of_nodes
        size_diff = size_new - size_old
        if size_diff <= 0:
            raise ValueError(f"Value of size_new needs to be larger than size_old. {size_new=}, {size_old=}")
        self.data = np.append(self.data, np.zeros((size_diff, size_old)), axis=0)
        self.data = np.append(self.data, np.zeros((size_new, size_diff)), axis=1)

    def add_edge(self, start_id: int, end_id: int, weight: float = 1, check_for_duplicate=False, allow_zero=True):
        if not weight and not allow_zero:
            raise ValueError("input argument 'weight' must be something else than 0.")
        if weight < 0 and not self.allow_negative_edges:
            weight = 0
        # make data matrix larger if necessary
        largest_index = max(start_id, end_id)
        if largest_index >= self.number_of_nodes:
            self._expand_data_table(largest_index + 1)
        if check_for_duplicate and self.data[start_id, end_id]:
            raise ValueError(f"Edge ({start_id},{end_id}) already exists")
        self.data[start_id, end_id] = weight
        return self

    def set_edge_weight(self, start_id: int, end_id: int, weight: float):
        self.add_edge(start_id, end_id, weight, check_for_duplicate=False)
        return self

    def remove_edge(self, start_id: int, end_id: int, strict=False):
        if strict and not self.data[start_id, end_id]:
            raise ValueError(f"No edge from '{start_id}' to '{end_id}' in graph.")
        self.data[start_id, end_id] = 0
        return self

    def remove_all_heads_but_max(self, node_id: int):
        # safe index and value
        max_head_id = self.get_max_head(node_id)
        if max_head_id is None:
            return self
        max_head_value = self.data[max_head_id, node_id]

        self.data[:, node_id] = 0
        self.data[max_head_id, node_id] = max_head_value  # restore value at index
        return self

    def get_edge_weight(self, start_id: int, end_id: int, strict=False) -> float:
        if not self.data[start_id, end_id] and strict:
            raise ValueError(f"No edge from '{start_id}' to '{end_id}' in graph.")
        return self.data[start_id, end_id]

    def get_max_head(self, node_id: int) -> int | None:
        if not self.has_head(node_id):
            return None
        return self._heads_slice(node_id).argmax()

    def add_node(self, index=0):
        """
        add a node, without a specified index it will be appended at the end
        """
        if index and index < self.number_of_nodes:
            raise AttributeError("There already exists a node at this position")
        else:
            index = self.number_of_nodes + 1
        self._expand_data_table(index)
        return self

    def delete_node(self, node_id):
        """
        actually deletes a node, aka node indexes change
        """
        self.data = np.delete(self.data, node_id, axis=0)
        self.data = np.delete(self.data, node_id, axis=1)
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

    def cut_incoming_to_node(self, node_id: int):
        self.data[:, node_id] = 0
        return self

    def has_head(self, node_id: int) -> bool:
        return bool(self._heads_slice(node_id).max())

    def attach_loose_nodes(self):
        for node in self.nodes:
            if self.has_head(node) and node != ROOT:
                self.add_edge(ROOT, node)
        return self

    def get_head_ids(self, node_id: int) -> np.ndarray[int]:
        return np.nonzero(self._heads_slice(node_id))[0]

    def get_dependent_ids(self, node_id: int) -> np.ndarray[int]:
        return np.nonzero(self._dependents_slice(node_id))[0]

    def has_dangling_nodes(self) -> bool:
        for node_id in self.nodes:
            if node_id == ROOT:
                continue
            if not self.has_head(node_id):
                return True
        return False

    def is_well_formed_tree(self) -> bool:
        no_root_head = not self.has_head(ROOT)
        num_edges_vs_num_nodes = (self.number_of_edges == (self.number_of_nodes - 1))
        no_dangling_nodes = not self.has_dangling_nodes()
        one_head_per_node = all([len(self.get_head_ids(node_id)) <= 1 for node_id in self.nodes])

        return no_root_head and num_edges_vs_num_nodes and no_dangling_nodes and one_head_per_node

    def find_cycle(self) -> List[int]:
        paths = {node_id: [[node_id, dependent] for dependent in self.get_dependent_ids(node_id)] for node_id in
                 self.nodes}

        for _ in self.nodes:  # any cycle is found after N iterations
            for node, old_paths in paths.items():
                new_paths = []
                for old_path in old_paths:
                    new_destinations = paths[old_path[-1]]
                    for new_destination in new_destinations:
                        new_path = old_path[:-1] + new_destination
                        if node in new_path[1:]:
                            cycle = new_path[:new_path.index(node, 1)]
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
    wdg = WeightedDirectedGraph.random(6).add_node(6).draw()
