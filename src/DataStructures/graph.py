import logging
from typing import List, Dict

import numpy as np

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
        max_head_value = self.data[max_head_id, node_id]

        # set all to 0
        if cycle is None:
            self.data[:, node_id] = 0
        else:
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

    def get_heads(self, node_id: int) -> List[int]:
        return list(self._heads_list(node_id))

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
            print(f"false because of:\n{root_has_head=}\n{num_edges_vs_num_nodes}\n{dangling_nodes}\n{more_than_one_head}")
            return False
        return True

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

    def contract(self, cycle: List[int]):
        """
        1: function CONTRACT(G = 〈V , A〉,C,σ):
        2: GC = G − C . subgraph of G excluding nodes in C
        3: Add vc to represent C . New node
        4: . Compute scores for arcs leaving vc
        5: for vd ∈ V − C : ∃v ′
        h ∈C 〈v ′
        h, vd 〉 ∈ A do
        6: Add 〈vc , vd 〉 to GC with σ(〈vc , vd 〉) = maxv ′
        h ∈C σ(〈v ′
        h, vd 〉
        7: end for
        8: . Compute scores for arcs entering vc
        9: for vh ∈ V − C : ∃v ′
        d ∈C 〈vh, v ′
        d 〉 ∈ A do
        10: Add 〈vh, vc 〉 to GC with
        11: σ(〈vh, vc 〉) = maxvd ∈C [σ(〈vh, vd 〉) + σ(C) − σ(〈h(vd ), vd 〉)]
        12: where h(vd ) is the head of vd in C
        13: and σ(C) = ∑
        vt ∈C σ(h(vt ), vt )
        14: end for
        15: return GC
        16: end function
        """
