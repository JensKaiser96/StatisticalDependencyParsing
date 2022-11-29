from typing import List, Tuple
from logging import Logger

from src.DataStructures.graph import WeightedDirectedGraph as WDGraph

LOGGER = Logger(__name__)


def mst(graph: WDGraph, recursion_depth=-1) -> WDGraph:
    if recursion_depth < 0:
        recursion_depth = graph.number_of_nodes ** 2
    if recursion_depth == 0:
        raise RecursionError(f"Recursion Error")
    """
    Chu-Liu-Edmonds Algorithm to get the maximal spanning tree for a given graph
    see. https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_file_3126280_download.html, slide 21
    """
    reduced_graph = reduce_graph(graph)
    cycle = reduced_graph.find_cycle()
    if not cycle:
        return reduced_graph

    contracted_graph, (cycle_breaker_from, cycle_breaker_to) = contract_graph(graph, cycle)
    print(f"Breaking cycle by removing edge ({cycle_breaker_from},{cycle_breaker_to})")
    return contracted_graph.remove_edge(cycle_breaker_from, cycle_breaker_to)
    y = mst(contracted_graph, recursion_depth - 1)
    # contract graph
    return y.remove_edge(cycle_breaker_from, cycle_breaker_to)


def reduce_graph(input_graph: WDGraph) -> WDGraph:
    output_graph = input_graph.copy()
    for node_id in input_graph.node_ids:
        output_graph.remove_all_heads_but_max(node_id)
    return output_graph


def contract_graph(input_graph: WDGraph, cycle: List[int]) -> Tuple[WDGraph, Tuple[int, int]]:
    """
    Same as input_graph but
       remove edges:
          going to the cycle
          leaving the cycle
       add/keep:
          max incomming edge
          max outgoing  edge to each node outside
    """
    # print(f"Contracting: \n{input_graph}\nto:\n")
    # compute weights from and to C
    max_in_weight_summed = 0
    max_in_weight_original = 0
    max_in_weight_from = 0
    max_in_weight_to = 0
    for node_id in input_graph.node_ids:
        if node_id in cycle:
            continue
        # remove all but max edge from cycle to node
        input_graph.remove_all_heads_but_max(node_id, cycle)

        # compute scores for arcs entering C
        for dependent in input_graph.get_dependents(node_id):
            if dependent not in cycle:
                continue
            # calculate edge into cycle
            summed_weight = input_graph.weight_to_cycle(cycle, node_id, dependent)
            if summed_weight > max_in_weight_summed:
                max_in_weight_original = input_graph.get_edge_weight(node_id, dependent)
                max_in_weight_summed = summed_weight
                max_in_weight_from = node_id
                max_in_weight_to = dependent
            # remove edge into cycle
            input_graph.remove_edge(node_id, dependent)
    # add winner edge back into graph
    if max_in_weight_original:
        input_graph.add_edge(max_in_weight_from, max_in_weight_to, max_in_weight_original)

    # calculate which edge breaks the cycle
    #print(f"Computing which edge to remove to break cycle:\n"
    #      f"{input_graph.get_heads(max_in_weight_to)} >-> {cycle}")
    for head in input_graph.get_heads(max_in_weight_to):
        if head in cycle:
            cycle_breaker_from = head
    #print(input_graph)
    return input_graph, (cycle_breaker_from, max_in_weight_to)


def resolve_cycle(input_graph: WDGraph, cycle_breaker_from ) -> WDGraph:
    """
        1: function RESOLVECYCLE(y ,C):
        2: Find a vertex vd in C s. t. 〈v ′
        h, vd 〉 ∈ y and 〈v ′′
        h , vd 〉 ∈ C
        3: return y ∪ C − {〈v ′′
        h , vd 〉}
        4: end functio
        """
    return input_graph


if __name__ == '__main__':
    for i in range(2, 20):
            wdg = WDGraph().random(i)
            wdg.draw()
            tree = mst(wdg)
            tree.draw()
            print(tree.is_well_formed_tree())
