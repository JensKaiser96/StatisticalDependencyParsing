from typing import List, Tuple
from logging import Logger

import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDGraph
from src.DataStructures.graph import ContractedDirectedWeightedGraph as CWDGraph

LOGGER = Logger(__name__)


def mst(graph: WDGraph) -> WDGraph | CWDGraph:
    """
    Chu-Liu-Edmonds Algorithm to get the maximal spanning tree for a given graph
    see. https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_file_3126280_download.html, slide 21
    """
    graph_max_heads = reduce_graph(graph)
    cycle = graph_max_heads.find_cycle()
    if not cycle:
        print("Graph Max Heads:\n", graph_max_heads)
        return graph_max_heads
    else:
        graph_contracted = CWDGraph(graph, cycle)
        y = mst(graph_contracted)
        return resolve_cycle(y, cycle)


def reduce_graph(input_graph: WDGraph | CWDGraph) -> WDGraph | CWDGraph:
    output_graph = input_graph.copy()
    for node_id in output_graph.node_ids:
        output_graph.remove_all_heads_but_max(node_id)
    return output_graph


def contract_graph(input_graph: WDGraph, cycle: List[int]) -> WDGraph:
    """
    DEPREACIATED U FUCK
    Same as input_graph but
       remove edges:
          going to the cycle
          leaving the cycle
       add/keep:
          max incomming edge
          max outgoing  edge to each node outside

    WE CAN'T YET DECIDE WHICH ARC TO REMOVE FROM WITHIN THE CYCLE
    set all the arcs containing nodes from the cycle to 0.
    contracted node is just going to be the lowest node id?!?!?
    CGraph:
    cycle: List[int]
    cycle_node_id: int
    in_arc_id: int
    out_arc_id: int
    we need to remember:
        where the max arc points to
    """
    # create CWDG, which already has cycle as node
    contracted_graph = CWDGraph(input_graph, cycle)

    # calculate max out head
    for node in cycle:
        input_graph.remove_all_heads_but_max()
        input_graph.get_dependent_ids()


    for node_id in input_graph.node_ids:
        if node_id in cycle:
            continue

        # compute scores for arcs entering C
        for dependent in input_graph.get_dependent_ids(node_id):
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

    graph_c = input_graph.copy()
    graph_c.cut(cycle)
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
        for dependent in input_graph.get_dependent_ids(node_id):
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
    for head in input_graph.get_head_ids(max_in_weight_to):
        if head in cycle:
            cycle_breaker_from = head
    return input_graph, (cycle_breaker_from, max_in_weight_to)  # todo error, referenced before assignment O.o


def resolve_cycle(graph: CWDGraph, cycle: List[int]) -> CWDGraph:
    # restore cycle to full thing
    graph.data[np.ix_(cycle, cycle)] = graph.internal_cycle_weights[np.ix_(cycle, cycle)]
    # set in arc
    in_arc_from, in_arc_to, in_arc_weight = graph.in_arc
    graph.add_edge(*graph.in_arc)
    # break cycle
    graph.remove_edge(cycle[cycle.index(in_arc_to)-1], in_arc_to)
    return graph


if __name__ == '__main__':
    for i in range(2, 50):
            wdg = WDGraph().random(i)
            is_tree = True
            try:
                tree = mst(wdg)
                is_tree = tree.is_well_formed_tree()
                if is_tree:
                    print(f"Graph of size: {i} to tree worked. :D")
            except UnboundLocalError as e:
                print(e)
                # is_tree = False
                tree = WDGraph()
            if not is_tree:
                print(f"before:\n{wdg}")
                print(f"after:\n{tree}")
                tree.draw()
