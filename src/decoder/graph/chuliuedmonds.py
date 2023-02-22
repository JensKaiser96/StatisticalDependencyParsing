import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDGraph


def mst(graph: WDGraph) -> WDGraph:
    """
    Chu-Liu-Edmonds Algorithm to get the maximal spanning tree for a given graph
    see. https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_file_3126280_download.html, slide 21
    """
    graph_max_heads = reduce(graph)
    cycle = graph_max_heads.find_cycle()
    if not cycle:
        return graph_max_heads
    else:
        graph_contracted, out_mapping, in_mapping = contract(graph, cycle)
        y = mst(graph_contracted)
        return resolve(y, cycle, out_mapping, in_mapping, graph_max_heads)


def reduce(input_graph: WDGraph) -> WDGraph:
    output_graph = input_graph.copy()
    for node_id in output_graph.nodes:
        output_graph.remove_all_heads_but_max(node_id)
    return output_graph


def contract(input_graph: WDGraph, cycle: list[int]) -> tuple[WDGraph, dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    contracted_graph = WDGraph(size=len(input_graph) + 1)
    cycle_node_id = len(contracted_graph) - 1
    out_mapping: dict[int, tuple[int, float]] = {}  # key is dependent, value is original (head, weight)
    in_mapping: dict[int, tuple[int, float]] = {}
    for head in input_graph.nodes:
        for dependent in input_graph.get_dependent_ids(head):
            if head in cycle and dependent in cycle:  # do nothing
                pass
            if head not in cycle and dependent not in cycle:  # just add edge
                contracted_graph.add_edge(head, dependent, input_graph.get_edge_weight(head, dependent))
            if head in cycle and dependent not in cycle:  # only add largest edge to each outside node
                out_weight = input_graph.get_edge_weight(head, dependent)
                if out_weight > contracted_graph.get_edge_weight(cycle_node_id, dependent):
                    contracted_graph.set_edge_weight(cycle_node_id, dependent, out_weight)
                    out_mapping[dependent] = head, out_weight
            if head not in cycle and dependent in cycle:  # only keep largest incoming edge, with summed cycle weight
                in_weight = input_graph.weight_to_cycle(cycle, head, dependent)
                if in_weight > contracted_graph.get_edge_weight(head, cycle_node_id):
                    contracted_graph.set_edge_weight(head, cycle_node_id, in_weight)
                    in_mapping[head] = (dependent, input_graph.get_edge_weight(head, dependent))
    return contracted_graph, out_mapping, in_mapping


def resolve(input_graph: WDGraph, cycle: list[int], out_mapping: dict[int, tuple[int, float]],
                in_mapping: dict[int, tuple[int, float]], max_tree: WDGraph):
    cycle_node_id = len(input_graph) - 1  # last node of graph is the cycle node
    # recreate all the original dependents of cycle
    for dependent in input_graph.get_dependent_ids(cycle_node_id):
        original_head, original_weight = out_mapping[dependent]
        input_graph.add_edge(original_head, dependent, original_weight)

    # recreate original head of cycle
    head = input_graph.get_head_ids(cycle_node_id)
    if len(head) != 1:
        print(f"Something went wrong {head}, {in_mapping}")
        head = 0
        original_weight = 1
        original_dependent = cycle[0]
    else:
        head = head[0]
        original_dependent, original_weight = in_mapping[head]
    input_graph.set_edge_weight(head, original_dependent, original_weight)

    # delete cycle_node
    input_graph.delete_node(cycle_node_id)

    # add weights from within the cycle, skipping the original_dependent node
    for i, node in enumerate(cycle):
        if node == original_dependent:
            continue
        dependent = node
        head = cycle[i-1]
        weight = max_tree.get_edge_weight(head, dependent)
        input_graph.set_edge_weight(head, dependent, weight)
    return input_graph
