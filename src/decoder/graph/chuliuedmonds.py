from typing import List

from src.DataStructures.graph import WeightedDirectedGraph as WDGraph


def mst(input_graph: WDGraph) -> WDGraph:
    """
    Chu-Liu-Edmonds Algorithm to get the maximal spanning tree for a given graph
    see. https://ilias3.uni-stuttgart.de/goto_Uni_Stuttgart_file_3126280_download.html, slide 21
    """
    reduced_graph = reduce_graph(input_graph)
    cycle = reduced_graph.find_cycle()
    if not cycle:
        return reduced_graph

    contracted_graph = contract_graph(input_graph, cycle)
    y = mst(contracted_graph)
    return resolve_cycle(y)


def reduce_graph(input_graph: WDGraph) -> WDGraph:
    output_graph = WDGraph()
    for node_id in input_graph.node_ids:
        output_graph.add_edge(**input_graph.copy_edge(input_graph.get_max_head(node_id), node_id))
    return output_graph


def contract_graph(input_graph: WDGraph, cycle: List[int]) -> WDGraph:
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
    # create new graph without cycle.
    output_graph = input_graph.copy()
    output_graph.cut_nodes(cycle)

    for node_id in input_graph.node_ids:
        if node_id in cycle:
            pass
            # compute scores for arcs entering C
        else:
            # compute scores for arcs leaving C
            output_graph.add_edge(input_graph.get_max_head(node_id, cycle), node_id)



    max_incomming_arc_weight = 0
    max_incomming_arc_head_id = 0
    for node_id in cycle:
        input_graph.get_max_head()
        max_incomming_arc_weight

    return input_graph


def resolve_cycle(input_graph: WDGraph) -> WDGraph:
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
    mst(WDGraph())

