"""
Code from:
https://raw.githubusercontent.com/tdozat/Parser-v3/master/scripts/chuliu_edmonds.py
only for comparing correctness of own implementation
"""
from src.DataStructures.graph import WeightedDirectedGraph as WDG
import numpy as np


# ***************************************************************
def tarjan(tree):
    """"""

    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []

    # -------------------------------------------------------------
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return

    # -------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles


# ===============================================================
# NOTE: i'm so sorry --Tim
def chuliu_edmonds(scores):
    """"""

    scores *= (1 - np.eye(scores.shape[0]))
    scores[0] = 0
    scores[0, 0] = 1
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)
    # print(scores)
    # print(cycles)
    if not cycles:
        return tree
    else:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # locations of cycle; (t) in [0,1]
        cycle = cycles.pop()
        # indices of cycle in original tree; (c) in t
        cycle_locs = np.where(cycle)[0]
        # heads of cycle in original tree; (c) in t
        cycle_subtree = tree[cycle]
        # scores of cycle in original tree; (c) in R
        cycle_scores = scores[cycle, cycle_subtree]
        # total score of cycle; () in R
        cycle_score = cycle_scores.prod()

        # locations of noncycle; (t) in [0,1]
        noncycle = np.logical_not(cycle)
        # indices of noncycle in original tree; (n) in t
        noncycle_locs = np.where(noncycle)[0]
        # print(cycle_locs, noncycle_locs)

        # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
        metanode_head_scores = scores[cycle][:, noncycle] / cycle_scores[:, None] * cycle_score
        # scores of cycle's potential dependents; (n x c) in R
        metanode_dep_scores = scores[noncycle][:, cycle]
        # best noncycle head for each cycle dependent; (n) in c
        metanode_heads = np.argmax(metanode_head_scores, axis=0)
        # best cycle head for each noncycle dependent; (n) in c
        metanode_deps = np.argmax(metanode_dep_scores, axis=1)

        # scores of noncycle graph; (n x n) in R
        subscores = scores[noncycle][:, noncycle]
        # pad to contracted graph; (n+1 x n+1) in R
        subscores = np.pad(subscores, ((0, 1), (0, 1)), 'constant')
        # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
        # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R-> (n) in R
        subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chuliu_edmonds(subscores)
        # head of the cycle; () in n
        # print(contracted_tree)
        cycle_head = contracted_tree[-1]
        # fixed tree: (n) in n+1
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_tree = -np.ones_like(tree)
        # print(0, new_tree)
        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < len(contracted_tree)
        # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
        # print(1, new_tree)
        # fixed tree with heads coming from the cycle: (n) in [0,1]
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        # print(2, new_tree)
        # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
        new_tree[cycle_locs] = tree[cycle_locs]
        # print(3, new_tree)
        # root of the cycle; (n)[() in n] in c = () in c
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        # print(4, new_tree)
        return new_tree


# ===============================================================
def chuliu_edmonds_one_root(scores):
    """"""

    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
    if len(roots_to_try) == 1:
        return tree

    # Look at all roots that are more likely than we would expect
    if len(roots_to_try) == 0:
        roots_to_try = np.where(scores[1:, 0] >= 1 / len(scores))[0] + 1
    # *sigh* just grab the most likely one
    if len(roots_to_try) == 0:
        roots_to_try = np.array([np.argmax(scores[1:, 0]) + 1])

    # -------------------------------------------------------------
    def set_root(scores, root):
        root_score = scores[root, 0]
        scores = np.array(scores)
        scores[1:, 0] = 0
        scores[root] = 0
        scores[root, 0] = 1
        return scores, root_score

    # -------------------------------------------------------------

    best_score, best_tree = -np.inf, None  # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = np.log(tree_probs).sum() + np.log(root_score) if tree_probs.all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree


def mst(in_tree: WDG):
    """
    interface with my datastructure
    """
    scores = in_tree.data
    flat_tree = chuliu_edmonds_one_root(scores)
    n = len(flat_tree)
    print(flat_tree)
    tree = np.zeros((n, n), dtype=float)
    for index, value in enumerate(flat_tree):
        tree[value, index] = 1
    tree[0, 0] = 0
    in_tree.data = tree
    return in_tree


# ***************************************************************
def main(n=10):
    """"""
    g = WDG()
    g.data = np.arange(16).reshape((4,4))
    np.fill_diagonal(g.data, 0)
    g.data[:,0] = 0
    print(g.data)
    g_ = mst(g)
    print(g_)
    assert g_.is_well_formed_tree()
    return
    for i in range(100):
        scores = np.random.randn(n, n)
        scores = np.exp(scores) / np.exp(scores).sum()
        scores *= (1 - np.eye(n))
        newtree = chuliu_edmonds_one_root(scores)
        cycles = tarjan(newtree)
        roots = np.where(np.equal(newtree[1:], 0))[0] + 1
        print(newtree, cycles, roots)
        assert not cycles
        assert len(roots) == 1
    return


# ***************************************************************
if __name__ == '__main__':
    """"""

    main()