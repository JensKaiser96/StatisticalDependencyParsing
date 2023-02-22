"""
[1] Greed is Good if Randomized: New Inference for Dependency Parsing, Zhang et al. 2014
"""

import threading

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.tools.CONLL06 import Sentence


def greedy(sentence: Sentence, score_tree, num_threads: int = 100):
    data = []
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=greed_is_good_threaded, args=(sentence, score_tree, i, data))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    max_tree = None
    max_score = 0

    for tree, score in data:
        if score > max_score:
            max_score = score
            max_tree = tree

    return max_tree


def greed_is_good_threaded(sentence: Sentence, score_tree, seed: int, data, max_repetitions=100):
    data.append(greed_is_good(sentence, score_tree, seed, max_repetitions))


def greed_is_good(sentence: Sentence, score_tree, seed: int, max_repetitions=100) -> tuple[WDG, float]:
    repetitions = 0
    tree = WDG.random_tree(size=len(sentence), seed=seed)
    best_tree = tree
    best_score = score_tree(tree)
    while repetitions < max_repetitions:
        repetitions += 1
        current_tree, current_score = ladder_step(sentence, score_tree)
        if current_score > best_score:
            best_tree = current_tree
        else:
            break
    return best_tree, best_score


def ladder_step(sentence: Sentence, score_tree) -> tuple[WDG, float]:
    tree = WDG.random_tree(size=len(sentence))
    best_score = score_tree(tree)
    sorted_nodes = dict(sorted(tree.get_root_distances().items(), key=lambda item: item[1]))
    for node in sorted_nodes.keys():
        if node == 0:
            continue
        best_head = tree.get_head(node)
        best_score = score_tree(tree)
        tree.remove_head(node)
        for new_head in tree.nodes:
            if new_head == node:
                continue
            tree.add_edge(new_head, node)
            score = score_tree(tree)
            if score > best_score:
                best_head = new_head
                best_score = score
            tree.remove_head(node)
        tree.add_edge(best_head, node)
    return tree, best_score
