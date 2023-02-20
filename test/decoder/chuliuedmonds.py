import unittest
from typing import List
import numpy as np
# https://github.com/tdozat/Parser-v3/blob/master/scripts/chuliu_edmonds.py

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.DataStructures.graph import ContractedDirectedWeightedGraph as CWDG
from src.decoder.graph.chuliuedmonds import mst, reduce


class ChuLiuEdmondsTest(unittest.TestCase):

    def test_reduce_graph(self):
        """
        wdg:           (0) -[10]-> (1) <-[1]- (2)
        expected:      (0) -[10]-> (1)        (2)
        """
        wdg = WDG()
        wdg.add_edge(0, 1, 10)
        wdg.add_edge(2, 1, 1)

        expected = WDG()
        expected.add_edge(0, 1, 10)

        self.assertEqual(reduce(wdg), expected)

        wdg.add_edge(0, 2, 1)

        expected.add_edge(0, 2, 1)
        self.assertEqual(reduce(wdg), expected)

    def test_contract_graph(self):
        """
            (0) <-[1>,<2]-> (1)
            | ^         ^  \
           [10] \      /   [1]
            |  [10]  [1]    |
            |    \   /     |
            L---> (2) <---/
            cycle = [0,1]

            expect:
            ???
        """
        wdg = WDG()
        wdg.add_edge(0, 1, 1)
        wdg.add_edge(1, 0, 2)
        cycle = [0, 1]
        wdg.add_edge(0, 2, 10)
        wdg.add_edge(1, 2, 1)
        wdg.add_edge(2, 0, 10)
        wdg.add_edge(2, 1, 1)

        print(CWDG(wdg, cycle))

    def test_resolve_graph(self):
        pass



    def test_mst(self):
        from test.decoder.chuliuedmonds_n2 import chuliu_edmonds_one_root, flat_tree_to_chonky_boiii
        wdg = WDG().random(10, seed=12345)
        scores = wdg.data
        gold_tree = chuliu_edmonds_one_root(scores)
        my_tree = mst(wdg).normalize()
        print(self.flat_tree_to_chonky_boiii(gold_tree))

        print(my_tree)
        exit(0)
        seed = 12345
        for i in range(2, 100):  # range(2, 20):
            wdg = WDG().random(i, seed=seed)
            my_tree = mst(wdg)
            # print(tree)
            self.assertTrue(tree.is_well_formed_tree())
