import unittest
from typing import List

from src.DataStructures.graph import WeightedDirectedGraph
from src.decoder.graph.chuliuedmonds import mst


class ChuLiuEdmondsTest(unittest.TestCase):
    def test(self):
        for i in range(2, 20):
            wdg = WeightedDirectedGraph().random(i)
            # print(wdg)
            tree = mst(wdg)
            # print(tree)
            print(tree.is_well_formed_tree())
        self.assertTrue(tree.is_well_formed_tree())
