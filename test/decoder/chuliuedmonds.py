import unittest
import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDG
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

    def test_nx_conversion(self):
        wdg = WDG.random(10)
        gx = wdg.to_nx()
        wdg_ = WDG.from_nx(gx)
        assert wdg == wdg_

    def test_mst_one(self):
        wdg = WDG()
        wdg.data = np.array(
            [[0, .31, .18, .20],
            [ 0,  0, .96, .65],
            [ 0, .65,  .0, .96],
            [ 0, .10, .29,  0]]
        )
        mst_wdg = mst(wdg)
        mst_gold = wdg.nx_cle()
        self.assertEqual(mst_wdg, mst_gold)

    def test_mst(self):
        seed = 12345
        for i in range(2, 100):  # range(2, 20):
            wdg = WDG.random(i, seed=seed-i)
            mst_wdg = mst(wdg)
            mst_gold = wdg.nx_cle()
            if mst_wdg != mst_gold:
                print(wdg.data)
                mst_wdg.draw()
                mst_gold.draw()
            self.assertEqual(mst_wdg, mst_gold)
            print(f"{i} ok")
