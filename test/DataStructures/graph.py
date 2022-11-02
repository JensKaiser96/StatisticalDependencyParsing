import unittest
from src.DataStructures.graph import WeightedDirectedGraph


class WeightedDirectedGraphTest(unittest.TestCase):

    @staticmethod
    def test_get_cycle():
        wdg1 = WeightedDirectedGraph()
        wdg1.add_edge(0, 1)
        wdg1.add_edge(1, 0)
        assert wdg1.get_cycle() == [0, 1]

        wdg2 = WeightedDirectedGraph()
        wdg2.add_edge(0, 1)
        wdg2.add_edge(1, 2)
        wdg2.add_edge(2, 1)
        assert wdg2.get_cycle() == [1, 2]

        wdg3 = WeightedDirectedGraph()
        wdg3.add_edge(0, 1)
        wdg3.add_edge(1, 2)
        wdg3.add_edge(2, 3)
        wdg3.add_edge(3, 4)
        wdg3.add_edge(4, 5)
        wdg3.add_edge(5, 1)
        assert wdg3.get_cycle()

        wdg4 = WeightedDirectedGraph()
        wdg4.add_edge(0, 1)
        wdg4.add_edge(1, 2)
        wdg4.add_edge(2, 3)
        wdg4.add_edge(1, 3)
        assert wdg4.get_cycle() == []


if __name__ == '__main__':
    unittest.main()
