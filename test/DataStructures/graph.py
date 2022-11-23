import unittest
from typing import List

from src.DataStructures.graph import WeightedDirectedGraph


class WeightedDirectedGraphTest(unittest.TestCase):
    def all_possible_starts(self, list_: List):
        """
        returns a list of every possible starting point for the given input list
        """
        permutations = []
        for i in range(len(list_)):
            permutations.append(list_[i:]+list_[:-len(list_)+i])
        return permutations

    def test_get_cycle(self):
        wdg1 = WeightedDirectedGraph()
        wdg1.add_edge(0, 1)
        wdg1.add_edge(1, 0)
        self.assertIn(wdg1.find_cycle(), self.all_possible_starts([0, 1]))

        wdg2 = WeightedDirectedGraph()
        wdg2.add_edge(0, 1)
        wdg2.add_edge(1, 2)
        wdg2.add_edge(2, 1)
        self.assertIn(wdg2.find_cycle(), self.all_possible_starts([1, 2]))

        wdg3 = WeightedDirectedGraph()
        wdg3.add_edge(0, 1)
        wdg3.add_edge(1, 2)
        wdg3.add_edge(2, 3)
        wdg3.add_edge(3, 4)
        wdg3.add_edge(4, 5)
        wdg3.add_edge(5, 1)
        self.assertIn(wdg3.find_cycle(), self.all_possible_starts([1, 2, 3, 4, 5]))

        wdg4 = WeightedDirectedGraph()
        wdg4.add_edge(0, 1)
        wdg4.add_edge(1, 2)
        wdg4.add_edge(2, 3)
        wdg4.add_edge(1, 3)
        self.assertEqual([], wdg4.find_cycle())


if __name__ == '__main__':
    unittest.main()
