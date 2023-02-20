import unittest
from typing import List
import numpy as np

from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.DataStructures.graph import ContractedDirectedWeightedGraph as CWDG
from src.decoder.graph.eisner import create_tree


class EisnerTest(unittest.TestCase):

    def test_create_tree(self):
        n = 4
        paths = np.array([
            [[0, 0, 0, 0],
             [0, 0, 1, 2],
             [0, 0, 0, 2],
             [0, 0, 0, 0]],
            [[0, 0, 0, 2],
             [0, 0, 1, 2],
             [0, 0, 0, 2],
             [0, 0, 0, 0]],
            [[0, 0, 0, 0],
             [0, 0, 1, 2],
             [0, 0, 0, 2],
             [0, 0, 0, 0]],
            [[0, 1, 1, 1],
             [0, 0, 2, 3],
             [0, 0, 0, 3],
             [0, 0, 0, 0]]
        ])
        create_tree(n, paths)
