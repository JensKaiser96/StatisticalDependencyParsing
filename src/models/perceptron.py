from itertools import permutations

import numpy as np

from src.decoder.graph.eisner import eisner
from src.features import template
from src.features.template import Templer
from src.tools.CONLL06 import TreeBank, Sentence
from src.DataStructures.graph import WeightedDirectedGraph as WDG


class Perceptron:
    weights: np.ndarray

    def __init__(self, templer: Templer):
        print(f"Initializing Perceptron with size {len(templer.features)}")
        self.templer = templer
        self.feature_dict = templer.feature_dict()
        self.weights = np.zeros((len(templer.features), ), dtype=int)

    def save_weights(self, path: str):
        np.save(path, self.weights, allow_pickle=True)
        return self

    def load_weights(self, path: str):
        self.weights = np.load(path, allow_pickle=True)
        return self

    def train(self, tree_bank: TreeBank, epochs: int = 1, logging=True):
        num_correct = 0
        num_incorrect = 0
        for epoch in range(epochs):
            for sentence in tree_bank:
                features = self._create_feature_vector(sentence)
                predicted_tree = eisner(len(sentence), self.score_tree(features))
                gold_tree = WDG()
                if gold_tree != predicted_tree:
                    gold_tree_indices = Templer.extract_feature_indices(gold_tree, features)
                    predicted_tree_indices = Templer.extract_feature_indices(predicted_tree, features)
                    self.weights[gold_tree_indices] += 1
                    self.weights[predicted_tree_indices] -= 1
                    num_incorrect += 1
                else:
                    num_correct += 1
            if logging:
                print(f"Epoch ({epoch}/{epochs}): "
                      f"{round(num_correct/(num_correct/num_incorrect), 2)}% correct "
                      f"({num_correct}/{num_correct+num_incorrect})")
            tree_bank.shuffle()

    def _create_feature_vector(self, sentence: Sentence) -> np.ndarray:
        n = len(sentence)
        features = np.ones((n, n, len(template.TemplateCollection), 5), dtype=int) * -1
        pairs = permutations(range(n), 2)
        for head_index, dependent_index in pairs:
            a = self.templer.extract_features(sentence[head_index], sentence[dependent_index], sentence)
            features[head_index, dependent_index] = a
        return features

    def predict(self, sentence: Sentence) -> WDG:
        return eisner(len(sentence), self.score_tree(self._create_feature_vector(sentence)))

    def score(self, features: np.ndarray) -> float:
        """
        input feature vector <psi> is not a one hot encoding, but a list of indices
        """
        indices = []
        for feature in features:
            try:
                self.feature_dict[tuple(feature)]
            except KeyError:
                pass
        if not indices:
            return 0
        return self.weights[indices].sum()

    def score_tree(self, feature_vectors) -> np.ndarray:
        scores = np.zeros((feature_vectors.shape[:2]))
        for i in range(feature_vectors.shape[0]):
            for j in range(feature_vectors.shape[0]):
                scores[i, j] = self.score(feature_vectors[i, j])
        return scores
