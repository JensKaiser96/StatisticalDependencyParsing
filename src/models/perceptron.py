from itertools import permutations

import numpy as np
from tqdm import tqdm

from src.decoder.graph.chuliuedmonds import mst
from test.decoder.chuliuedmonds_n2 import mst
from src.features.template import TemplateWizard
from src.tools.CONLL06 import TreeBank, Sentence
from src.DataStructures.graph import WeightedDirectedGraph as WDG


class Perceptron:
    weights: np.ndarray

    def __init__(self, feature_dict: dict[str, int]):
        self.feature_dict = feature_dict
        self.weights = np.ones((len(feature_dict), ), dtype=int)

    def save_weights(self, path: str):
        np.save(path, self.weights, allow_pickle=True)
        return self

    def load_weights(self, path: str):
        self.weights = np.load(path, allow_pickle=True)
        return self

    def train(self, tree_bank: TreeBank, epochs: int = 1, save_path="", logging=True):
        print(f"Training started with {epochs=}...")
        for epoch in range(epochs):
            num_correct = 0
            num_incorrect = 0
            uases = []
            for sentence in tqdm(tree_bank):
                predicted_tree = self.predict(sentence)
                gold_tree = WDG.from_sentence(sentence)
                # predicted_tree.draw()
                uases.append(predicted_tree.compare(gold_tree))
                if gold_tree != predicted_tree:
                    gold_tree_indices = self.get_feature_indices_from_tree(gold_tree, sentence)
                    predicted_tree_indices = self.get_feature_indices_from_tree(predicted_tree, sentence)
                    self.weights[gold_tree_indices] += 1
                    self.weights[predicted_tree_indices] -= 1
                    num_incorrect += 1
                else:
                    num_correct += 1
            if logging:
                print(f"\nEpoch ({epoch}/{epochs}): "
                      f"{round(num_correct/(num_correct+num_incorrect), 2)}% correct "
                      f"UAS: {np.array(uases).mean()}\n")
            tree_bank.shuffle()
            if save_path:
                self.save_weights(save_path)

    def _create_full_tree_from_sentence(self, sentence: Sentence) -> WDG:
        n = len(sentence)
        tree = WDG().random(size=n)
        all_arcs = permutations(range(n), 2)
        missed_features = 0
        for head, dependent in all_arcs:
            feature_keys = TemplateWizard.get_feature_keys(head, dependent, sentence)
            arc_weight = 0
            for feature_key in feature_keys:
                try:
                    feature_index = self.feature_dict[feature_key]
                    arc_weight += self.weights[feature_index]
                except KeyError:
                    missed_features += 1
            if dependent != 0 and (head == 0 and arc_weight > 0):  # mst only works if this is given
                tree.add_edge(head, dependent, arc_weight)
        return tree

    def predict(self, sentence: Sentence) -> WDG:
        full_predicted_tree = self._create_full_tree_from_sentence(sentence)
        return mst(full_predicted_tree).normalize()

    def score(self, features: np.ndarray=None, indices: list=None) -> float:
        """
        input feature vector <psi> is not a one hot encoding, but a list of indices
        """
        if indices is None:
            indices = []
            for feature in features:
                try:
                    indices.append(self.templer.feature_dict[tuple(feature)])
                except KeyError:
                    pass
        if not indices:
            return 0
        return max(self.weights[indices].sum(), 0)

    def get_feature_indices_from_tree(self, tree: WDG, sentence: Sentence) -> list[int]:
        """
        returns a list of all feature indices which occur in a given tree and sentence.
        Method is used during training, to get the indices for the weights which will be updated.
        in a way a reverse _create_full_tree_from_sentence
        """
        feature_indices = []
        missed_features = 0
        for token in sentence:
            for dependant_id in tree.get_dependent_ids(token.id_):
                dependant = sentence[dependant_id]
                feature_keys = TemplateWizard.get_feature_keys(token, dependant, sentence)
                for feature_key in feature_keys:
                    try:
                        feature_indices.append(self.feature_dict[feature_key])
                    except KeyError:
                        missed_features += 1
        #print(f"Missed {missed_features} features during extraction from tree")
        return feature_indices
