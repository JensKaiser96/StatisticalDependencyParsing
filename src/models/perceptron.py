from itertools import permutations

import numpy as np
from tqdm import tqdm

from src.decoder.graph.chuliuedmonds import mst
from src.features.template import TemplateWizard
from src.tools.CONLL06 import TreeBank, Sentence
from src.DataStructures.graph import WeightedDirectedGraph as WDG


class Perceptron:
    weights: np.ndarray

    def __init__(self, feature_dict: dict[str, int], logging=False):
        self.feature_dict = feature_dict
        self.weights = np.ones((len(feature_dict), ), dtype=int)
        self.logging = logging

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
            if logging:
                iterator = tqdm(tree_bank, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                iterator = tree_bank
            for sentence in iterator:
                predicted_tree = self.predict(sentence)
                gold_tree = WDG.from_sentence(sentence)
                #predicted_tree.draw()
                uases.append(predicted_tree.compare(gold_tree, verbose=False))
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
        # initialize with random values between 0 and 1, should not affect mst result due to low values
        # having these values prevents too sparse trees
        # tree = WDG.random(n)
        tree = WDG(size=n)
        all_arcs = permutations(range(n), 2)
        for head, dependent in all_arcs:
            feature_keys = TemplateWizard.get_feature_keys(head, dependent, sentence)
            arc_weight = 0
            for feature_key in feature_keys:
                try:
                    feature_index = self.feature_dict[feature_key]
                    arc_weight += self.weights[feature_index]
                except KeyError:
                    pass
            tree.add_edge(head, dependent, max(arc_weight, 1))
        tree.make_0_root()
        tree.attach_loose_nodes()
        return tree

    def predict(self, sentence: Sentence) -> WDG:
        full_tree = self._create_full_tree_from_sentence(sentence)
        pruned_tree = mst(full_tree).normalize()
        if not pruned_tree.is_well_formed_tree() and self.logging:
            print(f"Tree is not well formed")
        return pruned_tree

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
        if self.logging:
            print(f"Missed features during extraction:\t{missed_features}")
        return feature_indices
