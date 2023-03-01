from itertools import permutations

import os
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
        dir_name = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        np.save(path, self.weights, allow_pickle=True)
        return self

    def load_weights(self, path: str):
        self.weights = np.load(path, allow_pickle=True)
        return self

    def train(self, tree_bank: TreeBank, epochs: int = 1, save_path=""):
        last_uas = 0
        last_cct = 0
        for epoch in range(epochs):
            num_correct_trees = 0
            num_incorrect_trees = 0
            num_correct_edges = 0
            num_total_edges = 0
            iterator = tqdm(tree_bank,
                            desc=f"Epoch {epoch+1}/{epochs} | "
                                 f"prev. UAS: {last_uas}% | "
                                 f"prev. CCT: {last_cct}% ")
            for instance_id, sentence in enumerate(iterator):
                predicted_tree = self.predict(sentence.copy())
                gold_tree = sentence.to_tree()
                num_correct_edges += predicted_tree.count_common_edges(gold_tree)
                num_total_edges += predicted_tree.number_of_edges
                if gold_tree != predicted_tree:
                    predicted_tree_indices = self.get_feature_indices_from_tree(predicted_tree, sentence)
                    gold_tree_indices = self.get_feature_indices_from_tree(gold_tree, sentence)
                    self.weights[predicted_tree_indices] -= 1
                    self.weights[gold_tree_indices] += 1
                    num_incorrect_trees += 1
                else:
                    num_correct_trees += 1
            last_uas = round((num_correct_edges/num_total_edges)*100, 2)
            last_cct = round(num_correct_trees/(num_correct_trees+num_incorrect_trees)*100, 2)
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

    def annotate(self, tree_bank: TreeBank):
        print("Annotating Tree Bank")
        annotated_treebank = tree_bank.copy()
        for sentence in tqdm(annotated_treebank):
            tree = self.predict(sentence)
            sentence.set_heads(tree)
        return annotated_treebank

    def get_feature_indices_from_tree(self, tree: WDG, sentence: Sentence, strict=False) -> list[int]:
        """
        returns a list of all feature indices which occur in a given tree and sentence.
        Method is used during training, to get the indices for the weights which will be updated.
        in a way a reverse _create_full_tree_from_sentence

        In strict mode, an error is thrown if feature is not found. Can be enabled when extracting
        feature indices from gold trees.
        """
        feature_indices = []
        for token in sentence:
            for dependant_id in tree.get_dependent_ids(token.id_):
                feature_keys = TemplateWizard.get_feature_keys(token, dependant_id, sentence)
                for feature_key in feature_keys:
                    try:
                        feature_indices.append(self.feature_dict[feature_key])
                    except KeyError:
                        if strict:
                            raise ValueError(f"Could not find index for feature {feature_key}. (and you wanted me to bitch about it (: )")
        return feature_indices
