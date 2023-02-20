"""
Unigram:
hform, hpos         hform, hpos, dform, dpos    hpos, bpos, dpos
hform               hpos, dform, dpos           hpos, dpos, hpos+1, dpos-1
hpos                hform, dform, dpos          hpos, dpos, hpos-1, dpos-1
dform, dpos         hform, hpos, dform          hpos, dpos, hpos+1, dpos+1
dform               hform, hpos, dpos           hpos, dpos, hpos-1, dpos+1
dpos                hform, dform
                    hpos, dpos
"""
import os
from itertools import combinations

import numpy as np
from tqdm import tqdm

from src.tools.CONLL06 import Token, Sentence, TreeBank
from src.DataStructures.graph import WeightedDirectedGraph as WDG

HEAD = "HEAD"
DEPE = "DEPE"
FORM = "FORM"
POS = "POS"

HEAD_FORM = f"{HEAD}_{FORM}"
HEAD_POS = f"{HEAD}_{POS}"
DEPE_FORM = f"{DEPE}_{FORM}"
DEPE_POS = f"{DEPE}_{POS}"
BASIC_FEATURES = (HEAD_FORM, HEAD_POS, DEPE_FORM, DEPE_POS)

BETW = "BETW"
NEXT = "NEXT"
PREV = "PREV"

BETW_POS = f"{BETW}_{POS}"
HEAD_POS_NEXT = f"{HEAD}_{POS}_{NEXT}"
HEAD_POS_PREV = f"{HEAD}_{POS}_{PREV}"
DEPE_POS_NEXT = f"{DEPE}_{POS}_{NEXT}"
DEPE_POS_PREV = f"{DEPE}_{POS}_{PREV}"


class TemplateWizard:
    @staticmethod
    def basic_templates():
        """
        generates all combinations with length 1 to 4 of features
        """
        grams = []
        for n in range(1, 5):
            for combination in combinations(BASIC_FEATURES, n):
                grams.append(combination)
        return tuple(grams)

    @staticmethod
    def extended_templates():
        return (
            (HEAD_POS, BETW_POS, DEPE_POS),
            (HEAD_POS, DEPE_POS, HEAD_POS_NEXT, DEPE_POS_NEXT),
            (HEAD_POS, DEPE_POS, HEAD_POS_PREV, DEPE_POS_NEXT),
            (HEAD_POS, DEPE_POS, HEAD_POS_NEXT, DEPE_POS_PREV),
            (HEAD_POS, DEPE_POS, HEAD_POS_PREV, DEPE_POS_PREV)
        )

    TEMPLATES = basic_templates() + extended_templates()

    @staticmethod
    def get_feature_keys(head: Token | int, dependant: Token | int, sentence: Sentence) -> list[str]:
        """
        returns feature keys for an edge.
        Example:
        ["HEAD_FORM_<I>,", "HEAD_POS_<PP>,", ...
        "DEPE_FORM_<cats>,DEPE_POS_<NN>,", ...
        ]
        """
        if isinstance(head, int):
            head = sentence[head]
        if isinstance(dependant, int):
            dependant = sentence[dependant]
        keys = []
        for template in TemplateWizard.TEMPLATES:
            key = ""
            for feature in template:
                token = TemplateWizard._get_relevant_token(feature, head, dependant, sentence)
                if FORM in feature:
                    key += f"{feature}_<{token.form}>,"
                else:
                    key += f"{feature}_<{token.pos}>,"
            keys.append(key)
        return keys

    @staticmethod
    def _get_relevant_token(feature, head: Token, dependent: Token, sentence: Sentence) -> Token:
        if DEPE in feature:
            relevant_token = dependent
        elif HEAD in feature:
            relevant_token = head
        else:  # BETW
            betw_id = (head.id_ + dependent.id_) / 2
            if abs(head.id_ - betw_id) == 1:
                relevant_token = sentence[int(betw_id)]
            else:
                relevant_token = Token.create_none()
        if NEXT in feature:
            relevant_token = sentence.get_token_or_null_token(relevant_token.id_ + 1)
        if PREV in feature:
            relevant_token = sentence.get_token_or_null_token(relevant_token.id_ - 1)
        return relevant_token

    @staticmethod
    def create_feature_dict(tree_bank: TreeBank, path: str) -> dict[str, int]:
        if os.path.isfile(path):
            print(f"Found dict at given path, loading from file...")
            return TemplateWizard.load_dict(path)
        print(f"Creating new feature dict from tree bank, this might take a while...")
        feature_dict = {}
        index = 0
        for sentence in tqdm(tree_bank):
            for token in sentence:
                feature_keys = TemplateWizard.get_feature_keys(token.head, token, sentence)
                for feature_key in feature_keys:
                    if feature_key not in feature_dict:
                        feature_dict[feature_key] = index
                        index += 1
        TemplateWizard.save_dict(feature_dict, path)
        return feature_dict

    @staticmethod
    def save_dict(feature_dict: dict, path: str):
        with open(path, 'w', encoding="utf-8") as f_out:
            for key, value in feature_dict.items():
                f_out.write(f"{key}\t{value}\n")

    @staticmethod
    def load_dict(path: str) -> dict[str, int]:
        feature_dict = {}
        with open(path, 'r', encoding="utf-8") as f_in:
            for line in f_in.readlines():
                key, value = line.strip().split("\t")
                feature_dict[key] = int(value)
        return feature_dict

class Templer:
    features: np.ndarray
    tree_bank: TreeBank
    feature_dict: dict[tuple[int], int]
    template_dict = {}

    def __init__(self, tree_bank: TreeBank, path: str = "", logging=False):
        raise DeprecationWarning("FUCK THIS SHIT...")
        self.tree_bank = tree_bank
        self.features = np.array([], dtype=int)
        self.logging = logging
        if not path:
            return
        if os.path.isfile(path):
            print("File at path found. Im trusting you that it matches the provided TreeBank.")
            self._load_features(path)
        else:
            print("Creating new feature set ...")
            self.create_feature_set()
            self.to_file(path)
        self.compute_feature_dict()

    def compute_feature_dict(self):
        self.feature_dict = {tuple(key): i for i, key in enumerate(self.features)}

    def create_feature_set(self):
        feature_list = []
        for sentence in self.tree_bank:  # tqdm(self.tree_bank):
            for token in sentence:
                for feature in self.extract_features(head=sentence[token.head], dependent=token, sentence=sentence):
                    feature_list.append(feature)
        self.features = np.stack(feature_list)
        print(f"feature set complete, {self.features.shape}")

    def to_file(self, path: str, overwrite=False):
        if not overwrite and os.path.isfile(path):
            print(f"File '{path}' already exists, skipping dict creation")
            return
        np.save(path, np.array(self.features), allow_pickle=True)

    def _load_features(self, path: str):
        self.features = np.load(path, allow_pickle=True)

    def extract_features(self, head: Token, dependent: Token, sentence: Sentence) -> np.ndarray:  # todo list[tuple]
        features = []
        for template, template_index in self.template_dict.items():
            key = np.array([template_index, -1, -1, -1, -1])
            for feature_position, feature in enumerate(template):
                feature = feature.name
                current_token = Templer._get_relevant_token(feature, head, dependent, sentence)
                try:
                    if POS in feature:
                        key[feature_position + 1] = self.tree_bank.pos_dict[current_token.pos]
                    else:  # FORM
                        key[feature_position + 1] = self.tree_bank.form_dict[current_token.form]
                except KeyError:
                    key[feature_position + 1] = -1
                    if current_token.pos not in self.tree_bank.pos_dict and self.logging:
                        print(f"'{current_token.pos}' is out of POS vocabulary.")
                    if current_token.form not in self.tree_bank.form_dict and self.logging:
                        print(f"'{current_token.form}' is out of FORM vocabulary.")
            features.append(key)
        return np.array(features, dtype=int)

    @staticmethod
    def _get_relevant_token(feature, head: Token, dependent: Token, sentence: Sentence) -> Token:
        if DEPE in feature:
            relevant_token = dependent
        elif HEAD in feature:
            relevant_token = head
        else:  # BETW
            if head.id_ - dependent.id_ == 2:
                relevant_token = sentence.get_token_or_null_token(head.id_ - 1)
            elif dependent.id_ - head.id_ == 2:
                relevant_token = sentence.get_token_or_null_token(dependent.id_ - 1)
            else:  # head and dependent dont hug one token
                relevant_token = sentence[-1]
        if "NEXT" in feature:
            relevant_token = sentence.get_token_or_null_token(relevant_token.id_ + 1)
        if "PREV" in feature:
            relevant_token = sentence.get_token_or_null_token(relevant_token.id_ - 1)
        return relevant_token

    def extract_feature_indices(self, tree: WDG, features: np.ndarray) -> np.ndarray:
        """
        extracts all the feature indices actually used in the tree, the feature array contains feature
        indices for every possible arc, but the tree only uses a subset of those. Method is used to train
        perceptron
        """
        # todo, confirm that this acctually works, it seem like the root arc is missing
        indices = []
        arcs = np.where(tree.data > 0)
        for arc_pair in zip(arcs[0], arcs[1]):
            for feature in features[arc_pair]:
                try:
                    indices.append(self.feature_dict[tuple(feature)])
                except KeyError:
                    pass
        return np.array(indices, dtype=int)

    @staticmethod
    def used_features(tree: WDG, features):
        indices = []
        arcs = np.where(tree.data > 0)
        for i,j in zip(arcs[0], arcs[1]):
            for feature in features[i][j]:
                indices.append(feature)
        return np.array(indices, dtype=int)
