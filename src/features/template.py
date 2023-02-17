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
from enum import Enum, auto
from itertools import combinations, permutations

import numpy as np
from tqdm import tqdm

from src.tools.CONLL06 import Token, Sentence, TreeBank
from src.DataStructures.graph import WeightedDirectedGraph as WDG

HEAD = "HEAD"
DEPE = "DEPE"
FORM = "FORM"
POS = "POS"


class BasicTemplates(Enum):
    HEAD_FORM = auto()
    DEPE_FORM = auto()
    HEAD_POS = auto()
    DEPE_POS = auto()

    @staticmethod
    def templates() -> list[tuple["BasicTemplates"]]:
        """
        generates all combinations with length 1 to 4 of features
        """
        grams = []
        for n in range(1, 5):
            for combination in combinations(BasicTemplates, n):
                grams.append(combination)
        return grams


class ExtendedTemplates(Enum):
    BETW_POS = auto()
    HEAD_POS_PREV = auto()
    HEAD_POS_NEXT = auto()
    DEPE_POS_PREV = auto()
    DEPE_POS_NEXT = auto()

    @staticmethod
    def templates() -> list[tuple]:
        return [
            (BasicTemplates.HEAD_POS, ExtendedTemplates.BETW_POS, BasicTemplates.DEPE_POS),
            (BasicTemplates.HEAD_POS, BasicTemplates.DEPE_POS, ExtendedTemplates.HEAD_POS_NEXT, ExtendedTemplates.DEPE_POS_NEXT),
            (BasicTemplates.HEAD_POS, BasicTemplates.DEPE_POS, ExtendedTemplates.HEAD_POS_PREV, ExtendedTemplates.DEPE_POS_NEXT),
            (BasicTemplates.HEAD_POS, BasicTemplates.DEPE_POS, ExtendedTemplates.HEAD_POS_NEXT, ExtendedTemplates.DEPE_POS_PREV),
            (BasicTemplates.HEAD_POS, BasicTemplates.DEPE_POS, ExtendedTemplates.HEAD_POS_PREV, ExtendedTemplates.DEPE_POS_PREV)
        ]


TemplateCollection = BasicTemplates.templates() + ExtendedTemplates.templates()


class Templer:
    features: np.ndarray
    tree_bank: TreeBank
    template_dict = {t: i for i, t in enumerate(TemplateCollection)}

    def __init__(self, tree_bank: TreeBank, path: str = ""):
        self.tree_bank = tree_bank
        self.features = np.array([])
        if not path:
            return
        if os.path.isfile(path):
            print("File at path found. Im trusting you that it matches the provided TreeBank.")
            self._load_features(path)
        else:
            print("Creating new feature set ...")
            self.create_feature_set()
            self.to_file(path)

    def feature_dict(self) -> dict:
        return {key: i for i, key in enumerate(self.features)}

    def create_feature_set(self):
        for sentence in tqdm(self.tree_bank):
            for token in sentence:
                for feature in self.extract_features(head=sentence[token.head], dependent=token, sentence=sentence):
                    self.features = np.append(self.features, feature)

    def to_file(self, path: str, overwrite=False):
        if not overwrite and os.path.isfile(path):
            print(f"File '{path}' already exists, skipping dict creation")
            return
        np.save(path, np.array(self.features), allow_pickle=True)

    def _load_features(self, path: str):
        self.features = np.load(path, allow_pickle=True)

    def extract_features(self, head: Token, dependent: Token, sentence: Sentence) -> np.ndarray:
        features = []
        for template, template_index in self.template_dict.items():
            try:
                key = np.array([template_index, -1, -1, -1, -1])
                for feature_position, feature in enumerate(template):
                    feature = feature.name
                    current_token = Templer._get_relevant_token(feature, head, dependent, sentence)
                    if POS in feature:
                        key[feature_position + 1] = self.tree_bank.pos_dict[current_token.pos]
                    else:  # FORM
                        key[feature_position + 1] = self.tree_bank.form_dict[current_token.form]
                features.append(key)
            except KeyError:
                if current_token.pos not in self.tree_bank.pos_dict:
                    print(f"'{current_token.pos}' is out of POS vocabulary.")
                if current_token.form not in self.tree_bank.form_dict:
                    print(f"'{current_token.form}' is out of FORM vocabulary.")
        return np.array(features)

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

    @staticmethod
    def extract_feature_indices(tree: WDG, features: np.ndarray) -> np.ndarray:
        """
        extracts all the feature indices actually used in the tree, the feature array contains feature
        indices for every possible arc, but the tree only uses a subset of those. Method is used to train
        perceptron
        """
        indices = np.array([])
        arcs = np.where(tree.data > 0)
        for arc_pair in zip(arcs[0], arcs[1]):
            np.append(indices, features[arc_pair])
        return indices


class Templator:
    def __init__(self, form_dict_path: str, pos_dict_path: str):
        raise DeprecationWarning("Not working...")
        self.form_dict = self.read_dict(form_dict_path)
        self.pos_dict = self.read_dict(pos_dict_path)
        self.form_dict_size = len(self.form_dict)
        self.pos_dict_size = len(self.pos_dict)
        self.templates = BasicTemplates.templates() + ExtendedTemplates.templates()

        #self.start_index = dict()
        self.feature_vector_size = 0
        #self.set_feature_start_index()

        self.template_offset = dict()
        self.template_offsets()


    @staticmethod
    def read_dict(path: str) -> dict[str: int]:
        with open(path, mode='r', encoding="utf-8") as f_in:
            return {w.strip(): i for i, w in enumerate(f_in.readlines())}

    def set_feature_start_index(self):
        offset = 0
        for template in self.templates:
            for feature in template:
                self.start_index[(template, feature)] = offset
                feature_size = self.form_dict_size if "FORM" in feature.name else self.pos_dict_size
                offset += feature_size
        self.feature_vector_size = offset

    def form_index(self, token_form: str) -> int:
        try:
            return self.form_dict[token_form]
        except KeyError:
            return self.form_dict["_NONE_"]

    def pos_index(self, token_pos: str) -> int:
        try:
            return self.pos_dict[token_pos]
        except KeyError:
            return self.pos_dict["_NONE_"]

    def template_size(self, template: tuple[BasicTemplates]) -> int:
        size = 1
        for feature in template:
            if "FORM" in feature.name:
                size *= self.form_dict_size
            elif "POS" in feature.name:
                size *= self.pos_dict_size
            else:
                raise ValueError(f"Feature '{feature}' does not contain 'FORM' or 'POS'")
        return size

    def template_offsets(self):
        offset = 0
        for template in self.templates:
            self.template_offset[template] = offset
            offset += self.template_size(template)
        self.feature_vector_size = offset

    def create_features(self, sentence) -> np.ndarray:
        """
        returns an np.ndarray of shape n x n x 18 (num templates) with indexes for 1s
        -1 is for no index, (i,i),
        """
        n = len(sentence)
        features = np.ones((n, n, len(self.templates)), dtype=int) * - 1
        pairs = permutations(range(n), 2)
        for head_index, dependent_index in pairs:
            features[head_index, dependent_index] = self.create_feature_pair(sentence, head_index, dependent_index)
        return features

    def create_feature_pair(self, sentence: Sentence, head_index: int, dependant_index: int):
        features = []
        for template in self.templates:
            feature_index = 0
            for feature in template:
                if "HEAD" in feature.name:
                    token_index = head_index
                elif "DEPE" in feature.name:
                    token_index = dependant_index
                else:  # BETW
                    if head_index - dependant_index == 2:
                        token_index = head_index - 1
                    elif dependant_index - head_index == 2:
                        token_index = dependant_index - 1
                    else:  # head and dependent dont hug one token
                        token_index = -1

                if "NEXT" in feature.name:
                    token_index += 1
                elif "PREV" in feature.name:
                    token_index -= 1

                token = sentence[token_index]

                if "FORM" in feature.name:  # todo fix bug, indices are too high
                    feature_index += self.start_index[template, feature] + self.form_index(token.form)
                else:  # POS
                    feature_index += self.start_index[template, feature] + self.pos_index(token.pos)
            features.append(feature_index + self.template_offset[template])
        return features

    @staticmethod
    def extract_feature_indices(tree: WDG, features: np.ndarray) -> np.ndarray:
        """
        extracts all the feature indices actually used in the tree, the feature array contains feature
        indices for every possible arc, but the tree only uses a subset of those. Method is used to train
        perceptron
        """
        indices = np.array([])
        arcs = np.where(tree.data > 0)
        for arc_pair in zip(arcs[0], arcs[1]):
            np.append(indices, features[arc_pair])
        return indices
