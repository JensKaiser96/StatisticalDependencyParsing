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
from tqdm import tqdm

from src.tools.CONLL06 import Token, Sentence, TreeBank
from src.DataStructures.graph import WeightedDirectedGraph as WDG

# dots at the end prevent substring issues, ie HEAD. is not substring of HEADBIGRAM.
HEAD = "HEAD."
DEPE = "DEPE."
FORM = "FORM"
POS = "POS"

HEAD_FORM = f"{HEAD}_{FORM}"
HEAD_POS = f"{HEAD}_{POS}"
DEPE_FORM = f"{DEPE}_{FORM}"
DEPE_POS = f"{DEPE}_{POS}"

BASIC_FEATURES = (HEAD_FORM, HEAD_POS, DEPE_FORM, DEPE_POS)

BETW = "BETW."
NEXT = "NEXT."
PREV = "PREV."

BETW_POS = f"{BETW}_{POS}"
HEAD_POS_NEXT = f"{HEAD}_{POS}_{NEXT}"
HEAD_POS_PREV = f"{HEAD}_{POS}_{PREV}"
DEPE_POS_NEXT = f"{DEPE}_{POS}_{NEXT}"
DEPE_POS_PREV = f"{DEPE}_{POS}_{PREV}"

SIBLING = "SIBLING."
GRANDPARENT = "GRANDPARENT."
HEADBIGRAM = "HEADBIGRAM."
GRANDSIBLING = "GRANDSIBLING."
GRANDGRANDPARENT = "GRANDGRANDPARENT."
INNERAUNTI = "INNERAUNTI."
OUTERAUNTI = "OUTERAUNTI."

PARTIAL_HIGHER_ORDER_FEATURES = [SIBLING, GRANDPARENT, HEADBIGRAM, GRANDSIBLING, GRANDGRANDPARENT, INNERAUNTI, OUTERAUNTI]
HIGHER_ORDER_FEATURES = [f"{hof}_{POS}" for hof in PARTIAL_HIGHER_ORDER_FEATURES]


class TemplateWizard:
    @staticmethod
    def basic_templates() -> tuple[tuple[str]]:
        """
        generates all combinations with length 1 to 4 of features
        """
        grams = []
        for n in range(1, 5):
            for combination in combinations(BASIC_FEATURES, n):
                grams.append(combination)
        return tuple(grams)

    @staticmethod
    def extended_templates() -> tuple[tuple[str]]:
        return (
            (HEAD_POS, BETW_POS, DEPE_POS),
            (HEAD_POS, DEPE_POS, HEAD_POS_NEXT, DEPE_POS_NEXT),
            (HEAD_POS, DEPE_POS, HEAD_POS_PREV, DEPE_POS_NEXT),
            (HEAD_POS, DEPE_POS, HEAD_POS_NEXT, DEPE_POS_PREV),
            (HEAD_POS, DEPE_POS, HEAD_POS_PREV, DEPE_POS_PREV)
        )

    @staticmethod
    def higher_order_templates():
        # kinda hacky, todo fix, get basic templates by not copying code or
        # getting a name error due to TemplateWizard.basic_template() call
        basic_templates = []
        for n in range(1, 5):
            for combination in combinations(BASIC_FEATURES, n):
                basic_templates.append(combination)
        templates = []
        for higher_order_feature in HIGHER_ORDER_FEATURES:
            for basic_template in basic_templates:
                templates.append(tuple([*basic_template, higher_order_feature]))
        return tuple(templates)
    TEMPLATES = basic_templates() + extended_templates() + higher_order_templates()

    @staticmethod
    def get_feature_keys(head: Token | int, dependant: Token | int, sentence: Sentence, tree: WDG = None) -> list[str]:
        """
        returns feature keys for an edge.
        Example:
        ["HEAD_FORM_<I>,", "HEAD_POS_<PP>,", ...
        "DEPE_FORM_<cats>,DEPE_POS_<NN>,", ...
        ]
        """
        if not isinstance(head, Token):
            head = sentence.get_token_or_none_token(head)
        if not isinstance(dependant, Token):
            dependant = sentence.get_token_or_none_token(dependant)
        keys = []
        for template in TemplateWizard.TEMPLATES:
            key = f"|{head.id_ - dependant.id_}|_"
            for feature in template:
                token = TemplateWizard._get_relevant_token(feature, head, dependant, sentence, tree)
                if FORM in feature:
                    key += f"{feature}_<{token.form}>,"
                elif DEPE in feature:
                    key += f"{feature}_<{token.pos}>,"
            keys.append(key)
        return keys

    @staticmethod
    def _get_relevant_token(feature, head: Token, dependent: Token, sentence: Sentence, tree: WDG = None) -> Token:
        """
        This function could use some more ifs...
        """
        if DEPE in feature:
            relevant_token = dependent
        elif HEAD in feature:
            relevant_token = head
        else:  # BETW
            betw_id = (head.id_ + dependent.id_) / 2
            if abs(head.id_ - betw_id) == 1:
                relevant_token = sentence.get_token_or_none_token(int(betw_id))
            else:
                relevant_token = Token.create_none()
        if NEXT in feature:
            relevant_token = sentence.get_token_or_none_token(relevant_token.id_ + 1)
        if PREV in feature:
            relevant_token = sentence.get_token_or_none_token(relevant_token.id_ - 1)
        if SIBLING in feature:
            sibling = tree.get_next_sibling(relevant_token.id_)
            return sentence.get_token_or_none_token(sibling)
        if GRANDPARENT in feature:
            if not tree.has_head(relevant_token.id_):
                return Token.create_none()
            parent = tree.get_head(relevant_token.id_)
            if not tree.has_head(parent):
                return Token.create_none()
            grandparent = tree.get_head(parent)
            return sentence.get_token_or_none_token(grandparent)
        if HEADBIGRAM in feature:
            if head.id_ - 1 not in range(len(sentence)) or dependent.id_ + 1 not in range(len(sentence)):
                return Token.create_none()
            if not tree.get_edge_weight(head.id_-1, dependent.id_+1):
                return Token.create_none()
            if HEAD in feature:
                return sentence.get_token_or_none_token(head.id_-1)
            if DEPE in feature:
                return sentence.get_token_or_none_token(dependent.id_+1)
        if GRANDSIBLING in feature:
            # not sure what that is ._. like a child, but a brother?!?!?
            return Token.create_none()
        if GRANDGRANDPARENT in feature:
            if not tree.has_head(relevant_token.id_):
                return Token.create_none()
            parent = tree.get_head(relevant_token.id_)
            if not tree.has_head(parent):
                return Token.create_none()
            grandparent = tree.get_head(parent)
            if not tree.has_head(grandparent):
                return Token.create_none()
            grandgrandparent = tree.get_head(grandparent)
            return sentence.get_token_or_none_token(grandgrandparent)
        if OUTERAUNTI in feature:
            if not tree.has_head(relevant_token):
                return Token.create_none()
            parent = tree.get_head(relevant_token)
            if not tree.get_dependent_ids(relevant_token):
                return Token.create_none()
            child = max(tree.get_dependent_ids(relevant_token))
            outeraunti = max(tree.get_dependent_ids(parent))
            if child > outeraunti:
                return Token.create_none()
            return sentence.get_token_or_none_token(outeraunti)
        if INNERAUNTI in feature:
            if not tree.has_head(relevant_token):
                return Token.create_none()
            parent = tree.get_head(relevant_token)
            if not tree.get_dependent_ids(relevant_token):
                return Token.create_none()
            child = max(tree.get_dependent_ids(relevant_token))
            outeraunti = max(tree.get_dependent_ids(parent))
            if child < outeraunti:
                return Token.create_none()
            return sentence.get_token_or_none_token(outeraunti)
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
            tree = sentence.to_tree()
            for token in sentence:
                for dependent in tree.get_dependent_ids(token.id_):
                    feature_keys = TemplateWizard.get_feature_keys(token, dependent, sentence, tree)
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
