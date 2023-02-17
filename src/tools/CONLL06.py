import os
from dataclasses import dataclass
import random
from typing import List

MISSING_VALUE_CHAR = "_"
ROOT_VALUE_PLACEHOLDER = "_ROOT_"
NONE_VALUE_PLACEHOLDER = "_NONE_"

SPECIAL_VALUES = {MISSING_VALUE_CHAR, ROOT_VALUE_PLACEHOLDER, NONE_VALUE_PLACEHOLDER}


@dataclass
class Token:
    id_: int
    form: str
    lemma: str
    pos: str
    xpos: str
    morph: str
    head: int
    rel: str

    def __init__(self,
                 id_: int | str,
                 form: str = MISSING_VALUE_CHAR,
                 lemma: str = MISSING_VALUE_CHAR,
                 pos: str = MISSING_VALUE_CHAR,
                 xpos: str = MISSING_VALUE_CHAR,
                 morph: str = MISSING_VALUE_CHAR,
                 head: int | str = -1,
                 rel: str = MISSING_VALUE_CHAR
                 ):
        if isinstance(id_, str):
            self.id_ = int(id_)
        else:
            self.id_ = id_
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.xpos = xpos
        self.morph = morph
        if isinstance(head, str):
            self.head = self.head_from_str(head)
        else:
            self.head = head
        self.rel = rel

    @staticmethod
    def from_str(line: str):
        id_, form, lemma, pos, xpos, morph, head, rel, *_ = line.strip().split('\t')
        return Token(id_=id_, form=form, lemma=lemma, pos=pos, xpos=xpos, morph=morph, head=head, rel=rel)

    @staticmethod
    def head_from_str(s: str) -> int:
        if s in SPECIAL_VALUES:
            return -1
        else:
            return int(s)

    def head_to_str(self) -> str:
        if self.head == -1:
            return MISSING_VALUE_CHAR
        else:
            return str(self.head)

    @staticmethod
    def create_root() -> "Token":
        root_token = Token(0, *(ROOT_VALUE_PLACEHOLDER,)*7)
        return root_token

    @staticmethod
    def create_none() -> "Token":
        none_token = Token(-1, *(ROOT_VALUE_PLACEHOLDER,)*7)
        return none_token

    def __str__(self):
        return (f"{self.id_}\t"
                f"{self.form}\t"
                f"{self.lemma}\t"
                f"{self.pos}\t"
                f"{self.xpos}\t"
                f"{self.morph}\t"
                f"{self.head_to_str()}\t"
                f"{self.rel}\t"
                f"{MISSING_VALUE_CHAR}\t"
                f"{MISSING_VALUE_CHAR}")


@dataclass
class Sentence:
    tokens: List[Token]

    def __init__(self, tokens: List[Token] = None):
        self.tokens = tokens if tokens else [Token.create_root()]

    def __getitem__(self, index) -> Token:
        return self.tokens[index]

    def get_token_or_null_token(self, index):
        """
        returns Token at position <index>, if index is out of bounds, the special none token is returned
        """
        if index not in range(len(self)):
            return Token.create_none()
        else:
            return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def add_token(self, token: Token):
        self.tokens.append(token)
        return self


@dataclass
class TreeBank:
    sentences: List[Sentence]
    form_dict: dict[str, int]
    pos_dict: dict[str, int]
    form_set: set[str]
    pos_set: set[str]

    def __init__(self, sentences: List[Sentence] = None):
        self.sentences = sentences if sentences else []
        self.calculate_form_set()
        self.calculate_pos_set()
        self.calculate_form_dict()
        self.calculate_pos_dict()

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

    def calculate_form_dict(self, path=""):
        if path:
            self.load_form_dict(path)
            self.calculate_form_set()
        self.form_dict = {form: i for i, form in enumerate(self.form_set)}

    def calculate_pos_dict(self, path=""):
        if path:
            self.load_pos_dict(path)
            self.calculate_pos_set()
        self.pos_dict = {pos: i for i, pos in enumerate(self.pos_set)}

    @staticmethod
    def from_file(tree_bank_path: str, form_dict_path: str = "", pos_dict_path: str = ""):
        tree_bank = TreeBank()
        current_sentence = Sentence()
        with open(tree_bank_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    current_sentence.add_token(Token.from_str(line))
                else:
                    tree_bank.add_sentence(current_sentence)
                    current_sentence = Sentence()
        if form_dict_path:
            tree_bank.load_form_dict(form_dict_path)
        else:
            tree_bank.calculate_form_dict()
        if pos_dict_path:
            tree_bank.load_pos_dict(pos_dict_path)
        else:
            tree_bank.calculate_pos_dict()
        return tree_bank

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f_out:
            for sentence in self.sentences:
                for token in sentence.tokens:
                    f_out.write(str(token) + "\n")
                f_out.write("\n")

    def save_form_dict(self, path: str, overwrite=False):
        self.set_to_file(self.form_set, path, overwrite)

    def save_pos_dict(self, path: str, overwrite=False):
        self.set_to_file(self.pos_set, path, overwrite)

    def load_form_dict(self, path: str):
        with open(path, mode='r', encoding="utf-8") as f_in:
            self.form_set = {form.strip() for form in f_in.readlines()}
        self.calculate_form_dict()

    def load_pos_dict(self, path: str):
        with open(path, mode='r', encoding="utf-8") as f_in:
            self.pos_set = {pos.strip() for pos in f_in.readlines()}
        self.calculate_pos_dict()

    @staticmethod
    def set_to_file(d: set, path: str, overwrite=False):
        if not overwrite and os.path.isfile(path):
            print(f"File '{path}' already exists, skipping dict creation")
            return
        with open(path, mode='w', encoding="utf-8") as f_out:
            f_out.write("\n".join(d))

    @property
    def tokens(self) -> List[Token]:
        return [token for sentence in self.sentences for token in sentence.tokens]

    def calculate_pos_set(self):
        self.pos_set = set([token.pos for token in self.tokens]).union(SPECIAL_VALUES)

    def calculate_form_set(self):
        self.form_set = set([token.form for token in self.tokens]).union(SPECIAL_VALUES)

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def shuffle(self):
        random.shuffle(self.sentences)
        return self
