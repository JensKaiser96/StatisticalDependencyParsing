from dataclasses import dataclass
from typing import List

MISSING_VALUE_CHAR = "_"


@dataclass
class Token:
    id: int
    form: str = MISSING_VALUE_CHAR
    lemma: str = MISSING_VALUE_CHAR
    pos: str = MISSING_VALUE_CHAR
    xpos: str = MISSING_VALUE_CHAR
    morph: str = MISSING_VALUE_CHAR
    head: int = MISSING_VALUE_CHAR
    rel: str = MISSING_VALUE_CHAR

    def __init__(self, line: str = ""):
        if not line:
            return
        line = line.strip()
        id, form, lemma, pos, xpos, morph, head, rel, *_ = line.split('\t')
        self.id = int(id)
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.xpos = xpos
        self.morph = morph
        self.head_from_str(head)
        self.rel = rel

    def head_from_str(self, s: str):
        if s == MISSING_VALUE_CHAR:
            self.head = -1
        else:
            self.head = int(s)

    def head_to_str(self) -> str:
        if self.head == -1:
            return MISSING_VALUE_CHAR
        else:
            return str(self.head)

    @staticmethod
    def create_root() -> "Token":
        root = Token()
        root.id = 0
        return Token()

    def __str__(self):
        return (f""
                f"{self.id}\t"
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
        self.tokens = tokens if tokens else []

    def add_token(self, token: Token):
        self.tokens.append(token)


@dataclass
class TreeBank:
    sentences: List[Sentence]

    @property
    def tokens(self) -> List[Token]:
        return [token for sentence in self.sentences for token in sentence.tokens]

    def __init__(self, sentences: List[Sentence] = None):
        self.sentences = sentences if sentences else []

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
