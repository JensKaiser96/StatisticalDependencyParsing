from abc import ABC, abstractmethod

from src.tools.CONLL06 import Token, TreeBank


class Measure(ABC):
    def __call__(self, gold_data: TreeBank, test_data: TreeBank) -> float:
        num_correct = 0
        num_tokens = 0
        for gold_token, test_token in zip(gold_data.tokens, test_data.tokens):
            if gold_token.id != test_token.id or gold_token.form != test_token.form:
                raise TypeError("TreeBanks do not have a matching sequence of tokens/ids")
            if self.required_equality(gold_token, test_token):
                num_correct += 1
            num_tokens += 1
        return num_correct / num_tokens

    @staticmethod
    @abstractmethod
    def required_equality(token1: Token, token2: Token):
        pass


class UASMeasure(Measure):
    @staticmethod
    def required_equality(token1: Token, token2: Token):
        return token1.head == token2.head


class LASMeasure(Measure):
    @staticmethod
    def required_equality(token1: Token, token2: Token):
        return token1.head == token2.head and token1.rel == token2.rel


UAS = UASMeasure()
LAS = LASMeasure()
