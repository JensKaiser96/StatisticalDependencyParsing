from src.tools.CONLL06 import Token, Sentence, TreeBank


def read_file(path: str) -> TreeBank:
    tree_bank = TreeBank()
    current_sentence = Sentence()
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                current_sentence.add_token(Token(line))
            else:
                tree_bank.add_sentence(current_sentence)
                current_sentence = Sentence()
    return tree_bank


def write_file(path: str, tree_bank: TreeBank):
    with open(path, "w", encoding="utf-8") as f_out:
        for sentence in tree_bank.sentences:
            for token in sentence.tokens:
                f_out.write(str(token) + "\n")
            f_out.write("\n")
