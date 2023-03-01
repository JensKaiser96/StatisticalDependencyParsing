from src.models.perceptron import Perceptron
from src.tools.CONLL06 import TreeBank
from src.tools.measurements import UAS

english_treebank = {
    "train": "data/english/train/wsj_train.conll06",
    "dev": "data/english/dev/wsj_dev.conll06.gold",
    "test": "data/english/train/wsj_test.conll06.blind",
    "model": "data/models/english"
}
german_treebank = {
    "train": "data/german/train/tiger-2.2.train.conll06",
    "dev": "data/german/dev/tiger-2.2.dev.conll06.gold",
    "test": "data/german/test/tiger-2.2.test.conll06.blind",
    "model": "data/models/german"
}

treebank = german_treebank
load = False


def main():
    train_treebank = TreeBank.from_file(treebank["train"])
    dev_treebank = TreeBank.from_file(treebank["dev"])
    test_treebank = TreeBank.from_file(treebank["test"])

    model = Perceptron(train_treebank, treebank["model"])
    if load:
        model.load_weights()

    best_dev_score = 0
    dev_score = 0
    max_tries = 3
    tries = 0

    while dev_score > best_dev_score or tries < max_tries:
        model.train()

        dev_pred = model.annotate(dev_treebank)
        dev_score = UAS(dev_treebank, dev_pred)

        print(f"dev UAS: {dev_score}, best: {best_dev_score}")

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            tries = 0
            test_pred = model.annotate(test_treebank)
            test_pred.to_file(treebank["test"] + ".pred")
        else:
            tries += 1


if __name__ == '__main__':
    main()
