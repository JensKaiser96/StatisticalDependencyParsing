from src.features.template import Templer
from src.models.perceptron import Perceptron
from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.tools.CONLL06 import TreeBank, Sentence, Token
from src.tools.measurements import UAS, LAS


def main():
    tb_path = "data/english/train/wsj_train.only-projective.first-1k.conll06"
    form_dict_path = tb_path + ".form.dict"
    pos_dict_path = tb_path + ".pos.dict"
    tb = TreeBank.from_file(tb_path, form_dict_path, pos_dict_path)

    t = Templer(tb, tb_path + ".feature_set.npy")
    # t.create_feature_set()
    # t.to_file()

    tree = WDG().add_edge(0, 2).add_edge(2, 1).add_edge(2, 3)
    # print(tree)

    model = Perceptron(t)
    features = model._create_feature_vector(
        Sentence().add_token(
            Token(1, form="I", pos="PRP")).add_token(
            Token(2, form="love", pos="VB")).add_token(
            Token(3, form="cats", pos="NNP")
        )
    )
    print(features)

    model.train(tb, 5)


if __name__ == '__main__':
    main()
