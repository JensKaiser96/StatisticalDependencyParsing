from src.features.template import TemplateWizard
from src.models.perceptron import Perceptron
from src.DataStructures.graph import WeightedDirectedGraph as WDG
from src.tools.CONLL06 import TreeBank, Sentence, Token
from src.tools.measurements import UAS, LAS


def main():
    tb_path = "data/english/train/wsj_train.first-1k.conll06"
    tb = TreeBank.from_file(tb_path)
    feature_dict = TemplateWizard.create_feature_dict(tb, tb_path + ".feature_dict")

    tree_ = WDG().add_edge(0, 2).add_edge(2, 1).add_edge(2, 3)
    # print(tree)

    model = Perceptron(feature_dict)
    sentence = Sentence().add_token(
        Token(1, form="I", pos="PRP", head=2)).add_token(
        Token(2, form="love", pos="VB", head=0)).add_token(
        Token(3, form="cats", pos="NNP", head=2))

    #features = model._create_feature_vector(sentence)
    # print(features)
    tree = model.predict(sentence)

    #indices = t.extract_feature_indices(tree, features)
    #print(indices)
    # tree.draw()

    model.train(tb, 500, save_path="data/model/wsj_op_1k")


if __name__ == '__main__':
    main()
