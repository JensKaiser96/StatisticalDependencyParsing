import argparse

from src.features.template import TemplateWizard
from src.models.perceptron import Perceptron
from src.tools.CONLL06 import TreeBank
from src.tools.measurements import UAS


def annotate_tree_bank(tree_bank_path: str, feature_dict_path: str, model_path: str):
    tree_bank = TreeBank.from_file(tree_bank_path)
    gold_tree_bank = TreeBank.from_file(tree_bank_path)
    feature_dict = TemplateWizard.create_feature_dict(None, feature_dict_path)
    model = Perceptron(feature_dict)
    model.load_weights(model_path)
    model.annotate(tree_bank)
    print(f"The model achieved a UAS of : {UAS(gold_tree_bank, tree_bank)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluate_model.py',
        description='Evaluates a model on a gold tree bank',
        epilog='Bottom Text')
    parser.add_argument('tree_bank_path')
    parser.add_argument('feature_dict_path')
    parser.add_argument('model_path')
    args = parser.parse_args()
    evaluate_model(args.tree_bank_path, args.feature_dict_path, args.model_path)
