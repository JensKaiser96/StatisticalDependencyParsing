import argparse

from src.features.template import TemplateWizard
from src.models.perceptron import Perceptron
from src.tools.CONLL06 import TreeBank


def annotate_tree_bank(tree_bank_path: str, feature_dict_path: str, model_path: str, out_path: str):
    tree_bank = TreeBank.from_file(tree_bank_path)
    feature_dict = TemplateWizard.create_feature_dict(None, feature_dict_path)
    model = Perceptron(feature_dict)
    model.load_weights(model_path)
    model.annotate(tree_bank)
    tree_bank.to_file(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='annotate_tree_bank.py',
        description='Annotates a give tree bank',
        epilog='Bottom Text')
    parser.add_argument('tree_bank_path')
    parser.add_argument('feature_dict_path')
    parser.add_argument('model_path')
    parser.add_argument('out_path')
    args = parser.parse_args()
    annotate_tree_bank(args.tree_bank_path, args.feature_dict_path, args.model_path, args.out_path)
